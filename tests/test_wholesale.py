"""
Tests for the autonomous wholesaling platform.

All deterministic and offline: the company is driven by `tick()` directly
(no threads, no sleeps) with seeded RNGs, and with no ANTHROPIC_API_KEY the
agents use their heuristic fallbacks — so outcomes are reproducible.
"""

from __future__ import annotations

import json

from wholesale import config
from wholesale.core.company import Company
from wholesale.core.models import (
    Deal, Property, Seller, Stage, PIPELINE_ORDER,
)
from wholesale.integrations import IntegrationHub
from wholesale.contracts import ContractGenerator
from wholesale.compliance import ComplianceGate, gate as gate_mod
from wholesale.sourcing import build_rows, KEYWORDS
from wholesale.sourcing.county_records import dallas_notice_urls, public_search_url
from wholesale.outreach import draft, TEMPLATES
from wholesale.disposition import PLATFORMS, by_payout
from wholesale.sourcing.providers import get_provider, SyntheticProvider


def _make_deal(state="TX", price=300_000, market=400_000) -> Deal:
    prop = Property(
        address="123 Test St", city="Dallas", state=state, zip="75201",
        metro="TX-Dallas", beds=3, baths=2.0, sqft=1500, year_built=1980,
        property_type="SFR", distress="pre-foreclosure", est_market_value=market,
    )
    seller = Seller(name="Pat Test", motivation="distress", asking_price=market,
                    reachable_via="phone", flexibility=0.4, walk_floor=int(market * 0.6))
    return Deal(prop=prop, seller=seller, stage=Stage.SOURCED)


def test_company_boots_with_full_roster():
    co = Company(cfg=config, seed=7)
    assert len(co.agents) == 8
    assert "LEGAL" in co.agents and "BIZDEV" in co.agents
    # Every non-terminal stage has exactly one owning agent.
    for stage in PIPELINE_ORDER:
        if stage.is_terminal:
            continue
        assert stage in co._stage_owner, f"no owner for {stage}"


def test_snapshot_is_json_serializable_and_consistent():
    co = Company(cfg=config, seed=7)
    for _ in range(20):
        co.tick()
    snap = co.snapshot()
    # Round-trips through JSON (what the dashboard relies on).
    json.dumps(snap)
    # Stage counts account for every deal exactly once.
    assert sum(snap["counts"].values()) == len(co.deals)
    assert len(snap["agents"]) == 8
    # New panels are present and serialisable.
    assert "contacts" in snap and "dispo_platforms" in snap
    assert len(snap["dispo_platforms"]) == len(PLATFORMS)


def test_pipeline_closes_deals_deterministically():
    co = Company(cfg=config, seed=7)
    for _ in range(80):
        co.tick()
    # With seed 7 the chain reaches funded closings.
    assert co.closed_count >= 1
    # Revenue equals the sum of collected assignment fees on closed deals.
    booked = sum(d.assignment_fee for d in co.deals if d.stage == Stage.CLOSED_WON)
    assert co.revenue == booked
    # Some leads die — that's realistic, not a bug.
    assert any(d.stage == Stage.DEAD for d in co.deals)


def test_underwriting_uses_70_percent_rule():
    co = Company(cfg=config, seed=1)
    deal = _make_deal()
    co.agents["ANALYST"].handle  # ensure attribute exists
    deal.stage = Stage.UNDERWRITING
    co.agents["ANALYST"].handle(deal)
    uw = deal.uw
    assert uw.arv > 0 and uw.repair_estimate > 0
    # MAO must equal ARV*rule - repairs - target fee (no hallucinated numbers).
    expected_mao = round(uw.arv * config.ARV_RULE - uw.repair_estimate
                         - config.TARGET_ASSIGNMENT_FEE, -3)
    assert uw.mao == expected_mao
    assert deal.stage in (Stage.OUTREACH, Stage.DEAD)


def test_ceo_gate_holds_then_releases_high_value_deal():
    co = Company(cfg=config, seed=2)
    deal = _make_deal(price=300_000, market=400_000)
    deal.stage = Stage.UNDER_CONTRACT
    deal.contract_price = 300_000
    deal.assignment_fee = 20_000
    deal.needs_ceo = True  # above the $250k threshold
    co.deals.append(deal)

    co.tick()  # compliance reviews and escalates
    queue_ids = [d["id"] for d in co.snapshot()["ceo_queue"]]
    assert deal.id in queue_ids
    assert deal.ceo_decision is None

    assert co.decide(deal.id, "approved") is True
    co.tick()  # compliance releases the approved deal
    assert deal.stage in (Stage.DISPOSITION, Stage.ASSIGNED,
                          Stage.CLOSING, Stage.CLOSED_WON, Stage.DEAD)


def test_ceo_reject_kills_deal():
    co = Company(cfg=config, seed=3)
    deal = _make_deal(price=300_000)
    deal.stage = Stage.UNDER_CONTRACT
    deal.contract_price = 300_000
    deal.needs_ceo = True
    co.deals.append(deal)
    co.tick()
    assert co.decide(deal.id, "rejected") is True
    co.tick()
    assert deal.stage == Stage.DEAD


def test_integrations_are_dry_run_by_default():
    hub = IntegrationHub(armed=False)
    res = hub.crm_upsert(deal_id=1, name="x", stage="sourced", amount=1, contact="c")
    assert res.armed is False
    assert len(hub.outbox) == 1
    # Email/slack also record without sending.
    hub.email_seller(to="a@b.com", subject="s", body="hello")
    hub.slack_alert(channel="deals", text="hi")
    assert len(hub.outbox) == 3
    assert all(r.armed is False for r in hub.outbox)


def test_contract_generator_marks_drafts_and_fills_fields():
    deal = _make_deal()
    deal.contract_price = 120_000
    deal.assignment_fee = 30_000
    gen = ContractGenerator(buyer_entity="Keystone Property Partners LLC")
    psa = gen.purchase_agreement(deal)
    asg = gen.assignment_agreement(deal)
    # Every generated doc must carry the attorney-review banner.
    assert "ATTORNEY REVIEW REQUIRED" in psa
    assert "ATTORNEY REVIEW REQUIRED" in asg
    # Key economics are filled, not left as placeholders.
    assert "120,000" in psa
    assert "30,000" in asg
    assert "Keystone Property Partners LLC" in psa
    # Assignability + equitable-interest disclosure are present.
    assert "ASSIGNABLE" in psa and "equitable interest" in psa


def test_preforeclosure_packet_has_disclosures_and_legends():
    deal = _make_deal(state="GA")
    deal.contract_price = 120_000
    deal.assignment_fee = 30_000
    gen = ContractGenerator(buyer_entity="Keystone LLC")
    packet = gen.preforeclosure_packet(deal)
    assert set(packet) >= {"direct_mail_letter", "disclosure_addendum",
                           "purchase_agreement", "assignment_agreement"}
    # GA mailers must carry the SB 90 solicitation legend.
    assert "THIS IS A SOLICITATION" in packet["direct_mail_letter"]
    # Every disclosure is a flagged draft (not presented as legal advice).
    assert "NOT LEGAL ADVICE" in packet["disclosure_addendum"]
    # FL disclosure cites the equity-purchaser statute.
    fl = ContractGenerator().disclosure_addendum(_make_deal(state="FL"))
    assert "501.1377" in fl


def test_compliance_gate_blocks_until_fully_attested():
    g = ComplianceGate()
    # Default posture is BLOCKED.
    assert g.can_contact_real_owners("TX") is False
    assert set(g.missing("TX")) == set(gate_mod.REQUIRED_ATTESTATIONS)
    # Partial attestation still blocks.
    g.attest("TX", entity_formed=True, attorney_engaged=True)
    assert g.can_contact_real_owners("TX") is False
    # Full attestation clears that state only.
    g.attest("TX", **{a: True for a in gate_mod.REQUIRED_ATTESTATIONS})
    assert g.can_contact_real_owners("TX") is True
    assert g.can_contact_real_owners("FL") is False  # per-state isolation


def test_compliance_gate_rejects_unknown_attestation():
    g = ComplianceGate()
    try:
        g.attest("TX", not_a_real_flag=True)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_search_link_generator_builds_valid_google_urls():
    rows = build_rows("75215", site="zillow.com")
    assert len(rows) == len(KEYWORDS)
    for r in rows:
        # Properly URL-encoded Google search against the target site.
        assert r["google_url"].startswith("https://www.google.com/search?q=")
        assert "site%3Azillow.com" in r["google_url"]
        assert "75215" in r["google_url"]
    # Custom site + keyword list are honoured.
    rows2 = build_rows("Tampa, FL", site="realtor.com", keywords=["probate"])
    assert len(rows2) == 1 and rows2[0]["site"] == "realtor.com"


def test_county_records_builds_public_pdf_urls():
    docs = dallas_notice_urls("June", ["Dallas", "DeSoto"], max_per_city=3)
    assert len(docs) == 6
    for d in docs:
        assert d.url.startswith("https://www.dallascounty.org/")
        assert d.url.endswith(".pdf") and "June" in d.url
    assert public_search_url("TX-Dallas").startswith("https://")


def test_outreach_drafts_are_b2b_and_personalized():
    msg = draft("cowholesale", name="Jordan", market="Dallas, TX", area="75215")
    assert "Jordan" in msg and "75215" in msg
    assert "JV" in msg or "split" in msg
    # All three audiences are available and fill cleanly.
    for kind in TEMPLATES:
        assert "{" not in draft(kind, name="X", market="Dallas, TX")
    try:
        draft("homeowner_coldcall")  # not a supported (regulated) audience
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_bizdev_drafts_contacts_into_outbox():
    co = Company(cfg=config, seed=5)
    n_contacts = len(co.contacts)
    assert n_contacts > 0
    for _ in range(n_contacts):  # enough ticks to clear the queue
        co.tick()
    assert all(c.status == "drafted" for c in co.contacts)
    # Each draft was queued to the (dry-run) integration outbox as email.
    assert any(r.channel == "email" for r in co.integrations.outbox)


def test_company_seeds_real_dfw_targets():
    co = Company(cfg=config, seed=9)
    assert len(co.contacts) >= 10  # real public DFW businesses
    urls = [c.url for c in co.contacts]
    assert all(u.startswith("http") for u in urls)
    kinds = {c.kind for c in co.contacts}
    assert "cashbuyer" in kinds and "cowholesale" in kinds


def test_legal_analyst_returns_memo_with_disclaimer():
    co = Company(cfg=config, seed=1)
    memo = co.legal_review("TX", pre_foreclosure=True)
    assert memo["state"] == "TX"
    assert "NOT legal advice" in memo["disclaimer"]
    assert memo["source"] in ("claude", "heuristic")
    assert "foreclosure" in memo["analysis"].lower()


def test_dispo_platforms_have_payout_model_and_urls():
    assert len(PLATFORMS) >= 6
    for p in PLATFORMS:
        assert p.payout in ("assignment", "finder", "both")
        assert p.name and p.url
    # Assignment-spread filter includes 'both' platforms too.
    assert all(p.payout in ("assignment", "both") for p in by_payout("assignment"))


def test_lead_provider_factory_defaults_to_synthetic():
    p = get_provider(config.HQ_MARKETS, seed=3)
    assert isinstance(p, SyntheticProvider) and p.available()
    leads = p.fetch_leads("TX-Dallas", 3)
    assert len(leads) == 3
    prop, seller = leads[0]
    assert prop.est_market_value > 0 and seller.walk_floor > 0


def test_policy_engine_default_deny_and_status():
    from wholesale import policy
    assert policy.state_status("TX") == "allowed"
    assert policy.state_status("OH") == "caution"
    assert policy.state_status("NC") == "blocked"
    assert policy.state_status("CA") == "blocked"
    assert policy.state_status("ZZ") == "blocked"      # default deny
    assert policy.is_operable("TX") and not policy.is_operable("NC")


def test_scout_blocks_disallowed_market():
    from wholesale.core.models import Deal, Property, Seller, Stage
    co = Company(cfg=config, seed=1)
    prop = Property(address="1 X St", city="Charlotte", state="NC", zip="28201",
                    metro="NC-Charlotte", beds=3, baths=2.0, sqft=1400,
                    year_built=1985, property_type="SFR", distress="vacant",
                    est_market_value=300_000)
    seller = Seller(name="A B", motivation="distress", asking_price=280_000,
                    reachable_via="phone", flexibility=0.4, walk_floor=180_000)
    d = Deal(prop=prop, seller=seller, stage=Stage.SOURCED)
    co.agents["SCOUT"].handle(d)
    assert d.stage == Stage.DEAD
    assert any("Blocked market" in f for f in d.flags)


def test_underwriting_emits_tiers_and_route():
    co = Company(cfg=config, seed=2)
    deal = _make_deal(state="TX")
    deal.stage = Stage.UNDERWRITING
    co.agents["ANALYST"].handle(deal)
    uw = deal.uw
    assert uw.mao_conservative <= uw.mao <= uw.mao_aggressive
    assert uw.route in ("assignment", "double_close", "do_not_pursue")


def test_action_queue_aggregates_and_executes():
    co = Company(cfg=config, seed=7)
    for _ in range(80):
        co.tick()
    q = co.action_queue()
    assert q, "expected pending actions"
    # Sorted by priority (CEO approvals first).
    assert q[0]["priority"] <= q[-1]["priority"]
    # Snapshot exposes it for the dashboard.
    assert "action_queue" in co.snapshot()
    # A 'send' action marks the contact sent (draft-only).
    send = next(a for a in q if a["kind"] == "send")
    assert co.do_action(send["id"]) is True
    assert co.do_action(send["id"]) is False  # already sent
    # Unknown action ids are rejected.
    assert co.do_action("send:99999") is False
    assert co.do_action("bogus:1") is False


def test_decide_rejects_unknown_deal_and_bad_decision():
    co = Company(cfg=config, seed=4)
    assert co.decide(999999, "approved") is False
    deal = _make_deal()
    deal.stage = Stage.UNDER_CONTRACT
    deal.needs_ceo = True
    co.deals.append(deal)
    assert co.decide(deal.id, "maybe") is False  # invalid decision
