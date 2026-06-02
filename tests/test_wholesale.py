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
    assert len(co.agents) == 6
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
    assert len(snap["agents"]) == 6


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


def test_decide_rejects_unknown_deal_and_bad_decision():
    co = Company(cfg=config, seed=4)
    assert co.decide(999999, "approved") is False
    deal = _make_deal()
    deal.stage = Stage.UNDER_CONTRACT
    deal.needs_ceo = True
    co.deals.append(deal)
    assert co.decide(deal.id, "maybe") is False  # invalid decision
