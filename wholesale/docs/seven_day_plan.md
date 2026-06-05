# 7-Day plan — first income, the compliant fast way

Goal: a **realistic** first check in 5–7 days. The honest fastest route for a
brand-new operator is **co-wholesaling / connecting a buyer** (you split a fee
on someone else's deal), not originating a full assignment solo — that usually
takes 2–6 weeks. So this plan front-loads the buyer side, where cash moves
fastest, and starts your own pipeline in parallel.

What the **platform** does vs. what **you** must do: the agents source, draft,
underwrite, and match. *You* (a licensed-or-exempt human, with counsel) make the
calls, sign the contracts, and hit send. Pre-foreclosure homeowner contact stays
behind `compliance/gate.py`.

---

### Day 1 — Foundation & buyer list
- Confirm with a TX real-estate attorney: do you need a license to wholesale in
  Texas for your model, and is the PSA/assignment template OK? (You're forming
  the LLC already.)
- Build a **cash-buyer list** from public **Dallas County deed records** — pull
  recent cash purchases (no lien) in your ZIPs. These are verified active buyers.
  `wholesale/sourcing/county_records.py` points to the public sources.
- Generate motivated-seller search links for your ZIPs:
  `python -m wholesale.sourcing.search_links 75215`

### Day 2 — Get in front of buyers & partners
- Email/call 20–30 **active wholesalers** and **agents** in DFW using the
  drafts in `wholesale/outreach/` (`draft("cowholesale", ...)`,
  `draft("realtor", ...)`). Goal: "I bring buyers / I have buyers — let's JV."
- Post in 3–5 local REI Facebook groups + BiggerPockets that you have cash
  buyers for DFW SFRs.
- List yourself on a free disposition marketplace (OfferMarket) to see live
  deals you could place to a buyer.

### Day 3 — Match a live deal to a buyer (the money move)
- Find an existing under-contract deal needing a buyer (from Day-2 partners /
  marketplaces). Run it through the platform's underwriting sanity check
  (`IBBrain`-style numbers are in `wholesale` underwriting) to confirm the
  buyer's spread works.
- Present it to 5–10 matching buyers from your list. First yes → paper a
  **JV / assignment** with the wholesaler (attorney-reviewed template).

### Day 4 — Your own pipeline starts
- Pull this week's **public foreclosure notices** for Dallas County
  (`dallas_notice_urls("June", [...])`) and the public search portal.
- Skip-trace only via a **licensed provider** (not people-search sites), and
  only after `gate.attest("TX", ...)` is satisfied.
- Prepare **direct-mail** pieces to distressed owners (mail is TCPA-exempt and
  the safest first-touch). The agents draft; you mail.

### Day 5 — Follow up hard
- Re-touch every Day-2/3 contact. Deals die from silence, not from "no."
- For any seller lead that replies, the **Closer** drafts the conversation; you
  negotiate to a number at/under MAO and sign the assignable PSA.

### Day 6 — Close the loop
- Push your matched deal toward closing/assignment; coordinate title/escrow
  (the Coordinator agent tracks it; a title company/closing attorney executes).
- Collect the **JV split / assignment fee** on anything that funds.

### Day 7 — Review & systematize
- Tally pipeline on the dashboard. Double down on whichever channel produced a
  reply (buyers vs. agents vs. mail).

---

## The compliance gate (non-negotiable before homeowner contact)
Before the platform will mark TX homeowner outreach as allowed, attest in code:

```python
from wholesale.compliance import ComplianceGate
g = ComplianceGate()
g.attest("TX", entity_formed=True, licensed_or_exempt=True,
         attorney_engaged=True, contracts_reviewed=True,
         foreclosure_law_reviewed=True, tcpa_dnc_process=True,
         data_source_compliant=True)
assert g.can_contact_real_owners("TX")
```

Until then: **buyer-side and B2B outreach run freely; homeowner cold-contact
does not.** That ordering is also just good business — buyers first, deals
second.

## Realistic expectations
- Most likely Day 5–7 outcome: a **co-wholesale split** ($2k–$8k) or a signed
  contract you can assign next week — not a guaranteed solo $30k assignment in 7
  days. Anyone promising the latter from a cold start is selling a course.
