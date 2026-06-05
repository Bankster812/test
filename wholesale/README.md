# wholesale — an autonomous US real-estate wholesaling company

A multi-agent platform that runs the **entire wholesale value chain** end to
end, with a **live CEO dashboard** showing what every agent is doing in real
time. You are the CEO: the company runs itself and only escalates the calls
that are genuinely yours.

```
 source ─▶ underwrite ─▶ outreach ─▶ negotiate ─▶ contract
        ─▶ compliance/CEO gate ─▶ disposition ─▶ assign ─▶ close
```

## Run it

```bash
python -m wholesale.run                 # company + dashboard at http://localhost:8200
python -m wholesale.run --no-dashboard  # headless, streams activity to stdout
python -m wholesale.run --ticks 60 --seed 7   # fast, reproducible batch run
```

No build step, no third-party dependencies — pure standard library (same
philosophy as `neuromorphic/brain_web.py`).

## The workforce

| Agent | Role | Owns stage(s) |
|-------|------|---------------|
| **Ava** — Acquisitions Scout | Sourcing & lead qualification | sourced |
| **Marcus** — Underwriting Analyst | ARV, repairs & Maximum Allowable Offer | underwriting |
| **Sofia** — Acquisition Manager | Seller outreach & negotiation | outreach, negotiation |
| **Dana** — Compliance & Risk | Legal review & CEO escalation gate | under_contract |
| **Theo** — Disposition Manager | Cash-buyer matching & assignment | disposition |
| **Priya** — Transaction Coordinator | Title, escrow & closing | assigned, closing |

Each agent is **Claude-API powered** for judgement and messaging (seller
outreach copy, qualification notes, go/no-go rationale) and **deterministic**
for the money (ARV, the 70% rule, MAO, buyer matching) — so financial outputs
are never hallucinated. With no `ANTHROPIC_API_KEY` set, agents fall back to
heuristics and the company keeps operating.

## The dashboard

`http://localhost:8200` shows, refreshed live:

- **KPIs** — revenue (collected assignment fees), closed-won, active deals,
  pipeline value, average fee, dead deals.
- **Agent grid** — each agent's live status (idle / thinking / acting), current
  task, current deal, lifetime action count.
- **Pipeline kanban** — every deal as a chip in its current stage.
- **CEO approval queue** — deals over the approval threshold (default $250K) or
  carrying risk flags wait here with **Approve / Reject** buttons.
- **Activity feed** — the company's running narration, colour-coded.
- **Integration outbox** — every CRM/email/Slack action the company *would*
  send (dry-run).

## You are the CEO

- Deals whose contract price ≥ `WS_CEO_APPROVAL_USD` (default $250,000), or that
  Compliance flags, are **held** in the CEO queue until you approve or reject
  them from the dashboard.
- Everything else the agents handle autonomously.

## Safe by default

Outbound CRM (HubSpot), email (Gmail) and Slack go through adapters in
`wholesale/integrations/`. They are **dormant**: each records the intended
action to the dashboard outbox and returns a dry-run result **without touching
the network**. Going live is a deliberate act — set `WS_INTEGRATIONS_ARMED=1`
*and* wire each adapter's `_live_*` method to a real transport (e.g. route
through the HubSpot / Gmail / Slack MCP tools). Until then, nothing leaves the
machine.

## Configuration (env vars)

| Var | Default | Meaning |
|-----|---------|---------|
| `ANTHROPIC_API_KEY` | — | enable Claude-powered agents |
| `WS_LLM_MODEL` | `claude-sonnet-4-6` | agent reasoning model |
| `WS_DASHBOARD_PORT` | `8200` | dashboard port |
| `WS_TICK_SECONDS` | `2.0` | wall-clock seconds per operating tick |
| `WS_CEO_APPROVAL_USD` | `250000` | contract price that triggers CEO sign-off |
| `WS_INTEGRATIONS_ARMED` | `0` | arm outbound adapters (still need wiring) |
| `WS_COMPANY_NAME` / `WS_CEO_NAME` | … | branding |

## Architecture

```
wholesale/
  run.py              entrypoint (company + dashboard, or headless)
  config.py           flat constants, env-overridable
  llm.py              stdlib Claude Messages client (+ heuristic fallback)
  core/
    models.py         Property, Seller, Buyer, Deal, Stage, Underwriting
    eventbus.py       thread-safe activity log (the "nervous system")
    company.py        orchestrator: tick loop, financials, CEO gate, snapshot
  agents/             one specialist per value-chain stage
  data/               synthetic market feed + cash-buyer book (the real-data seam)
  integrations/       dormant CRM / email / Slack adapters
  dashboard/          stdlib HTTP server + single-page live UI
```

The tick loop processes stages in **reverse** pipeline order, so a deal advances
at most one stage per tick — making the value chain visible on the board rather
than teleporting end-to-end.

### Going live (real operations)

This platform models and *operates the workflow*; it does not, by itself,
constitute a legal real-estate business. Before any live deal you still need:
entity formation, a business bank account, real lead sources (county
pre-foreclosure / tax-delinquent lists, skip-tracing), a real cash-buyer list,
assignable purchase contracts reviewed by counsel, and compliance with the
wholesaling/licensing rules of each state you operate in (these vary and several
states have tightened them). The seams to plug all of this in are
`data/market.py` (lead source), `data/buyers.py` (buyer list), and
`integrations/` (CRM/email/Slack) — swap the synthetic pieces for real adapters
and the agent logic is unchanged.
