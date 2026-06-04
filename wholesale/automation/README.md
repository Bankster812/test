# Automation layer — n8n / Make as the "hands", the Python platform as the "brain"

The cleanest design splits responsibilities:

```
        ┌─────────────────────────────────────────────┐
        │  wholesale/ (Python)  =  THE BRAIN           │
        │  • agents, underwriting (70% rule, MAO)      │
        │  • compliance gate, legal triage             │
        │  • contract + disclosure generation          │
        │  • Action Queue (what needs a human)         │
        │  exposes:  GET /api/state   POST /api/action │
        └───────────────┬─────────────────────────────┘
                        │ HTTP (JSON)
        ┌───────────────┴─────────────────────────────┐
        │  n8n  (or Make)  =  THE HANDS / ORCHESTRATION│
        │  • cron triggers (run the cadence)           │
        │  • webhook intake (form replies, new leads)  │
        │  • connectors (Gmail, Slack, HubSpot, Sheets)│
        │  • daily "Action Queue" digest to you        │
        └──────────────────────────────────────────────┘
```

**Why this split:** the brain should own judgement, math, and compliance (testable, version-controlled Python). n8n/Make are excellent at the glue — scheduling, fan-out to 100s of connectors, webhooks — but you don't want business logic or legal gating buried in a visual flow. Keep decisions in code; keep plumbing in n8n.

## What n8n does here (compliant pieces only)
1. **Cadence** — cron node pings the platform / triggers a tick batch on schedule.
2. **Digest** — pulls `GET /api/state`, formats the **Action Queue**, and sends you a Slack/email summary: "5 things need you today, est. $X."
3. **Lead intake** — a webhook node receives new leads (e.g., from a public-records export or a form) and posts them to the platform.
4. **Reply capture** — inbound email/SMS replies land on a webhook → logged against the deal.
5. **Send-on-approval** — only *after* you approve in the cockpit does n8n actually dispatch via Gmail/HubSpot. The send node sits **behind** the human gate, never before it.

## What n8n does NOT do
- It does not bypass the compliance gate or auto-send to homeowners.
- It does not sign contracts or move money.
- Business logic (underwriting, go/no-go, disclosures) stays in Python.

## Files
- `n8n_workflow.json` — importable starter workflow (Schedule → HTTP GET state →
  build digest → send to your Slack/email webhook). Set the `HOST` and webhook
  URL after import.

## Make.com alternative
Make ("scenarios") maps 1:1: Scheduler module → HTTP module (GET /api/state) →
Iterator over `action_queue` → Router → Slack/Email module. Same gating rules.

## Going live checklist
1. Self-host n8n (Docker) or use n8n.cloud / Make.
2. Expose the platform's dashboard server to n8n (same network, or a tunnel).
3. Import `n8n_workflow.json`, set HOST + your Slack/email webhook.
4. Keep `WS_INTEGRATIONS_ARMED=0` until you've cleared the compliance gate and
   wired real send credentials — then arm one channel at a time.
