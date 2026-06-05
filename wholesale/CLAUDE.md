# Keystone Property Partners — Claude Code Guide

**READ THIS FIRST in any new session. It contains the full project context.**

---

## What this is

An autonomous US real-estate wholesaling company. 11 AI agents run the full value chain.
The human operator is the **CEO** — the company handles everything and only escalates
decisions that are genuinely the CEO's.

```
source → underwrite → outreach → negotiate → contract
       → compliance/CEO gate → disposition → assign → close
```

**Company:** Keystone Property Partners | **Markets:** TX-Dallas, FL-Tampa (+ OH-caution, NC-blocked)

---

## Quick start

```bash
python -m wholesale.run                       # dashboard at http://localhost:8200
python -m wholesale.run --no-dashboard        # headless
python -m wholesale.run --ticks 40 --seed 7   # fast deterministic batch
python -m wholesale.run --subscription        # use Claude.ai subscription
pytest                                        # 24 tests, offline, no API key needed
```

---

## Architecture

```
wholesale/
  run.py              CLI entrypoint
  config.py           flat constants (all WS_* env-overridable)
  llm.py              3-backend: API → subscription CLI → heuristic
  core/
    models.py         Property, Seller, Buyer, Deal, Stage, Underwriting
    eventbus.py       thread-safe activity log
    company.py        tick loop, financials, CEO gate, snapshot
    cockpit.py        CEO action-queue builder
  agents/             11 specialists (see roster below)
  data/               market feed + cash-buyer book
  integrations/       dormant CRM/email/Slack/dispo adapters
  dashboard/          stdlib HTTP server + live SPA (page.py)
  contracts/          PSA/assignment templates + state disclosures
  sourcing/           county records, search links, lead providers
  outreach/           B2B templates + DFW cash-buyer targets
  disposition/        platform list + submission logic
  compliance/         ComplianceGate (per-state attestation)
  policy/             default-deny JSON rules
  automation/         n8n workflow JSON
  docs/               7-day plan, dispo guide, sourcing guide, legal triage
```

---

## The 11-agent roster

| Code | Name | Team | Owns |
|------|------|------|------|
| CHIEF | Atlas (Chief of Staff) | Executive | Org health, no pipeline stage |
| SCOUT | Ava (Acquisitions Scout) | Acquisitions | sourced |
| ANALYST | Marcus (Underwriting Analyst) | Acquisitions | underwriting |
| CLOSER | Sofia (Acquisition Manager) | Acquisitions | outreach, negotiation |
| COMPLY | Dana (Compliance & Risk) | Compliance | under_contract |
| DISPO | Theo (Disposition Manager) | Disposition | disposition |
| COORD | Priya (Transaction Coordinator) | Transaction | assigned, closing |
| LEGAL | Counsel-AI | Compliance | advisory |
| FOREIGN | Mei (Foreign Payee Specialist) | Capital | advisory |
| BIZDEV | Riley (Business Development) | Capital | B2B outreach |
| QA | Quinn (QA Auditor) | Governance | audit |

Key files: `agents/orchestrator.py` (CHIEF), `agents/sourcing.py`, `agents/underwriting.py`,
`agents/negotiation.py`, `agents/compliance.py`, `agents/disposition.py`, `agents/transaction.py`

---

## Key config (env vars)

| Var | Default | Meaning |
|-----|---------|---------|
| `ANTHROPIC_API_KEY` | — | enable Claude-powered reasoning |
| `WS_LLM_MODEL` | `claude-sonnet-4-6` | model |
| `WS_LLM_BACKEND` | auto | `api` / `cli` / `off` |
| `WS_LEAD_SOURCE` | `live` | `live` = wait for real data; `demo` = synthetic |
| `WS_LEADS_CSV` | — | path to CSV file for lead import |
| `WS_DASHBOARD_PORT` | `8200` | dashboard port |
| `WS_TICK_SECONDS` | `2.0` | wall-clock seconds per tick |
| `WS_CEO_APPROVAL_USD` | `250000` | deal price that triggers CEO sign-off |
| `WS_INTEGRATIONS_ARMED` | `0` | arm outbound adapters |

---

## What is already built and working

- Full 11-agent pipeline (CHIEF/Atlas, SCOUT/Ava, ANALYST/Marcus, CLOSER/Sofia,
  COMPLY/Dana, DISPO/Theo, COORD/Priya, LEGAL, FOREIGN/Mei, BIZDEV/Riley, QA/Quinn)
- Deal lifecycle: sourced → underwriting → outreach → negotiation → under_contract
  → disposition → assigned → closing → closed_won / dead
- 70% rule underwriting + 3-tier MAO (aggressive/base/conservative)
- Default-deny policy engine (TX/FL operable, GA under review, OH caution, NC blocked)
- ComplianceGate with per-state attestations + disclosure kit
- Contract generator (PSA + assignment + state disclosures)
- CEO action queue (approve/reject, sign, send outreach)
- Pause/resume/step controls
- Disposition platforms + buyer matching + dispo submission adapter
- Atlas/CHIEF oversight (org health: green/amber/red, team rollups, QA failures)
- Foreign payee readiness (Mei — EIN/ITIN, W-8BEN-E, FIRPTA, title-company payout)
- QA/Audit agent (Quinn — arithmetic + policy + artifact checks)
- 3-backend LLM (API / CLI subscription / heuristic)
- Integration adapters (HubSpot/Gmail/Slack) — dormant, outbox-only
- n8n workflow JSON (automation/n8n_workflow.json)
- County scoring (sourcing/county_score.py)
- Real DFW B2B targets (12 public cash-buyer contacts in outreach/dfw_targets.py)
- 24 passing tests (offline, seeded, no API key)
- Live dashboard: Command Center, Agents, Pipeline, Deals, Outreach,
  Disposition, Legal, Contracts, Settings

---

## Pending work — implement these in order

### 1. Windows one-click launcher (HIGHEST PRIORITY)

User wants a desktop icon that opens the dashboard. No terminal.

Create these files at **repo root** (not inside wholesale/):

**`launcher.pyw`** — `.pyw` runs Python without a console window on Windows:
```python
import os, subprocess, sys, time, webbrowser, urllib.request
PORT = int(os.environ.get("WS_DASHBOARD_PORT", "8200"))
ROOT = os.path.dirname(os.path.abspath(__file__))
server = subprocess.Popen([sys.executable, "-m", "wholesale.run", "--port", str(PORT)], cwd=ROOT)
for _ in range(40):
    try:
        urllib.request.urlopen(f"http://localhost:{PORT}/api/state", timeout=1); break
    except: time.sleep(0.5)
webbrowser.open(f"http://localhost:{PORT}")
try: server.wait()
except KeyboardInterrupt: server.terminate()
```

**`setup.bat`** — Windows one-time setup:
1. Check Python 3.10+ (offer `winget install Python.Python.3.12` if absent)
2. `python -m pip install anthropic --quiet`
3. Create Desktop shortcut via PowerShell `WScript.Shell.CreateShortcut`:
   - Target: `pythonw.exe`
   - Arguments: `"<full-path-to-launcher.pyw>"`
   - WorkingDirectory: repo root
   - Icon: `shell32.dll,23`

**`launch.sh`** — Mac/Linux: `python3 launcher.pyw`

### 2. Subscription-first LLM default

Change `_pick_backend()` in `wholesale/llm.py`:
```python
def _pick_backend(self) -> str:
    forced = os.environ.get("WS_LLM_BACKEND", "").lower()
    if forced in ("api", "cli", "off"):
        if forced == "off": return "heuristic"
        if forced == "cli" and not self._cli: return "heuristic"
        if forced == "api" and not self.api_key: return "heuristic"
        return forced
    if self.api_key:
        return "api"
    # Auto-detect subscription CLI (guard against recursive Claude Code subprocess)
    if self._cli and not os.environ.get("CLAUDE_CODE_ENTRYPOINT"):
        return "cli"
    return "heuristic"
```

### 3. Real data mode + CSV import

No synthetic leads by default. System starts empty and waits for real data.

**`wholesale/config.py`** — add:
```python
LEAD_SOURCE = os.environ.get("WS_LEAD_SOURCE", "live")  # "live" | "demo"
LEADS_CSV_PATH = os.environ.get("WS_LEADS_CSV", "")
```

**`wholesale/sourcing/providers.py`** — add `CsvProvider(LeadProvider)`:
- Reads CSV at `LEADS_CSV_PATH`; columns: `address, city, state, zip, metro, beds, baths,
  sqft, year_built, property_type, distress, est_market_value, seller_name, motivation,
  asking_price, reachable_via`
- `available()` → True if file exists
- `fetch_leads(n)` → reads up to n unprocessed rows, returns `[(Property, Seller)]`
- Advances cursor; returns `[]` when exhausted

**`wholesale/core/company.py`** — change tick loop:
```python
if cfg.LEAD_SOURCE == "demo":
    if self.tick_count % cfg.NEW_LEADS_TICKS == 1:
        for prop, seller in self.market.next_leads(cfg.LEADS_PER_BATCH):
            self.deals.append(Deal(prop=prop, seller=seller))
else:
    provider = get_provider()
    if provider.available():
        for prop, seller in provider.fetch_leads(cfg.LEADS_PER_BATCH):
            self.deals.append(Deal(prop=prop, seller=seller))
```

**Dashboard** — add "Import Leads" card to Command Center when pipeline empty.
New endpoints: `POST /api/leads/import` (CSV text body) and `POST /api/leads/add` (single).

### 4. Per-agent detail panels + chat in dashboard

**`wholesale/agents/base.py`** — add:
```python
def receive_message(self, text: str) -> str:
    system = (f"You are {self.name}, {self.role}. The CEO is speaking to you directly. "
              f"Reply in 2-3 sentences. Status: {self.status}. Task: {self.current_task}.")
    reply = self.reason(system, text, max_tokens=200,
                        fallback=f"Understood. I'm currently {self.current_task}.")
    self.say(f"[CEO→{self.code}] {text} | [{self.code}→CEO] {reply}", level="info")
    return reply
```

**`wholesale/dashboard/server.py`** — add `POST /api/agent/message`:
```python
elif self.path == "/api/agent/message":
    code = str(data.get("code", "")).upper()
    msg = str(data.get("message", "")).strip()
    agent = _company.agents.get(code)
    result = {"ok": bool(agent and msg),
              "reply": agent.receive_message(msg) if (agent and msg) else ""}
```

**`wholesale/dashboard/page.py`** — Agents view redesign:
- Left panel: agent list (color dot, name, role/team, status badge, handled count)
- Right panel on click: name/role/desc, status+task+deal, last 10 events from this agent,
  Atlas panel (when CHIEF selected) shows full oversight report, chat input at bottom

### 5. Atlas oversight panel in Command Center

Add to Command Center section of `page.py` (uses `state.oversight`):
```
ATLAS · Chief of Staff    ● GREEN / AMBER / RED
"[oversight note]"
Teams: Acquisitions N active | Disposition N active | ...
QA failures: N   Policy blocks: N   Needs human: N
```

### 6. First-contact CEO gate

When a deal first reaches OUTREACH stage, pause and ask CEO approval before any
homeowner contact. Add `first_contact_approval` action type to `compliance/gate.py`
and `core/cockpit.py`. Dashboard shows "Ready for Your Decision" priority queue.

---

## Testing convention

- All tests in `tests/test_wholesale.py`
- Run: `WS_LLM_BACKEND=off pytest tests/ -q`
- Must stay at 24 passing, all offline, no API key

## Safe-by-default principle

All outbound adapters are dormant. Set `WS_INTEGRATIONS_ARMED=1` AND wire `_live_*`
methods in `wholesale/integrations/` to arm them. Nothing leaves the machine by default.

## Foreign payee note

The operator is a German non-resident. Mei (FOREIGN agent) tracks EIN/ITIN status,
W-8BEN-E, FIRPTA withholding, and title-company payout routing for non-residents.
Dashboard: /api/foreign endpoint returns readiness report.
