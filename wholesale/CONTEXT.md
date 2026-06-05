# Keystone Property Partners — Full Session Context

This document captures the complete requirements, architecture decisions, and pending work
from the founding session. A new Claude Code session should read this first.

---

## What this platform is

An autonomous US real-estate wholesaling company. 11 AI agents cover the full value chain:

```
source → underwrite → outreach → negotiate → contract
       → compliance/CEO gate → disposition → assign → close
```

The human operator is the **CEO**. The system runs autonomously and only escalates
decisions that are genuinely the CEO's (first seller contact, deals over $250K, compliance flags).

**Company name:** Keystone Property Partners  
**Target markets:** TX-Dallas, FL-Tampa, GA-Atlanta (watch-only), OH-Columbus (caution), NC-Charlotte (blocked)

---

## The 11-agent roster

| Code | Name | Team | Owns | File |
|------|------|------|------|------|
| CHIEF | Atlas (Chief of Staff) | Executive | Org health & accountability | `agents/orchestrator.py` |
| SCOUT | Ava (Acquisitions Scout) | Acquisitions | sourced | `agents/sourcing.py` |
| ANALYST | Marcus (Underwriting Analyst) | Acquisitions | underwriting | `agents/underwriting.py` |
| CLOSER | Sofia (Acquisition Manager) | Acquisitions | outreach, negotiation | `agents/negotiation.py` |
| COMPLY | Dana (Compliance & Risk) | Compliance | under_contract | `agents/compliance.py` |
| DISPO | Theo (Disposition Manager) | Disposition | disposition | `agents/disposition.py` |
| COORD | Priya (Transaction Coordinator) | Transaction | assigned, closing | `agents/transaction.py` |
| LEGAL | Counsel-AI | Compliance | advisory | `agents/legal.py` |
| FOREIGN | Mei (Foreign Payee Specialist) | Capital | advisory | `agents/foreign_payee.py` |
| BIZDEV | Riley (Business Development) | Capital | B2B outreach | `agents/bizdev.py` |
| QA | Quinn (QA Auditor) | Governance | audit | `agents/qa.py` |

---

## Architecture

- **Entry point**: `wholesale/run.py` → `python -m wholesale.run`
- **Dashboard**: stdlib HTTP server at `:8200`, single-page app in `dashboard/page.py`
- **Tick loop**: `core/company.py` — processes stages in reverse pipeline order (one stage/deal/tick)
- **LLM**: `llm.py` — 3 backends: API (ANTHROPIC_API_KEY) → CLI (claude subscription) → heuristic fallback
- **Policy engine**: default-deny JSON rules in `policy/` — blocks NC, flags OH
- **Integrations**: all dormant by default (`WS_INTEGRATIONS_ARMED=0`); log to outbox, never touch network
- **Compliance gate**: `compliance/gate.py` — per-state attestation before any homeowner contact
- **Contracts**: `contracts/` — PSA + assignment templates + state-specific disclosures (TX, FL, GA, OH, NC)

---

## LLM backend — IMPORTANT

The LLM client (`llm.py`) currently selects backend in this priority order:
1. `WS_LLM_BACKEND` env var if set (api/cli/off)
2. `ANTHROPIC_API_KEY` if set → API backend
3. Falls back to heuristic (no LLM)

**Pending change the user wants**: auto-select `cli` (Claude subscription) when the `claude`
binary is in PATH and no API key is set. Guard against recursive execution when running
inside Claude Code itself by checking `CLAUDE_CODE_ENTRYPOINT` env var.

```python
# Desired _pick_backend() logic:
if self.api_key:
    return "api"
if self._cli and not os.environ.get("CLAUDE_CODE_ENTRYPOINT"):
    return "cli"   # auto-detect subscription
return "heuristic"
```

---

## Windows one-click launcher — PENDING (not yet built)

The user wants to double-click a desktop icon and have the dashboard open in Chrome.
No terminal, no commands. Planned files:

**`launcher.pyw`** (repo root — `.pyw` runs Python without a console window on Windows):
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

**`setup.bat`** (Windows one-time setup):
1. Check for Python 3.10+ (offer winget install if absent)
2. `pip install anthropic --quiet`
3. Create desktop shortcut via PowerShell `WScript.Shell.CreateShortcut()` pointing to `pythonw.exe launcher.pyw`

**`launch.sh`** (Mac/Linux): `python3 launcher.pyw`

---

## Real data sourcing — PENDING (not yet built)

The user does NOT want synthetic/fake lead generation as the default.
The platform currently generates random leads from `data/market.py` every 3 ticks.

**Required changes:**

1. **Config** (`config.py`): add `LEAD_SOURCE = os.environ.get("WS_LEAD_SOURCE", "live")`
   - `"live"` = start empty, only real data
   - `"demo"` = synthetic auto-generation (for testing)

2. **CSV import**: `CsvProvider` class in `sourcing/providers.py`
   - Reads a CSV at `WS_LEADS_CSV` path
   - Columns: `address, city, state, zip, metro, beds, baths, sqft, year_built, property_type, distress, est_market_value, seller_name, motivation, asking_price, reachable_via`
   - Returns `LeadProvider.fetch_leads()` compatible output

3. **Dashboard import UI**: "Import Leads" section in Command Center
   - CSV upload field → POST `/api/leads/import`
   - Single-property manual form → POST `/api/leads/add`
   - When pipeline is empty: onboarding card "No leads yet — import a CSV or let Scout scan county records"

4. **Company tick loop** (`core/company.py`): switch on `LEAD_SOURCE`:
   ```python
   if cfg.LEAD_SOURCE == "demo":
       # existing synthetic generation
   else:
       # only pull from real provider
   ```

---

## Dashboard improvements — PENDING (not yet built)

The user wants per-agent detail panels with agent chat. Current state: basic grid, no detail.

**Required:**

1. **Agents view redesign** (`dashboard/page.py`):
   - Left panel: scrollable agent roster list (color dot, name, status, team, handled count)
   - Right panel (on click): agent detail
     - Name, role, description, team badge
     - Current status + task + deal
     - Last 10 event-bus messages from this agent
     - When CHIEF selected: full Atlas oversight report (health, team rollups, QA failures)
     - **Chat input at bottom**: "Message [Agent Name]..." → POST `/api/agent/message`
     - Chat reply rendered inline

2. **Atlas oversight panel** in Command Center (populated from `state.oversight`):
   ```
   ATLAS · Chief of Staff     ● GREEN / AMBER / RED
   "Note from last oversight cycle"
   Teams: Acquisitions 4 active | Disposition 2 active
   QA failures: 0   Policy blocks: 2   Needs human: 2
   ```

3. **Simulation/live mode badge**: small indicator in top nav showing
   "LIVE MODE" (green) or "DEMO MODE — synthetic data" (amber)

**New endpoint needed** (`dashboard/server.py`):
```
POST /api/agent/message  { "code": "CHIEF", "message": "What's the pipeline status?" }
→ { "ok": true, "reply": "...", "agent": "CHIEF" }
```

**New method needed** (`agents/base.py`):
```python
def receive_message(self, text: str) -> str:
    system = f"You are {self.name}, {self.role}. The CEO is speaking to you. Reply in 2-3 sentences."
    reply = self.reason(system, text, max_tokens=200,
                        fallback=f"Understood. I'm currently {self.current_task}.")
    self.say(f"[CEO→{self.code}] {text} | [{self.code}→CEO] {reply}", level="info")
    return reply
```

---

## CEO first-decision gate — PARTIALLY BUILT

The system already gates deals over $250K and compliance-flagged deals. The user wants
the **first homeowner contact** to also require CEO approval. When a deal reaches OUTREACH
stage for the first time, it should pause and ask the CEO "Should we send a letter to this seller?"

- `compliance/gate.py`: add `first_contact_approval` action type
- `core/cockpit.py`: add first-contact actions with full property details + draft outreach letter
- Dashboard: "Ready for Your Decision" priority queue at the top of Command Center

---

## State policy (policy engine)

- **TX, FL**: fully operable (assignment + double-close)
- **GA**: under review — check 2025-26 licensing rules with counsel
- **OH**: caution — operable with enhanced disclosures
- **NC**: blocked

Policy files: `policy/market_rules.json`, `policy/outreach_rules.json`, `policy/payout_rules.json`

---

## Foreign payee setup (Mei/FOREIGN agent)

The operator is a German non-resident. Mei tracks:
- EIN/ITIN status
- W-8BEN-E completion
- FIRPTA withholding applicability
- Title-company payout routing (required for non-resident)
- 30% withholding risk flag

---

## n8n automation layer

`automation/n8n_workflow.json` — Action Queue digest workflow. Sends daily summary of
pending CEO decisions via webhook. Wire to n8n/Make instance for live use.

---

## Pending work checklist

- [ ] Windows launcher (`launcher.pyw`, `setup.bat`, `launch.sh`)
- [ ] Subscription-first LLM default (auto-detect `claude` CLI)
- [ ] CSV lead import (`CsvProvider` + dashboard upload UI)
- [ ] Real-data mode in config (`LEAD_SOURCE=live` default)
- [ ] Per-agent detail panels in dashboard
- [ ] Agent chat (`receive_message` + `/api/agent/message`)
- [ ] Atlas oversight panel in Command Center
- [ ] Live/demo mode indicator in nav bar
- [ ] First-contact CEO gate (outreach stage)
- [ ] Repo separation: wholesale → `bankster812/keystone-property-partners`

## What IS complete and working

- [x] 11-agent pipeline (CHIEF, SCOUT, ANALYST, CLOSER, COMPLY, DISPO, COORD, LEGAL, FOREIGN, BIZDEV, QA)
- [x] Full deal lifecycle (sourced → closed_won / dead)
- [x] 70% rule underwriting + 3-tier MAO
- [x] Default-deny policy engine (state-based)
- [x] ComplianceGate with per-state attestations
- [x] State disclosures (TX §5.0205, FL §501.1377, OH §5301.95)
- [x] Contract generator (PSA + assignment)
- [x] CEO action queue (approve/reject deals, sign contracts, send outreach)
- [x] Pause/resume/step controls
- [x] Disposition platforms + buyer matching
- [x] Dispo submission adapter (dry-run → live)
- [x] Atlas/CHIEF oversight (org health: green/amber/red, team rollups, QA failures)
- [x] Foreign payee readiness (Mei/FOREIGN)
- [x] QA/Audit agent (Quinn)
- [x] 3-backend LLM (API / CLI / heuristic)
- [x] Integration adapters (HubSpot/Gmail/Slack) — dormant, outbox-only
- [x] n8n workflow JSON
- [x] County scoring (`sourcing/county_score.py`)
- [x] Real DFW B2B targets (12 public cash-buyer contacts)
- [x] 24 passing tests (offline, seeded, no API key needed)
- [x] Live CEO dashboard at :8200 (Command Center, Agents, Pipeline, Deals, Outreach, Disposition, Legal, Contracts, Settings)

---

## How to run (current state)

```bash
# Heuristic mode (no API key, fully offline):
WS_LLM_BACKEND=off python -m wholesale.run --ticks 20 --seed 7

# With Claude subscription:
python -m wholesale.run --subscription

# Dashboard:
python -m wholesale.run
# open http://localhost:8200

# Tests:
WS_LLM_BACKEND=off python -m pytest tests/test_wholesale.py -q
```

---

## Instruction for a new Claude Code session

If you are reading this in a new session on `keystone-property-partners`:

1. This is the **Keystone Property Partners** wholesale platform. Read this file top to bottom.
2. The pending work checklist above is your task list.
3. Start with the Windows launcher (`launcher.pyw` + `setup.bat`) — highest user priority.
4. Then fix LLM default (subscription-first).
5. Then build the dashboard agent-chat and Atlas panel.
6. Then add CSV lead import and real-data mode.
7. Run `WS_LLM_BACKEND=off python -m pytest tests/ -q` after each change — all 24 must pass.
8. The user is the CEO. Do not ask for permission before implementing — just build it.
