"""
config — company-wide constants for the wholesale platform
==========================================================

A flat module of constants (mirrors the convention in
`neuromorphic/config.py`). Tunable via environment variables so the
company can be reconfigured without touching code.
"""

from __future__ import annotations

import os

# ----------------------------------------------------------------------------
# Company identity
# ----------------------------------------------------------------------------
COMPANY_NAME   = os.environ.get("WS_COMPANY_NAME", "Keystone Property Partners")
CEO_NAME       = os.environ.get("WS_CEO_NAME", "CEO")          # the human operator
HQ_MARKETS     = ["TX-Dallas", "FL-Tampa", "GA-Atlanta", "OH-Columbus", "NC-Charlotte"]

# ----------------------------------------------------------------------------
# Underwriting rules (classic wholesale economics)
# ----------------------------------------------------------------------------
ARV_RULE             = 0.70    # Maximum Allowable Offer = ARV*ARV_RULE - repairs - fee
MIN_ASSIGNMENT_FEE   = 8_000   # don't bother below this spread, USD
TARGET_ASSIGNMENT_FEE = 15_000 # what we aim to clear per deal, USD
MAX_REHAB_RATIO      = 0.45    # walk if repairs > 45% of ARV (too hairy)

# Deals whose contract price exceeds this require CEO sign-off
CEO_APPROVAL_THRESHOLD = float(os.environ.get("WS_CEO_APPROVAL_USD", "250000"))

# ----------------------------------------------------------------------------
# Simulation / operating cadence
# ----------------------------------------------------------------------------
TICK_SECONDS    = float(os.environ.get("WS_TICK_SECONDS", "2.0"))   # wall-clock per tick
NEW_LEADS_TICKS = int(os.environ.get("WS_LEAD_INTERVAL", "3"))      # source leads every N ticks
LEADS_PER_BATCH = int(os.environ.get("WS_LEADS_PER_BATCH", "2"))
MAX_ACTIVITY_LOG = 400                                              # ring-buffer size

# ----------------------------------------------------------------------------
# LLM (Claude API) — agents reason and write outreach via this
# ----------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
LLM_MODEL         = os.environ.get("WS_LLM_MODEL", "claude-sonnet-4-6")
LLM_MAX_TOKENS    = int(os.environ.get("WS_LLM_MAX_TOKENS", "600"))
LLM_TIMEOUT       = int(os.environ.get("WS_LLM_TIMEOUT", "40"))

# ----------------------------------------------------------------------------
# Integrations — dormant until armed (see wholesale.integrations)
# ----------------------------------------------------------------------------
# When False, adapters log the intended action but never touch the network.
INTEGRATIONS_ARMED = os.environ.get("WS_INTEGRATIONS_ARMED", "0") == "1"

# ----------------------------------------------------------------------------
# Dashboard
# ----------------------------------------------------------------------------
DASHBOARD_HOST = os.environ.get("WS_DASHBOARD_HOST", "0.0.0.0")
DASHBOARD_PORT = int(os.environ.get("WS_DASHBOARD_PORT", "8200"))
