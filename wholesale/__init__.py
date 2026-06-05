"""
wholesale — an autonomous US real-estate wholesaling company
============================================================

A multi-agent platform that runs the full wholesale value chain end to end:

    source  ->  underwrite  ->  outreach  ->  negotiate  ->  contract
            ->  match buyer  ->  assign  ->  close

Each stage is owned by an autonomous agent (Claude-API powered, with a
deterministic heuristic fallback so the company still operates with no
network/key). A live web dashboard shows what every agent is doing in
real time. The human operator is the CEO: high-stakes decisions escalate
to a CEO approval queue.

Run it:

    python -m wholesale.run                 # company + dashboard on :8200
    python -m wholesale.run --no-dashboard  # headless, logs to stdout

Nothing leaves the machine by default. Outbound CRM / email / Slack go
through adapters in `wholesale.integrations`, which run in dry-run mode
until explicitly armed.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
