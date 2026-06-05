"""
run — boot the autonomous wholesaling company
=============================================

    python -m wholesale.run                  # company + dashboard on :8200
    python -m wholesale.run --no-dashboard    # headless, prints activity
    python -m wholesale.run --ticks 40        # run N ticks headless then exit
    python -m wholesale.run --seed 7          # reproducible deal flow

Environment:
    ANTHROPIC_API_KEY        enable Claude-powered agent reasoning
    WS_LLM_MODEL             default claude-sonnet-4-6
    WS_DASHBOARD_PORT        default 8200
    WS_TICK_SECONDS          default 2.0
    WS_INTEGRATIONS_ARMED=1  arm CRM/email/Slack adapters (still need wiring)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

from . import config
from .core.company import Company


def _headless(company: Company, ticks: int | None) -> None:
    seen = 0
    try:
        n = 0
        while ticks is None or n < ticks:
            company.tick()
            for ev in company.bus.recent(after_seq=seen):
                seen = ev.seq
                print(f"  [{ev.agent:<26}] {ev.message}")
            n += 1
            if ticks is None:
                time.sleep(config.TICK_SECONDS)
    except KeyboardInterrupt:
        pass
    f = company.snapshot()["financials"]
    print(f"\n== {company.cfg.COMPANY_NAME} ==")
    print(f"   leads={f['total_leads']} closed={f['closed']} "
          f"dead={f['dead']} revenue=${f['revenue']:,} "
          f"pipeline=${f['pipeline_fee_value']:,}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Autonomous real-estate wholesaling company")
    ap.add_argument("--no-dashboard", action="store_true", help="run headless")
    ap.add_argument("--ticks", type=int, default=None,
                    help="headless: run N ticks fast then exit")
    ap.add_argument("--seed", type=int, default=None, help="seed deal flow")
    ap.add_argument("--port", type=int, default=config.DASHBOARD_PORT)
    ap.add_argument("--host", default=config.DASHBOARD_HOST)
    ap.add_argument("--subscription", action="store_true",
                    help="use your Claude subscription via the local `claude` CLI "
                         "(no API key needed)")
    args = ap.parse_args(argv)

    if args.subscription:
        os.environ["WS_LLM_BACKEND"] = "cli"

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    company = Company(cfg=config, seed=args.seed)

    llm = company.agents["SCOUT"].llm
    llm_state = (llm.describe() if llm.is_available()
                 else "heuristic (use --subscription for your Claude plan, "
                      "or set ANTHROPIC_API_KEY)")
    print(f"\n{config.COMPANY_NAME} — CEO: {config.CEO_NAME}")
    print(f"Agents: {', '.join(a.name for a in company.agents.values())}")
    print(f"Reasoning: {llm_state}")
    print(f"Markets: {', '.join(config.HQ_MARKETS)}\n")

    if args.no_dashboard or args.ticks is not None:
        _headless(company, args.ticks)
        return 0

    from .dashboard.server import serve
    httpd = serve(company, args.host, args.port)
    company.start()
    url = f"http://localhost:{args.port}"
    print(f"Dashboard:  {url}   (Ctrl-C to stop)\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down…")
    finally:
        company.stop()
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
