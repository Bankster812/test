#!/usr/bin/env python3
"""
driver — verify Camoufox can reach a target site without bot-detection
======================================================================

The harness future agents (and humans) run to confirm Camoufox is set up
correctly and a real listing-page renders. Three subcommands:

    python driver.py probe              # bot.sannysoft.com fingerprint test
    python driver.py zillow [STATE]     # hit Zillow foreclosures, save shot
    python driver.py extract [STATE]    # run the Zillow scraper, print rows

Outputs land in ./screenshots/ next to this driver.

Exit codes:
    0  success
    1  bot-detection signals present
    2  Camoufox / Playwright not installed
    3  network failure

Headless by default. `--no-headless` to watch (needs Xvfb on Linux).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHOTS = HERE / "screenshots"
SHOTS.mkdir(exist_ok=True)


def _check_camoufox() -> None:
    try:
        import camoufox  # noqa: F401
    except ImportError:
        print("ERROR: camoufox not installed.\n"
              "  pip install camoufox[geoip]\n"
              "  python -m camoufox fetch\n"
              "See SKILL.md for offline-install fallback.", file=sys.stderr)
        sys.exit(2)


def _open_browser(headless: bool):
    from camoufox.sync_api import Camoufox
    from camoufox.addons import DefaultAddons
    cm = Camoufox(
        headless=headless,
        humanize=True,
        os="windows",
        geoip=False,
        exclude_addons=[DefaultAddons.UBO],
        firefox_user_prefs={
            "security.cert_pinning.enforcement_level": 0,
            "security.enterprise_roots.enabled": True,
        },
    )
    browser = cm.__enter__()
    ctx = browser.new_context(ignore_https_errors=True)
    return cm, browser, ctx


def cmd_probe(args) -> int:
    """Hit bot.sannysoft.com and screenshot the fingerprint report."""
    _check_camoufox()
    cm, browser, ctx = _open_browser(args.headless)
    try:
        page = ctx.new_page()
        page.goto("https://bot.sannysoft.com/", timeout=30000,
                  wait_until="domcontentloaded")
        page.wait_for_timeout(4000)
        shot = SHOTS / "sannysoft.png"
        page.screenshot(path=str(shot), full_page=True)
        print(f"[ok] fingerprint report → {shot}")
        return 0
    except Exception as e:
        print(f"[!] probe failed: {e}", file=sys.stderr)
        return 3
    finally:
        cm.__exit__(None, None, None)


def cmd_zillow(args) -> int:
    """Hit Zillow foreclosures for STATE (default TX) and check for blocks."""
    _check_camoufox()
    state = args.state.lower()
    url = f"https://www.zillow.com/{state}/foreclosures/"
    cm, browser, ctx = _open_browser(args.headless)
    try:
        page = ctx.new_page()
        page.goto(url, timeout=45000, wait_until="domcontentloaded")
        page.wait_for_timeout(5000)
        title = page.title()
        print(f"[*] {url}")
        print(f"[*] title: {title}")
        shot = SHOTS / f"zillow_{state}.png"
        page.screenshot(path=str(shot), full_page=False)
        print(f"[*] screenshot → {shot}")
        # detection probe
        cards = page.locator("[data-testid='property-card']").count()
        print(f"[*] property-cards in DOM: {cards}")
        # the keywords below appear in PerimeterX SDK code too; only
        # fail if BOTH a visible challenge wall AND zero cards.
        html_lc = page.content().lower()
        visible_wall = any(w in html_lc for w in
            ["press and hold", "are you human", "please verify you are",
             "this page can't load google maps correctly"])
        if visible_wall and cards == 0:
            print("[!] visible bot-wall detected")
            return 1
        if cards == 0:
            print("[!] zero cards — Zillow may have changed selector or "
                  "we're being soft-blocked")
            return 1
        print(f"[ok] real listings rendered ({cards} cards)")
        return 0
    except Exception as e:
        print(f"[!] zillow probe failed: {e}", file=sys.stderr)
        return 3
    finally:
        cm.__exit__(None, None, None)


def cmd_extract(args) -> int:
    """Run the production ZillowScraper and print the first 5 listings."""
    _check_camoufox()
    # Make sure the package is importable from this script's location.
    repo_root = HERE.parents[3]  # .../wholesale/.claude/skills/<name>/ → repo
    sys.path.insert(0, str(repo_root))
    from wholesale.sourcing.zillow import ZillowScraper
    state = args.state.upper()
    market = f"{state}-Unknown"
    scraper = ZillowScraper(markets=[market])
    print(f"[*] scraping ZillowScraper(markets=[{market!r}])...")
    rows = scraper.fetch(limit=10)
    print(f"[ok] got {len(rows)} listings")
    for prop, seller in rows[:5]:
        print(f"    {prop.address}, {prop.city} {prop.state} {prop.zip}  "
              f"${seller.asking_price:,}  {prop.distress}")
    if not rows:
        print("[!] no rows — see screenshots/zillow_*.png for what Zillow returned")
        return 1
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Camoufox sourcing driver")
    p.add_argument("--no-headless", dest="headless",
                   action="store_false", default=True)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("probe", help="bot.sannysoft.com fingerprint test")
    sp.set_defaults(fn=cmd_probe)

    sp = sub.add_parser("zillow", help="Hit Zillow foreclosures page")
    sp.add_argument("state", nargs="?", default="TX")
    sp.set_defaults(fn=cmd_zillow)

    sp = sub.add_parser("extract", help="Run production ZillowScraper")
    sp.add_argument("state", nargs="?", default="TX")
    sp.set_defaults(fn=cmd_extract)

    args = p.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
