"""
browser — Playwright-based stealth browser for sourcing scrapes
==============================================================

Wraps Playwright (when installed) with anti-detection measures so the
Scout can pull listings from sites that block plain HTTP clients
(Zillow, Realtor, Redfin, Auction.com, etc.). All scrapers funnel through
the same `StealthBrowser` so detection bypass lives in one place.

Optional dependency: `playwright` (and `playwright install chromium`).
If absent, `is_available()` returns False and browser-based scrapers
gracefully no-op — the scout falls back to non-JS sources (HUD, HomePath).

Anti-detection measures:
  - playwright-stealth patches (navigator.webdriver, plugins, languages)
  - rotating User-Agent + viewport
  - realistic mouse jitter on page load
  - optional proxy via WS_BROWSER_PROXY env var
  - explicit waits (no fixed sleeps that pattern-match as bot)

LEGAL NOTE — operating a stealth browser against sites whose Terms
prohibit automated access is the user's responsibility. This module is
provided as a sourcing seam; the operator must decide which sources to
enable and whether their use complies with applicable law and ToS.
"""

from __future__ import annotations

import logging
import os
import random
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger("wholesale.sourcing.browser")

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
]

_VIEWPORTS = [
    {"width": 1920, "height": 1080},
    {"width": 1366, "height": 768},
    {"width": 1536, "height": 864},
    {"width": 1440, "height": 900},
]


def _have_playwright() -> bool:
    try:
        import playwright  # noqa: F401
        return True
    except ImportError:
        return False


class StealthBrowser:
    """Context-managed Playwright browser with anti-detection patches.

    Usage:
        with StealthBrowser() as b:
            html = b.fetch("https://www.zillow.com/homes/for_sale/...")

    The returned HTML is fully rendered (post-JS, post-XHR).
    """

    def __init__(self, headless: bool = True, timeout_ms: int = 30000) -> None:
        self.headless = headless
        self.timeout_ms = timeout_ms
        self._pw = None
        self._browser = None
        self._context = None

    @classmethod
    def is_available(cls) -> bool:
        return _have_playwright()

    def __enter__(self) -> "StealthBrowser":
        if not _have_playwright():
            raise RuntimeError(
                "Playwright not installed. Run: pip install playwright && "
                "playwright install chromium")
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        launch_args = {"headless": self.headless}
        proxy = os.environ.get("WS_BROWSER_PROXY", "").strip()
        if proxy:
            launch_args["proxy"] = {"server": proxy}
        self._browser = self._pw.chromium.launch(
            args=["--disable-blink-features=AutomationControlled"], **launch_args)
        ua = random.choice(_USER_AGENTS)
        vp = random.choice(_VIEWPORTS)
        self._context = self._browser.new_context(
            user_agent=ua, viewport=vp, locale="en-US",
            timezone_id="America/Chicago",
            extra_http_headers={
                "Accept-Language": "en-US,en;q=0.9",
                "Sec-Ch-Ua": '"Chromium";v="124", "Not-A.Brand";v="99"',
                "Sec-Ch-Ua-Mobile": "?0",
                "Sec-Ch-Ua-Platform": '"Windows"',
            },
        )
        self._context.add_init_script(_STEALTH_PATCH)
        return self

    def __exit__(self, *exc) -> None:
        try:
            if self._context: self._context.close()
            if self._browser: self._browser.close()
            if self._pw: self._pw.stop()
        except Exception:
            pass

    def fetch(self, url: str, wait_selector: str | None = None) -> str:
        """Navigate to `url` and return fully-rendered HTML.

        If `wait_selector` is given, waits for that CSS selector before
        returning (use for SPAs that need a particular element rendered).
        """
        page = self._context.new_page()
        try:
            page.goto(url, timeout=self.timeout_ms, wait_until="domcontentloaded")
            if wait_selector:
                try:
                    page.wait_for_selector(wait_selector, timeout=self.timeout_ms)
                except Exception:
                    pass
            # human-ish jitter before reading
            page.mouse.move(random.randint(100, 800), random.randint(100, 600))
            page.wait_for_timeout(random.randint(800, 1800))
            return page.content()
        finally:
            page.close()


# Anti-fingerprint script injected into every page before any site script
# runs. Patches the most common bot-detection signals.
_STEALTH_PATCH = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
Object.defineProperty(navigator, 'plugins', {
    get: () => [1, 2, 3, 4, 5].map(() => ({name: 'plugin'}))
});
Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
window.chrome = {runtime: {}};
const originalQuery = window.navigator.permissions && window.navigator.permissions.query;
if (originalQuery) {
    window.navigator.permissions.query = (params) =>
        params.name === 'notifications'
            ? Promise.resolve({state: Notification.permission})
            : originalQuery(params);
}
"""


@contextmanager
def stealth_browser(headless: bool = True):
    """Convenience context manager: `with stealth_browser() as b: ...`."""
    with StealthBrowser(headless=headless) as b:
        yield b
