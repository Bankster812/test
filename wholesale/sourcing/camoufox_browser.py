"""
camoufox_browser — Camoufox-based stealth browser for sourcing scrapes
======================================================================

Wraps the Camoufox patched Firefox build (https://github.com/daijro/camoufox)
to bypass enterprise bot-detection (Cloudflare, PerimeterX, DataDome, Akamai).
Unlike playwright-stealth — which only patches detection signals at the
JavaScript level — Camoufox patches at the C++ Firefox level: navigator
fingerprint, WebGL renderer, canvas, audio context, fonts, timezone, locale.
Result: real listings pages render the same as for a human user.

OPTIONAL DEPENDENCY: `camoufox` (and `python -m camoufox fetch` to grab the
patched Firefox binary, ~630 MB). If absent, `is_available()` returns False
and browser-driven providers gracefully skip (Scout falls back to non-JS
sources: HUD Home Store, Fannie HomePath).

USAGE:
    from wholesale.sourcing.camoufox_browser import CamoufoxBrowser
    with CamoufoxBrowser() as b:
        html = b.fetch("https://www.zillow.com/tx/foreclosures/",
                       wait_selector="[data-testid='property-card']")

ENV VARS:
    WS_BROWSER_PROXY        proxy URL (e.g. http://user:pass@host:port) for
                            residential-proxy rotation
    WS_BROWSER_OS           "windows" | "macos" | "linux" (default: windows)
    WS_BROWSER_HEADLESS     "0" to show a window (default headless)
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("wholesale.sourcing.camoufox")


def _have_camoufox() -> bool:
    try:
        import camoufox  # noqa: F401
        return True
    except ImportError:
        return False


class CamoufoxBrowser:
    """Context-managed Camoufox session with anti-detection baked in.

    Camoufox patches navigator.webdriver, WebGL/Canvas fingerprint, audio
    context, fonts, timezone, locale, and many other detection vectors at
    the Firefox C++ level — much more robust than JS-patch approaches.
    """

    def __init__(self, headless: bool | None = None, timeout_ms: int = 45000,
                 humanize: bool = True, os_profile: str | None = None) -> None:
        if headless is None:
            headless = os.environ.get("WS_BROWSER_HEADLESS", "1") == "1"
        self.headless = headless
        self.timeout_ms = timeout_ms
        self.humanize = humanize
        self.os_profile = os_profile or os.environ.get("WS_BROWSER_OS", "windows")
        self._browser = None
        self._ctx = None

    @classmethod
    def is_available(cls) -> bool:
        return _have_camoufox()

    def __enter__(self) -> "CamoufoxBrowser":
        if not _have_camoufox():
            raise RuntimeError(
                "Camoufox not installed. Install with:\n"
                "  pip install camoufox[geoip]\n"
                "  python -m camoufox fetch")
        from camoufox.sync_api import Camoufox
        from camoufox.addons import DefaultAddons

        launch_kwargs: dict[str, Any] = {
            "headless": self.headless,
            "humanize": self.humanize,
            "os": self.os_profile,
            "geoip": False,
            # Skip default add-on download (uBlock) to keep first-run fast and
            # avoid extra GitHub API calls.
            "exclude_addons": [DefaultAddons.UBO],
            "firefox_user_prefs": {
                "security.cert_pinning.enforcement_level": 0,
                "security.enterprise_roots.enabled": True,
            },
        }
        proxy = os.environ.get("WS_BROWSER_PROXY", "").strip()
        if proxy:
            launch_kwargs["proxy"] = {"server": proxy}

        self._cm = Camoufox(**launch_kwargs)
        self._browser = self._cm.__enter__()
        # Behind corporate / dev TLS-proxies, ignore the unknown-issuer error
        # so the scraper still works in sandboxed environments.
        self._ctx = self._browser.new_context(ignore_https_errors=True)
        return self

    def __exit__(self, *exc) -> None:
        try:
            if self._ctx:
                self._ctx.close()
        except Exception:
            pass
        try:
            if self._cm:
                self._cm.__exit__(*exc)
        except Exception:
            pass

    def fetch(self, url: str, wait_selector: str | None = None,
              hydrate_ms: int = 4000) -> str:
        """Navigate and return fully-rendered HTML.

        If `wait_selector` is given, waits for that CSS selector to appear.
        Otherwise sleeps `hydrate_ms` to let JS/XHR finish populating the DOM.
        """
        page = self._ctx.new_page()
        try:
            page.goto(url, timeout=self.timeout_ms, wait_until="domcontentloaded")
            if wait_selector:
                try:
                    page.wait_for_selector(wait_selector, timeout=self.timeout_ms)
                except Exception:
                    logger.debug("wait_selector %r timed out on %s",
                                 wait_selector, url)
            page.wait_for_timeout(hydrate_ms)
            return page.content()
        finally:
            page.close()

    def screenshot(self, url: str, path: str,
                   wait_selector: str | None = None) -> None:
        """Visit `url` and save a screenshot to `path` (for debugging)."""
        page = self._ctx.new_page()
        try:
            page.goto(url, timeout=self.timeout_ms, wait_until="domcontentloaded")
            if wait_selector:
                try:
                    page.wait_for_selector(wait_selector, timeout=self.timeout_ms)
                except Exception:
                    pass
            page.wait_for_timeout(3000)
            page.screenshot(path=path, full_page=False)
        finally:
            page.close()
