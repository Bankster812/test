"""
WebLearner — 24/7 internet learning for IB content
====================================================
Continuously scrapes high-quality IB content from the web and feeds it
into the brain's STDP ingestion pipeline.

Sources scraped:
  - Financial news (Reuters, Bloomberg headlines, FT)
  - SEC filings (8-K, 10-K, S-4 merger filings)
  - M&A announcements
  - Investment bank research (publicly accessible summaries)
  - Academic finance papers (SSRN abstracts)
  - IB career/education sites (WSO, M&I summaries)

Requires: requests (stdlib fallback: urllib)
Optional: beautifulsoup4 for better HTML parsing
"""

from __future__ import annotations

import re
import time
import logging
from dataclasses import dataclass, field
from urllib.request import urlopen, Request
from urllib.error import URLError
from urllib.parse import urlencode, urljoin

logger = logging.getLogger("ib_brain.web_learner")


@dataclass
class WebArticle:
    url:     str
    title:   str
    text:    str
    source:  str
    fetched_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Curated IB learning sources
# ---------------------------------------------------------------------------
IB_SOURCES = [
    # Financial news — publicly accessible headlines and summaries
    {
        "name":    "Reuters Finance",
        "url":     "https://www.reuters.com/finance/",
        "type":    "news",
    },
    {
        "name":    "SEC EDGAR M&A Filings",
        "url":     "https://efts.sec.gov/LATEST/search-index?q=%22merger+agreement%22&dateRange=custom&startdt={start}&enddt={end}&forms=S-4,SC+TO-T",
        "type":    "sec",
    },
    {
        "name":    "SSRN Finance Abstracts",
        "url":     "https://www.ssrn.com/index.cfm/en/finance/",
        "type":    "academic",
    },
    {
        "name":    "Mergers & Acquisitions Magazine",
        "url":     "https://www.themiddlemarket.com/",
        "type":    "news",
    },
]

# IB-specific search queries for web scraping
IB_SEARCH_TOPICS = [
    "M&A transaction multiples 2024",
    "leveraged buyout market conditions",
    "DCF valuation methodology",
    "acquisition premium statistics",
    "private equity deal structure",
    "WACC calculation sector",
    "investment banking pitch book",
    "merger accretion dilution analysis",
    "LBO debt capacity",
    "comparable company analysis multiples",
]


class WebLearner:
    """
    Fetches and prepares IB web content for brain ingestion.

    Parameters
    ----------
    config : ib_config module
    max_articles_per_session : int
        Cap on articles fetched per learning session.
    request_delay : float
        Seconds to wait between requests (be a good citizen).
    """

    def __init__(
        self,
        config,
        max_articles_per_session: int = 20,
        request_delay: float = 2.0,
    ):
        self.cfg         = config
        self.max_art     = max_articles_per_session
        self.delay       = request_delay
        self._seen_urls: set[str] = set()
        self._bs4_avail  = self._check_bs4()

    # ------------------------------------------------------------------
    # Main fetch interface
    # ------------------------------------------------------------------

    def fetch_session(self, topics: list[str] | None = None) -> list[WebArticle]:
        """
        Run one learning session: fetch articles on the given topics.
        Returns list of WebArticle objects ready for DocumentIngestion.
        """
        topics = topics or IB_SEARCH_TOPICS
        articles: list[WebArticle] = []

        for topic in topics:
            if len(articles) >= self.max_art:
                break
            try:
                new = self._fetch_topic(topic)
                articles.extend(new)
                time.sleep(self.delay)
            except Exception as e:
                logger.debug(f"Web fetch failed for topic '{topic}': {e}")

        logger.info(f"WebLearner: fetched {len(articles)} articles")
        return articles

    def fetch_url(self, url: str, source_name: str = "web") -> WebArticle | None:
        """Fetch a single URL and return a WebArticle."""
        if url in self._seen_urls:
            return None
        try:
            html  = self._get(url)
            title = self._extract_title(html)
            text  = self._extract_text(html)
            if len(text) < 100:
                return None
            self._seen_urls.add(url)
            return WebArticle(url=url, title=title, text=text[:8000], source=source_name)
        except Exception as e:
            logger.debug(f"Failed to fetch {url}: {e}")
            return None

    def fetch_sec_filing(self, ticker: str) -> list[WebArticle]:
        """
        Fetch recent SEC filings (8-K, S-4) for a given ticker.
        Returns list of WebArticles from EDGAR full-text search.
        """
        url = (
            f"https://efts.sec.gov/LATEST/search-index?"
            f"q=%22{ticker}%22&forms=8-K,S-4&dateRange=custom&"
            f"startdt=2020-01-01&enddt=2099-12-31"
        )
        article = self.fetch_url(url, f"SEC EDGAR ({ticker})")
        return [article] if article else []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_topic(self, topic: str) -> list[WebArticle]:
        """Fetch articles related to a topic via DuckDuckGo HTML search."""
        # DuckDuckGo HTML search (no API key needed)
        query = topic + " investment banking"
        url   = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
        try:
            html  = self._get(url)
            links = self._extract_links(html, "duckduckgo.com")
            articles = []
            for link in links[:3]:    # max 3 articles per topic
                art = self.fetch_url(link, f"web/{topic[:20]}")
                if art:
                    articles.append(art)
                time.sleep(self.delay * 0.5)
            return articles
        except Exception:
            return []

    def _get(self, url: str, timeout: int = 10) -> str:
        """HTTP GET with browser-like headers."""
        req = Request(url, headers={
            "User-Agent": "Mozilla/5.0 (compatible; IBBrainLearner/1.0)",
            "Accept":     "text/html,application/xhtml+xml",
        })
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        # Try UTF-8, fall back to latin-1
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("latin-1", errors="ignore")

    def _extract_title(self, html: str) -> str:
        m = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
        return m.group(1).strip() if m else "Untitled"

    def _extract_text(self, html: str) -> str:
        if self._bs4_avail:
            return self._extract_text_bs4(html)
        # Fallback: strip HTML tags
        text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>",  " ", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _extract_text_bs4(self, html: str) -> str:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)

    def _extract_links(self, html: str, exclude_domain: str = "") -> list[str]:
        """Extract href links from HTML, filtering out non-article URLs."""
        links = re.findall(r'href=["\']([^"\']+)["\']', html)
        cleaned = []
        for link in links:
            if (link.startswith("http") and
                    exclude_domain not in link and
                    not any(x in link for x in ["#", "javascript:", "mailto:", ".pdf",
                                                 "login", "signup", "subscribe"])):
                cleaned.append(link)
        return list(dict.fromkeys(cleaned))  # deduplicate

    @staticmethod
    def _check_bs4() -> bool:
        try:
            import bs4  # noqa: F401
            return True
        except ImportError:
            return False
