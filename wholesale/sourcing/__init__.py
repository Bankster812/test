"""Compliant lead-discovery helpers (search-link generation, no scraping)."""

from __future__ import annotations

from .search_links import KEYWORDS, build_rows, write_csv

__all__ = ["KEYWORDS", "build_rows", "write_csv"]
