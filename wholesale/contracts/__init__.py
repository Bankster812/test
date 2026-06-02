"""
contracts — draft assignment & purchase agreements
==================================================

Generates *draft* documents from a `Deal`. These are starting templates,
NOT legal advice: every document carries a banner requiring review by a
licensed real-estate attorney in the property's state before use. State
law on contract assignment, equitable interest, and (critically) any
pre-foreclosure transaction varies and is strictly enforced.
"""

from __future__ import annotations

from .generator import ContractGenerator

__all__ = ["ContractGenerator"]
