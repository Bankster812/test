"""Legales Content-Automation-Toolkit.

Analysiert das *Erfolgsprinzip* (Format/Struktur) guter Inhalte und hilft,
daraus eigene, neue Posts zu erstellen, über die offizielle Instagram Graph API
zu veröffentlichen und die Performance auszuwerten.

Es kopiert keinen fremden Content. Siehe README.md.
"""

from .content_patterns import ContentPattern, PatternLibrary, Hook, CallToAction
from .brief_generator import ContentBrief, generate_brief

__all__ = [
    "ContentPattern",
    "PatternLibrary",
    "Hook",
    "CallToAction",
    "ContentBrief",
    "generate_brief",
]
