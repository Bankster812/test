"""Aus einem Format-Prinzip + DEINEM neuen Thema einen originalen Brief erzeugen.

Der Brief ist eine Produktionsanleitung für *eigenen* Content: eigene Aufnahmen,
eigene Motive, eigener Text. Es wird nichts aus fremdem Material übernommen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .content_patterns import ContentPattern, HookType


# Vorlagen für Hook-Formulierungen je Prinzip. Platzhalter werden mit DEINEM
# Thema gefüllt — das Ergebnis ist origineller Text, kein fremdes Zitat.
_HOOK_TEMPLATES = {
    HookType.QUESTION: "Wusstest du, dass {subject} {angle}?",
    HookType.BOLD_CLAIM: "{subject} machst du wahrscheinlich falsch — hier ist warum.",
    HookType.PROBLEM: "Das größte Problem mit {subject}: {angle}.",
    HookType.CURIOSITY_GAP: "Niemand redet über diesen einen Punkt bei {subject}...",
    HookType.RESULT_FIRST: "So sieht {subject} aus, wenn es funktioniert — der Weg dahin:",
    HookType.PATTERN_BREAK: "Stop. Bevor du {subject} weitermachst, sieh dir das an.",
}

_CTA_TEXT = {
    "follow": "Folge für mehr zu diesem Thema.",
    "save": "Speichere das für später.",
    "share": "Teile das mit jemandem, der es braucht.",
    "comment": "Schreib deine Meinung in die Kommentare.",
    "link_in_bio": "Mehr dazu über den Link in der Bio.",
    "dm": "Schreib mir 'INFO' per DM für die Details.",
}


@dataclass
class ContentBrief:
    """Konkrete Produktionsanleitung für ein eigenes Stück Content."""

    title: str
    subject: str
    hook_line: str
    beats: List[str]               # Storyboard-Schritte
    target_seconds: float
    cta_line: str
    suggested_caption: str
    suggested_hashtags: List[str] = field(default_factory=list)
    based_on_pattern: str = ""

    def to_text(self) -> str:
        lines = [
            f"# Brief: {self.title}",
            f"Thema: {self.subject}",
            f"Format-Prinzip: {self.based_on_pattern}",
            f"Ziel-Länge: ~{self.target_seconds:.0f}s",
            "",
            f"HOOK (0–3s): {self.hook_line}",
            "",
            "STORYBOARD:",
        ]
        for i, beat in enumerate(self.beats, 1):
            lines.append(f"  {i}. {beat}")
        lines += [
            "",
            f"CALL-TO-ACTION: {self.cta_line}",
            "",
            f"CAPTION-VORSCHLAG: {self.suggested_caption}",
            f"HASHTAGS: {' '.join(self.suggested_hashtags)}",
            "",
            "Hinweis: Eigene Aufnahmen/Motive verwenden. Kein fremdes Material kopieren.",
        ]
        return "\n".join(lines)


def _hashtags_for(topic_space: str, subject: str) -> List[str]:
    base = {
        "how-to": ["#tutorial", "#tipps", "#lernen"],
        "opinion": ["#meinung", "#debatte"],
        "story": ["#story", "#storytime"],
    }.get(topic_space, ["#content"])
    # Aus dem Thema ein einfaches Stichwort ableiten.
    slug = "".join(c for c in subject.lower() if c.isalnum() or c == " ").split()
    if slug:
        base = base + ["#" + slug[0]]
    return base


def generate_brief(pattern: ContentPattern, subject: str, angle: str = "") -> ContentBrief:
    """Erzeuge aus einem Format-Prinzip und deinem Thema einen originalen Brief.

    Args:
        pattern: Das strukturelle Erfolgsprinzip (siehe content_patterns).
        subject: DEIN neues Thema/Motiv (z.B. "Espresso zuhause").
        angle: Optionaler Blickwinkel/Detail, das den Hook konkretisiert.
    """
    angle = angle or "es einfacher geht als gedacht"
    hook_template = _HOOK_TEMPLATES[pattern.hook.type]
    hook_line = hook_template.format(subject=subject, angle=angle)

    beats = [step.replace("Ergebnis", f"{subject}-Ergebnis") for step in pattern.structure]

    cta_line = _CTA_TEXT.get(pattern.call_to_action.value, "")

    caption = f"{subject}: {angle.capitalize()}. {cta_line}"
    hashtags = _hashtags_for(pattern.topic_space, subject)

    return ContentBrief(
        title=f"{pattern.name} — {subject}",
        subject=subject,
        hook_line=hook_line,
        beats=beats,
        target_seconds=pattern.pacing_seconds,
        cta_line=cta_line,
        suggested_caption=caption,
        suggested_hashtags=hashtags,
        based_on_pattern=pattern.name,
    )
