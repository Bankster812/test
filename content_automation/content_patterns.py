"""Erfolgsprinzipien von Content strukturiert erfassen — legal, ohne Kopieren.

Wir speichern hier ausschließlich *strukturelle* Eigenschaften eines Formats
(Art des Hooks, Aufbau, Pacing, Call-to-Action), niemals fremdes Material.
Aus diesen Prinzipien werden später eigene, neue Inhalte abgeleitet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import List, Optional


class HookType(str, Enum):
    """Art des Aufhängers in den ersten ~3 Sekunden."""

    QUESTION = "question"            # offene Frage an die Zielgruppe
    BOLD_CLAIM = "bold_claim"        # provokante/starke Aussage
    PROBLEM = "problem"              # konkretes Schmerzproblem benennen
    CURIOSITY_GAP = "curiosity_gap"  # Neugierlücke ("Das wusste ich nicht...")
    RESULT_FIRST = "result_first"    # Ergebnis/Transformation zuerst zeigen
    PATTERN_BREAK = "pattern_break"  # visueller/auditiver Bruch


class CallToAction(str, Enum):
    """Gewünschte Handlung am Ende."""

    FOLLOW = "follow"
    SAVE = "save"
    SHARE = "share"
    COMMENT = "comment"
    LINK_IN_BIO = "link_in_bio"
    DM = "dm"


@dataclass
class Hook:
    """Beschreibt das Erfolgsprinzip eines Hooks — nicht den konkreten Wortlaut."""

    type: HookType
    # Eigene, abstrahierte Beschreibung des Prinzips (kein fremdes Zitat).
    principle: str
    seconds: float = 3.0


@dataclass
class ContentPattern:
    """Ein wiederverwendbares Format-Prinzip.

    Beispiel: "kurzes Tutorial mit Ergebnis-zuerst-Hook, 3 Schritten und Save-CTA".
    Enthält bewusst KEIN fremdes Material — nur die Struktur.
    """

    name: str
    hook: Hook
    structure: List[str]                 # z.B. ["Problem", "3 Schritte", "Ergebnis"]
    pacing_seconds: float                # Ziel-Länge des Clips
    call_to_action: CallToAction
    target_audience: str
    topic_space: str                     # in welchem Themenfeld funktioniert das
    notes: str = ""

    # Performance-Gewichtung, die sich über Insights aktualisiert (siehe insights.py).
    score: float = 0.0
    samples: int = 0

    def update_score(self, performance: float) -> None:
        """Laufenden Mittelwert der Performance dieses Prinzips fortschreiben.

        ``performance`` ist eine normalisierte Kennzahl (z.B. Saves/Reach), die
        in insights.py berechnet wird. So lernt die Library, welche *eigenen*
        Formate tatsächlich funktionieren.
        """
        self.samples += 1
        # Inkrementeller Mittelwert.
        self.score += (performance - self.score) / self.samples


@dataclass
class PatternLibrary:
    """Sammlung von Format-Prinzipien, sortierbar nach Erfolg."""

    patterns: List[ContentPattern] = field(default_factory=list)

    def add(self, pattern: ContentPattern) -> None:
        self.patterns.append(pattern)

    def best(self, topic_space: Optional[str] = None, limit: int = 3) -> List[ContentPattern]:
        """Die erfolgreichsten Prinzipien, optional gefiltert nach Themenfeld."""
        candidates = self.patterns
        if topic_space is not None:
            candidates = [p for p in candidates if p.topic_space == topic_space]
        return sorted(candidates, key=lambda p: p.score, reverse=True)[:limit]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        data = [_pattern_to_dict(p) for p in self.patterns]
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PatternLibrary":
        path = Path(path)
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(patterns=[_pattern_from_dict(d) for d in raw])


def _pattern_to_dict(p: ContentPattern) -> dict:
    d = asdict(p)
    d["hook"]["type"] = p.hook.type.value
    d["call_to_action"] = p.call_to_action.value
    return d


def _pattern_from_dict(d: dict) -> ContentPattern:
    hook = Hook(
        type=HookType(d["hook"]["type"]),
        principle=d["hook"]["principle"],
        seconds=d["hook"].get("seconds", 3.0),
    )
    return ContentPattern(
        name=d["name"],
        hook=hook,
        structure=list(d["structure"]),
        pacing_seconds=d["pacing_seconds"],
        call_to_action=CallToAction(d["call_to_action"]),
        target_audience=d["target_audience"],
        topic_space=d["topic_space"],
        notes=d.get("notes", ""),
        score=d.get("score", 0.0),
        samples=d.get("samples", 0),
    )


def starter_library() -> PatternLibrary:
    """Ein paar bewährte, generische Format-Prinzipien als Ausgangspunkt.

    Bewusst abstrakt gehalten — das sind allgemein bekannte Strukturen, kein
    Abbild eines bestimmten fremden Videos.
    """
    lib = PatternLibrary()
    lib.add(ContentPattern(
        name="Ergebnis-zuerst Mini-Tutorial",
        hook=Hook(HookType.RESULT_FIRST,
                  principle="Zeige in Sekunde 1 das fertige Ergebnis, dann erklären."),
        structure=["Ergebnis zeigen", "Problem benennen", "3 schnelle Schritte", "Ergebnis wiederholen"],
        pacing_seconds=22.0,
        call_to_action=CallToAction.SAVE,
        target_audience="Einsteiger im Themenfeld",
        topic_space="how-to",
    ))
    lib.add(ContentPattern(
        name="Provokante These + Beweis",
        hook=Hook(HookType.BOLD_CLAIM,
                  principle="Starte mit einer These, die der Zielgruppe widerspricht."),
        structure=["These", "Warum das überrascht", "Beweis/Beispiel", "Fazit + Frage"],
        pacing_seconds=30.0,
        call_to_action=CallToAction.COMMENT,
        target_audience="Fortgeschrittene mit Meinung",
        topic_space="opinion",
    ))
    lib.add(ContentPattern(
        name="Neugierlücke + Auflösung",
        hook=Hook(HookType.CURIOSITY_GAP,
                  principle="Deute etwas Unerwartetes an, löse es erst am Ende auf."),
        structure=["Andeutung", "Spannung aufbauen", "Auflösung", "CTA"],
        pacing_seconds=18.0,
        call_to_action=CallToAction.FOLLOW,
        target_audience="breites Publikum",
        topic_space="story",
    ))
    return lib
