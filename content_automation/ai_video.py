"""Aus einem Content-Brief eine fertige KI-Video-Produktionsanleitung erzeugen.

Wandelt einen Brief (siehe brief_generator) in konkrete Text-to-Video-Prompts,
ein Voiceover-Skript und On-Screen-Text um — einsetzbar in kostenlosen KI-Tools.

Erzeugt NEUE, eigene Videos im erfolgreichen Format. Es wird kein fremdes Reel
reproduziert und kein fremdes Material eingespeist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .brief_generator import ContentBrief


# Kostenlose / mit Gratis-Kontingent nutzbare KI-Video-Tools (Stand pflegbar).
# Kein Tool wird von hier automatisch bedient — die Prompts sind so gebaut, dass
# du sie 1:1 hineinkopieren kannst.
FREE_AI_VIDEO_TOOLS = [
    ("Runway", "Text-to-Video / Image-to-Video, kurzer Clip, gutes Motion", "runwayml.com"),
    ("Pika", "Kurze, stilisierte Clips aus Text/Bild", "pika.art"),
    ("Kling", "Längere, realistische Text-to-Video-Clips (Gratis-Credits)", "klingai.com"),
    ("Luma Dream Machine", "Text/Image-to-Video, flüssige Kamerafahrten", "lumalabs.ai"),
    ("Canva Magic Media", "Video + Text-Overlays + Musik in einem Editor", "canva.com"),
    ("CapCut", "Zusammenschnitt, Untertitel, TTS-Voiceover (kostenlos)", "capcut.com"),
    ("ElevenLabs", "KI-Voiceover, natürliche Stimme (Gratis-Kontingent)", "elevenlabs.io"),
]


@dataclass
class VideoScene:
    """Eine Szene = ein Beat aus dem Brief, übersetzt in einen Video-Prompt."""

    index: int
    seconds: float
    visual_prompt: str      # Text-to-Video-Prompt für DIESE Szene
    on_screen_text: str     # eingeblendeter Text
    voiceover: str          # gesprochener Satz


@dataclass
class AIVideoPlan:
    title: str
    aspect_ratio: str
    total_seconds: float
    scenes: List[VideoScene]
    voiceover_script: str
    caption: str
    hashtags: List[str] = field(default_factory=list)

    def to_text(self) -> str:
        lines = [
            f"# KI-Video-Plan: {self.title}",
            f"Format: {self.aspect_ratio} (Hochkant, für Reels/Shorts/TikTok)",
            f"Gesamtlänge: ~{self.total_seconds:.0f}s",
            "",
            "## Szenen (je eine in ein Text-to-Video-Tool geben)",
        ]
        for s in self.scenes:
            lines += [
                f"\nSzene {s.index} (~{s.seconds:.0f}s):",
                f"  VISUAL-PROMPT : {s.visual_prompt}",
                f"  ON-SCREEN-TEXT: {s.on_screen_text}",
                f"  VOICEOVER     : {s.voiceover}",
            ]
        lines += [
            "",
            "## Voiceover-Skript (am Stück, z.B. für ElevenLabs/CapCut-TTS)",
            self.voiceover_script,
            "",
            f"## Caption: {self.caption}",
            f"## Hashtags: {' '.join(self.hashtags)}",
            "",
            "## Kostenlose Tools",
        ]
        for name, desc, url in FREE_AI_VIDEO_TOOLS:
            lines.append(f"  - {name} ({url}): {desc}")
        lines += [
            "",
            "Workflow: 1) Szenen einzeln generieren  2) in CapCut/Canva schneiden  "
            "3) Voiceover + On-Screen-Text drauf  4) als MP4 exportieren  "
            "5) mit instagram_publisher.py posten.",
            "",
            "Hinweis: NEUES Video im Format — kein fremdes Reel reproduzieren.",
        ]
        return "\n".join(lines)


def _visual_prompt_for(beat: str, subject: str) -> str:
    """Aus einem Storyboard-Beat einen anschaulichen Text-to-Video-Prompt bauen."""
    beat_clean = beat.strip().rstrip(".")
    return (
        f"Cinematic vertical 9:16 clip about '{subject}'. Scene: {beat_clean}. "
        f"Bright, modern, high-contrast, dynamic camera, social-media style, no text overlay."
    )


def plan_from_brief(brief: ContentBrief) -> AIVideoPlan:
    """Erzeuge aus einem Brief einen kompletten KI-Video-Produktionsplan."""
    n = max(1, len(brief.beats))
    per_scene = brief.target_seconds / n

    scenes: List[VideoScene] = []
    voiceover_parts: List[str] = []

    # Szene 1 trägt den Hook als Voiceover.
    for i, beat in enumerate(brief.beats, 1):
        if i == 1:
            vo = brief.hook_line
        elif i == len(brief.beats):
            vo = brief.cta_line
        else:
            vo = f"{beat}."
        voiceover_parts.append(vo)
        scenes.append(VideoScene(
            index=i,
            seconds=round(per_scene, 1),
            visual_prompt=_visual_prompt_for(beat, brief.subject),
            on_screen_text=beat if i > 1 else brief.hook_line,
            voiceover=vo,
        ))

    return AIVideoPlan(
        title=brief.title,
        aspect_ratio="9:16",
        total_seconds=brief.target_seconds,
        scenes=scenes,
        voiceover_script=" ".join(voiceover_parts),
        caption=brief.suggested_caption,
        hashtags=brief.suggested_hashtags,
    )
