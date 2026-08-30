"""Schreibt das "kostenlos + international" Preset in eine bestehende config.toml.

Bewusst zeilenbasiert statt über einen TOML-Writer: die Vorlage von
MoneyPrinterTurbo besteht zum größten Teil aus Kommentaren, die bei einem
Round-Trip durch tomli-w verloren gingen. Wir ersetzen nur die Werte von
Schlüsseln, die wir kennen, und lassen alles andere unangetastet.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# Ausgabe-Defaults: englisches Voiceover + englische Untertitel, 9:16 Hochformat.
# BeVietnamPro-Bold ist die mitgelieferte Schrift mit sauberer lateinischer
# Glyphenabdeckung; die MicrosoftYaHei-Voreinstellung ist auf CJK ausgelegt.
UI_PRESET = {
    "video_language": '"en-US"',
    "voice_name": '"en-US-AvaMultilingualNeural-Female"',
    "font_name": '"BeVietnamPro-Bold.ttf"',
    "video_aspect_pexels": '"9:16"',
    "subtitle_enabled": "true",
    "subtitle_position": '"bottom"',
    "font_size": "72",
    "text_fore_color": '"#FFFFFF"',
    "stroke_color": '"#000000"',
    "stroke_width": "2.0",
    "video_clip_duration": "3",
    "bgm_type": '"random"',
    "bgm_volume": "0.18",
}

ROOT_PRESET = {
    "subtitle_provider": '"edge"',
    "video_source": '"pexels"',
}


def set_key(lines: list[str], key: str, value: str, section: str | None) -> list[str]:
    """Setzt key = value innerhalb von section (None = vor der ersten Section).

    Auskommentierte Vorgaben (`# font_size = 60`) werden reaktiviert, damit das
    Preset nicht unter einem toten Kommentar begraben wird.
    """
    in_section = section is None
    pattern = re.compile(rf"^\s*#?\s*{re.escape(key)}\s*=")
    header = re.compile(r"^\s*\[([^\]]+)\]\s*$")
    out: list[str] = []
    done = False
    insert_at = None

    for i, line in enumerate(lines):
        m = header.match(line)
        if m:
            if in_section and not done and insert_at is None:
                insert_at = i
            in_section = m.group(1) == section
        elif in_section and not done and pattern.match(line):
            out.append(f"{key} = {value}\n")
            done = True
            continue
        out.append(line)

    if not done:
        if section is None:
            out.insert(insert_at if insert_at is not None else 0, f"{key} = {value}\n")
        else:
            # Section anlegen, falls sie fehlt.
            if not any(header.match(l) and header.match(l).group(1) == section for l in out):
                out.append(f"\n[{section}]\n")
            for i, line in enumerate(out):
                m = header.match(line)
                if m and m.group(1) == section:
                    out.insert(i + 1, f"{key} = {value}\n")
                    break
    return out


def main() -> int:
    cfg_path = Path(os.environ["MPT_CFG"])
    lines = cfg_path.read_text(encoding="utf-8").splitlines(keepends=True)

    for key, value in ROOT_PRESET.items():
        lines = set_key(lines, key, value, None)
    for key, value in UI_PRESET.items():
        lines = set_key(lines, key, value, "ui")

    applied = ["englisches Voiceover + Untertitel (en-US), 9:16, Pexels, Edge-TTS"]

    pexels = os.environ.get("MPT_PEXELS_KEY", "").strip()
    if pexels:
        lines = set_key(lines, "pexels_api_keys", f'["{pexels}"]', None)
        applied.append("Pexels-Key gesetzt")

    provider = os.environ.get("MPT_LLM_PROVIDER", "").strip()
    llm_key = os.environ.get("MPT_LLM_KEY", "").strip()
    if provider:
        lines = set_key(lines, "llm_provider", f'"{provider}"', None)
        applied.append(f"LLM-Anbieter: {provider}")
        if llm_key:
            lines = set_key(lines, f"{provider}_api_key", f'"{llm_key}"', None)
            applied.append("LLM-Key gesetzt")

    cfg_path.write_text("".join(lines), encoding="utf-8")

    print("Preset geschrieben nach", cfg_path)
    for item in applied:
        print("  -", item)
    if not pexels:
        print("  ! Kein Pexels-Key: pexels_api_keys in config.toml nachtragen.")
    if provider and provider != "ollama" and not llm_key:
        print(f"  ! Kein LLM-Key: {provider}_api_key in config.toml nachtragen.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
