#!/usr/bin/env bash
# Erzeugt ein fertiges 9:16-Kurzvideo mit englischem Voiceover und englischen
# Untertiteln – ohne WebUI, direkt aus dem Terminal.
#
#   bash neues-video.sh "5 morning habits that changed my life"
#   bash neues-video.sh "topic" 3          # 3 Varianten desselben Themas
#
# Umgebungsvariablen:
#   MPT_DIR     Installationsverzeichnis (Standard: ~/MoneyPrinterTurbo)
#   MPT_VOICE   Edge-TTS-Stimme (Standard: en-US-AvaMultilingualNeural-Female)
#   MPT_ASPECT  9:16 (Standard) | 16:9 | 1:1

set -euo pipefail

MPT_DIR="${MPT_DIR:-$HOME/MoneyPrinterTurbo}"
VOICE="${MPT_VOICE:-en-US-AvaMultilingualNeural-Female}"
ASPECT="${MPT_ASPECT:-9:16}"
SUBJECT="${1:-}"
COUNT="${2:-1}"

if [ -z "$SUBJECT" ]; then
  echo "Nutzung: bash neues-video.sh \"Dein Thema auf Englisch\" [Anzahl]" >&2
  exit 1
fi
[ -d "$MPT_DIR" ] || { echo "Nicht gefunden: $MPT_DIR – erst setup.sh ausführen." >&2; exit 1; }

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
cd "$MPT_DIR"

# --video-language en-US erzwingt das englische Skript unabhängig davon, in
# welcher Sprache das Thema formuliert ist.
uv run python cli.py \
  --video-subject "$SUBJECT" \
  --video-language "en-US" \
  --voice-name "$VOICE" \
  --video-aspect "$ASPECT" \
  --video-count "$COUNT" \
  --font-name "BeVietnamPro-Bold.ttf" \
  --subtitle-enabled \
  --bgm-type random
