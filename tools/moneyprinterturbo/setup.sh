#!/usr/bin/env bash
# MoneyPrinterTurbo – Ein-Kommando-Setup für macOS und Linux.
#
#   bash tools/moneyprinterturbo/setup.sh
#
# Nicht-interaktiv (z. B. für Server):
#   MPT_DIR=~/MoneyPrinterTurbo MPT_NONINTERACTIVE=1 \
#   MPT_PEXELS_KEY=... MPT_LLM_PROVIDER=gemini MPT_LLM_KEY=... \
#     bash tools/moneyprinterturbo/setup.sh

set -euo pipefail

REPO_URL="https://github.com/harry0703/MoneyPrinterTurbo.git"
MPT_DIR="${MPT_DIR:-$HOME/MoneyPrinterTurbo}"
KIT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

log()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m  %s\n' "$*"; }
die()  { printf '\033[1;31mXX\033[0m  %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 1. ffmpeg
ensure_ffmpeg() {
  if command -v ffmpeg >/dev/null 2>&1; then
    log "ffmpeg gefunden: $(ffmpeg -version | head -1)"
    return
  fi
  log "ffmpeg fehlt – installiere ..."
  if [ "$(uname -s)" = "Darwin" ]; then
    command -v brew >/dev/null 2>&1 || die "Homebrew fehlt. Installiere es von https://brew.sh und starte das Skript erneut."
    brew install ffmpeg
  elif command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -qq && sudo apt-get install -y --no-install-recommends ffmpeg
  elif command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y ffmpeg
  elif command -v pacman >/dev/null 2>&1; then
    sudo pacman -S --noconfirm ffmpeg
  else
    die "Kein bekannter Paketmanager. Bitte ffmpeg manuell installieren: https://ffmpeg.org/download.html"
  fi
  command -v ffmpeg >/dev/null 2>&1 || die "ffmpeg-Installation fehlgeschlagen."
}

# ------------------------------------------------------------------- 2. uv
ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    log "uv gefunden: $(uv --version)"
    return
  fi
  log "uv fehlt – installiere (Paketmanager für Python) ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  command -v uv >/dev/null 2>&1 || die "uv-Installation fehlgeschlagen. Öffne ein neues Terminal und starte erneut."
}

# ------------------------------------------------------------ 3. Repo holen
ensure_repo() {
  if [ -d "$MPT_DIR/.git" ]; then
    log "Update vorhandener Installation in $MPT_DIR ..."
    git -C "$MPT_DIR" pull --ff-only || warn "git pull übersprungen (lokale Änderungen?)."
  else
    log "Klone MoneyPrinterTurbo nach $MPT_DIR ..."
    git clone "$REPO_URL" "$MPT_DIR"
  fi
}

# ------------------------------------------------------- 4. Abhängigkeiten
install_deps() {
  log "Installiere Python-Abhängigkeiten (dauert beim ersten Mal einige Minuten) ..."
  ( cd "$MPT_DIR" && uv sync --frozen )
}

# ------------------------------------------------------------- 5. Konfig
configure() {
  local cfg="$MPT_DIR/config.toml"
  if [ ! -f "$cfg" ]; then
    cp "$MPT_DIR/config.example.toml" "$cfg"
    log "config.toml aus Vorlage erstellt."
  else
    log "Vorhandene config.toml wird beibehalten (nur fehlende Werte werden ergänzt)."
  fi

  local pexels="${MPT_PEXELS_KEY:-}" provider="${MPT_LLM_PROVIDER:-}" llmkey="${MPT_LLM_KEY:-}"

  if [ "${MPT_NONINTERACTIVE:-0}" != "1" ]; then
    echo
    echo "──────────── Kostenlose Zugänge ────────────"
    echo "1) Pexels (Videomaterial, gratis): https://www.pexels.com/api/"
    [ -n "$pexels" ] || read -r -p "   Pexels API-Key (Enter = später eintragen): " pexels
    echo
    echo "2) Sprachmodell für die Skripte. Kostenlose Optionen:"
    echo "     gemini  – Google AI Studio, großzügiges Gratis-Kontingent (empfohlen)"
    echo "     groq    – sehr schnell, kostenloses Kontingent"
    echo "     ollama  – komplett lokal, kein Key, benötigt Ollama + Modell"
    [ -n "$provider" ] || read -r -p "   Anbieter [gemini]: " provider
    provider="${provider:-gemini}"
    if [ "$provider" != "ollama" ] && [ -z "$llmkey" ]; then
      case "$provider" in
        gemini) echo "   Key holen: https://aistudio.google.com/app/apikey" ;;
        groq)   echo "   Key holen: https://console.groq.com/keys" ;;
      esac
      read -r -p "   API-Key (Enter = später eintragen): " llmkey
    fi
    echo "────────────────────────────────────────────"
    echo
  fi

  MPT_CFG="$cfg" MPT_PEXELS_KEY="$pexels" MPT_LLM_PROVIDER="${provider:-}" MPT_LLM_KEY="$llmkey" \
    uv run --project "$MPT_DIR" python "$KIT_DIR/apply_preset.py"
}

# ------------------------------------------------------------- 6. Start
main() {
  command -v git >/dev/null 2>&1 || die "git fehlt."
  ensure_ffmpeg
  ensure_uv
  ensure_repo
  install_deps
  configure

  # printf statt cat<<HEREDOC: ein Heredoc gibt \033 wörtlich aus.
  printf '\n\033[1;32mFertig.\033[0m MoneyPrinterTurbo liegt in: %s\n\n' "$MPT_DIR"
  cat <<MSG
  WebUI starten:      bash $KIT_DIR/start.sh
  Video per Kommando: bash $KIT_DIR/neues-video.sh "Dein Thema"

Konfiguration nachträglich ändern: $MPT_DIR/config.toml
MSG
}

main "$@"
