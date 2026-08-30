#!/usr/bin/env bash
# Startet die MoneyPrinterTurbo WebUI und öffnet sie im Browser.
set -euo pipefail
MPT_DIR="${MPT_DIR:-$HOME/MoneyPrinterTurbo}"
[ -d "$MPT_DIR" ] || { echo "Nicht gefunden: $MPT_DIR – erst setup.sh ausführen." >&2; exit 1; }
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
exec bash "$MPT_DIR/webui.sh"
