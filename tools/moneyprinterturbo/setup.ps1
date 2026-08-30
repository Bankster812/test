# MoneyPrinterTurbo - Ein-Kommando-Setup fuer Windows (PowerShell).
#
#   powershell -ExecutionPolicy Bypass -File tools\moneyprinterturbo\setup.ps1
#
# Nicht-interaktiv:
#   $env:MPT_PEXELS_KEY="..."; $env:MPT_LLM_KEY="..."
#   powershell -ExecutionPolicy Bypass -File tools\moneyprinterturbo\setup.ps1 -NonInteractive

[CmdletBinding()]
param(
    [string]$MptDir = "$HOME\MoneyPrinterTurbo",
    [switch]$NonInteractive
)

$ErrorActionPreference = "Stop"
$RepoUrl = "https://github.com/harry0703/MoneyPrinterTurbo.git"
$KitDir  = Split-Path -Parent $MyInvocation.MyCommand.Path

function Write-Step { param($m) Write-Host "==> $m" -ForegroundColor Cyan }
function Write-Warn { param($m) Write-Host "!!  $m" -ForegroundColor Yellow }
function Die       { param($m) Write-Host "XX  $m" -ForegroundColor Red; exit 1 }

function Test-Command { param($n) $null -ne (Get-Command $n -ErrorAction SilentlyContinue) }

# PATH im laufenden Prozess auffrischen. winget und die uv-Installer schreiben in
# die Benutzer-PATH-Variable, die erst eine neue Shell sieht - ohne das hier
# faende das Skript die gerade installierten Programme nicht.
function Update-Path {
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user    = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machine;$user;$HOME\.local\bin;$HOME\.cargo\bin"
}

# ------------------------------------------------------------------- ffmpeg
function Install-Ffmpeg {
    if (Test-Command ffmpeg) {
        Write-Step "ffmpeg gefunden: $((ffmpeg -version 2>&1 | Select-Object -First 1))"
        return
    }
    Write-Step "ffmpeg fehlt - installiere ueber winget ..."
    if (-not (Test-Command winget)) {
        Die @"
winget nicht verfuegbar. Zwei Auswege:
  1. 'App-Installer' aus dem Microsoft Store installieren, dann dieses Skript erneut starten.
  2. ffmpeg manuell von https://www.gyan.dev/ffmpeg/builds/ laden, entpacken,
     und den Pfad zur ffmpeg.exe spaeter in config.toml als ffmpeg_path eintragen.
"@
    }
    winget install --id Gyan.FFmpeg -e --source winget `
        --accept-package-agreements --accept-source-agreements
    Update-Path
    if (-not (Test-Command ffmpeg)) {
        Write-Warn "ffmpeg wurde installiert, ist aber noch nicht im PATH dieser Sitzung."
        Write-Warn "Schliesse dieses Fenster, oeffne ein neues PowerShell und starte das Skript erneut."
        exit 1
    }
}

# ----------------------------------------------------------------------- uv
function Install-Uv {
    if (Test-Command uv) {
        Write-Step "uv gefunden: $(uv --version)"
        return
    }
    Write-Step "uv fehlt - installiere (Paketmanager fuer Python) ..."
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    Update-Path
    if (-not (Test-Command uv)) { Die "uv-Installation fehlgeschlagen. Neues PowerShell oeffnen und erneut starten." }
}

# --------------------------------------------------------------- Repo holen
function Get-Repo {
    if (Test-Path (Join-Path $MptDir ".git")) {
        Write-Step "Update vorhandener Installation in $MptDir ..."
        git -C $MptDir pull --ff-only
        if ($LASTEXITCODE -ne 0) { Write-Warn "git pull uebersprungen (lokale Aenderungen?)." }
    } else {
        Write-Step "Klone MoneyPrinterTurbo nach $MptDir ..."
        git clone $RepoUrl $MptDir
        if ($LASTEXITCODE -ne 0) { Die "git clone fehlgeschlagen." }
    }
}

# -------------------------------------------------------------- Konfiguration
function Set-Config {
    $cfg = Join-Path $MptDir "config.toml"
    if (-not (Test-Path $cfg)) {
        Copy-Item (Join-Path $MptDir "config.example.toml") $cfg
        Write-Step "config.toml aus Vorlage erstellt."
    } else {
        Write-Step "Vorhandene config.toml wird beibehalten."
    }

    $pexels   = $env:MPT_PEXELS_KEY
    $provider = $env:MPT_LLM_PROVIDER
    $llmKey   = $env:MPT_LLM_KEY

    if (-not $NonInteractive) {
        Write-Host ""
        Write-Host "---------------- Kostenlose Zugaenge ----------------"
        Write-Host "1) Pexels (Videomaterial, gratis): https://www.pexels.com/api/"
        if (-not $pexels) { $pexels = Read-Host "   Pexels API-Key (Enter = spaeter eintragen)" }
        Write-Host ""
        Write-Host "2) Sprachmodell fuer die Skripte. Kostenlose Optionen:"
        Write-Host "     gemini  - Google AI Studio, grosszuegiges Gratis-Kontingent (empfohlen)"
        Write-Host "     groq    - sehr schnell, kostenloses Kontingent"
        Write-Host "     ollama  - komplett lokal, kein Key"
        if (-not $provider) { $provider = Read-Host "   Anbieter [gemini]" }
        if (-not $provider) { $provider = "gemini" }
        if ($provider -ne "ollama" -and -not $llmKey) {
            switch ($provider) {
                "gemini" { Write-Host "   Key holen: https://aistudio.google.com/app/apikey" }
                "groq"   { Write-Host "   Key holen: https://console.groq.com/keys" }
            }
            $llmKey = Read-Host "   API-Key (Enter = spaeter eintragen)"
        }
        Write-Host "-----------------------------------------------------"
        Write-Host ""
    }
    if (-not $provider) { $provider = "gemini" }

    $env:MPT_CFG          = $cfg
    $env:MPT_PEXELS_KEY   = $pexels
    $env:MPT_LLM_PROVIDER = $provider
    $env:MPT_LLM_KEY      = $llmKey

    uv run --project $MptDir python (Join-Path $KitDir "apply_preset.py")
    if ($LASTEXITCODE -ne 0) { Die "Preset konnte nicht geschrieben werden." }
}

# --------------------------------------------------------------------- main
if (-not (Test-Command git)) { Die "git fehlt. Installieren: winget install --id Git.Git -e" }
Install-Ffmpeg
Install-Uv
Get-Repo

Write-Step "Installiere Python-Abhaengigkeiten (beim ersten Mal einige Minuten) ..."
Push-Location $MptDir
try {
    uv sync --frozen
    if ($LASTEXITCODE -ne 0) { Die "uv sync fehlgeschlagen." }
} finally { Pop-Location }

Set-Config

Write-Host ""
Write-Host "Fertig." -ForegroundColor Green
Write-Host "MoneyPrinterTurbo liegt in: $MptDir"
Write-Host ""
Write-Host "  WebUI starten:      $MptDir\webui.bat"
Write-Host "  Video per Kommando: powershell -File $KitDir\neues-video.ps1 ""Dein Thema"""
Write-Host ""
Write-Host "Konfiguration aendern: $MptDir\config.toml"
