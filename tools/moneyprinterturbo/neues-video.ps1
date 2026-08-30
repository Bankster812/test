# Erzeugt ein fertiges 9:16-Kurzvideo mit englischem Voiceover und englischen
# Untertiteln - ohne WebUI, direkt aus PowerShell.
#
#   powershell -File neues-video.ps1 "5 morning habits that changed my life"
#   powershell -File neues-video.ps1 "topic" -Count 3

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Subject,
    [int]$Count = 1,
    [string]$MptDir = "$HOME\MoneyPrinterTurbo",
    [string]$Voice  = "en-US-AvaMultilingualNeural-Female",
    [ValidateSet("9:16", "16:9", "1:1")]
    [string]$Aspect = "9:16"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $MptDir)) {
    Write-Host "Nicht gefunden: $MptDir - erst setup.ps1 ausfuehren." -ForegroundColor Red
    exit 1
}

$env:Path = "$env:Path;$HOME\.local\bin;$HOME\.cargo\bin"

Push-Location $MptDir
try {
    # --video-language en-US erzwingt das englische Skript unabhaengig davon,
    # in welcher Sprache das Thema formuliert ist.
    uv run python cli.py `
        --video-subject $Subject `
        --video-language "en-US" `
        --voice-name $Voice `
        --video-aspect $Aspect `
        --video-count $Count `
        --font-name "BeVietnamPro-Bold.ttf" `
        --subtitle-enabled `
        --bgm-type random
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally { Pop-Location }
