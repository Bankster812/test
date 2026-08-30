# MoneyPrinterTurbo – Setup-Kit

Fertiges Setup für [MoneyPrinterTurbo](https://github.com/harry0703/MoneyPrinterTurbo):
Aus einem Thema entsteht ein 9:16-Kurzvideo mit **englischem Voiceover und
englischen Text-Overlays** – passend für YouTube Shorts, TikTok, Instagram
Reels und Facebook Reels.

Das Kit klont das Original-Repo, installiert alle Abhängigkeiten und schreibt
eine vorkonfigurierte `config.toml`. Der Upstream-Code wird **nicht** in dieses
Repo kopiert, damit `git pull` im Zielverzeichnis weiter funktioniert.

## Installation

```bash
bash tools/moneyprinterturbo/setup.sh
```

Das Skript installiert bei Bedarf `ffmpeg` und `uv`, klont nach
`~/MoneyPrinterTurbo`, installiert die Python-Abhängigkeiten und fragt zwei
API-Keys ab (siehe unten). Anderes Zielverzeichnis:
`MPT_DIR=/pfad/dahin bash tools/moneyprinterturbo/setup.sh`.

Nicht-interaktiv:

```bash
MPT_NONINTERACTIVE=1 MPT_PEXELS_KEY=xxx MPT_LLM_PROVIDER=gemini MPT_LLM_KEY=yyy \
  bash tools/moneyprinterturbo/setup.sh
```

## Nutzung

```bash
# Grafische Oberfläche (empfohlen zum Ausprobieren) → http://127.0.0.1:8501
bash tools/moneyprinterturbo/start.sh

# Oder direkt aus dem Terminal
bash tools/moneyprinterturbo/neues-video.sh "5 morning habits that changed my life"
bash tools/moneyprinterturbo/neues-video.sh "why most diets fail" 3   # 3 Varianten
```

Die fertigen Videos liegen in `~/MoneyPrinterTurbo/storage/tasks/<task-id>/final-1.mp4`.

Andere Stimme oder anderes Format:

```bash
MPT_VOICE=en-US-AndrewMultilingualNeural-Male MPT_ASPECT=16:9 \
  bash tools/moneyprinterturbo/neues-video.sh "topic"
```

## Was kostet nichts

| Baustein | Dienst | Kosten |
|---|---|---|
| Voiceover | Edge TTS (Microsoft) | gratis, kein Key |
| Untertitel | Edge / Whisper lokal | gratis |
| Videomaterial | Pexels, Pixabay, Coverr | gratis, Key nötig |
| Hintergrundmusik | mitgelieferte Tracks | gratis |
| Skript-Generierung | Gemini / Groq / Ollama | Gratis-Kontingent bzw. lokal |

Die zwei Keys, die `setup.sh` abfragt:

1. **Pexels** – <https://www.pexels.com/api/> (Registrierung, sofort nutzbar)
2. **Sprachmodell**, eine der Optionen:
   - `gemini` – <https://aistudio.google.com/app/apikey> (empfohlen, großzügiges Gratis-Kontingent)
   - `groq` – <https://console.groq.com/keys> (sehr schnell)
   - `ollama` – komplett lokal, kein Key; vorher `ollama pull llama3.1` und
     `ollama_model_name` in `config.toml` setzen

Keys lassen sich jederzeit in `~/MoneyPrinterTurbo/config.toml` ändern.
**Diese Datei nie committen** – sie enthält Klartext-Keys.

## Voreingestellte Werte

`setup.sh` schreibt über `apply_preset.py` folgendes Preset:

| Einstellung | Wert |
|---|---|
| Sprache Skript + Untertitel | `en-US` |
| Stimme | `en-US-AvaMultilingualNeural-Female` |
| Format | 9:16, Untertitel unten |
| Schrift | `BeVietnamPro-Bold.ttf`, 72 px, weiß, schwarze Kontur |
| Materialquelle | Pexels |
| Musik | zufälliger mitgelieferter Track, Lautstärke 0.18 |

Weitere englische Stimmen: `en-US-AndrewMultilingualNeural-Male`,
`en-US-JennyNeural-Female`, `en-US-GuyNeural-Male`, `en-GB-SoniaNeural-Female`.

Die Standard-Schrift ist bewusst nicht `MicrosoftYaHeiBold.ttc` – die ist auf
CJK ausgelegt. `BeVietnamPro-Bold` rendert lateinische Schrift sauber.

## Veröffentlichen auf den Plattformen

MoneyPrinterTurbo bringt eine Anbindung an [upload-post.com](https://upload-post.com/)
mit, die zu TikTok, Instagram und YouTube postet (`upload_post_*` in
`config.toml`). Der Dienst ist **kostenpflichtig**. Solange nur die
Video-Erstellung gratis sein soll: Dateien herunterladen und manuell hochladen.

## Warum MoneyPrinterTurbo und nicht MoneyPrinterV2

Beide Projekte wurden geprüft. Sie lösen unterschiedliche Aufgaben:

**MoneyPrinterTurbo** (harry0703, MIT-Lizenz) ist eine Video-Produktions-Pipeline:
WebUI, CLI und REST-API, ~20 austauschbare LLM-Anbieter, mehrere Materialquellen,
Untertitel-Engine, Hintergrundmusik, Batch-Betrieb. Genau das, was für
plattformübergreifenden Kurzvideo-Content gebraucht wird.

**MoneyPrinterV2** (FujiwaraChoki, AGPL-3.0) ist eher eine Automatisierungs-Suite:
Twitter-Bot, Affiliate-Marketing, Kaltakquise bei lokalen Firmen – und als ein
Modul davon ein YouTube-Shorts-Generator. Die Videoerzeugung ist deutlich
einfacher gehalten, und das Hochladen läuft über Selenium-Browserautomatisierung
mit einem echten Firefox-Profil. Das ist fragil und verstößt gegen die
Nutzungsbedingungen der Plattformen; im schlechtesten Fall kostet es den Account.

Kein Hybrid, aus zwei Gründen. Erstens funktional: MPV2 hat für die eigentliche
Videoerstellung nichts, was MPT nicht besser kann. Zweitens rechtlich: MPV2 steht
unter AGPL-3.0, MPT unter MIT. Code aus MPV2 in MPT zu übernehmen würde das
Ergebnis unter die AGPL zwingen, inklusive Offenlegungspflicht beim
Netzwerkbetrieb. Der einzige Teil von MPV2, der reizvoll wäre – der kostenlose
Auto-Uploader – ist genau der Teil, von dem abzuraten ist.

## Fehlersuche

**`edge_tts stream timed out` / Zertifikatsfehler** – Netzwerk blockiert die
WebSocket-Verbindung zu `speech.platform.bing.com`. In Firmennetzen oder hinter
einem MITM-Proxy das Proxy-CA in `certifi` eintragen oder `[proxy]` in
`config.toml` konfigurieren.

**`ffmpeg` nicht gefunden** – `ffmpeg_path` in `config.toml` auf die
ausführbare Datei setzen.

**Video kürzer als das Voiceover** – normal; die Pipeline loopt die Clips
automatisch, bis die Audiolänge erreicht ist.

**Port 8501 belegt** – `webui.sh` weicht selbstständig auf 8502–8599 aus.
