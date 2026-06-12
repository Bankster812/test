# Content Automation Toolkit

Ein **legales, nachhaltiges** Toolkit, um aus dem *Erfolgsprinzip* guter Inhalte
eigene, neue Posts/Reels zu erstellen, über die **offizielle Instagram Graph API**
zu veröffentlichen und die Performance auszuwerten — ohne fremden Content zu kopieren.

> Kernidee: Wir analysieren **Struktur und Format** (Hook, Aufbau, Pacing, Call-to-Action),
> nicht das konkrete Material. Daraus entstehen **eigene** Inhalte mit eigenen Motiven.

## Was dieses Toolkit ausdrücklich NICHT tut

- Es lädt keine fremden Videos herunter, schneidet sie um und postet sie neu.
  Das wäre Urheberrechtsverletzung und führt zu Account-Sperren.
- Es scrapt Instagram nicht über die normale App/Webseite. Das verstößt gegen die
  Nutzungsbedingungen.
- Es kauft keine Follower und automatisiert kein Spam-Verhalten.

All das würde dein Ziel (echte Reichweite, Monetarisierung) sabotieren, nicht erreichen.

## Der legale Workflow

```
1. ANALYSE   →  Erfolgsprinzip eines Formats strukturiert erfassen
                (was macht den Hook stark? Aufbau? CTA?)  → content_patterns.py
2. BRIEF     →  Aus dem Prinzip + DEINEM neuen Thema einen
                originalen Skript-/Post-Brief generieren     → brief_generator.py
3. PRODUKTION→  Du (oder ein Tool) erstellst das eigene Video/Bild
                (eigene Aufnahmen, eigene Motive)
4. POSTEN    →  Über die offizielle Graph API veröffentlichen → instagram_publisher.py
5. MESSEN    →  Insights ziehen, Prinzipien nach Erfolg gewichten → insights.py
6. WIEDERHOLEN → pipeline.py orchestriert die Schleife
```

## Setup

```bash
pip install -r requirements.txt
```

Für das Posten/Auslesen brauchst du:

1. Einen **Instagram-Business- oder Creator-Account**, verknüpft mit einer Facebook-Seite.
2. Eine **Meta-Developer-App** (https://developers.facebook.com) mit den Berechtigungen
   `instagram_basic`, `instagram_content_publish`, `instagram_manage_insights`.
3. Ein **langlebiges Access Token** und deine **IG User ID**.

Lege diese als Umgebungsvariablen ab (oder in eine nicht eingecheckte `.env`):

```bash
export IG_USER_ID="17841400000000000"
export IG_ACCESS_TOKEN="EAAB..."
```

## Schnellstart

```bash
# Demo ohne echte API-Calls (zeigt Analyse → Brief)
python -m content_automation.pipeline --demo

# Echten Reel-Post planen (benötigt gültige Credentials)
python -m content_automation.pipeline --publish-reel \
    --video-url "https://deinserver.example/dein-eigenes-reel.mp4" \
    --caption "Dein origineller Text #deinhashtag"
```

## Dateien

| Datei                     | Zweck                                                       |
|---------------------------|-------------------------------------------------------------|
| `content_patterns.py`     | Erfolgsprinzipien strukturiert erfassen & bewerten          |
| `brief_generator.py`      | Aus Prinzip + neuem Thema einen originalen Brief erzeugen   |
| `instagram_publisher.py`  | Eigene Inhalte über die offizielle Graph API veröffentlichen|
| `insights.py`             | Performance-Daten ziehen und Prinzipien gewichten           |
| `pipeline.py`             | Den gesamten Loop orchestrieren + CLI                       |

## Rechtlicher Hinweis

Du bist verantwortlich für die Inhalte, die du veröffentlichst. Dieses Toolkit ist
darauf ausgelegt, **eigene** Inhalte zu erstellen, die sich an erfolgreichen *Formaten*
orientieren — nicht daran, fremdes Material zu reproduzieren. Halte dich an die
Instagram-Plattformrichtlinien und das Urheberrecht.
