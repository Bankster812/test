# Maskenmodell — Stark-EX-Merkmale auf einer KTM-EXC-Maske

Parametrisches, druckbares 3D-Modell einer Scheinwerfermaske mit den beiden
Merkmalen, die übertragen werden sollten:

1. **Blinkerhalter-Aussparungen** — je drei offene Schlitze übereinander in
   der Seitenkante, in die sich der Halter von außen einschiebt. Die drei
   Positionen sind die Höhenverstellung.
2. **Rückseite** — oberer Befestigungsbock mittig, vier Schraubdome rings um
   den Scheinwerferausschnitt, dazu frei setzbare Befestigungslaschen.

Alles kommt aus `params.json`. Kein Maß ist hart verdrahtet.

---

## Stand der Dinge, ehrlich

**Was aus den Bildern stammt:** Form und Merkmale sind aus den
Referenzfotos abgegriffen (Carbon-Version von CMT Composit, dazu ein
Kunststoff-Render desselben Teils). Der Charakter der Form stimmt: breite
flache Oberkante, ausgestellte Schultern, größte Breite im oberen Drittel,
Auslauf nach unten, hohes schmales Oval im unteren Drittel.

**Was geschätzt ist:** die absoluten Maße. Aus Fotos lässt sich kein
Maßstab gewinnen. Voreingestellt sind 245 × 232 × 72 mm — plausibel, aber
nicht gemessen.

**Zur Links/Rechts-Frage:** auf dem Kunststoff-Render ist eine Seite
abweichend ausgeführt (zusätzlicher flacher Halter mit Doppelloch). Die
Carbon-Version ist symmetrisch mit je drei Schlitzen — die ist hier
umgesetzt, auf beiden Seiten gleich. `selftest.py` prüft die Symmetrie
ausdrücklich nach.

**Nicht erreichbar:** `C:\Users\schwa\OneDrive\Documents\MASKE STARK VARG EX`.
Das Modell läuft in einem entfernten Container ohne Zugriff auf deinen
Rechner. Liegen dort STL/STEP/OBJ der Maske, häng sie an — dann übernimmt
`apply_to_scan.py` die Merkmale in die echte Geometrie, und das
Nachbauen entfällt.

---

## Schnellstart

```bash
pip install -r requirements.txt

python3 build.py                 # STEP + STL + SVG + Konturen nach out/
python3 selftest.py              # misst das Ergebnis nach
```

Einzelne Werte ändern, ohne die Datei anzufassen:

```bash
python3 build.py --set wall=3.4 --set slot_depth=16 --set slot_pitch=18
python3 build.py --split --gauges --outlines
```

## Der schnellste Weg zur richtigen Form

Nicht Maße abtippen — **fotografieren**:

```bash
python3 trace_outline.py --bild maske.jpg --breite 245 --vorschau
python3 build.py
```

Maske flach auf hellen Untergrund, von gerade oben fotografieren, möglichst
weit weg und herangezoomt. Das Werkzeug erkennt den Umriss, rechnet ihn auf
die Einheitsbox und schreibt ihn in `params.json`; mit `--breite` setzt es
zugleich den Maßstab. `--vorschau` legt den gefundenen Umriss zur Kontrolle
über das Foto.

Danach nur noch die Merkmale nachmessen (→ `MESSANLEITUNG.md`): Lage und
Größe der Schlitze, Scheinwerferausschnitt, Befestigungspunkte.

## Ablauf bis zum passenden Teil

1. **Umriss** über `trace_outline.py`, oder Maße von Hand eintragen.
2. **Prüfen** mit `python3 selftest.py`.
3. **Passproben drucken** — `python3 build.py --gauges` erzeugt vier kleine
   Ausschnitte (je 20–40 min statt Stunden):
   - `probe_aussparungen_rechts` / `_links` — Blinkerhalter einschieben
   - `probe_befestigungsbock` — oberer Anbindungspunkt
   - `probe_scheinwerferdom` — Kante des Ausschnitts und ein Dom
4. **Korrigieren**, 2–3 wiederholen.
5. **Ganze Maske drucken.**

Schritt 3 ist der, den man gern überspringt und dann zweimal acht Stunden
druckt.

## Übertragung auf eine echte Maske

Sobald ein Netz der echten Maske vorliegt:

```bash
python3 apply_to_scan.py --scan maske.stl --align auto
python3 apply_to_scan.py --scan maske.stl --nur aussparungen
```

Das Netz muss **geschlossen** sein — boolesche Operationen brauchen das.
Sonst vorher in Meshmixer, Blender oder dem PrusaSlicer reparieren.
`--align auto` legt X/Y mittig und die Hinterkante auf Z = 0; Feinjustage
über `--rotate`, `--translate`, `--scale`. Angeformte Felder werden an der
konvexen Hülle des Scans bündig beschnitten.

## Drucken

| | Empfehlung | warum |
|---|---|---|
| Material | **ASA**, ersatzweise PETG | UV, Wärme, Vibration. PLA kriecht am Motorrad schon in der Sonne |
| Düse / Schicht | 0,4–0,6 mm / 0,2–0,25 mm | |
| Wandlinien | so viele, dass 3,0 mm voll werden (0,45 mm → 7) | die Festigkeit sitzt in den Wänden |
| Infill | 15–20 %, Gyroid | |
| Lage | **Vorderseite (Wölbung) nach unten** | die Flanken laufen dann flach an, die Sichtfläche bekommt die glatte Druckbettoberfläche |
| Stützen | für die Dome und den Bock innen | mit `--rear none` und `--set light_boss=false` entfällt fast alles |
| Nachbearbeitung | Schlitze mit einer Feile auf Endmaß | gedruckte Schlitze fallen zu eng aus |

**Wärme:** mit H4-Halogen kann ASA in Lampennähe weich werden. Mit LED-Einsatz
kein Thema.

### Kleines Druckbett

Die Maske ist rund 245 × 233 × 72 mm. Passt das nicht:

```bash
python3 build.py --split
```

Teilt bei X = 0 in zwei Hälften mit Steckzungen (Spiel über
`split_clearance`, Standard 0,2 mm). Zungenpositionen an der gekrümmten
Unterkante, wo an der Trennebene kein Material steht, werden automatisch
übersprungen. Verkleben: ASA mit Aceton-Schlicker oder 2K-Epoxid, Naht innen
mit Glasgewebe und Epoxid hinterlegen. Jede Hälfte ist ~139 × 233 × 71 mm.

## Blinker und Zulassung

Die Schlitze nehmen einen Halter auf — wie weit der Blinker damit nach außen
kommt, hängt an deinem Halter, nicht am Modell. Wer stattdessen einen
Blinker mit Gewindeschaft direkt anschrauben will, schaltet über
`--stalk` die Schaftaufnahme dazu; dann rechnet `selftest.py` den Abstand
der leuchtenden Flächen gegen **240 mm** — den Wert, der für vordere
Fahrtrichtungsanzeiger am Krad üblicherweise zwischen den Innenkanten
verlangt wird.

Das ist eine **Rechenhilfe, keine Zulassungsaussage.** Maßgeblich ist dein
Prüfer; je nach Ausführung kommen Anforderungen an Höhe über Fahrbahn,
Abstand zum Abblendlicht und Sichtwinkel hinzu.

## Dateien

```
params.json          alle Maße — deine Stellschraube
trace_outline.py     Umriss aus einem Foto abgreifen
build.py             erzeugt STEP / STL / SVG / DXF, Teilung, Passproben
selftest.py          misst nach: Wandstärke, Aussparungen, Symmetrie, Teilung
apply_to_scan.py     überträgt die Merkmale auf ein echtes Netz
MESSANLEITUNG.md     was am realen Teil zu messen ist
ktm_mask/
  geometry.py        Kontur, Parallelflächen — reine Mathematik
  params.py          Parametermodell samt Plausibilitätsprüfung
  shell.py           Grundkörper, Kavität, Scheinwerferausschnitt
  slots.py           Blinkerhalter-Aussparungen (Übertragung Stark EX)
  rearmount.py       Bock, Dome, Laschen (Übertragung Rückseite)
  blinker.py         optionale Schaftaufnahme für Schraubblinker
  features.py        Merkmalsliste — gilt für beide Wege
  assembly.py        Zusammenbau, Teilung, Bericht
  gauges.py          Passproben
out/                 Ergebnisse
```

`out/*.step` öffnest du in Fusion 360, FreeCAD, SolidWorks oder Onshape.
`out/*.stl` geht direkt in den Slicer. `out/kontur_*.dxf` sind
maßstabsgetreue Schablonen zum Ausdrucken bei 100 %.

## Was das Modell nicht ist

- **Keine Kopie einer OEM-Geometrie.** Die CAD-Daten von KTM und Stark
  Future sind nicht öffentlich. Das hier ist eine eigene Konstruktion, die
  die auf den Fotos sichtbaren Merkmale nachbildet.
- **Nicht am Fahrzeug erprobt.** Passung und Festigkeit musst du prüfen.
- **Kein Ersatz für die Abnahme.** Umbauten an der Beleuchtung sind
  abnahmepflichtig.
