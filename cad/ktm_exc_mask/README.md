# Modifizierte KTM-EXC-Scheinwerfermaske

Parametrisches, druckbares 3D-Modell einer EXC-Scheinwerfermaske mit zwei
Übertragungen:

1. **Rückseite** — eine austauschbare hintere Aufnahme (Rahmen, Laschen oder
   Gummiband-Durchbrüche), die an die Stelle der originalen EXC-Rückseite tritt.
2. **Seitliche Blinker-Aussparungen** im Stil der Stark-EX-Maske: flach
   ausgefräste Tasche in der Flanke, vollflächig hinterlegt, mit
   Schaftbohrung und Verdrehsicherung.

Alles wird aus `params.json` erzeugt. Kein Wert ist hart verdrahtet.

---

## Zuerst das Wichtige: die Grundform ist geschätzt

Die Bilder, auf die sich dein „hieraus“ bezog, sind bei mir nicht angekommen —
im Arbeitsverzeichnis lag nichts. Die Maße der **Grundform** in `params.json`
sind deshalb plausible Schätzwerte in der Größenordnung einer EXC-Maske
(241 × 228 × 115 mm), **keine abgenommenen Maße**. Sie passen so nicht an dein
Motorrad.

Was dagegen fertig und belastbar ist: die gesamte Mechanik drumherum. Du
misst deine Maske ab (→ `MESSANLEITUNG.md`), trägst die Werte ein, und das
Modell baut sich mit deinen Maßen neu auf — inklusive Nachmessen des
Ergebnisses. Der Weg dahin ist unten beschrieben.

Wenn du mir die Bilder nachreichst oder — deutlich besser — einen Scan deiner
EXC-Maske als STL hast, ist `apply_to_scan.py` der direkte Weg: dann wird
nichts nachgebaut, sondern die beiden Merkmale werden in dein echtes Netz
hineingerechnet.

---

## Schnellstart

```bash
pip install cadquery numpy trimesh manifold3d

cd cad/ktm_exc_mask

python3 build.py                 # STEP + STL + SVG-Ansichten nach out/
python3 selftest.py              # misst das Ergebnis nach
```

Einzelne Werte ändern, ohne die Datei anzufassen:

```bash
python3 build.py --set wall=3.6 --set blinker_z=-42 --set pocket_w=52
python3 build.py --rear tabs --split --gauges
```

## Der Ablauf, der zum passenden Teil führt

1. **Messen.** `MESSANLEITUNG.md` durchgehen, Werte in `params.json` eintragen.
2. **Prüfen.** `python3 selftest.py` — meldet zu dünne Wände, durchbrechende
   Taschen, freistehende Teile, zu kleines Druckbett.
3. **Passproben drucken.** `python3 build.py --gauges` erzeugt vier kleine
   Ausschnitte (je 20–40 min statt 8–10 h):
   - `probe_blinker_rechts` / `_links` — Blinker einschrauben, Sitz prüfen
   - `probe_rueckseite` — Lochbild gegen die Halterung am Motorrad halten
   - `probe_vorderkante` — Wandstärke und Randverlauf beurteilen
4. **Korrigieren**, Schritt 2–3 wiederholen.
5. **Ganze Maske drucken.**

Schritt 3 ist der, den man gern überspringt und dann zweimal acht Stunden
druckt.

## Übertragung auf eine echte Maske

Sobald du ein Netz deiner EXC-Maske hast (3D-Scan, Photogrammetrie oder ein
fremdes Modell):

```bash
python3 apply_to_scan.py --scan meine_exc_maske.stl --align auto
python3 apply_to_scan.py --scan meine_exc_maske.stl --nur blinker
```

Das Netz muss **geschlossen** sein — boolesche Operationen brauchen das. Ist
es das nicht, vorher in Meshmixer, Blender oder dem PrusaSlicer reparieren.

`--align auto` legt X/Y mittig und die Vorderkante auf Z = 0. Feinjustage über
`--rotate 0,0,90`, `--translate 0,0,-5`, `--scale 1.02`. Die angeformten
Felder werden an der konvexen Hülle des Scans bündig beschnitten; auf den
konvexen Flanken einer Maske deckt die sich mit der Außenhaut (gemessen:
Median 0,0 mm, Maximum 0,66 mm Abweichung).

Die Rückseite `plate` setzt voraus, dass die hintere Kontur in `params.json`
zu deinem Scan passt — sonst `--nur blinker` benutzen oder auf `tabs`
umstellen.

## Varianten der Rückseite

| `rear_mount_type` | was es ist | wofür |
|---|---|---|
| `plate` | hinterer Rahmen mit Innenausschnitt, angeformten Laschen, Kabeldurchführung | Standard — kommt einer übernommenen Rückseite am nächsten |
| `tabs`  | nur die Laschen, direkt in der Flanke verankert | leichter, weniger Stützmaterial |
| `strap` | verstärkte Durchbrüche für Gummibänder | wie EXC ab Werk |
| `none`  | Rückseite offen | wenn du die Aufnahme selbst konstruierst |

## Drucken

| | Empfehlung | warum |
|---|---|---|
| Material | **ASA**, ersatzweise PETG | UV, Wärme, Vibration. PLA ist am Motorrad ungeeignet — es kriecht schon in der Sonne |
| Düse / Schicht | 0,4–0,6 mm / 0,2–0,25 mm | |
| Wandlinien | so viele, dass 3,0 mm voll werden (0,45 mm Breite → 7) | die Festigkeit sitzt in den Wänden, nicht im Infill |
| Infill | 15–20 %, Gyroid | |
| Lage | **Stirnfläche nach unten** | die Flanken laufen dann mit ~18° zur Senkrechten — unkritisch. Die Sichtfläche bekommt die glatte Druckbettoberfläche |
| Stützen | nur bei `plate` nötig (Rahmen kragt ~30 mm nach innen) | mit `--rear tabs` entfällt fast alles |
| Nachbearbeitung | Schaftbohrung auf Endmaß aufbohren (10,4 → 10,5 mm) | gedruckte Löcher fallen zu eng aus |

**Wärme:** mit H4-Halogen wird es im Gehäuse heiß genug, dass ASA in
Lampennähe weich werden kann. Mit LED-Einsatz ist das kein Thema.

### Kleines Druckbett

Die Maske ist 241 × 228 × 115 mm. Passt das nicht:

```bash
python3 build.py --split
```

Teilt bei X = 0 in zwei Hälften mit fünf Steckzungen (Spiel über
`split_clearance`, Standard 0,2 mm). Verkleben: ASA mit Aceton-Schlicker oder
2K-Epoxid, Naht innen mit einem Streifen Glasgewebe und Epoxid hinterlegen.
Jede Hälfte ist 136 × 228 × 115 mm.

## Blinker und Zulassung

`selftest.py` und `build.py` rechnen den Abstand der leuchtenden Flächen aus
und vergleichen ihn mit **240 mm** — dem Wert, der für vordere
Fahrtrichtungsanzeiger am Krad üblicherweise zwischen den Innenkanten
gefordert wird.

Das ist eine **Rechenhilfe, keine Zulassungsaussage.** Maßgeblich ist, was
dein Prüfer sagt; je nach Ausführung kommen Anforderungen an Höhe über
Fahrbahn, Abstand zum Abblendlicht und Sichtwinkel hinzu. Mit den
Standardwerten (55 mm Schaft, 22° nach außen) kommt das Modell auf 301 mm —
schmalere Schäfte oder ein flacherer Winkel drücken das schnell unter die
Grenze. Der Prüfwert steht in `out/bericht.json`.

## Dateien

```
params.json          alle Maße — das ist deine Stellschraube
build.py             erzeugt STEP / STL / SVG, Teilung, Passproben
selftest.py          misst das Ergebnis nach (Wandstärke, Taschenboden, Bohrungen)
apply_to_scan.py     überträgt die Merkmale auf ein echtes Netz
MESSANLEITUNG.md     was am realen Teil zu messen ist
ktm_mask/
  geometry.py        Querschnitte, Parallelkurven — reine Mathematik
  params.py          Parametermodell samt Plausibilitätsprüfung
  shell.py           Grundkörper, Aushöhlung, Scheinwerferausschnitt
  blinker.py         Blinkeraufnahme (Übertragung Stark EX)
  rearmount.py       Rückseite (Übertragung Spendermaske)
  features.py        Merkmalsliste — gilt für beide Wege
  assembly.py        Zusammenbau, Teilung, Bericht
  gauges.py          Passproben
out/                 Ergebnisse
```

`out/*.step` öffnest du in Fusion 360, FreeCAD, SolidWorks oder Onshape und
änderst dort weiter. `out/*.stl` geht direkt in den Slicer.

## Was das Modell nicht ist

- **Keine Kopie einer OEM-Geometrie.** Die Formen von KTM und Stark Future
  sind nicht öffentlich; hier ist nichts davon abgeleitet. Das Modell ist eine
  eigene Konstruktion mit den beschriebenen Merkmalen.
- **Nicht am Fahrzeug erprobt.** Passung und Festigkeit musst du prüfen. Eine
  Scheinwerfermaske ist ein sichtrelevantes Bauteil am Fahrwerk.
- **Kein Ersatz für die Abnahme.** Umbauten an der Beleuchtung sind
  eintragungs- bzw. abnahmepflichtig.
