# Messanleitung

Was am realen Teil abzunehmen ist, damit aus dem Modell ein passendes Teil
wird. Jeder Abschnitt nennt die Parameter in `params.json`, die er füttert.

**Werkzeug:** Messschieber (150 mm reicht), Stahllineal, zwei Winkel oder
Buchstapel bekannter Höhe, Filzstift. Nützlich: Konturenlehre.

**Grundregel:** alles in Millimetern, Nachkommastelle ruhig mitnehmen. Lieber
einmal zu oft nachmessen als zweimal acht Stunden drucken.

---

## 1. Grundform — `sections`

Der aufwendigste Teil, dafür der wichtigste.

1. Maske mit der **Stirnfläche nach unten** auf eine ebene Platte legen.
2. **Gesamttiefe** messen: höchster Punkt der Hinterkante über der Platte.
   Das ist der Betrag des letzten `z`-Werts (Standard: 115).
3. Auf fünf bis sechs Höhen über der Platte jeweils **Breite** und **Höhe**
   der Maske messen. Die Höhen sind die `z`-Werte, negativ eingetragen.
   Praktisch: zwei gleich hohe Bücher/Klötze links und rechts, Lineal quer
   darüber, dann mit dem Messschieber an dieser Ebene messen.

| `z` | wo messen | trägt |
|---|---|---|
| `0` | direkt an der Vorderkante | `width`, `height` |
| `-20` | 20 mm über der Platte | `width`, `height` |
| `-45` | 45 mm | `width`, `height` |
| `-75` | 75 mm | `width`, `height` |
| `-100` | 100 mm | `width`, `height` |
| `-115` | Hinterkante | `width`, `height` |

Die Zeilen in `params.json` lauten
`[z, breite, höhe, exponent, y_offset, einzug_unten]`.

**`exponent`** — wie kastig der Querschnitt ist. 2,0 = reine Ellipse,
3,0 = leicht kantig, 4,5 = deutlich kastig mit gerundeten Ecken. Nicht
messen, sondern vergleichen (siehe unten).

**`y_offset`** — wie weit die Mitte dieses Querschnitts gegenüber der
Vorderkante nach unten gewandert ist. Maske auf die Seite legen, Mitte der
Vorderkante und Mitte der Hinterkante anzeichnen, Höhenunterschied messen,
negativ eintragen.

**`einzug_unten`** — wie stark die Maske nach unten schmaler wird.
0 = gar nicht, 0,3 = deutlich (Standard), 0,45 = stark keilförmig.

### Exponent und Einzug ohne Messschieber prüfen

```bash
python3 build.py --outlines
```

schreibt für jeden Querschnitt eine DXF-Datei nach `out/`. Bei **100 %**
ausdrucken (Inkscape, LibreOffice Draw, jedes CAD — nicht „an Seite
anpassen"), ausschneiden und in der jeweiligen Tiefe an die Maske halten.
Abweichung sichtbar → `exponent` und `einzug_unten` anpassen, neu erzeugen.
Zwei, drei Runden reichen erfahrungsgemäß.

## 2. Wandstärke — `wall`

An der Vorderkante mit dem Messschieber abgreifen. Serienmasken liegen bei
2,0–2,5 mm; **für den Druck 3,0–3,5 mm eintragen**, gedrucktes Material
verträgt weniger Biegung als Spritzguss.

## 3. Scheinwerferausschnitt — `light_*`

| messen | Parameter |
|---|---|
| runder Ausschnitt: Durchmesser | `light_shape="round"`, `light_d` |
| eckiger Ausschnitt: Breite, Höhe, Eckradius | `light_shape="rect"`, `light_w`, `light_h`, `light_r` |
| Mitte des Ausschnitts über der Maskenmitte | `light_y` |
| Lochkreis der Scheinwerferschrauben | `light_screw_bcd`, `light_screws` |
| Schraubendurchmesser | `light_screw_d` (Kernloch, ca. 0,8 × Gewinde) |

`light_lip` ist der Kragen, gegen den sich der Scheinwerfer von innen legt —
so breit wählen, dass er trägt, ohne ins Sichtfeld zu ragen (6–8 mm).

## 4. Blinkeraufnahme — Übertragung von der Stark-EX-Maske

Am **Spenderteil** (Stark-EX-Maske oder deinem Blinker) abnehmen:

| messen | Parameter |
|---|---|
| Länge × Höhe der Aussparung | `pocket_w`, `pocket_h` |
| Eckradius der Aussparung | `pocket_r` (≥ `pocket_h`/2 ergibt ein Oval) |
| Tiefe der Aussparung ab Außenhaut | `pocket_depth` |
| Gewinde des Blinkerschafts | `bolt_hole_d` = Gewindemaß + 0,4 mm |
| Verdrehsicherungsstift: Ø und Abstand zur Achse | `antirot_d`, `antirot_offset` |
| Schaftlänge ab Auflagefläche | `blinker_stalk_len` |
| Durchmesser der leuchtenden Fläche | `blinker_lens_d` |

Am **Zielteil** (deiner EXC-Maske) festlegen, wo es sitzen soll:

| messen | Parameter |
|---|---|
| Abstand der Blinkermitte zur Vorderkante | `blinker_z` (negativ) |
| Höhe der Blinkermitte über der Maskenmitte | `blinker_y` |
| Winkel der Achse nach außen/vorn | `blinker_yaw_deg` (0 = quer, 22 = Standard) |
| Neigung nach oben/unten | `blinker_pitch_deg` |

`pocket_floor_t` (Restwand hinter dem Taschenboden, Standard 3 mm) und
`boss_depth` (Materialtiefe an der Schaftbohrung, Standard 9 mm) sind
Konstruktionswerte, keine Messwerte — mehr davon heißt stabiler und schwerer.

**Vor dem Vollausdruck:** `python3 build.py --gauges`, dann
`probe_blinker_rechts.stl` drucken und den Blinker wirklich einschrauben.

## 5. Rückseite — Übertragung von der Spendermaske

Zuerst entscheiden, welche Variante (`rear_mount_type`, siehe README).

Für `plate` und `tabs`:

| messen | Parameter |
|---|---|
| Anzahl der Befestigungspunkte | `tab_count` |
| Lage der Punkte als Winkel, 0° = rechts, 90° = oben | `tab_angles_deg` |
| Schraubendurchmesser | `tab_hole_d` = Gewindemaß + 0,5 mm |
| Breite und Dicke der Laschen | `tab_w`, `tab_t` |
| wie weit die Lasche nach innen reicht | `tab_len` |
| Öffnung im hinteren Rahmen, Anteil der Außenkontur | `plate_aperture` (0,62 = 62 %) |
| Durchmesser der Kabeldurchführung | `cable_hole_d` |

Für `strap` zusätzlich `strap_angles_deg`, `strap_z` und die Schlitzmaße
`strap_slot_w` / `strap_slot_h` nach dem Haken deines Gummibands.

Die Winkel bekommst du am einfachsten so: Maske von hinten fotografieren,
Bild in einem Zeichenprogramm öffnen, Mittelpunkt festlegen, Winkel zu jedem
Befestigungspunkt ablesen. 0° zeigt nach rechts, gegen den Uhrzeigersinn
positiv.

## 6. Druckbett — `bed_x`, `bed_y`, `bed_z`

Das nutzbare Bauvolumen deines Druckers eintragen. `selftest.py` sagt dir
dann, ob das Teil in irgendeiner Lage passt oder ob `--split` nötig ist.

---

## Zum Schluss

```bash
python3 selftest.py
```

Meldet, was ein Rendering nicht zeigt: ob die Wand wirklich so stark ist wie
eingestellt, ob der Taschenboden trägt, ob die Bohrungen durchgehen, ob nach
der Teilung zwei zusammenhängende Hälften herauskommen und ob der
Blinkerabstand den Richtwert erreicht.
