# Messanleitung

Was am realen Teil abzunehmen ist. Jeder Abschnitt nennt die Parameter in
`params.json`, die er füttert.

**Werkzeug:** Messschieber (150 mm reicht), Stahllineal, Filzstift.

**Grundregel:** alles in Millimetern. Lieber einmal zu oft nachmessen als
zweimal acht Stunden drucken.

---

## 1. Umriss — am besten gar nicht messen

Statt Punkt für Punkt zu vermessen: fotografieren.

```bash
python3 trace_outline.py --bild maske.jpg --breite 245 --vorschau
```

- Maske flach auf hellen, gleichmäßigen Untergrund, Wölbung nach oben
- von **gerade oben** fotografieren, weit weg stehen und heranzoomen
  (aus der Nähe verzerrt die Perspektive die Ränder)
- `--breite` ist die **einmal gemessene Gesamtbreite** in mm; die Höhe wird
  aus dem Seitenverhältnis des Fotos berechnet, oder du gibst `--hoehe` an
- `--vorschau` schreibt ein Kontrollbild mit dem gefundenen Umriss

Das füllt `outline`, `width` und `height`.

Erkennt das Werkzeug den falschen Bereich, hilft `--schwelle 0.6` (Wert
zwischen 0 und 1; kleiner = nur sehr Dunkles zählt als Objekt).

### Von Hand, falls kein Foto möglich

`outline` ist eine Liste `[x, y]`, normiert auf **−0,5 … +0,5** in beiden
Achsen, gegen den Uhrzeigersinn, Ursprung in der Mitte der Kontur. Punkte
dichter setzen, wo die Kontur ihre Richtung ändert (Schultern, Taille).
`width` und `height` skalieren sie auf Millimeter.

## 2. Wölbung — `levels`

Wie weit die Maske nach vorn gewölbt ist. Eine Zeile ist
`[z, verkleinerung, verschiebung_nach_oben]`:

- `z` = Tiefe, 0 an der **Hinterkante**, positiv nach vorn
- `verkleinerung` = wie groß die Kontur auf dieser Tiefe noch ist (1,0 = voll)
- `verschiebung_nach_oben` = wandert der Scheitel nach oben

Messen: Maske mit der Hinterkante auf eine ebene Platte legen. Der höchste
Punkt über der Platte ist die Gesamttiefe (letzter `z`-Wert, voreingestellt
72). Dann in ein paar Höhen die Breite messen und ins Verhältnis zur
Gesamtbreite setzen — das ist die `verkleinerung`.

Schneller geht es mit den Schablonen:

```bash
python3 build.py --outlines
```

schreibt für jede Ebene eine DXF nach `out/`. Bei **100 %** ausdrucken
(nicht „an Seite anpassen"), ausschneiden und in der jeweiligen Tiefe an die
Maske halten.

## 3. Wandstärke — `wall`

An der Kante abgreifen. Serienteile liegen bei 2,0–2,5 mm; **für den Druck
3,0–3,5 mm eintragen**, gedrucktes Material verträgt weniger Biegung als
Spritzguss.

## 4. Blinkerhalter-Aussparungen — die Übertragung

Das Kernstück. Am Spenderteil abnehmen:

| messen | Parameter |
|---|---|
| Anzahl der Schlitze übereinander | `slot_count` |
| Höhe der **mittleren** Aussparung über der Maskenmitte | `slot_y` |
| Abstand von Schlitzmitte zu Schlitzmitte | `slot_pitch` |
| Höhe eines Schlitzes | `slot_h` |
| Tiefe von der Außenkante nach innen | `slot_depth` |
| Beginn in Längsrichtung, ab Hinterkante | `slot_z0` |
| Länge in Längsrichtung | `slot_z_len` |

Die Schlitze gehen **durch die Flanke hindurch** — dahinter liegt der
Hohlraum, kein Boden. `slot_depth` ist deshalb kein Sackloch, sondern wie
weit der Umriss zurückgeschnitten wird.

`slot_fillet` (Eckverrundung, Standard 1,2 mm) ist ein Konstruktionswert:
scharfe Ecken sind an einem schwingenden Bauteil Rissausgangspunkte.

**Vor dem Vollausdruck:** `python3 build.py --gauges`, dann
`probe_aussparungen_rechts.stl` drucken und den Halter wirklich einschieben.

## 5. Scheinwerferausschnitt — `light_*`

| messen | Parameter |
|---|---|
| Breite des Ovals | `light_w` |
| Höhe des Ovals | `light_h` |
| Eckradius (≥ Breite/2 ergibt ein reines Oval) | `light_r` |
| Mitte des Ovals **unter** der Maskenmitte (negativ) | `light_y` |

Die vier Schraubdome:

| messen | Parameter |
|---|---|
| seitlicher Abstand Dommitte zur Ovalmitte | `light_boss_dx` |
| Höhenabstand Dommitte zur Ovalmitte | `light_boss_dy` |
| Außendurchmesser des Doms | `light_boss_d` |
| Kernloch (ca. 0,8 × Gewinde) | `light_boss_hole_d` |

`light_boss_t` (Aufbau nach innen, Standard 8 mm) und `light_lip`
(Auflagekragen, gegen den sich der Scheinwerfer von hinten legt) sind
Konstruktionswerte.

## 6. Rückseite

**Oberer Befestigungsbock:**

| messen | Parameter |
|---|---|
| Höhe über der Maskenmitte | `top_bracket_y` |
| Breite × Höhe des Sockels | `top_bracket_w`, `top_bracket_h` |
| Durchgangsloch | `top_bracket_hole_d` |

**Weitere Befestigungspunkte** trägst du als Liste `tab_positions` ein, je
Punkt `[x, y]` in Millimetern von der Maskenmitte aus. Voreingestellt ist
ein Punkt unten mittig. Dazu `tab_len`, `tab_w`, `tab_t`, `tab_hole_d`.

Braucht die Maske keine angeformten Laschen: `rear_mount_type` auf `none`.

## 7. Druckbett — `bed_x`, `bed_y`, `bed_z`

Nutzbares Bauvolumen eintragen. `selftest.py` sagt dann, ob das Teil in
irgendeiner Lage passt oder ob `--split` nötig ist.

---

## Zum Schluss

```bash
python3 selftest.py
```

Meldet, was ein Rendering nicht zeigt: ob der Körper geschlossen ist, ob die
Wand wirklich so stark ist wie eingestellt, ob alle Aussparungen frei sind
und die Stege dazwischen stehen, ob beide Seiten gespiegelt gleich sind, ob
nach der Teilung zwei zusammenhängende Hälften herauskommen und ob das
Ergebnis aufs Bett passt.
