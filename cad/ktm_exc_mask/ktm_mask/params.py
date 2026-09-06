"""Alle Masse der Maske an einer Stelle.

Jeder Wert ist entweder GEMESSEN (am realen Teil abgenommen) oder GESCHAETZT
(Startwert, bis gemessen wurde). Die Startwerte hier sind Schaetzungen auf
Basis der ueblichen Groessenordnung einer KTM-EXC-Scheinwerfermaske — sie
passen NICHT ohne Nachmessen. Siehe MESSANLEITUNG.md.

Aendern: params.json bearbeiten und `python3 build.py` erneut laufen lassen.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path

from .geometry import Section

# Superellipsen-Querschnitte der EXC-Grundmaske:
#   [z, breite, hoehe, exponent, y_offset, einzug_unten]
DEFAULT_SECTIONS = [
    [0.0,    240.0, 228.0, 4.2, 0.0,   0.26],   # vordere Kante
    [-20.0,  238.0, 224.0, 4.0, -1.0,  0.26],
    [-45.0,  228.0, 211.0, 3.7, -3.0,  0.28],
    [-75.0,  206.0, 187.0, 3.3, -6.0,  0.29],
    [-100.0, 180.0, 160.0, 3.1, -9.0,  0.30],
    [-115.0, 166.0, 146.0, 3.0, -11.0, 0.30],   # hintere Kante
]


@dataclass
class MaskParams:
    # ---------------------------------------------------------------- Grundkoerper
    sections: list = field(default_factory=lambda: [list(s) for s in DEFAULT_SECTIONS])
    wall: float = 3.0                 # Wandstaerke (FDM: 3.0 = 8 Bahnen bei 0.4er Duese)
    section_points: int = 72          # Aufloesung der Loft-Splines

    # ------------------------------------------------- Scheinwerferausschnitt vorn
    light_cut: bool = True
    light_shape: str = "round"        # "round" | "rect"
    light_d: float = 152.0            # bei "round": Durchmesser
    light_w: float = 150.0            # bei "rect": Breite
    light_h: float = 112.0            # bei "rect": Hoehe
    light_r: float = 26.0             # bei "rect": Eckradius
    light_y: float = 12.0             # Mitte des Ausschnitts ueber Maskenmitte
    light_lip: float = 7.0            # Auflagekragen innen (0 = keiner)
    light_lip_t: float = 5.0          # Dicke des Kragens
    light_screws: int = 4             # Schraubdome fuer den Scheinwerfer
    light_screw_d: float = 4.3        # Kernloch M5-Blechschraube
    light_screw_bcd: float = 176.0    # Lochkreisdurchmesser
    light_screw_boss_d: float = 11.0
    light_screw_boss_t: float = 7.0

    # ------------------------------------ Blinkeraufnahme (Uebernahme Stark-EX-Maske)
    blinker: bool = True
    blinker_z: float = -38.0          # Tiefenposition auf der Flanke
    blinker_y: float = 8.0            # Hoehenposition auf der Flanke
    blinker_yaw_deg: float = 22.0     # Achse nach aussen/vorn geschwenkt
    blinker_pitch_deg: float = 0.0    # Achse nach oben (+) / unten (-)
    pocket_w: float = 48.0            # Aussparung: Breite
    pocket_h: float = 34.0            # Aussparung: Hoehe
    pocket_r: float = 10.0            # Aussparung: Eckradius (>= h/2 ergibt Oval)
    pocket_depth: float = 4.5         # Aussparungstiefe ab Aussenhaut
    pocket_margin: float = 6.0        # Verstaerkungsfeld ueber die Tasche hinaus
    pocket_floor_t: float = 3.0       # Restwand hinter dem Taschenboden
    boss_d: float = 30.0              # Schaftverstaerkung: Durchmesser
    boss_depth: float = 9.0           # Schaftverstaerkung: Tiefe ab Taschenboden
    bolt_hole_d: float = 10.4         # Durchgangsloch Blinkerschaft (M10)
    antirot: bool = True              # Verdrehsicherungsstift
    antirot_d: float = 4.4
    antirot_offset: float = 12.0      # Abstand zur Schaftachse
    nut_pocket: bool = False          # Sechskanttasche innen
    nut_af: float = 17.0              # Schluesselweite
    nut_depth: float = 8.0

    # ---------------------------------------------------- ECE-Kontrollwerte (Blinker)
    blinker_stalk_len: float = 55.0   # Schaftlaenge ab Auflageflaeche
    blinker_lens_d: float = 32.0      # Durchmesser der leuchtenden Flaeche
    ece_min_inner_spacing: float = 240.0

    # ---------------------------------------- Rueckseite (Uebernahme der Spendermaske)
    rear_mount_type: str = "plate"    # "plate" | "tabs" | "strap" | "none"
    plate_t: float = 4.0              # Dicke des hinteren Rahmens
    plate_aperture: float = 0.62      # Innenausschnitt als Anteil der Rueckkontur
    tab_count: int = 4
    tab_angles_deg: list = field(default_factory=lambda: [58.0, 122.0, 238.0, 302.0])
    tab_len: float = 24.0             # Laenge nach innen
    tab_w: float = 20.0
    tab_t: float = 5.0
    tab_hole_d: float = 6.5           # M6 mit Spiel
    strap_angles_deg: list = field(default_factory=lambda: [70.0, 110.0, 250.0, 290.0])
    strap_z: float = -92.0            # Tiefenlage der Gummiband-Durchbrueche
    strap_pad_w: float = 22.0         # Verstaerkungsfeld: Breite (tangential)
    strap_pad_h: float = 18.0         # Verstaerkungsfeld: Hoehe (in Z)
    strap_pad_depth: float = 5.0      # Verstaerkung nach innen
    strap_slot_w: float = 13.0        # Durchbruch fuer den Gummibandhaken
    strap_slot_h: float = 4.5
    cable_hole_d: float = 18.0        # Kabeldurchfuehrung im hinteren Rahmen
    cable_hole_y_frac: float = -0.72  # Position, Anteil der halben Rahmenhoehe

    # --------------------------------------------------------------------- Druck
    split: bool = False               # in zwei Haelften teilen (kleines Druckbett)
    split_clearance: float = 0.2      # Spiel der Steckverbindung
    split_tongues: int = 5
    split_tongue_l: float = 16.0      # Ueberstand je Seite
    split_tongue_w: float = 12.0
    split_tongue_t: float = 1.8       # Dicke, muss < wall sein
    bed_x: float = 250.0              # Druckbett fuer die Warnung im Report
    bed_y: float = 210.0
    bed_z: float = 210.0
    density: float = 1.07             # ASA, g/cm3 — fuer die Gewichtsschaetzung

    # --------------------------------------------------------------- Tessellierung
    stl_tolerance: float = 0.05
    stl_angular_tolerance: float = 0.12

    # ------------------------------------------------------------------- Methoden
    @property
    def section_objs(self):
        return [Section.from_list(s) for s in self.sections]

    @property
    def z_rear(self) -> float:
        return min(s[0] for s in self.sections)

    @property
    def depth(self) -> float:
        return abs(self.z_rear)

    @classmethod
    def load(cls, path=None) -> "MaskParams":
        if path is None:
            return cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known - {"_kommentar"}
        if unknown:
            raise ValueError(f"unbekannte Parameter in {path}: {sorted(unknown)}")
        data.pop("_kommentar", None)
        return cls(**data)

    def save(self, path) -> None:
        data = asdict(self)
        data["_kommentar"] = ("Masse in mm. Nach dem Aendern: python3 build.py "
                              "--params params.json")
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False),
                              encoding="utf-8")

    def validate(self) -> list:
        """Gibt eine Liste von Warnungen zurueck (leere Liste = alles plausibel)."""
        w = []
        zs = [s[0] for s in self.sections]
        if len(self.sections) < 2:
            w.append("mindestens zwei Querschnitte noetig")
        if zs != sorted(zs, reverse=True):
            w.append("sections muessen nach z absteigend sortiert sein (0, -20, -45 ...)")
        if self.wall < 1.6:
            w.append(f"Wandstaerke {self.wall} mm ist fuer ein Bauteil am "
                     "Fahrwerk sehr duenn (>= 2.5 mm empfohlen)")
        if self.split and self.split_tongue_t >= self.wall:
            w.append(f"split_tongue_t ({self.split_tongue_t}) muss kleiner sein "
                     f"als wall ({self.wall})")
        min_w = min(s[1] for s in self.sections)
        if self.light_cut and self.light_shape == "round" and self.light_d > min_w:
            w.append("Scheinwerferausschnitt ist breiter als die Maske")
        if self.rear_mount_type not in ("plate", "tabs", "strap", "none"):
            w.append(f"rear_mount_type '{self.rear_mount_type}' unbekannt")
        if self.blinker and self.pocket_floor_t < 1.5:
            w.append(f"pocket_floor_t {self.pocket_floor_t} mm — der Taschenboden "
                     "traegt den Blinker und sollte >= 2 mm sein")
        if self.blinker and self.boss_depth < self.pocket_floor_t:
            w.append("boss_depth kleiner als pocket_floor_t — die "
                     "Schaftverstaerkung bringt so nichts")
        if self.blinker and self.pocket_depth > self.wall + 8.0:
            w.append(f"pocket_depth {self.pocket_depth} mm ragt weit in den "
                     "Innenraum — Freigang zum Scheinwerfer pruefen")
        return w
