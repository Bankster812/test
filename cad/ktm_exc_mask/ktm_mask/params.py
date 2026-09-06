"""Alle Masse der Maske an einer Stelle.

Die Umrisskontur und die Merkmale sind aus den Referenzbildern der
Stark-VARG-EX-Maske abgegriffen (Carbon-Version von CMT Composit sowie ein
Kunststoff-Render desselben Teils). Abgegriffen heisst: aus Fotos
massstabslos entnommen — die FORM stimmt in ihrem Charakter, die absoluten
MASSE sind Schaetzwerte, bis am realen Teil gemessen wurde.

`width` und `height` skalieren die gesamte Kontur; alle Merkmale sind an
diese Kontur gekoppelt und wandern mit. Nachmessen und diese beiden Werte
setzen bringt daher schon sehr viel. Siehe MESSANLEITUNG.md, zum Abgreifen
aus einem Foto siehe trace_outline.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path

from .geometry import Level, ensure_ccw

# Umriss der Maske von vorn, normiert auf eine Einheitsbox (-0.5 … +0.5).
# Abgegriffen aus der Frontalansicht der Carbon-Maske: breite, leicht
# gewoelbte Oberkante, ausgestellte Schultern, groesste Breite im oberen
# Drittel, dann nach unten zulaufend.
DEFAULT_OUTLINE = [
    # Oberkante: breit und flach
    [0.000,  0.500], [0.110,  0.499], [0.210,  0.496], [0.290,  0.490],
    # Schulter
    [0.352,  0.476], [0.404,  0.452], [0.444,  0.416], [0.472,  0.368],
    # groesste Breite im oberen Drittel, danach fast senkrechte Flanke
    [0.492,  0.310], [0.500,  0.245], [0.500,  0.170], [0.496,  0.090],
    [0.488,  0.010], [0.476, -0.070],
    # Taille, dann Auslauf nach unten
    [0.458, -0.148], [0.433, -0.222], [0.401, -0.292], [0.363, -0.356],
    [0.318, -0.412], [0.268, -0.456], [0.213, -0.487], [0.160, -0.500],
    [0.080, -0.503], [0.000, -0.504],
    [-0.080, -0.503], [-0.160, -0.500], [-0.213, -0.487], [-0.268, -0.456],
    [-0.318, -0.412], [-0.363, -0.356], [-0.401, -0.292], [-0.433, -0.222],
    [-0.458, -0.148], [-0.476, -0.070], [-0.488,  0.010], [-0.496,  0.090],
    [-0.500,  0.170], [-0.500,  0.245], [-0.492,  0.310], [-0.472,  0.368],
    [-0.444,  0.416], [-0.404,  0.452], [-0.352,  0.476], [-0.290,  0.490],
    [-0.210,  0.496], [-0.110,  0.499],
]

# Woelbung: die Kontur wird nach vorn hin kleiner. Die Maske ist ein flaches
# Schild — der Anstieg passiert nahe am Rand, die Mitte ist vergleichsweise
# flach.  [z, verkleinerung, verschiebung nach oben]
DEFAULT_LEVELS = [
    [0.0,  1.000,  0.0],
    [18.0, 0.960,  1.5],
    [34.0, 0.902,  3.0],
    [48.0, 0.816,  4.5],
    [58.0, 0.686,  6.0],
    [65.0, 0.505,  7.0],
    [70.0, 0.285,  8.0],
    [72.0, 0.060,  8.0],
]


@dataclass
class MaskParams:
    # ------------------------------------------------------------- Grundkoerper
    outline: list = field(default_factory=lambda: [list(p) for p in DEFAULT_OUTLINE])
    levels: list = field(default_factory=lambda: [list(v) for v in DEFAULT_LEVELS])
    width: float = 245.0              # Gesamtbreite der Kontur
    height: float = 232.0             # Gesamthoehe der Kontur
    wall: float = 3.0                 # Wandstaerke (FDM: 3.0 = 7 Bahnen bei 0.45)
    spline_outline: bool = True       # Kontur als Spline glaetten statt Polygonzug

    # ------------------------------------------- Scheinwerferausschnitt (hohes Oval)
    light_cut: bool = True
    light_w: float = 59.0             # Breite des Ausschnitts
    light_h: float = 136.0            # Hoehe des Ausschnitts
    light_r: float = 27.0             # Eckradius (>= Breite/2 ergibt ein Oval)
    light_y: float = -32.0            # Mitte des Ausschnitts unter der Maskenmitte
    light_lip: float = 6.0            # Auflagekragen innen (0 = keiner)
    light_lip_t: float = 4.0

    # Befestigungsdome des Scheinwerfers: zwei oben, zwei unten neben dem Oval
    light_boss: bool = True
    light_boss_dx: float = 44.0       # seitlicher Abstand zur Ausschnittmitte
    light_boss_dy: float = 52.0       # Hoehenabstand zur Ausschnittmitte
    light_boss_d: float = 12.0
    light_boss_t: float = 8.0
    light_boss_hole_d: float = 5.2

    # ------------------- Blinkerhalter-Aussparungen (Uebernahme Stark-EX-Maske)
    # Offene Schlitze in der Seitenkante, je drei uebereinander. Der Halter
    # schiebt sich von aussen ein; die drei Positionen sind die
    # Hoehenverstellung. Beide Seiten sind gespiegelt gleich — die
    # abweichende Seite auf dem Kunststoff-Render wird nicht uebernommen.
    slots: bool = True
    slot_count: int = 3
    slot_y: float = -6.0              # Hoehe der mittleren Aussparung
    slot_pitch: float = 17.0          # Abstand der Aussparungen zueinander
    slot_h: float = 6.5               # Hoehe einer Aussparung
    slot_depth: float = 14.0          # Tiefe von der Aussenkante nach innen
    slot_z0: float = -2.0             # Beginn in Laengsrichtung (hinter der Kante)
    slot_z_len: float = 26.0          # Laenge in Laengsrichtung
    slot_fillet: float = 1.2          # Eckverrundung, gegen Kerbwirkung

    # ----------------------------------- optionale Schaftaufnahme fuer den Blinker
    # Zusaetzlich zu den Schlitzen: Durchgang fuer einen Blinker mit Gewinde-
    # schaft, falls du keinen einschiebbaren Halter benutzt.
    stalk: bool = False
    stalk_y: float = -6.0
    stalk_z: float = 22.0
    stalk_yaw_deg: float = 20.0
    stalk_pitch_deg: float = 0.0
    pocket_w: float = 44.0
    pocket_h: float = 32.0
    pocket_r: float = 10.0
    pocket_depth: float = 4.0
    pocket_margin: float = 6.0
    pocket_floor_t: float = 3.0
    boss_d: float = 28.0
    boss_depth: float = 9.0
    bolt_hole_d: float = 10.4
    antirot: bool = True
    antirot_d: float = 4.4
    antirot_offset: float = 12.0

    # ---------------------------------------------------- ECE-Kontrollwerte
    blinker_stalk_len: float = 55.0
    blinker_lens_d: float = 32.0
    ece_min_inner_spacing: float = 240.0

    # ------------------------------- Rueckseite (Uebernahme der Spendermaske)
    # Oberer Befestigungsbock in der Mitte, wie auf den Bildern: erhabener
    # Sockel an der Innenseite mit Durchgangsloch.
    top_bracket: bool = True
    top_bracket_y: float = 96.0       # Hoehe ueber der Maskenmitte
    top_bracket_w: float = 34.0
    top_bracket_h: float = 22.0
    top_bracket_t: float = 10.0       # Aufbau nach innen
    top_bracket_hole_d: float = 8.5

    rear_mount_type: str = "tabs"     # "tabs" | "none"
    tab_positions: list = field(default_factory=lambda: [[0.0, -104.0]])
    tab_len: float = 26.0
    tab_w: float = 20.0
    tab_t: float = 5.0
    tab_hole_d: float = 6.5

    # ---------------------------------------------------------------- Druck
    split: bool = False
    split_clearance: float = 0.2
    split_tongues: int = 4
    split_tongue_l: float = 16.0
    split_tongue_w: float = 12.0
    split_tongue_t: float = 1.8
    bed_x: float = 250.0
    bed_y: float = 210.0
    bed_z: float = 210.0
    density: float = 1.07             # ASA, g/cm3

    # ------------------------------------------------------------ Tesselierung
    outline_points: int = 0           # 0 = Kontur unveraendert, >0 = neu abtasten
    stl_tolerance: float = 0.05
    stl_angular_tolerance: float = 0.12

    # ---------------------------------------------------------------- Methoden
    @property
    def outline_ccw(self):
        return ensure_ccw([tuple(p) for p in self.outline])

    @property
    def level_objs(self):
        return [Level.from_list(v) for v in self.levels]

    @property
    def z_rear(self) -> float:
        return min(v[0] for v in self.levels)

    @property
    def z_front(self) -> float:
        return max(v[0] for v in self.levels)

    @property
    def depth(self) -> float:
        return self.z_front - self.z_rear

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
        data["_kommentar"] = ("Masse in mm. Kontur normiert auf -0.5 … +0.5, "
                              "skaliert ueber width/height. Nach dem Aendern: "
                              "python3 build.py && python3 selftest.py")
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False),
                              encoding="utf-8")

    def validate(self) -> list:
        """Liste von Warnungen (leer = alles plausibel)."""
        w = []
        if len(self.outline) < 8:
            w.append("outline hat zu wenige Punkte fuer eine brauchbare Kontur")
        zs = [v[0] for v in self.levels]
        if zs != sorted(zs):
            w.append("levels muessen nach z aufsteigend sortiert sein")
        if len(self.levels) < 2:
            w.append("mindestens zwei Ebenen noetig")
        if any(v[1] <= 0 for v in self.levels[:-1]):
            w.append("nur die vorderste Ebene darf die Verkleinerung 0 haben")
        if self.wall < 1.6:
            w.append(f"Wandstaerke {self.wall} mm ist fuer ein Bauteil am "
                     "Fahrwerk sehr duenn (>= 2.5 mm empfohlen)")
        if self.split and self.split_tongue_t >= self.wall:
            w.append(f"split_tongue_t ({self.split_tongue_t}) muss kleiner "
                     f"sein als wall ({self.wall})")
        if self.slots:
            if self.slot_depth <= self.wall:
                w.append("slot_depth ist nicht tiefer als die Wand — die "
                         "Aussparung schneidet nichts frei")
            span = (self.slot_count - 1) * self.slot_pitch + self.slot_h
            if span > self.height * 0.5:
                w.append(f"die {self.slot_count} Aussparungen ueberspannen "
                         f"{span:.0f} mm — passt das noch auf die Flanke?")
        if self.light_cut and self.light_w > self.width * 0.6:
            w.append("Scheinwerferausschnitt ist sehr breit fuer diese Kontur")
        if self.stalk and self.pocket_floor_t < 1.5:
            w.append(f"pocket_floor_t {self.pocket_floor_t} mm — der "
                     "Taschenboden traegt den Blinker, >= 2 mm einplanen")
        if self.rear_mount_type not in ("tabs", "none"):
            w.append(f"rear_mount_type '{self.rear_mount_type}' unbekannt")
        return w
