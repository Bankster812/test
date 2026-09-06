"""Zusammenbau, Teilung fuer kleine Druckbetten und Pruefbericht."""

from __future__ import annotations

from itertools import permutations

import cadquery as cq
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps

from .geometry import interp_section
from .shell import build_shell, outer_solid
from . import blinker, features

BIG = 1000.0


def build_mask(p) -> cq.Workplane:
    """Komplette modifizierte EXC-Maske: Grundkoerper plus alle Merkmale."""
    outer = outer_solid(p)
    body = build_shell(p, outer=outer)
    for feat in features.adders(p):
        body = body.union(feat.solid.intersect(outer) if feat.clip else feat.solid)
    for feat in features.cutters(p):
        body = body.cut(feat.solid)
    return body


def volume_mm3(obj, tolerance: float = 1e-5) -> float:
    """Volumen eines Koerpers.

    Nicht `Shape.Volume()` benutzen: dessen Standardtoleranz liefert auf den
    Spline-Flaechen dieser Maske rund 2,5 % zu viel. Mit vorgegebener
    Toleranz konvergiert der Wert und deckt sich mit der Summe der geteilten
    Haelften.
    """
    shape = obj.val().wrapped if hasattr(obj, "val") else obj.wrapped
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props, tolerance, True)
    return props.Mass()


def fits_on_bed(dims, bed) -> bool:
    """Passt das Teil in irgendeiner Lage aufs Bett? Es darf gedreht werden."""
    return any(all(d <= b + 1e-6 for d, b in zip(perm, bed))
               for perm in permutations(dims))


def _half_space(x_positive: bool) -> cq.Workplane:
    off = 0.0 if x_positive else -BIG
    return (cq.Workplane("XY")
            .center(off + BIG / 2.0, 0.0)
            .box(BIG, BIG, BIG, centered=(True, True, True))
            .translate((0, 0, 0)))


def _tongue_boxes(p, clearance: float = 0.0):
    """Steckzungen entlang der Trennebene x = 0, oben und unten verteilt."""
    z_front, z_rear = p.section_objs[0].z, p.z_rear
    n = max(1, p.split_tongues)
    boxes = []
    for i in range(n):
        t = (i + 0.5) / n
        z = z_front + t * (z_rear - z_front)
        sec = interp_section(p.section_objs, z)
        top = i % 2 == 0
        y_surface = sec.y_offset + (sec.height / 2.0 if top else -sec.height / 2.0)
        y_mid = y_surface - (p.wall / 2.0 if top else -p.wall / 2.0)
        box = (cq.Workplane("XY")
               .box(2.0 * p.split_tongue_l + 2 * clearance,
                    p.split_tongue_t + 2 * clearance,
                    p.split_tongue_w + 2 * clearance)
               .translate((0.0, y_mid, z)))
        boxes.append(box)
    return boxes


def split_halves(p, body: cq.Workplane):
    """Teilt die Maske an x = 0 in zwei ineinandersteckende Haelften."""
    right_space, left_space = _half_space(True), _half_space(False)

    tongues = None
    for b in _tongue_boxes(p, 0.0):
        tongues = b if tongues is None else tongues.union(b)
    sockets = None
    for b in _tongue_boxes(p, p.split_clearance):
        sockets = b if sockets is None else sockets.union(b)

    tongue_mat = tongues.intersect(body).intersect(left_space)
    right = body.intersect(right_space).union(tongue_mat)
    left = body.intersect(left_space).cut(sockets)
    return right, left


def report(p, body: cq.Workplane) -> dict:
    """Kennzahlen und Warnungen zum fertigen Modell."""
    solid = body.val()
    bb = solid.BoundingBox()
    vol_cm3 = volume_mm3(body) / 1000.0
    bodies = len(solid.Solids())

    warnings = list(p.validate())
    if bodies != 1:
        warnings.append(
            f"{bodies} getrennte Koerper — ein Teil haengt in der Luft. "
            "Positionen von Laschen/Verstaerkungen pruefen.")
    fits = fits_on_bed((bb.xlen, bb.ylen, bb.zlen), (p.bed_x, p.bed_y, p.bed_z))
    if not fits and not p.split:
        warnings.append(
            f"passt nicht auf das Druckbett ({p.bed_x} x {p.bed_y} x {p.bed_z} mm) "
            "— mit --split in zwei Haelften teilen")

    ece = blinker.ece_check(p) if p.blinker else None
    if ece and not ece["erfuellt"]:
        warnings.append(
            f"Blinker-Innenabstand {ece['innenkanten_abstand_mm']} mm liegt unter "
            f"dem Richtwert {ece['richtwert_mm']} mm — Schaft je Seite um "
            f"{ece['fehlende_schaftlaenge_je_seite_mm']} mm verlaengern")

    return {
        "abmessungen_mm": {"x": round(bb.xlen, 1), "y": round(bb.ylen, 1),
                           "z": round(bb.zlen, 1)},
        "volumen_cm3": round(vol_cm3, 1),
        "materialbedarf_g": round(vol_cm3 * p.density, 0),
        "koerper": bodies,
        "wandstaerke_mm": p.wall,
        "rueckseite": p.rear_mount_type,
        "passt_aufs_bett": fits,
        "ece_blinker": ece,
        "warnungen": warnings,
    }
