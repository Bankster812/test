"""Zusammenbau, Teilung fuer kleine Druckbetten und Pruefbericht."""

from __future__ import annotations

from itertools import permutations

import cadquery as cq
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps

from .geometry import level_points
from .shell import build_shell, outer_solid
from . import blinker, features

BIG = 1000.0


def build_mask(p) -> cq.Workplane:
    """Komplette modifizierte Maske: Grundkoerper plus alle Merkmale."""
    outer = outer_solid(p)
    body = build_shell(p, outer=outer)
    for feat in features.adders(p, outer=outer):
        body = body.union(feat.solid)
    for feat in features.cutters(p, outer=outer):
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
    n = max(1, p.split_tongues)
    boxes = []
    for i in range(n):
        t = (i + 0.5) / n
        z = p.z_rear + t * (p.z_front - p.z_rear)
        pts = level_points(p.outline_ccw, p.width, p.height, p.level_objs, z)
        # Ober- bzw. Unterkante der Kontur nahe der Trennebene
        near = [(x, y) for x, y in pts if abs(x) < p.width * 0.12] or pts
        top = i % 2 == 0
        y_surface = max(y for _, y in near) if top else min(y for _, y in near)
        y_mid = y_surface - (p.wall / 2.0 if top else -p.wall / 2.0)
        boxes.append(
            cq.Workplane("XY")
            .box(2.0 * p.split_tongue_l + 2 * clearance,
                 p.split_tongue_t + 2 * clearance,
                 p.split_tongue_w + 2 * clearance)
            .translate((0.0, y_mid, z)))
    return boxes


MIN_TONGUE_VOL = 60.0   # mm^3 — darunter ist es kein Zapfen, sondern ein Splitter


def _useful_tongues(p, body, right_space, left_space):
    """Zungenpositionen, die die Trennebene wirklich ueberbruecken.

    Eine Zunge sitzt an der Ober- oder Unterkante der Kontur. An der
    gekrummten Unterkante steht direkt an der Trennebene nicht ueberall
    Material — eine Zunge dort faellt als loser Splitter ab statt die
    Haelften zu verbinden. Darum wird jede Position vorher geprueft: es
    muss beiderseits der Trennebene genug Material im Kasten liegen.
    """
    keep = []
    for index, box in enumerate(_tongue_boxes(p, 0.0)):
        material = box.intersect(body)
        try:
            right = volume_mm3(material.intersect(right_space))
            left = volume_mm3(material.intersect(left_space))
        except Exception:
            continue
        if min(right, left) >= MIN_TONGUE_VOL:
            keep.append(index)
    return keep


def split_halves(p, body: cq.Workplane):
    """Teilt die Maske an x = 0 in zwei ineinandersteckende Haelften."""
    right_space, left_space = _half_space(True), _half_space(False)
    usable = _useful_tongues(p, body, right_space, left_space)
    if not usable:
        raise ValueError("keine brauchbare Zungenposition gefunden — "
                         "split_tongues erhoehen oder Masse pruefen")

    plain = _tongue_boxes(p, 0.0)
    loose = _tongue_boxes(p, p.split_clearance)
    tongues, sockets = None, None
    for index in usable:
        tongues = plain[index] if tongues is None else tongues.union(plain[index])
        sockets = loose[index] if sockets is None else sockets.union(loose[index])

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

    ece = blinker.ece_check(p) if p.stalk else None
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
        "aussparungen_je_seite": p.slot_count if p.slots else 0,
        "passt_aufs_bett": fits,
        "ece_blinker": ece,
        "warnungen": warnings,
    }
