#!/usr/bin/env python3
"""Misst das gebaute Modell nach und meldet Konstruktionsfehler.

Nach jeder Parameteraenderung sinnvoll:  python3 selftest.py

Geprueft wird, was man einem Rendering nicht ansieht: ob der Koerper
geschlossen ist, ob die Wand wirklich so stark ist wie eingestellt, ob der
Taschenboden traegt und ob die Bohrungen durchgehen.
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cadquery as cq  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402

from ktm_mask import blinker  # noqa: E402
from trimesh.proximity import closest_point_naive  # noqa: E402

from ktm_mask.assembly import (build_mask, fits_on_bed, split_halves,  # noqa: E402
                               volume_mm3)
from ktm_mask.params import MaskParams  # noqa: E402
from ktm_mask.shell import hollow, outer_solid  # noqa: E402

TOL = 0.35  # zulaessige Abweichung durch die STL-Tesselierung, mm


class Result:
    def __init__(self):
        self.rows = []

    def check(self, name, ok, detail=""):
        self.rows.append((name, bool(ok), detail))
        print(f"  {'ok  ' if ok else 'FEHL'}  {name:<44} {detail}")
        return ok

    @property
    def failed(self):
        return [r for r in self.rows if not r[1]]


def to_mesh(obj, p) -> trimesh.Trimesh:
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as fh:
        path = fh.name
    cq.exporters.export(obj, path, exportType="STL",
                        tolerance=p.stl_tolerance,
                        angularTolerance=p.stl_angular_tolerance)
    mesh = trimesh.load(path, force="mesh")
    Path(path).unlink(missing_ok=True)
    return mesh


def crossings(mesh, origin, direction, eps=1e-9):
    """Abstaende aller Flaechendurchstosse entlang eines Strahls, sortiert.

    Moeller-Trumbore, ueber alle Dreiecke auf einmal gerechnet. Bewusst
    ohne trimesh.ray, damit die Pruefung keine Zusatzpakete (rtree) braucht.
    """
    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)

    tri = mesh.triangles                     # (n, 3, 3)
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    e1, e2 = v1 - v0, v2 - v0
    pvec = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, pvec)
    parallel = np.abs(det) < eps
    inv_det = np.divide(1.0, det, out=np.zeros_like(det), where=~parallel)

    tvec = origin - v0
    u = np.einsum("ij,ij->i", tvec, pvec) * inv_det
    qvec = np.cross(tvec, e1)
    v = np.einsum("j,ij->i", direction, qvec) * inv_det
    t = np.einsum("ij,ij->i", e2, qvec) * inv_det

    hit = (~parallel) & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 1e-7)
    d = np.sort(t[hit])
    if d.size == 0:
        return d
    # Mehrfachtreffer auf gemeinsamen Dreieckskanten zusammenfassen
    keep = [d[0]]
    for value in d[1:]:
        if value - keep[-1] > 1e-3:
            keep.append(value)
    return np.array(keep)


def run(p: MaskParams) -> Result:
    r = Result()
    print(f"\nPruefe Modell  (Rueckseite={p.rear_mount_type}, wall={p.wall} mm)")

    for w in p.validate():
        print(f"  hinw  Parameterwarnung: {w}")

    mask = build_mask(p)
    solid = mask.val()
    r.check("Modell ist ein zusammenhaengender Koerper",
            len(solid.Solids()) == 1, f"{len(solid.Solids())} Koerper")

    mesh = to_mesh(mask, p)
    r.check("STL ist geschlossen (druckbar)", mesh.is_watertight,
            f"{len(mesh.faces)} Dreiecke")
    r.check("STL hat konsistente Normalen", mesh.volume > 0,
            f"{mesh.volume / 1000:.1f} cm3")
    exact = volume_mm3(mask)
    r.check("Modell und STL beschreiben denselben Koerper",
            abs(exact - mesh.volume) / exact < 0.02,
            f"exakt {exact / 1000:.1f} cm3, STL {mesh.volume / 1000:.1f} cm3")

    # --- Wandstaerke ueber die ganze Haut -----------------------------------
    # gemessen als Abstand von Punkten der Aussenhaut zur Kavitaetsflaeche,
    # also senkrecht zur Oberflaeche und nicht achsparallel
    outer = outer_solid(p)
    m_outer = to_mesh(outer, p)
    m_cav = to_mesh(outer.cut(hollow(p, outer)), p)
    pts, _ = trimesh.sample.sample_surface(m_outer, 2500, seed=1)
    pts = pts[pts[:, 2] > p.z_rear + 4.0]          # hintere Deckflaeche ignorieren
    _, dist, _ = closest_point_naive(m_cav, pts)
    p1, med = np.percentile(dist, 1), float(np.median(dist))
    r.check("Wandstaerke im Mittel wie eingestellt", abs(med - p.wall) <= 0.3,
            f"Median {med:.2f} mm, eingestellt {p.wall} mm")
    r.check("keine duennen Stellen in der Wand", p1 >= p.wall * 0.7,
            f"1%-Quantil {p1:.2f} mm, min {dist.min():.2f} mm")

    if p.blinker:
        n = np.array(blinker.axis(p, +1))
        mp = np.array(blinker.mount_point(p, +1))

        # --- Boden der Blinkertasche traegt ---------------------------------
        # seitlich neben der Schaftbohrung messen, dort darf nichts offen sein
        up = np.array([0.0, 1.0, 0.0])
        tangent = np.cross(n, up)
        tangent /= np.linalg.norm(tangent)
        offset = tangent * (p.pocket_w / 2.0 - p.pocket_margin * 0.5)
        d = crossings(mesh, mp + n * 60.0 + offset, -n)
        if len(d) >= 2:
            floor = d[1] - d[0]
            r.check("Taschenboden hat Material",
                    floor >= p.pocket_floor_t - TOL,
                    f"gemessen {floor:.2f} mm, gefordert {p.pocket_floor_t} mm")
        else:
            r.check("Taschenboden hat Material", False,
                    f"Strahl traf {len(d)} Flaechen — Tasche bricht durch")

        # --- Tasche ist so tief wie eingestellt -----------------------------
        d_pocket = crossings(mesh, mp + n * 60.0 + offset, -n)
        if len(d_pocket) >= 1:
            depth = d_pocket[0] - 60.0 + 0.0
            r.check("Aussparungstiefe stimmt",
                    abs(depth - p.pocket_depth) <= TOL + 0.4,
                    f"gemessen {depth:.2f} mm, eingestellt {p.pocket_depth} mm")

        # --- Schaftbohrung geht durch ---------------------------------------
        d = crossings(mesh, mp + n * 60.0, -n)
        r.check("Schaftbohrung ist durchgehend", len(d) <= 2,
                f"{len(d)} Flaechen auf der Achse (0 oder 2 = frei)")

    # --- Teilung ------------------------------------------------------------
    if p.split:
        right, left = split_halves(p, mask)
        ok = (len(right.val().Solids()) == 1 and len(left.val().Solids()) == 1)
        r.check("beide Haelften sind je ein Koerper", ok,
                f"{len(right.val().Solids())} / {len(left.val().Solids())}")
        whole = volume_mm3(mask)
        total = volume_mm3(right) + volume_mm3(left)
        loss = (whole - total) / whole
        r.check("Teilung verliert kein Material", 0 <= loss < 0.02,
                f"{loss * 100:.2f} % Differenz (Steckspiel)")
        bb = right.val().BoundingBox()
        dims = (bb.xlen, bb.ylen, bb.zlen)
        r.check("Haelfte passt aufs Druckbett",
                fits_on_bed(dims, (p.bed_x, p.bed_y, p.bed_z)),
                f"{dims[0]:.0f} x {dims[1]:.0f} x {dims[2]:.0f} mm auf "
                f"{p.bed_x:.0f} x {p.bed_y:.0f} x {p.bed_z:.0f} mm")

    # --- ECE-Rechenhilfe ----------------------------------------------------
    if p.blinker:
        e = blinker.ece_check(p)
        r.check("Blinkerabstand erreicht den Richtwert", e["erfuellt"],
                f"{e['innenkanten_abstand_mm']} mm (Richtwert "
                f"{e['richtwert_mm']} mm)")
    return r


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    path = Path(args[0]) if args else Path("params.json")
    p = MaskParams.load(path) if path.exists() else MaskParams()
    p.split = True
    failed = run(p).failed
    print()
    if failed:
        print(f"{len(failed)} Pruefung(en) fehlgeschlagen:")
        for name, _, detail in failed:
            print(f"  - {name}: {detail}")
        return 1
    print("alle Pruefungen bestanden")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
