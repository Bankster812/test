#!/usr/bin/env python3
"""Uebertraegt die Merkmale auf eine ECHTE Maske (Scan oder fremdes Modell).

Das ist der Weg, der zaehlt, sobald du ein Netz deiner EXC-Maske hast: statt
die Maske nachzubauen, werden Blinkeraufnahme und Rueckseite in dein
vorhandenes Netz hineingerechnet.

    python3 apply_to_scan.py --scan exc_maske.stl --align auto
    python3 apply_to_scan.py --scan exc_maske.stl --nur blinker
    python3 apply_to_scan.py --scan exc_maske.stl --rotate 0,0,90 --translate 0,0,-5

Ablauf
------
1. Netz laden, reparieren, auf Geschlossenheit pruefen (Voraussetzung fuer
   boolesche Operationen).
2. In das Koordinatensystem des Modells bringen: X quer, Y hoch, Z laengs
   mit der Maskenvorderkante bei Z = 0.  `--align auto` macht das grob,
   Feinjustage ueber --rotate / --translate / --scale.
3. Dieselben Merkmalskoerper wie beim parametrischen Modell bilden, an der
   konvexen Huelle des Scans buendig beschneiden und verrechnen.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cadquery as cq  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402

from ktm_mask import features  # noqa: E402
from ktm_mask.params import MaskParams  # noqa: E402

ENGINE = "manifold"


def to_mesh(obj, p) -> trimesh.Trimesh:
    """CadQuery-Koerper zu Dreiecksnetz."""
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as fh:
        path = fh.name
    cq.exporters.export(obj, path, exportType="STL",
                        tolerance=p.stl_tolerance,
                        angularTolerance=p.stl_angular_tolerance)
    mesh = trimesh.load(path, force="mesh")
    Path(path).unlink(missing_ok=True)
    return mesh


def load_scan(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise SystemExit(f"{path}: keine einzelne Netzgeometrie")
    print(f"[i] geladen: {len(mesh.faces)} Dreiecke, "
          f"{mesh.extents[0]:.1f} x {mesh.extents[1]:.1f} x {mesh.extents[2]:.1f} mm")
    return mesh


def repair(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    if not mesh.is_watertight:
        print("[i] Netz ist offen — es wird versucht, die Loecher zu schliessen")
        mesh.fill_holes()
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fix_winding(mesh)
    return mesh


def align(mesh: trimesh.Trimesh, mode: str, scale, rotate, translate):
    if scale and scale != 1.0:
        mesh.apply_scale(scale)
    if rotate:
        for axis, ang in zip(((1, 0, 0), (0, 1, 0), (0, 0, 1)), rotate):
            if abs(ang) > 1e-9:
                mesh.apply_transform(trimesh.transformations.rotation_matrix(
                    np.radians(ang), axis))
    if mode == "auto":
        lo, hi = mesh.bounds
        centre = (lo + hi) / 2.0
        # X/Y mittig, Vorderkante der Maske auf Z = 0
        mesh.apply_translation((-centre[0], -centre[1], -hi[2]))
        print("[i] ausgerichtet: X/Y mittig, Vorderkante auf Z = 0")
    if translate:
        mesh.apply_translation(translate)
    lo, hi = mesh.bounds
    print(f"[i] Lage jetzt: X [{lo[0]:.1f} … {hi[0]:.1f}]  "
          f"Y [{lo[1]:.1f} … {hi[1]:.1f}]  Z [{lo[2]:.1f} … {hi[2]:.1f}]")
    return mesh


def parse_triplet(text):
    if not text:
        return None
    parts = [float(v) for v in text.replace(";", ",").split(",")]
    if len(parts) != 3:
        raise SystemExit(f"'{text}': drei durch Komma getrennte Zahlen erwartet")
    return parts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", required=True, help="STL/OBJ/PLY/3MF der echten Maske")
    ap.add_argument("--params", default="params.json")
    ap.add_argument("--out", default="out/exc_maske_uebertragen.stl")
    ap.add_argument("--align", choices=("auto", "none"), default="auto")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--rotate", help="Grad um X,Y,Z — z. B. 0,0,90")
    ap.add_argument("--translate", help="Verschiebung X,Y,Z in mm")
    ap.add_argument("--nur", choices=("aussparungen", "rueckseite", "alles"),
                    default="alles", help="nur ein Merkmal uebertragen")
    ap.add_argument("--clip", choices=("huelle", "keine"), default="huelle",
                    help="angeformte Felder buendig an der Aussenhaut beschneiden")
    ap.add_argument("--force", action="store_true",
                    help="auch mit offenem Netz rechnen (Ergebnis meist unbrauchbar)")
    args = ap.parse_args(argv)

    params_path = Path(args.params)
    p = MaskParams.load(params_path) if params_path.exists() else MaskParams()
    if args.nur == "aussparungen":
        p.rear_mount_type = "none"
        p.top_bracket = False
        p.light_boss = False
    elif args.nur == "rueckseite":
        p.slots = False
        p.stalk = False

    mesh = repair(load_scan(Path(args.scan)))
    if not mesh.is_watertight:
        msg = ("Netz ist nach der Reparatur noch offen. Boolesche Operationen "
               "brauchen ein geschlossenes Netz — erst in Meshmixer, Blender "
               "oder PrusaSlicer reparieren.")
        if not args.force:
            raise SystemExit(f"[x] {msg}")
        print(f"[!] {msg} (--force gesetzt, es wird trotzdem gerechnet)")
    else:
        print(f"[i] Netz ist geschlossen, Volumen {mesh.volume / 1000:.1f} cm3")

    mesh = align(mesh, args.align, args.scale,
                 parse_triplet(args.rotate), parse_triplet(args.translate))

    clip_body = mesh.convex_hull if args.clip == "huelle" else None
    if clip_body is not None:
        print("[i] angeformte Felder werden an der konvexen Huelle beschnitten")

    result = mesh
    for feat in features.adders(p):
        solid = to_mesh(feat.solid, p)
        if feat.clip and clip_body is not None:
            solid = trimesh.boolean.intersection([solid, clip_body], engine=ENGINE)
            if solid.is_empty or len(solid.faces) == 0:
                print(f"[!] {feat.name}: liegt ausserhalb des Scans — uebersprungen. "
                      "Ausrichtung oder Position im Parametersatz pruefen.")
                continue
        result = trimesh.boolean.union([result, solid], engine=ENGINE)
        print(f"[+] angeformt: {feat.name}")

    for feat in features.cutters(p):
        result = trimesh.boolean.difference([result, to_mesh(feat.solid, p)],
                                            engine=ENGINE)
        print(f"[-] abgezogen: {feat.name}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.export(out)

    print(f"\n  Ergebnis      {out}")
    print(f"  Dreiecke      {len(result.faces)}")
    print(f"  geschlossen   {result.is_watertight}")
    print(f"  Volumen       {result.volume / 1000:.1f} cm3  "
          f"(~{result.volume / 1000 * p.density:.0f} g)")
    e = result.extents
    print(f"  Groesse       {e[0]:.1f} x {e[1]:.1f} x {e[2]:.1f} mm")
    if not result.is_watertight:
        print("  [!] Ergebnis ist nicht geschlossen — vor dem Slicen reparieren")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
