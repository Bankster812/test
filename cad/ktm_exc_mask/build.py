#!/usr/bin/env python3
"""Baut die modifizierte KTM-EXC-Maske aus params.json.

Beispiele
---------
    python3 build.py                                  # alles mit Standardwerten
    python3 build.py --set wall=3.4 --set blinker_z=-42
    python3 build.py --rear tabs --split --gauges
    python3 build.py --formats step                   # nur STEP fuer Fusion/FreeCAD
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cadquery as cq  # noqa: E402
import trimesh  # noqa: E402

from ktm_mask import gauges  # noqa: E402
from ktm_mask.assembly import build_mask, report, split_halves  # noqa: E402
from ktm_mask.geometry import level_points  # noqa: E402
from ktm_mask.params import MaskParams  # noqa: E402

FORMATS = {"step": "STEP", "stl": "STL", "svg": "SVG", "3mf": "3MF",
           "brep": "BREP"}
VIEWS = {"iso": (-1.0, -0.9, 0.6), "vorn": (0.0, 0.0, 1.0),
         "seite": (1.0, 0.0, 0.0)}


def apply_overrides(p: MaskParams, assignments) -> MaskParams:
    """--set key=value auf die passenden Typen abbilden."""
    types = {f.name: f.type for f in fields(MaskParams)}
    for item in assignments or []:
        if "=" not in item:
            raise SystemExit(f"--set braucht key=value, bekam '{item}'")
        key, raw = item.split("=", 1)
        key = key.strip()
        if key not in types:
            raise SystemExit(f"unbekannter Parameter '{key}'")
        current = getattr(p, key)
        if isinstance(current, bool):
            value = raw.strip().lower() in ("1", "true", "ja", "yes", "on")
        elif isinstance(current, int) and not isinstance(current, bool):
            value = int(float(raw))
        elif isinstance(current, float):
            value = float(raw)
        elif isinstance(current, list):
            value = json.loads(raw)
        else:
            value = raw
        setattr(p, key, value)
    return p


def tidy_stl(path: Path) -> None:
    """Nullflaechen-Fetzen aus dem Netz entfernen.

    An Stellen, wo eine Verschneidung tangential auslaeuft, laesst der
    Kernel entartete Flaechen stehen. Sie haben kein Volumen und keine
    offenen Kanten, machen das Netz aber ungeschlossen — und damit fuer den
    Slicer unbrauchbar. `clean()` bekommt sie nicht weg, hier fallen sie
    beim Zerlegen in Komponenten heraus.
    """
    mesh = trimesh.load(path, force="mesh")
    parts = [q for q in mesh.split(only_watertight=False) if abs(q.volume) > 1.0]
    if len(parts) == len(mesh.split(only_watertight=False)):
        return
    tidy = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
    tidy.merge_vertices()
    tidy.remove_unreferenced_vertices()
    tidy.export(path)
    print(f"[i] {path.name}: {len(mesh.split(only_watertight=False)) - len(parts)} "
          f"Nullflaechen-Fetzen entfernt, geschlossen={tidy.is_watertight}")


def export(obj, out_dir: Path, stem: str, formats, p: MaskParams) -> list:
    written = []
    for fmt in formats:
        kind = FORMATS[fmt]
        if fmt == "svg":
            for view, direction in VIEWS.items():
                path = out_dir / f"{stem}_{view}.svg"
                cq.exporters.export(obj, str(path), exportType="SVG", opt={
                    "width": 900, "height": 900, "marginLeft": 20,
                    "marginTop": 20, "showAxes": False,
                    "projectionDir": direction, "strokeWidth": 0.4,
                    "strokeColor": (40, 40, 40), "hiddenColor": (200, 200, 200),
                    "showHidden": False})
                written.append(path)
            continue
        path = out_dir / f"{stem}.{fmt}"
        if fmt == "stl":
            cq.exporters.export(obj, str(path), exportType="STL",
                                tolerance=p.stl_tolerance,
                                angularTolerance=p.stl_angular_tolerance)
            tidy_stl(path)
        else:
            cq.exporters.export(obj, str(path), exportType=kind)
        written.append(path)
    return written


def export_outlines(p: MaskParams, out_dir: Path) -> list:
    """Konturen der Woelbungsebenen als DXF, massstabsgetreu in Millimetern.

    Bei 100 % ausdrucken und an die echte Maske halten: so siehst du ohne
    Messschieber, ob Breite, Hoehe und der Verlauf der Kontur stimmen.
    """
    written = []
    for lvl in p.level_objs:
        if lvl.scale < 0.2:
            continue                      # der Scheitel taugt nicht als Schablone
        pts = level_points(p.outline_ccw, p.width, p.height, p.level_objs, lvl.z)
        wire = cq.Workplane("XY").spline(pts, periodic=True)
        path = out_dir / f"kontur_z{lvl.z:.0f}mm.dxf"
        cq.exporters.exportDXF(wire, str(path))
        written.append(path)
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", default="params.json",
                    help="Parameterdatei (Standard: params.json)")
    ap.add_argument("--out", default="out", help="Ausgabeverzeichnis")
    ap.add_argument("--formats", default="step,stl,svg",
                    help="Kommaliste aus " + ", ".join(FORMATS))
    ap.add_argument("--set", action="append", metavar="KEY=WERT",
                    help="einzelnen Parameter ueberschreiben, mehrfach moeglich")
    ap.add_argument("--rear", choices=("tabs", "none"),
                    help="Variante der Rueckseite")
    ap.add_argument("--no-slots", action="store_true",
                    help="Blinkerhalter-Aussparungen weglassen")
    ap.add_argument("--stalk", action="store_true",
                    help="zusaetzlich die Schaftaufnahme fuer einen "
                         "Schraubblinker einarbeiten")
    ap.add_argument("--split", action="store_true",
                    help="in zwei Haelften teilen (kleines Druckbett)")
    ap.add_argument("--gauges", action="store_true",
                    help="zusaetzlich die Passproben exportieren")
    ap.add_argument("--outlines", action="store_true",
                    help="Querschnittskonturen als DXF zum Ausdrucken (1:1)")
    ap.add_argument("--stem", default="exc_maske_modifiziert",
                    help="Dateiname ohne Endung")
    args = ap.parse_args(argv)

    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    bad = [f for f in formats if f not in FORMATS]
    if bad:
        raise SystemExit(f"unbekanntes Format: {bad}, moeglich: {list(FORMATS)}")

    params_path = Path(args.params)
    p = MaskParams.load(params_path) if params_path.exists() else MaskParams()
    if not params_path.exists():
        print(f"[i] {params_path} nicht gefunden — Standardwerte werden benutzt")
    p = apply_overrides(p, args.set)
    if args.rear:
        p.rear_mount_type = args.rear
    if args.no_slots:
        p.slots = False
    if args.stalk:
        p.stalk = True
    if args.split:
        p.split = True

    for w in p.validate():
        print(f"[!] {w}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[i] Maske wird aufgebaut …")
    mask = build_mask(p)

    rep = report(p, mask)
    (out_dir / "bericht.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")

    written = export(mask, out_dir, args.stem, formats, p)

    if p.split:
        print("[i] Maske wird geteilt …")
        right, left = split_halves(p, mask)
        for half, name in ((right, "haelfte_rechts"), (left, "haelfte_links")):
            n_solids = len(half.val().Solids())
            if n_solids != 1:
                print(f"[!] {name}: {n_solids} Koerper — Steckzungen pruefen")
            written += export(half, out_dir, f"{args.stem}_{name}",
                              [f for f in formats if f != "svg"], p)

    if args.outlines:
        print("[i] Konturen werden exportiert …")
        written += export_outlines(p, out_dir)

    if args.gauges:
        print("[i] Passproben werden erzeugt …")
        for name, fn in gauges.ALL.items():
            written += export(fn(p, mask), out_dir, f"probe_{name}",
                              [f for f in formats if f in ("stl", "step")], p)

    d = rep["abmessungen_mm"]
    print(f"\n  Groesse       {d['x']} x {d['y']} x {d['z']} mm")
    print(f"  Volumen       {rep['volumen_cm3']} cm3  (~{rep['materialbedarf_g']:.0f} g "
          f"bei Dichte {p.density})")
    print(f"  Koerper       {rep['koerper']}")
    print(f"  Rueckseite    {rep['rueckseite']}")
    print(f"  Aussparungen  {rep['aussparungen_je_seite']} je Seite")
    if rep["ece_blinker"]:
        e = rep["ece_blinker"]
        mark = "ok" if e["erfuellt"] else "ZU ENG"
        print(f"  Blinker       Innenabstand {e['innenkanten_abstand_mm']} mm "
              f"(Richtwert {e['richtwert_mm']}) — {mark}")
    for w in rep["warnungen"]:
        print(f"  [!] {w}")
    print("\n  geschrieben:")
    for path in written:
        print(f"    {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
