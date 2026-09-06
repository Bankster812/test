#!/usr/bin/env python3
"""Umriss der Maske aus einem Foto abgreifen und in params.json schreiben.

Das ist der genaueste Weg zur richtigen Form, ohne die Maske zu scannen:

  1. Maske flach auf einen hellen, gleichmaessigen Untergrund legen
     (weisses Papier, Bettlaken). Stirnflaeche nach oben.
  2. Von GERADE OBEN fotografieren, moeglichst weit weg und herangezoomt —
     aus der Naehe fotografiert werden die Raender perspektivisch verzerrt.
  3. Ein Lineal mit ins Bild legen, oder die Breite der Maske einmal messen.

      python3 trace_outline.py --bild maske.jpg --breite 245 --vorschau

  Danach `python3 build.py` — das Modell hat jetzt deinen Umriss.

Erkannt wird der dunkle Bereich vor hellem Grund. Kontrolliere das Ergebnis
mit `--vorschau`: das erzeugte PNG legt den gefundenen Umriss ueber das Foto.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from ktm_mask.params import MaskParams  # noqa: E402


def load_gray(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError:
        raise SystemExit("Pillow fehlt:  pip install pillow")
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=float) / 255.0


def silhouette(gray: np.ndarray, threshold: float | None):
    """Maske des dunklen Objekts vor hellem Grund."""
    if threshold is None:
        # Schwelle zwischen den beiden Helligkeitsgipfeln (Otsu)
        hist, edges = np.histogram(gray, bins=256, range=(0.0, 1.0))
        total = hist.sum()
        omega = np.cumsum(hist) / total
        mu = np.cumsum(hist * np.arange(256)) / total
        mu_t = mu[-1]
        denom = omega * (1.0 - omega)
        denom[denom == 0] = 1e-12
        between = (mu_t * omega - mu) ** 2 / denom
        threshold = float(np.argmax(between)) / 255.0
    return gray < threshold, threshold


def contour_from_rows(mask: np.ndarray, min_run: int = 12):
    """Umriss aus den Zeilenraendern.

    Fuer jede Bildzeile den linken und rechten Rand des Objekts nehmen und
    beides zu einem geschlossenen Umlauf zusammensetzen. Reicht fuer eine
    Maskensilhouette voellig aus und braucht weder OpenCV noch SciPy.
    """
    rows = []
    for y in range(mask.shape[0]):
        xs = np.flatnonzero(mask[y])
        if xs.size >= min_run:
            rows.append((y, xs.min(), xs.max()))
    if len(rows) < 20:
        raise SystemExit("kein zusammenhaengendes Objekt gefunden — "
                         "Schwelle mit --schwelle setzen oder Foto pruefen")
    right = [(x1, y) for y, _, x1 in rows]
    left = [(x0, y) for y, x0, _ in reversed(rows)]
    return np.array(right + left, dtype=float)


def resample(points: np.ndarray, count: int) -> np.ndarray:
    """Gleichmaessig nach Bogenlaenge neu abtasten."""
    closed = np.vstack([points, points[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, dist[-1], count, endpoint=False)
    out = np.empty((count, 2))
    for axis in (0, 1):
        out[:, axis] = np.interp(targets, dist, closed[:, axis])
    return out


def normalise(points: np.ndarray):
    """Auf die Einheitsbox -0.5 … +0.5 bringen; Y nach oben drehen."""
    x, y = points[:, 0], -points[:, 1]        # Bildzeilen laufen nach unten
    w = x.max() - x.min()
    h = y.max() - y.min()
    nx = (x - (x.min() + x.max()) / 2.0) / w
    ny = (y - (y.min() + y.max()) / 2.0) / h
    return np.stack([nx, ny], axis=1), w, h


def preview(path: Path, gray, mask, points_px, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(gray, cmap="gray")
    axes[0].plot(points_px[:, 0], points_px[:, 1], "-", lw=1.6, color="#d33")
    axes[0].plot(np.append(points_px[:, 0], points_px[0, 0]),
                 np.append(points_px[:, 1], points_px[0, 1]), "-", lw=1.6,
                 color="#d33")
    axes[0].set_title(f"gefundener Umriss ({len(points_px)} Punkte)")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("erkannter Objektbereich")
    for ax in axes:
        ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out, dpi=110, facecolor="white")
    print(f"[i] Vorschau: {out}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bild", required=True, help="Foto der Maske von vorn")
    ap.add_argument("--params", default="params.json")
    ap.add_argument("--punkte", type=int, default=64,
                    help="Zahl der Konturpunkte (Standard 64)")
    ap.add_argument("--breite", type=float,
                    help="gemessene Gesamtbreite der Maske in mm")
    ap.add_argument("--hoehe", type=float,
                    help="gemessene Gesamthoehe in mm (sonst aus dem "
                         "Seitenverhaeltnis des Fotos)")
    ap.add_argument("--schwelle", type=float,
                    help="Helligkeitsschwelle 0…1 (sonst automatisch)")
    ap.add_argument("--vorschau", action="store_true",
                    help="Kontrollbild schreiben")
    ap.add_argument("--trocken", action="store_true",
                    help="nur anzeigen, params.json nicht aendern")
    args = ap.parse_args(argv)

    image = Path(args.bild)
    gray = load_gray(image)
    mask, used = silhouette(gray, args.schwelle)
    print(f"[i] Bild {gray.shape[1]} x {gray.shape[0]} px, "
          f"Schwelle {used:.3f}, Objektanteil {mask.mean() * 100:.1f} %")

    raw = contour_from_rows(mask)
    points_px = resample(raw, args.punkte)
    outline, w_px, h_px = normalise(points_px)

    params_path = Path(args.params)
    p = MaskParams.load(params_path) if params_path.exists() else MaskParams()
    p.outline = [[round(float(x), 4), round(float(y), 4)] for x, y in outline]

    if args.breite:
        p.width = float(args.breite)
        p.height = float(args.hoehe) if args.hoehe else round(
            args.breite * h_px / w_px, 1)
        print(f"[i] Groesse gesetzt: {p.width} x {p.height} mm "
              f"(Seitenverhaeltnis im Foto {w_px / h_px:.3f})")
    else:
        print(f"[i] Groesse unveraendert ({p.width} x {p.height} mm). Mit "
              "--breite die gemessene Breite angeben.")

    warnings = p.validate()
    for warning in warnings:
        print(f"[!] {warning}")

    if args.vorschau:
        preview(image, gray, mask, points_px,
                image.with_name(image.stem + "_umriss.png"))

    if args.trocken:
        print("[i] --trocken gesetzt, params.json bleibt unveraendert")
    else:
        p.save(params_path)
        print(f"[i] {params_path} geschrieben — jetzt: python3 build.py")
    return 1 if warnings else 0


if __name__ == "__main__":
    raise SystemExit(main())
