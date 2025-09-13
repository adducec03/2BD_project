#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizza il disegno contenuto in un JSON di input con campo 'disegnoJson'.
- Converte da px a mm usando (in ordine di priorità):
    1) scale_mm_per_px
    2) scale_cm_per_px * 10
    3) mediana di (misure_mm / misure_px)
    4) fallback 1 px = 1 mm
- Mostra il contorno e le quote (mm) lungo i lati.
- Opzione: mostrare gli angoli ai vertici.

Usage:
    python visualize_disegno.py input.json [--show-angles]
"""

import json
import math
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _to_float_list(seq):
    out = []
    for v in (seq or []):
        try:
            out.append(float(v))
        except Exception:
            pass
    return out


def load_disegno(json_path: Path):
    """Ritorna (disegno:dict). Gestisce disegnoJson come stringa o dict."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    raw = data.get("disegnoJson")
    if isinstance(raw, str):
        dj = json.loads(raw)
    elif isinstance(raw, dict):
        dj = raw
    else:
        raise ValueError("Campo 'disegnoJson' mancante o non valido.")
    if not isinstance(dj, dict) or "disegno" not in dj:
        raise ValueError("Struttura 'disegnoJson' non valida (manca 'disegno').")
    D = dj["disegno"]

    # normalizza alcuni campi utili
    for k in ("scale_mm_per_px", "scale_cm_per_px"):
        if k in D:
            try:
                D[k] = float(D[k])
            except Exception:
                D[k] = None

    for k in ("misure", "misure_px", "misure_mm", "angoli"):
        if k in D and isinstance(D[k], list):
            D[k] = _to_float_list(D[k])

    # vertici come (x,y) float
    if "vertici" not in D or not isinstance(D["vertici"], list):
        raise ValueError("Mancano i 'vertici' nel disegno.")
    V = []
    for v in D["vertici"]:
        if isinstance(v, dict) and "x" in v and "y" in v:
            V.append((float(v["x"]), float(v["y"])))
    if len(V) < 3:
        raise ValueError("Numero di vertici insufficiente.")
    D["_vertici_px"] = V

    # lines (opzionali)
    lines = []
    for L in D.get("lines", []) or []:
        try:
            lines.append(((float(L["x1"]), float(L["y1"])),
                          (float(L["x2"]), float(L["y2"]))))
        except Exception:
            pass
    D["_lines_px"] = lines

    return D


def detect_scale_mm_per_px(D: dict) -> float:
    """Determina il fattore mm/px con più strategie."""
    s_mm = D.get("scale_mm_per_px")
    if isinstance(s_mm, (int, float)) and s_mm > 0:
        return float(s_mm)
    s_cm = D.get("scale_cm_per_px")
    if isinstance(s_cm, (int, float)) and s_cm > 0:
        return float(s_cm) * 10.0
    mm = D.get("misure_mm")
    px = D.get("misure_px")
    if isinstance(mm, list) and isinstance(px, list) and len(mm) == len(px) and len(px) > 0:
        pairs = [(float(m), float(p)) for m, p in zip(mm, px) if float(p) > 0]
        if pairs:
            ratios = [m / p for m, p in pairs]  # mm/px
            return float(np.median(ratios))
    # fallback prudente
    return 1.0


def to_mm(points_px, mm_per_px: float):
    return [(x * mm_per_px, y * mm_per_px) for (x, y) in points_px]


def build_edges_mm(D: dict, mm_per_px: float):
    """
    Ritorna:
      edges_mm:   lista di ((x1,y1),(x2,y2)) in mm
      labels_mm:  lista stesse dimensioni degli edges, con lunghezze in mm
    Usa 'lines' se presenti, altrimenti i lati del poligono 'vertici'.
    Per le etichette usa 'misure_mm' se coerenti, altrimenti calcola dalla geometria.
    """
    # 1) geometria
    if D["_lines_px"]:
        lines_mm = [(
            (a[0] * mm_per_px, a[1] * mm_per_px),
            (b[0] * mm_per_px, b[1] * mm_per_px)
        ) for (a, b) in D["_lines_px"]]
    else:
        Vpx = D["_vertici_px"]
        lines_mm = []
        for i in range(len(Vpx)):
            a = Vpx[i]
            b = Vpx[(i + 1) % len(Vpx)]
            lines_mm.append((
                (a[0] * mm_per_px, a[1] * mm_per_px),
                (b[0] * mm_per_px, b[1] * mm_per_px),
            ))

    # 2) etichette
    misure_mm = D.get("misure_mm") or []
    if len(misure_mm) == len(lines_mm):
        labels_mm = list(misure_mm)
    else:
        # calcola dalle coordinate
        labels_mm = []
        for (p, q) in lines_mm:
            dx = q[0] - p[0]
            dy = q[1] - p[1]
            labels_mm.append(math.hypot(dx, dy))
    return lines_mm, labels_mm


def plot_disegno(D: dict, show_angles=False):
    mm_per_px = detect_scale_mm_per_px(D)
    Vmm = to_mm(D["_vertici_px"], mm_per_px)
    edges_mm, labels_mm = build_edges_mm(D, mm_per_px)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)

    # poligono (contorno)
    xs = [p[0] for p in Vmm] + [Vmm[0][0]]
    ys = [p[1] for p in Vmm] + [Vmm[0][1]]
    ax.plot(xs, ys, "-k", lw=2, label="contorno")

    # quote/etichette per ogni lato
    for (p, q), Lmm in zip(edges_mm, labels_mm):
        # segmento
        ax.plot([p[0], q[0]], [p[1], q[1]], lw=1, color="#555555", alpha=0.6)

        # etichetta al centro con rotazione lungo il lato
        mx, my = (p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0
        ang = math.degrees(math.atan2(q[1] - p[1], q[0] - p[0]))
        text = f"{Lmm:.1f} mm"
        ax.text(
            mx, my, text,
            rotation=ang,
            rotation_mode="anchor",
            ha="center", va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.5", alpha=0.9)
        )

    # opzionale: angoli ai vertici (se presenti in input)
    if show_angles:
        angoli = D.get("angoli") or []
        if len(angoli) == len(Vmm):
            for (x, y), a in zip(Vmm, angoli):
                ax.text(
                    x, y, f"{a:.1f}°",
                    ha="center", va="center",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.5", alpha=0.85)
                )

    # info sulla scala
    ax.text(
        0.02, 0.98,
        f"Scala: {mm_per_px:.6f} mm/px",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.9)
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.set_title("Disegno (mm)")

    # per assomigliare ai sistemi con origine in alto a sinistra
    ax.invert_yaxis()

    # margini
    allx = [p[0] for p in Vmm]
    ally = [p[1] for p in Vmm]
    pad_x = (max(allx) - min(allx)) * 0.08
    pad_y = (max(ally) - min(ally)) * 0.08
    ax.set_xlim(min(allx) - pad_x, max(allx) + pad_x)
    ax.set_ylim(max(ally) + pad_y, min(ally) - pad_y)

    plt.tight_layout()
    plt.show()


def main():

    input_json="new_version/input.json"
    show_angles=True

    D = load_disegno(Path(input_json))
    plot_disegno(D, show_angles=show_angles)


if __name__ == "__main__":
    main()