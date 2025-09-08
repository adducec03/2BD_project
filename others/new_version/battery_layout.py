
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Battery pack grouping & layout visualizer.

Input
-----
- CSV with columns "x,y" for cell centers.
- Integers S (number of groups in series) and P (cells in parallel / per group).

Constraints enforced
--------------------
1) Ogni gruppo ha esattamente P celle.
2) Ogni gruppo è connesso per adiacenza (distanza ~ 2R).
3) I collegamenti tra gruppi consecutivi vengono massimizzati in modo greedy,
   rispettando il limite di sicurezza: max 4 collegamenti per cella.
4) Se S*P > #celle, il programma esce con errore.

Usage
-----
python battery_layout.py --csv /path/out.csv --S 14 --P 17 --radius 9 --save fig.png

Note: l'algoritmo è euristico (greedy) ma produce layout "a linea" che abbracciano
il gruppo precedente, massimizzando i contatti tra gruppi adiacenti.
"""

import argparse
import math
from typing import List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors


def build_adjacency(coords: np.ndarray, diam: float, tol: float = 0.25):
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=diam + tol)
    adj: List[Set[int]] = [set() for _ in range(len(coords))]
    for i, j in pairs:
        d = float(np.linalg.norm(coords[i] - coords[j]))
        if abs(d - diam) <= tol + 1e-9:
            adj[i].add(j)
            adj[j].add(i)
    return adj


def color_for_group(i: int) -> str:
    palette = (list(mcolors.TABLEAU_COLORS.values())
               + list(mcolors.CSS4_COLORS.values()))
    return palette[i % len(palette)]


def layout_groups(coords: np.ndarray, S: int, P: int, R: float,
                  start_idx: int = None,
                  adj_tol: float = 0.25,
                  w_touch_prev: float = 5.0,
                  w_along: float = 1.2,
                  w_border: float = 0.2):
    N = len(coords)
    diam = 2 * R
    adj = build_adjacency(coords, diam, adj_tol)

    need = S * P
    if need > N:
        raise SystemExit(f"ERRORE: celle richieste ({need}) > celle disponibili ({N}).")

    if start_idx is None:
        # top-left (y minimo poi x minimo)
        start_idx = int(np.lexsort((coords[:, 0], coords[:, 1]))[0])

    # direzione principale per "linee"
    pca = PCA(n_components=2).fit(coords)
    d1 = pca.components_[0]
    d1 = d1 / (np.linalg.norm(d1) + 1e-9)

    assigned = np.full(N, -1, dtype=int)
    external_links_count = np.zeros(N, dtype=int)
    cross_edges: List[Tuple[int, int]] = []

    def build_group(seed: int, g_id: int, prev_group: Set[int]) -> Set[int]:
        group: Set[int] = set([seed])
        assigned[seed] = g_id

        def current_dir() -> np.ndarray:
            if len(group) == 1:
                return d1
            G = coords[list(group)]
            pc = PCA(n_components=2).fit(G).components_[0]
            return pc / (np.linalg.norm(pc) + 1e-9)

        def candidates() -> Set[int]:
            cands = set()
            for u in group:
                for v in adj[u]:
                    if assigned[v] == -1:
                        cands.add(v)
            return cands

        while len(group) < P:
            cand_set = candidates()
            if not cand_set:
                # fallback robusto: cerca la non assegnata più vicina al gruppo
                unassigned = np.where(assigned == -1)[0]
                if len(unassigned) == 0:
                    break
                G = coords[list(group)].mean(axis=0)
                dists = np.linalg.norm(coords[unassigned] - G, axis=1)
                idx = int(unassigned[np.argmin(dists)])
                cand_set = {idx}

            vdir = current_dir()
            best_v = None
            best_score = -1e18

            for v in cand_set:
                # preferisci toccare il gruppo precedente distribuendo i contatti
                touch_prev = 0
                if prev_group:
                    for u in adj[v]:
                        if u in prev_group and external_links_count[v] < 4 and external_links_count[u] < 4:
                            touch_prev += 1

                vec = coords[v] - coords[list(group)][0]
                if np.linalg.norm(vec) == 0:
                    along = 0.0
                else:
                    vec = vec / (np.linalg.norm(vec) + 1e-9)
                    along = abs(float(np.dot(vec, vdir)))

                borderiness = max(0, 6 - len(adj[v]))

                score = (w_touch_prev * touch_prev
                         + w_along * along
                         + w_border * borderiness)

                if score > best_score:
                    best_score = score
                    best_v = v

            v = best_v
            if v is None:
                break
            group.add(v)
            assigned[v] = g_id

            # registra collegamenti verso prev_group rispettando limite 4 per cella
            if prev_group:
                neighbors_prev = [u for u in adj[v] if u in prev_group]
                neighbors_prev.sort(key=lambda u: external_links_count[u])
                for u in neighbors_prev:
                    if external_links_count[v] >= 4:
                        break
                    if external_links_count[u] >= 4:
                        continue
                    external_links_count[v] += 1
                    external_links_count[u] += 1
                    cross_edges.append((u, v))

        return group

    groups: List[Set[int]] = []

    g0 = build_group(start_idx, 0, prev_group=set())
    groups.append(g0)

    for g_id in range(1, S):
        prev = groups[-1]
        seed = None
        # scegli un seed che massimizza i contatti col prev group
        best = (-1, -1)
        for u in prev:
            for v in adj[u]:
                if assigned[v] != -1:
                    continue
                possible = sum((w in prev) and (external_links_count[w] < 4) for w in adj[v])
                if possible > best[0]:
                    best = (possible, v)
        if best[0] >= 0:
            seed = best[1]
        else:
            # fallback: punto non assegnato più vicino al centro del prev group
            Gprev = coords[list(prev)].mean(axis=0)
            un = np.where(assigned == -1)[0]
            seed = int(un[np.argmin(np.linalg.norm(coords[un] - Gprev, axis=1))])

        gi = build_group(seed, g_id, prev_group=prev)
        groups.append(gi)

    # info collegamenti tra gruppi consecutivi
    links_between = []
    for k in range(1, len(groups)):
        a = groups[k - 1]
        b = groups[k]
        count = 0
        for (u, v) in cross_edges:
            if (u in a and v in b) or (u in b and v in a):
                count += 1
        links_between.append(count)

    return groups, cross_edges, assigned, links_between


def plot_layout(coords: np.ndarray, groups: List[Set[int]],
                cross_edges: List[Tuple[int, int]],
                assigned: np.ndarray, R: float,
                links_between: List[int], save: str = None, title: str = ""):
    fig, ax = plt.subplots(figsize=(11, 7), dpi=140)

    # groups
    for g_id, gset in enumerate(groups):
        color = color_for_group(g_id)
        for idx in gset:
            x, y = coords[idx]
            ax.add_patch(Circle((x, y), R, facecolor=color, edgecolor='black', linewidth=0.5))

    # unused cells
    for idx in range(len(coords)):
        if assigned[idx] == -1:
            x, y = coords[idx]
            ax.add_patch(Circle((x, y), R, facecolor='white', edgecolor='black', linewidth=0.35, alpha=0.6))

    # cross links
    for u, v in cross_edges:
        x1, y1 = coords[u]
        x2, y2 = coords[v]
        ax.plot([x1, x2], [y1, y2], color='black', linewidth=1.0)

    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])

    subtitle = ""
    if links_between:
        subtitle = f" — collegamenti medi tra gruppi: {np.mean(links_between):.1f}"
    ax.set_title(title + subtitle)

    plt.tight_layout()
    if save:
        plt.savefig(save, bbox_inches="tight", dpi=160)
    plt.show()


def main():
    

    csv = "new_version/out.csv"
    radius = 9.0
    S = 48
    P = 7
    adj_tol = 0.25


    df = pd.read_csv(csv)
    coords = df[['x', 'y']].to_numpy()
    groups, cross_edges, assigned, links_between = layout_groups(
        coords, S, P, radius, adj_tol=adj_tol)
    
    if links_between:
        print("Collegamenti tra gruppi consecutivi:", ", ".join(str(int(c)) for c in links_between))
        print(f"Media collegamenti: {float(np.mean(links_between)):.2f}")

    plot_layout(coords, groups, cross_edges, assigned, radius, links_between,
                save=None,
                title=f"{S}S{P}P")


if __name__ == "__main__":
    main()

