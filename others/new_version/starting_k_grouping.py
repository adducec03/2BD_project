#!/usr/bin/env python3
"""
Stage 1 – contact-graph construction + coarse k-way grouping.
Solo le parti usate dal programma principale.
"""

from pathlib import Path
import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from networkx.algorithms.coloring import greedy_color
import pymetis  # usato da rb_exact_partition

# -------------------------------------------------------------
# Costanti usate da build_contact_graph
# -------------------------------------------------------------
R   = 9.0   # mm cell radius
EPS = 0.2   # slack in adjacency threshold


# -------------------------------------------------------------
# I/O e grafi di contatto
# -------------------------------------------------------------
def load_centres(csv_path: Path) -> np.ndarray:
    """x,y CSV ▸ ndarray (N,2)"""
    return np.loadtxt(csv_path, delimiter=",", skiprows=1)


def build_contact_graph(centres: np.ndarray) -> nx.Graph:
    """
    Edge ↔ centres closer than 2R+EPS.
    Each edge carries a weight = 1/‖Δx‖.
    """
    N = centres.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(N))

    thr = 2 * R + EPS
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(centres[i] - centres[j])
            if d <= thr:
                w = 1.0 / d
                G.add_edge(i, j, weight=w)
    return G


# -------------------------------------------------------------
# Palette e plotting
# -------------------------------------------------------------
def make_big_palette(n: int, seed: int | None = None):
    # 1) colorcet se disponibile
    try:
        import colorcet as cc  # pip install colorcet
        base = list(cc.glasbey)
        if n <= len(base):
            return [mpl.colors.to_rgb(c) for c in base[:n]]
    except Exception:
        pass

    # 2) seaborn se disponibile
    try:
        import seaborn as sns  # pip install seaborn
        return sns.color_palette("husl", n)
    except Exception:
        pass

    # 3) fallback HSV
    hues = np.linspace(0, 1, n, endpoint=False)
    return [mpl.colors.hsv_to_rgb((h, 0.65, 0.95)) for h in hues]


def plot_groups(poly, centres, part_of, S, group_color=None, title="k-way grouping"):
    fig, ax = plt.subplots(figsize=(7, 7))
    # boundary
    ax.plot(*poly.exterior.xy, color="k", lw=2)

    # colori gruppo
    if group_color is None:
        palette = make_big_palette(S)
        group_color = {g: palette[g % len(palette)] for g in range(S)}

    # cerchi
    for idx, (x, y) in enumerate(centres):
        gid = part_of[idx] if isinstance(part_of, dict) else part_of[idx]
        col = group_color[gid]
        ax.add_patch(plt.Circle((x, y), R, facecolor=col, edgecolor="k", lw=0.4))

    # label gruppi
    for g in range(S):
        members = [i for i in (part_of.keys() if isinstance(part_of, dict) else range(len(part_of)))]
        members = [i for i in members if (part_of[i] if isinstance(part_of, dict) else part_of[i]) == g]
        if members:
            cx, cy = centres[members].mean(axis=0)
            ax.text(
                cx,
                cy,
                str(g),
                ha="center",
                va="center",
                fontsize=7,
                color="black",
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.5),
            )

    ax.set_aspect("equal")
    ax.set_title(f"{title} (S={S})")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------
# Colori evitando adiacenze tra gruppi
# -------------------------------------------------------------
def build_group_adjacency_graph(G: nx.Graph, part_of: dict[int, int], S: int) -> nx.Graph:
    """
    G       : contact graph (nodes = cells)
    part_of : dict {node_id -> group_id}
    S       : number of groups
    Returns: group adjacency graph H with node-set {0..S-1}
    """
    H = nx.Graph()
    H.add_nodes_from(range(S))
    for u, v in G.edges:
        gu, gv = part_of[u], part_of[v]
        if gu != gv:
            H.add_edge(gu, gv)
    return H


def color_groups_avoiding_adjacent(
    G: nx.Graph, part_of: dict[int, int], S: int, palette=None, strategy="saturation_largest_first"
):
    """
    Ritorna: dict {group_id -> RGB} in cui gruppi adiacenti hanno colori diversi.
    """
    H = build_group_adjacency_graph(G, part_of, S)
    col_idx = greedy_color(H, strategy=strategy)
    ncols = max(col_idx.values()) + 1 if col_idx else 1

    if palette is None or len(palette) < ncols:
        palette = make_big_palette(ncols)

    return {g: palette[col_idx.get(g, 0)] for g in range(S)}


# -------------------------------------------------------------
# Selezione N celle (peeling periferia) per ottenere esattamente S*P
# -------------------------------------------------------------
def drop_periphery_iterative(centres, N_keep):
    """
    Iterativamente rimuove celle dal bordo finché restano esattamente N_keep.
    – rimuove il vertice di grado più basso sul bordo a ogni step
    – ricalcola i gradi a ogni iterazione
    – mantiene connesso il grafo rimanente
    """
    N = len(centres)
    if N_keep >= N:
        return np.ones(N, bool)

    keep = np.ones(N, bool)
    G = build_contact_graph(centres)

    cx, cy = centres.mean(axis=0)
    d2_cent = np.square(centres[:, 0] - cx) + np.square(centres[:, 1] - cy)

    to_drop = N - N_keep
    for _ in range(to_drop):
        sub = G.subgraph(np.where(keep)[0])

        deg = {v: sub.degree(v) for v in sub}
        min_deg = min(deg.values())
        cand = [v for v, d in deg.items() if d == min_deg]

        # tie-breaker: più lontani dal centro prima
        cand.sort(key=lambda v: d2_cent[v], reverse=True)

        # scegli il primo che mantiene il grafo connesso
        for v in cand:
            keep[v] = False
            if nx.is_connected(sub.subgraph([u for u in sub if keep[u]])):
                break
            keep[v] = True

    return keep


# -------------------------------------------------------------
# Partizionamento esatto in gruppi con PyMetis (ricorsivo)
# -------------------------------------------------------------
def rb_exact_partition(G: nx.Graph, groups: list[int], P: int) -> dict[int, int]:
    """
    Ricorsiva bisezione in foglie di grandezza P.
    Ritorna part_of (vertex -> group_id) con dimensioni esatte e alta contiguità.
    """
    n = len(groups)
    S_here = n // P
    if S_here == 1:
        gid = rb_exact_partition.gid_counter
        rb_exact_partition.gid_counter += 1
        return {v: gid for v in groups}

    # quanti gruppi a sinistra/destra (multipli di P)
    S_left = S_here // 2
    S_right = S_here - S_left
    target_left = S_left * P
    target_right = S_right * P

    # CSR locale per il sottografo indotto
    idx = {v: i for i, v in enumerate(groups)}
    xadj, adjncy, eweights = [0], [], []
    for v in groups:
        for nb, data in G[v].items():
            if nb in idx:
                adjncy.append(idx[nb])
                w = data.get("weight", 1.0)
                eweights.append(int(w * 10000))
        xadj.append(len(adjncy))

    vweights = [1] * n
    tpwgts = [target_left / float(n), target_right / float(n)]

    # bisezione 2-way con target
    _, parts2 = pymetis.part_graph(
        2, xadj=xadj, adjncy=adjncy, eweights=eweights, vweights=vweights, tpwgts=tpwgts, recursive=True, contiguous=True
    )
    left = [groups[i] for i, p in enumerate(parts2) if p == 0]
    right = [groups[i] for i, p in enumerate(parts2) if p == 1]

    # piccole correzioni se ±1 dal target
    if len(left) != target_left:
        need = target_left - len(left)
        if need > 0:
            border_candidates = [v for v in right if any(u in left for u in G[v])]
            move = border_candidates[:need] if len(border_candidates) >= need else right[:need]
            left += move
            right = [v for v in right if v not in move]
        elif need < 0:
            need = -need
            border_candidates = [v for v in left if any(u in right for u in G[v])]
            move = border_candidates[:need] if len(border_candidates) >= need else left[:need]
            right += move
            left = [v for v in left if v not in move]

    # ricorsione
    L = rb_exact_partition(G, left, P)
    R = rb_exact_partition(G, right, P)
    return {**L, **R}


# contatore statico per nuovi group id
rb_exact_partition.gid_counter = 0