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
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib import cm

R=9.0


import time
import csv
import numpy as np
import networkx as nx
from collections import defaultdict


# --- PALETTE CONDIVISA (stessa usata in series_ordering) ---------------------
def build_series_palette(S: int):
    """
    Restituisce una palette deterministica con priorità:
    1) colorcet.glasbey (se disponibile)
    2) matplotlib tab20
    """
    try:
        import colorcet as cc
        palette = [mpl.colors.to_rgb(c) for c in cc.glasbey]
        return {g: palette[g % len(palette)] for g in range(S)}
    except Exception:
        palette = [cm.tab20(i % 20) for i in range(S)]
        return {g: palette[g % len(palette)] for g in range(S)}

# ---------- timing semplice ----------
class Timer:
    def __init__(self): self.t0 = {}
    def start(self, key): self.t0[key] = time.perf_counter()
    def stop(self, key):  return time.perf_counter() - self.t0[key]

# ---------- interfacce tra gruppi ----------
def compute_group_interfaces(G: nx.Graph, part_of: dict[int,int], S: int):
    """
    Ritorna:
      W[g,h] = # contatti cella–cella tra gruppo g e h (simmetrica; diag=0)
      pair_edges[(g,h)] = lista delle coppie (u,v) con u∈g, v∈h
    """
    W = np.zeros((S, S), dtype=int)
    pair_edges = defaultdict(list)
    for u, v in G.edges:
        gu, gv = part_of[u], part_of[v]
        if gu == gv: 
            continue
        a, b = (gu, gv) if gu < gv else (gv, gu)
        W[a, b] += 1; W[b, a] += 1
        pair_edges[(a, b)].append((u, v) if gu < gv else (v, u))
    return W, pair_edges

def contact_stats(G: nx.Graph, part_of: dict[int,int], S: int):
    """
    Statistiche sintetiche sulle interfacce tra gruppi.
    """
    W, _ = compute_group_interfaces(G, part_of, S)
    # valori solo sopra-diagonale
    mask = np.triu(np.ones_like(W, dtype=bool), k=1)
    vals = W[mask]
    # se non ci sono interfacce, evita nan
    if vals.size == 0:
        return {
            "interfaces_total": 0,
            "interfaces_mean": 0.0,
            "interfaces_min": 0,
            "interfaces_max": 0,
            "interfaces_num_pairs": 0,
            "num_single_contact_pairs": 0
        }
    return {
        "interfaces_total": int(vals.sum()),              # somma contatti tra tutti i pair g–h
        "interfaces_mean": float(vals.mean()),            # media contatti per pair
        "interfaces_min": int(vals.min()),
        "interfaces_max": int(vals.max()),
        "interfaces_num_pairs": int((vals > 0).sum()),    # quante coppie di gruppi sono adiacenti
        "num_single_contact_pairs": int((vals == 1).sum())# interfacce “deboli”: 1 solo contatto
    }

# ---------- gruppi spezzati ----------
def split_groups(G: nx.Graph, part_of: dict[int,int], S: int):
    """
    Restituisce elenco dei gruppi non contigui (numero componenti > 1)
    e una mappa g -> numero componenti.
    """
    split = []
    comp_count = {}
    for g in range(S):
        members = [u for u, gg in part_of.items() if gg == g]
        if not members:
            comp_count[g] = 0
            continue
        H = G.subgraph(members)
        ncomp = nx.number_connected_components(H)
        comp_count[g] = ncomp
        if ncomp > 1:
            split.append(g)
    return split, comp_count


# -------------------------------------------------------------
# I/O e grafi di contatto
# -------------------------------------------------------------
def load_centres(csv_path: Path) -> np.ndarray:
    """x,y CSV ▸ ndarray (N,2)"""
    return np.loadtxt(csv_path, delimiter=",", skiprows=1)


def build_contact_graph(centres: np.ndarray, R: float, EPS: float) -> nx.Graph:
    """
    Edge ↔ centres closer than 2R+EPS.
    (GRAFO NON PESATO: nessun attributo 'weight' sugli archi)
    """
    N = centres.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(N))

    thr = 2 * R + EPS
    for i in range(N):
        for j in range(i + 1, N):
            d = float(np.linalg.norm(centres[i] - centres[j]))
            if d <= thr:
                G.add_edge(i, j) 
    return G




def plot_contact_graph_unweighted(centres, R=9.0, EPS=0.2,
                                  edge_lw=1.0, edge_color="0.4",
                                  node_ec="k", node_fc="white", node_lw=0.6,
                                  figsize=(7,7), title=None):
    centres = np.asarray(centres, float)
    G = build_contact_graph(centres, R, EPS)

    fig, ax = plt.subplots(figsize=figsize)

    # archi (tutti uguali)
    for u, v in G.edges:
        x1, y1 = centres[u]
        x2, y2 = centres[v]
        ax.plot([x1, x2], [y1, y2], '-', lw=edge_lw, color=edge_color, alpha=0.9)

    # nodi
    for (x, y) in centres:
        ax.add_patch(Circle((x, y), R, facecolor=node_fc, edgecolor=node_ec, lw=node_lw))

    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title or f"Grafo di contatto (non pesato) – soglia {2*R+EPS:.1f} mm")
    plt.tight_layout()
    return fig, ax


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


def plot_groups(poly, centres, part_of, S,
                group_color=None,
                show_labels=False,          # << nuovo: niente numeri per default
                title=None):                 # << nuovo: nessun titolo per default
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(*poly.exterior.xy, color="k", lw=2)

    # colori gruppo: se non passati, usa la palette condivisa
    if group_color is None:
        group_color = build_series_palette(S)

    # celle colorate per gruppo
    for idx, (x, y) in enumerate(centres):
        gid = part_of[idx] if isinstance(part_of, dict) else part_of[idx]
        ax.add_patch(plt.Circle((x, y), R, facecolor=group_color[gid], edgecolor="k", lw=0.4))

    # etichette opzionali (OFF per default)
    if show_labels:
        for g in range(S):
            members = [i for i in (part_of.keys() if isinstance(part_of, dict) else range(len(part_of)))]
            members = [i for i in members if (part_of[i] if isinstance(part_of, dict) else part_of[i]) == g]
            if members:
                cx, cy = centres[members].mean(axis=0)
                ax.text(cx, cy, str(g),
                        ha="center", va="center", fontsize=7, color="black", weight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.5))

    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title)    # di default None → nessun titolo
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
def drop_periphery_iterative(centres, N_keep,R,EPS):
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
    G = build_contact_graph(centres, R, EPS)

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
    Versione NON pesata: tutti gli archi equivalenti per METIS.
    """
    n = len(groups)
    S_here = n // P
    if S_here == 1:
        gid = rb_exact_partition.gid_counter
        rb_exact_partition.gid_counter += 1
        return {v: gid for v in groups}

    S_left  = S_here // 2
    S_right = S_here - S_left
    target_left  = S_left  * P
    target_right = S_right * P

    # CSR del sottografo indotto su 'groups'
    idx = {v: i for i, v in enumerate(groups)}
    xadj, adjncy = [0], []
    for v in groups:
        for nb in G[v]:
            if nb in idx:
                adjncy.append(idx[nb])
        xadj.append(len(adjncy))

    # Pesi vertice unitari (facoltativi)
    vweights = [1] * n
    # Target di bilanciamento tra le due parti (in frazione di nodi)
    tpwgts = [target_left / float(n), target_right / float(n)]

    # Bisezione 2-way NON pesata
    # NB: pymetis.part_graph tipicamente supporta recursive=True.
    #     eweights viene semplicemente OMMESSO.
    _, parts2 = pymetis.part_graph(
        2,
        xadj=xadj,
        adjncy=adjncy,
        vweights=vweights,
        tpwgts=tpwgts,
        recursive=True
    )

    left  = [groups[i] for i, p in enumerate(parts2) if p == 0]
    right = [groups[i] for i, p in enumerate(parts2) if p == 1]

    # Micro-correzioni ±1 dal target (come avevi già)
    if len(left) != target_left:
        need = target_left - len(left)
        if need > 0:
            border_candidates = [v for v in right if any(u in left  for u in G[v])]
            move = border_candidates[:need] if len(border_candidates) >= need else right[:need]
            left += move
            right = [v for v in right if v not in move]
        elif need < 0:
            need = -need
            border_candidates = [v for v in left  if any(u in right for u in G[v])]
            move = border_candidates[:need] if len(border_candidates) >= need else left[:need]
            right += move
            left = [v for v in left if v not in move]

    L = rb_exact_partition(G, left,  P)
    R = rb_exact_partition(G, right, P)
    return {**L, **R}


# contatore statico per nuovi group id
rb_exact_partition.gid_counter = 0