#!/usr/bin/env python3
"""
Stage 1 – contact-graph construction + coarse k-way grouping.
"""

import sys
import math
import numpy as np
import networkx as nx               # pip install networkx
from pathlib import Path
import matplotlib as mpl
import matplotlib.cm as cm        # colour maps
import matplotlib.pyplot as plt
import heapq, random
from collections import defaultdict

import numpy as np, networkx as nx, pymetis
import circle_packing as cp

R   = 9.0          # mm cell radius
EPS = 0.2          # slack in adjacency threshold
S =42
P = 10

# -------------------------------------------------------------
# Configurations to try for test purposes
# 43S11P
# 29S16P
# 59S8P
# 18S18P
# 54S6P
# 10S30P
# 78S4P
# 8S8P
# 59S8P
# 34S14P
# -------------------------------------------------------------




# -------------------------------------------------------------
# Visualise the k-way grouping
# -------------------------------------------------------------


def plot_groups(poly, centres, part_of, S, group_color=None, title="k-way grouping"):
    fig, ax = plt.subplots(figsize=(7,7))
    # boundary
    ax.plot(*poly.exterior.xy, color='k', lw=2)

    # if caller did not compute colors, fall back to palette only
    if group_color is None:
        palette = make_big_palette(S)
        group_color = {g: palette[g % len(palette)] for g in range(S)}

    # draw circles
    for idx, (x, y) in enumerate(centres):
        gid = part_of[idx] if isinstance(part_of, dict) else part_of[idx]
        col = group_color[gid]
        ax.add_patch(plt.Circle((x, y), R, facecolor=col, edgecolor='k', lw=0.4))

    # optional: put group labels at the group centroid
    for g in range(S):
        members = [i for i,p in part_of.items()] if isinstance(part_of, dict) else list(range(len(part_of)))
        members = [i for i in members if (part_of[i] if isinstance(part_of, dict) else part_of[i]) == g]
        if members:
            cx, cy = centres[members].mean(axis=0)
            ax.text(cx, cy, str(g), ha='center', va='center',
                    fontsize=7, color='black', weight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.5))

    ax.set_aspect('equal')
    ax.set_title(f"{title} (S={S})")
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def make_big_palette(n: int, seed: int|None = None):
    # 1) best: colorcet glasbey (if installed)
    try:
        import colorcet as cc  # pip install colorcet
        base = list(cc.glasbey)   # ~256 very distinct colors
        if n <= len(base):
            return [mpl.colors.to_rgb(c) for c in base[:n]]
        else:
            # fallback to HSV if you need more
            pass
    except Exception:
        pass

    # 2) very good: seaborn
    try:
        import seaborn as sns  # pip install seaborn
        return sns.color_palette("husl", n)  # or: "hls"
    except Exception:
        pass

    # 3) fallback: HSV evenly spaced
    rng = np.random.default_rng(seed)
    hues = np.linspace(0, 1, n, endpoint=False)
    # fixed saturation/value to ensure good contrast
    return [mpl.colors.hsv_to_rgb((h, 0.65, 0.95)) for h in hues]

from networkx.algorithms.coloring import greedy_color

def color_groups_avoiding_adjacent(G: nx.Graph, part_of: dict[int,int], S: int,
                                   palette=None, strategy="saturation_largest_first"):
    """
    Returns: dict {group_id -> RGB tuple} such that adjacent groups have different colors.
    """
    H = build_group_adjacency_graph(G, part_of, S)
    # Greedy coloring of the *group* graph
    col_idx = greedy_color(H, strategy=strategy)
    ncols   = max(col_idx.values()) + 1

    # If no palette provided, build one with at least ncols colors
    if palette is None or len(palette) < ncols:
        palette = make_big_palette(ncols)

    # Map each group to its color
    group_color = {g: palette[col_idx[g]] for g in range(S)}
    return group_color


def build_group_adjacency_graph(G: nx.Graph, part_of: dict[int,int], S: int) -> nx.Graph:
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

# ----------------------------------------------------------------------
def load_centres(csv_path: Path) -> np.ndarray:
    """x,y CSV ▸ ndarray (N,2)"""
    return np.loadtxt(csv_path, delimiter=',', skiprows=1)

def build_contact_graph(centres: np.ndarray) -> nx.Graph:
    """
    Edge ↔ centres closer than 2R+EPS.
    Each edge carries a weight = 1/‖Δx‖  (or 1/‖Δx‖²).
    """
    N   = centres.shape[0]
    G   = nx.Graph()
    G.add_nodes_from(range(N))

    thr = 2*R + EPS
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(centres[i]-centres[j])
            if d <= thr:
                w = 1.0 / d                # or 1.0 / d**2
                G.add_edge(i, j, weight=w)
    return G


# ----------------------------------------------------------------------
# Balanced k-way partition (sizes differ by at most 1)
# ----------------------------------------------------------------------
def greedy_balanced_partition(G: nx.Graph, k: int) -> dict[int, int]:
    """
    Return a dict node_id  ->  group_id   with
    |group_i| ∈ { floor(N/k), ceil(N/k) }.
    Works component-by-component exactly like the earlier routine,
    but honours a per-group capacity.
    """
    N        = G.number_of_nodes()
    base     = N // k
    extra    = N %  k                       # the first `extra` groups get one more
    capacity = [base+1]*extra + [base]*(k-extra)

    # (re)number groups 0…k-1 in the order their seeds are chosen
    capacities = {g:capacity[g] for g in range(k)}

    # ------------ choose one seed per group ---------------------------
    #   • largest component first
    #   • farthest-point heuristic inside each component
    seeds   = []
    comps   = list(nx.connected_components(G))
    comps.sort(key=len, reverse=True)

    for CC in comps:
        sub   = G.subgraph(CC)
        nodes = list(sub)

        # if some groups are still without a seed, pick inside *this* CC
        while len(seeds) < k and nodes:
            if not seeds:                         # first seed = first node
                seeds.append(nodes[0])
            else:                                 # farthest from existing seeds
                dist = {v: min(nx.shortest_path_length(sub, v, s)
                                for s in seeds if s in sub)
                        for v in nodes}
                seeds.append(max(dist, key=dist.get))
                nodes.remove(seeds[-1])

    # any component that still has un-seeded nodes will be handled later

    # ------------ multi-source BFS with capacities --------------------
    assign     = {s:i for i,s in enumerate(seeds)}   # seed→group
    frontiers  = {s:{s} for s in seeds}              # current BFS wave fronts

    todo = set(G) - set(seeds)
    while todo:
        progress = False
        for s in list(frontiers.keys()):
            gid = assign[s]
            if capacities[gid] == 0:                 # this group full?
                continue
            this_front, new_front = frontiers[s], set()
            for v in this_front:
                for nb in G[v]:
                    if nb in todo:
                        assign[nb]   = gid
                        capacities[gid] -= 1
                        todo.remove(nb)
                        new_front.add(nb)
                        progress = True
                        if capacities[gid] == 0:     # full now – stop growing
                            break
                if capacities[gid] == 0:
                    break
            frontiers[s] = new_front
        if not progress:       # disconnected leftovers → give them free slots
            v = todo.pop()
            # first group with remaining capacity
            gid = next(g for g,c in capacities.items() if c>0)
            assign[v] = gid
            capacities[gid] -= 1
            frontiers[v] = {v}

    return assign

# ----------------------------------------------------------------------


# ----------------------------------------------------------------
# 2. call PyMetis for balanced k-way split
# ----------------------------------------------------------------
# 2.  call PyMetis for balanced k-way split  (robust to v2023-2025)
# ----------------------------------------------------------------
def metis_k_partition(G: nx.Graph, k: int) -> list[int]:
    """Return a list of length |V| with part-id 0…k-1 (version-proof)."""

    # --- build CSR arrays ----------------------------------------
    xadj, adjncy, eweights = [0], [], []
    for v in G.nodes:
        for nb in G.adj[v]:
            adjncy.append(nb)
            eweights.append(int(G[v][nb]["weight"] * 10_000))
        xadj.append(len(adjncy))

    n_cuts, parts = pymetis.part_graph(
        k,
        xadj=xadj,
        adjncy=adjncy,
        eweights=eweights,
        recursive=False,
        contiguous=True,        # fine for both versions
    )

    # ----- cope with name change --------------------------------
    return list(parts)                        # ensure plain list
# ----------------------------------------------------------------------

# --------------------------------------------------------------------
# Heal a METIS assignment so that each group is connected **and**
# has exactly the required size P.
# --------------------------------------------------------------------
# --------------------------------------------------------------------
def metis_connected_parts(G: nx.Graph, S: int, P: int,
                          weight_key="weight",
                          rng=random.Random(1234),
                          max_outer=8):
    """
    Balanced & connected k-way partition built on top of METIS.

    Returns
    -------
    part_of : {vertex_id : group_id}
    """
    N = G.number_of_nodes()
    if S * P != N:
        raise ValueError(f"S*P must equal |V|  ({S}×{P} ≠ {N})")

    # ---- METIS seed ---------------------------------------------------
    base_parts = metis_k_partition(G, S)
    part_of = {v: base_parts[v] for v in range(N)}

    # helper lambdas ----------------------------------------------------
    vertices  = lambda g: [v for v,p in part_of.items() if p == g]
    size      = lambda g: len(vertices(g))
    connected = lambda g: nx.is_connected(G.subgraph(vertices(g)))

    # ---------- 2.  multi-pass repair loop -----------------------------
    for outer in range(max_outer):
        changed = False

        # (a) fix connectivity by merging islands -----------------------
        for v in range(N):
            g = part_of[v]
            if any(part_of[n] == g for n in G[v]):
                continue                       # v not an island
            # choose neighbour group with smallest extra edge cut
            cand = [(sum(1 for n in G[v] if part_of[n] != gg), gg)
                    for gg in {part_of[n] for n in G[v]}]
            if cand:
                _, g2 = min(cand)
                part_of[v] = g2
                changed = True

        # (b) balance sizes  (while keeping connectivity if possible) ---
        excess  = [g for g in range(S) if size(g) > P]
        deficit = [g for g in range(S) if size(g) < P]

        for g_def in deficit:
            need = P - size(g_def)
            while need:
                # candidate donors = neighbours of current part
                border = [(v, part_of[v]) for v in G
                          if part_of[v] in excess
                          and any(part_of[n] == g_def for n in G[v])]
                if not border:
                    break
                # pick vertex with smallest cut increase
                best_v, g_src = min(border,
                                   key=lambda t: sum(1 for n in G[t[0]]
                                                     if part_of[n] not in
                                                        (t[1], g_def)))
                # move it
                part_of[best_v] = g_def
                changed = True
                need -= 1

                # update lists
                if size(g_src) == P:
                    excess.remove(g_src)
                if need == 0:
                    break

        # -------------- fallback micro-moves if still unbalanced -------
        if any(size(g) != P for g in range(S)):
            # pick first surplus group + first deficit group
            g_sur = next(g for g in range(S) if size(g) > P)
            g_def = next(g for g in range(S) if size(g) < P)

            # choose surplus vertex v that is *closest* to deficit part
            cand_v = [v for v in vertices(g_sur)
                      if any(part_of[n] == g_def for n in G[v])]
            if not cand_v:
                # if no direct neighbour, just take a random surplus vertex
                cand_v = vertices(g_sur)
            v = min(cand_v, key=lambda v: G.degree(v))   # low degree better
            part_of[v] = g_def        # move v to deficit

            # donor may now be split; patch by pulling back one neighbour
            donors_nb = [n for n in G[v] if part_of[n] == g_sur]
            if donors_nb:
                # move the neighbour with highest connectivity back
                w = max(donors_nb, key=lambda n: G.degree(n))
                part_of[w] = g_sur    # restores connection in most cases

            changed = True    # we did a move → iterate again

        # check convergence ---------------------------------------------
        if all(size(g) == P and connected(g) for g in range(S)):
            break
        if not changed:
            # No further progress ⇒ relax connectivity requirement for
            # a single problematic group by one vertex and retry.
            for g in range(S):
                if size(g) > P:
                    # drop a peripheral vertex from the surplus group
                    v = max(vertices(g), key=lambda v: G.degree(v, weight='weight'))
                    neigh_g = max((part_of[n] for n in G[v] if size(part_of[n])<P),
                                  default=None)
                    if neigh_g is not None:
                        part_of[v] = neigh_g
                        break

    # final sizes must match
    for g in range(S):
        assert size(g) == P, f"group {g} size {size(g)} != P"

    return part_of
# --------------------------------------------------------------------
# --------------------------------------------------------------------

# ----------------------------------------------------------------------

def merge_islands(G, parts, k):
    part_of = parts[:]                 # copy
    sizes   = [parts.count(i) for i in range(k)]

    changed = True
    while changed:
        changed=False
        for v in G.nodes:
            gv = part_of[v]
            # if v is an island (no neighbour in same group)
            if all(part_of[n]!=gv for n in G.adj[v]):
                # move it to the majority group around it
                counts = {}
                for n in G.adj[v]:
                    gn = part_of[n]
                    if sizes[gn] < sizes[gv]:   # keep balance
                        counts[gn] = counts.get(gn,0)+1
                if counts:
                    g_new = max(counts, key=counts.get)
                    part_of[v] = g_new
                    sizes[gv]-=1; sizes[g_new]+=1
                    changed=True
    return part_of
# ----------------------------------------------------------------------

def select_cells_for_SP(centres, S, P, strategy="random"):
    """
    Return a boolean mask of length N saying which cells to KEEP to obtain
    exactly S*P cells.  Simple strategies:
      • 'random'  : uniform random subset
      • 'densest' : keep cells with highest degree in the contact graph
    """
    N_keep = S*P
    if N_keep > len(centres):
        raise ValueError(f"Need {N_keep} cells but only {len(centres)} exist")

    if strategy == "random":
        idx_keep = np.random.choice(len(centres), N_keep, replace=False)

    elif strategy == "densest":
        G = build_contact_graph(centres)
        deg = np.array([G.degree(v) for v in range(len(centres))])
        idx_keep = np.argsort(deg)[-N_keep:]          # top-degree nodes

    else:
        raise ValueError("unknown selection strategy")

    mask = np.zeros(len(centres), bool)
    mask[idx_keep] = True
    return mask
# ----------------------------------------------------------------------
def drop_periphery_iterative(centres, N_keep):
    """
    Iteratively peel cells from the boundary until exactly N_keep remain.
    – removes the lowest-degree boundary vertex at each step
    – recomputes degrees every iteration
    – ensures the remaining graph stays connected
    """
    N   = len(centres)
    if N_keep >= N:
        return np.ones(N, bool)          # nothing to drop

    keep = np.ones(N, bool)              # boolean mask we will update
    G    = build_contact_graph(centres)  # full graph once; we’ll take subgraphs

    cx, cy = centres.mean(axis=0)
    d2_cent = np.square(centres[:,0]-cx) + np.square(centres[:,1]-cy)

    to_drop = N - N_keep
    for _ in range(to_drop):
        sub = G.subgraph(np.where(keep)[0])

        # current degrees in the *remaining* graph
        deg = {v: sub.degree(v) for v in sub}

        # candidate vertices: minimal degree
        min_deg = min(deg.values())
        cand    = [v for v,d in deg.items() if d == min_deg]

        # tie-breaker: farthest-from-centroid first
        cand.sort(key=lambda v: d2_cent[v], reverse=True)

        # pick the first candidate whose removal keeps the graph connected
        for v in cand:
            keep[v] = False
            if nx.is_connected(sub.subgraph([u for u in sub if keep[u]])):
                break      # accepted
            keep[v] = True # undo → try next candidate

    return keep
# ----------------------------------------------------------------------

def geodesic_capacity_partition(G: nx.Graph, S: int, P: int,
                                weight_key="weight",
                                rng=random.Random(1234)):
    """
    Return `parts` – list of length |V| with group‐id 0..S-1.

    Guarantees:
        • each part size   == P
        • each part subgraph is connected (multi-source Dijkstra grow)
    """
    N = G.number_of_nodes()
    if S * P != N:
        raise ValueError(f"S*P must equal |V|  ({S}×{P} ≠ {N})")

    # ---------- 1. choose S seeds with farthest-point on graph distance -----
    seeds = []
    # pick a random node as the first seed
    seeds.append(rng.choice(list(G.nodes)))

    while len(seeds) < S:
        # distance of every vertex to the closest already-chosen seed
        min_d = {}
        for v in G.nodes:
            # single-source shortest-path lengths from each existing seed
            d_to_seeds = [
                nx.single_source_shortest_path_length(G, s).get(v, 1e9)
                for s in seeds
            ]
            min_d[v] = min(d_to_seeds)
        farthest = max(min_d, key=min_d.get)
        seeds.append(farthest)

    # ------------------------------------------------------------------
    # 2. multi-source BFS growth honouring per-group capacity
    # ------------------------------------------------------------------
    capacity = {g: P for g in range(S)}  
    assign     = {}                    # vertex -> group id
    frontiers  = {}                    # current wave-front per seed
    seed2gid   = {}                    # seed vertex -> its group id

    for gid, seed in enumerate(seeds):
        assign[seed]   = gid
        frontiers[seed] = {seed}
        seed2gid[seed] = gid
        capacity[gid] -= 1             # reserve one slot for the seed

    todo = set(G.nodes) - set(seeds)
    while todo:
        progressed = False
        # iterate over a *copy* of seeds because we may append new wave fronts
        for seed in list(frontiers.keys()):
            gid        = seed2gid[seed]          # <-- get group id
            if capacity[gid] == 0:
                continue                         # this group already full
            new_ring = set()
            for v in frontiers[seed]:
                for nb in G[v]:
                    if nb not in assign:
                        assign[nb] = gid
                        capacity[gid] -= 1
                        todo.discard(nb)
                        new_ring.add(nb)
                        progressed = True
                        if capacity[gid] == 0:   # group just filled
                            break
                if capacity[gid] == 0:
                    break
            if new_ring:
                frontiers[seed] = new_ring
        if not progressed:                       # disconnected leftovers
            v  = todo.pop()
            gid = min((g for g,c in capacity.items() if c>0),
                      key=lambda g: capacity[g])  # pick any group w/ room
            assign[v] = gid
            capacity[gid] -= 1
            frontiers[v]  = {v}                   # start a new frontier
            seed2gid[v]   = gid                   # and remember its gid

    return assign

# ----------------------------------------------------------------------
def rb_exact_partition(G: nx.Graph, groups: list[int], P: int) -> dict[int,int]:
    """
    Recursively bisect the vertex set 'groups' into len(groups)/P leaves of size P.
    Returns part_of (vertex -> group_id) with exact sizes and high contiguity.
    """
    # groups is the list of vertices of a current subproblem
    n = len(groups)
    S_here = n // P
    if S_here == 1:
        # one leaf ⇒ assign all to the same group id
        gid = rb_exact_partition.gid_counter
        rb_exact_partition.gid_counter += 1
        return {v: gid for v in groups}

    # choose how many parts go left/right (balance by size in multiples of P)
    S_left = S_here // 2
    S_right = S_here - S_left
    target_left = S_left * P
    target_right = S_right * P

    # Build local CSR for the induced subgraph
    idx = {v:i for i,v in enumerate(groups)}
    xadj, adjncy, eweights = [0], [], []
    for v in groups:
        for nb, data in G[v].items():
            if nb in idx:
                adjncy.append(idx[nb])
                w = data.get('weight', 1.0)
                eweights.append(int(w*10000))
        xadj.append(len(adjncy))

    vweights = [1]*n
    tpwgts   = [target_left/float(n), target_right/float(n)]

    # 2-way bisection with targets
    _, parts2 = pymetis.part_graph(
        2, xadj=xadj, adjncy=adjncy, eweights=eweights,
        vweights=vweights, tpwgts=tpwgts, recursive=True, contiguous=True
    )
    left  = [groups[i] for i,p in enumerate(parts2) if p==0]
    right = [groups[i] for i,p in enumerate(parts2) if p==1]

    # Minor corrections if one side is ±1 from target due to rounding
    # (usually unnecessary when vweights=1 and tpwgts are multiples of 1/n)
    if len(left) != target_left:
        # move boundary vertices to correct exact sizes (1-2 moves)
        # choose vertices adjacent across the cut to minimise damage
        need = target_left - len(left)
        if need > 0:
            # move 'need' vertices from right to left
            border_candidates = [v for v in right if any(u in left for u in G[v])]
            move = border_candidates[:need] if len(border_candidates)>=need else right[:need]
            left += move
            right = [v for v in right if v not in move]
        elif need < 0:
            need = -need
            border_candidates = [v for v in left if any(u in right for u in G[v])]
            move = border_candidates[:need] if len(border_candidates)>=need else left[:need]
            right += move
            left = [v for v in left if v not in move]

    # recurse
    L = rb_exact_partition(G, left, P)
    R = rb_exact_partition(G, right, P)
    return {**L, **R}

# static counter for new group ids
rb_exact_partition.gid_counter = 0


# ----------------------------------------------------------------------
def main():
    # ----- LOAD ---------------------------------------------------------
    poly, _ = cp.load_boundary(Path("new_version/input.json"))
    centres_all = load_centres(Path("new_version/out.csv"))
    N_all = len(centres_all)

    # ----- SELECT EXACTLY S*P CELLS ------------------------------------
    keep_mask = drop_periphery_iterative(centres_all, S*P)
    centres   = centres_all[keep_mask]        # array of shape (S*P,2)
    print(f"Kept {len(centres)} cells, discarded {N_all-len(centres)} extras")

    # -------------------------------------------------------------------
    G   = build_contact_graph(centres)        # only those kept cells
    #parts = metis_k_partition(G, S)           # S balanced parts
    #parts = geodesic_capacity_partition(G, S, P)
    #part_of = metis_connected_parts(G, S, P)   # healing on top of METIS

    # parts is a list (len = S*P).  Convert to dict {node_id: group_id}
    #part_dict = {i:part_of[i] for i in range(len(part_of))}
    rb_exact_partition.gid_counter = 0
    part_of = rb_exact_partition(G, list(G.nodes()), P)

    group_color = color_groups_avoiding_adjacent(G, part_of, S)


    # ----- Sanity check -------------------------------------------------
    #group_sizes = [parts.count(g) for g in range(S)]
    #group_sizes = [list(part_of.values()).count(g) for g in range(S)]
    #print("Group sizes:", group_sizes)        # should all be P

    # ----- PLOT ---------------------------------------------------------
    plot_groups(poly, centres, part_of, S, group_color=group_color, title=f"S={S}, P={P} (adjacency-colored)")
# ----------------------------------------------------------------------

if __name__ == "__main__":
    main()