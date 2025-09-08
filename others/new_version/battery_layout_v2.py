
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Battery pack layout — v2 (wavefront serpentine)

Caratteristiche principali:
- Gruppi connessi per adiacenza (distanza ~ 2R, tolleranza robusta).
- Crescita "a serpentina" che abbraccia il gruppo precedente (wavefront).
- Target dei collegamenti tra gruppi parte da 2*P e viene ridotto in automatico
  solo se necessario per evitare gruppi spezzati o dead-end.
- Limite di sicurezza: max 4 collegamenti per cella.
- Plot dei gruppi (colori), celle non usate (bianco) e collegamenti (tratti neri).
- Statistiche dei collegamenti tra gruppi consecutivi (min, max, media).

Uso:
  python battery_layout_v2.py --csv out.csv --S 31 --P 7 --radius 9 --save fig.png
"""

import argparse
from collections import deque
from typing import List, Set, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA


def robust_adjacency(coords: np.ndarray, R: float, tol: float = 0.8):
    """Celle adiacenti se distanza in [2R - tol, 2R + tol]."""
    N = len(coords)
    adj = [set() for _ in range(N)]
    r = 2 * R + tol
    tree = cKDTree(coords)
    lists = tree.query_ball_tree(tree, r=r)
    for i in range(N):
        for j in lists[i]:
            if j <= i:
                continue
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if abs(d - 2 * R) <= tol:
                adj[i].add(j)
                adj[j].add(i)
    return adj


def color_for_group(i: int) -> str:
    palette = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    return palette[i % len(palette)]


def grow_group_wavefront(coords: np.ndarray, adj, R: float, g_id: int,
                         assigned: np.ndarray, ext_deg: np.ndarray, cross_edges: list,
                         seed: int, prev_group: Set[int], bias_vec: np.ndarray,
                         desired_links: int):
    """Cresce un gruppo connesso abbracciando prev_group.
    Ritorna (group_set, links_made)."""
    PENDING_LIMIT = 2000  # guardia contro loop in casi patologici

    def links_possible(v, prev_group):
        c = 0
        for u in adj[v]:
            if u in prev_group and ext_deg[u] < 4 and ext_deg[v] < 4:
                c += 1
        return c

    group: Set[int] = set([seed])
    assigned[seed] = g_id
    made_links = 0

    # prova collegamenti del seed
    if prev_group:
        neigh_prev = [u for u in adj[seed] if u in prev_group]
        neigh_prev.sort(key=lambda u: ext_deg[u])
        for u in neigh_prev:
            if ext_deg[u] < 4 and ext_deg[seed] < 4 and made_links < desired_links:
                cross_edges.append((u, seed))
                ext_deg[u] += 1
                ext_deg[seed] += 1
                made_links += 1

    def frontier():
        cand = set()
        for u in group:
            for v in adj[u]:
                if assigned[v] == -1:
                    cand.add(v)
        return cand

    def centroid():
        return coords[list(group)].mean(axis=0)

    steps = 0
    while steps < PENDING_LIMIT and len(group) < TARGET_P:
        steps += 1
        cands = list(frontier())
        if not cands:
            # espandi alla cella non assegnata più vicina via BFS per mantenere connettività
            q = deque(list(group))
            seen = set(group)
            parent = {}
            target = None
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if v in seen:
                        continue
                    seen.add(v)
                    parent[v] = u
                    if assigned[v] == -1:
                        target = v
                        q.clear()
                        break
                    q.append(v)
            if target is None:
                break
            # collega al gruppo con il primo step della path
            v = target
            while parent.get(v) not in group and v in parent:
                v = parent[v]
            # ora v è adiacente al gruppo
            group.add(v)
            assigned[v] = g_id
            continue

        Gc = centroid()
        best = None
        best_score = -1e18
        for v in cands:
            vec = coords[v] - Gc
            if np.linalg.norm(vec) > 0:
                align = float(np.dot(vec / (np.linalg.norm(vec) + 1e-9), bias_vec))
            else:
                align = 0.0
            touch_prev = links_possible(v, prev_group)
            border = max(0, 6 - len(adj[v]))
            score = 6.0 * touch_prev + 1.0 * align + 0.15 * border
            if score > best_score:
                best_score = score
                best = v

        v = best
        group.add(v)
        assigned[v] = g_id

        if prev_group:
            neigh_prev = [u for u in adj[v] if u in prev_group]
            neigh_prev.sort(key=lambda u: ext_deg[u])
            for u in neigh_prev:
                if ext_deg[u] < 4 and ext_deg[v] < 4 and made_links < desired_links:
                    cross_edges.append((u, v))
                    ext_deg[u] += 1
                    ext_deg[v] += 1
                    made_links += 1

    return group, made_links


def build_layout(coords: np.ndarray, S: int, P: int, R: float, tol: float = 0.8):
    N = len(coords)
    if S * P > N:
        raise SystemExit(f"ERRORE: celle richieste ({S*P}) > disponibili ({N}).")

    adj = robust_adjacency(coords, R, tol=tol)

    # inizializza
    assigned = np.full(N, -1, dtype=int)
    ext_deg = np.zeros(N, dtype=int)    # gradi esterni (max 4)
    cross_edges: List[Tuple[int, int]] = []

    # assi PCA e direzione a serpentina
    pca = PCA(n_components=2).fit(coords)
    d1 = pca.components_[0]
    d1 = d1 / (np.linalg.norm(d1) + 1e-9)

    # seed iniziale: alto-sinistra
    start_idx = int(np.lexsort((coords[:, 0], coords[:, 1]))[0])

    # primo gruppo
    global TARGET_P
    TARGET_P = P
    g0, _ = grow_group_wavefront(coords, adj, R, 0, assigned, ext_deg, cross_edges,
                                 seed=start_idx, prev_group=set(), bias_vec=d1,
                                 desired_links=0)
    groups = [g0]

    # target collegamenti (rilassato se necessario)
    desired_links = 2 * P

    for gid in range(1, S):
        prev = groups[-1]
        # seed: adiacente al prev con più potenziali link
        seed = None
        best = (-1, None)
        for u in prev:
            for v in adj[u]:
                if assigned[v] == -1:
                    pot = sum((w in prev) and (ext_deg[w] < 4) for w in adj[v])
                    if pot > best[0]:
                        best = (pot, v)
        if best[1] is None:
            # fallback: non dovrebbe capitare spesso
            un = np.where(assigned == -1)[0]
            cen = coords[list(prev)].mean(axis=0)
            seed = int(un[np.argmin(np.linalg.norm(coords[un] - cen, axis=1))])
        else:
            seed = best[1]

        bias = d1 if (gid % 2 == 0) else -d1

        # prova con desired_links, se il gruppo non raggiunge P celle connesse rilassa e riprova
        dlinks = desired_links
        ok = False
        for _attempt in range(6):
            ext_bak = ext_deg.copy()
            asg_bak = assigned.copy()
            ce_bak = list(cross_edges)
            g, made = grow_group_wavefront(coords, adj, R, gid, assigned, ext_deg, cross_edges,
                                           seed=seed, prev_group=prev, bias_vec=bias,
                                           desired_links=dlinks)
            if len(g) == P:
                groups.append(g)
                ok = True
                break
            # rollback e rilassa
            ext_deg[:] = ext_bak
            assigned[:] = asg_bak
            cross_edges[:] = ce_bak
            dlinks = max(0, dlinks - max(1, P // 2))
        if not ok:
            # accetta comunque (taglio a P se serve)
            g, _ = grow_group_wavefront(coords, adj, R, gid, assigned, ext_deg, cross_edges,
                                        seed=seed, prev_group=prev, bias_vec=bias,
                                        desired_links=0)
            groups.append(set(list(g)[:P]))

    # collegamenti tra gruppi consecutivi
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


def plot_layout(coords: np.ndarray, groups, cross_edges, assigned, R: float,
                links_between, title: str, save: str = None):
    fig, ax = plt.subplots(figsize=(11, 7), dpi=140)

    for gid, g in enumerate(groups):
        for idx in g:
            x, y = coords[idx]
            ax.add_patch(Circle((x, y), R, facecolor=color_for_group(gid),
                                edgecolor='black', linewidth=0.5))

    for idx in range(len(coords)):
        if assigned[idx] == -1:
            x, y = coords[idx]
            ax.add_patch(Circle((x, y), R, facecolor='white',
                                edgecolor='black', linewidth=0.35, alpha=0.6))

    for (u, v) in cross_edges:
        x1, y1 = coords[u]
        x2, y2 = coords[v]
        ax.plot([x1, x2], [y1, y2], color='black', linewidth=1.0)

    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])

    if links_between:
        subtitle = f" — collegamenti medi: {float(np.mean(links_between)):.1f}  (min={min(links_between)}, max={max(links_between)})"
    else:
        subtitle = ""
    ax.set_title(title + subtitle)
    plt.tight_layout()
    if save:
        plt.savefig(save, bbox_inches="tight", dpi=160)
    plt.show()


def main():

    csv = "new_version/out.csv"
    radius = 9.0
    S = 10
    P = 7
    tol = 0.25


    df = pd.read_csv(csv)
    coords = df[['x', 'y']].to_numpy()

    df = pd.read_csv(csv)
    coords = df[['x', 'y']].to_numpy()


    groups, cross_edges, assigned, links_between = build_layout(coords, S, P, radius, tol=tol)

    if links_between:
        print("Collegamenti tra gruppi consecutivi:", ", ".join(map(str, links_between)))
        print(f"Media: {float(np.mean(links_between)):.2f}")


    plot_layout(coords, groups, cross_edges, assigned, radius, links_between,
                title=f"{S}S{P}P", save=None)


if __name__ == "__main__":
    main()
