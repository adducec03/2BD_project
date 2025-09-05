
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CP-SAT battery layout optimizer

- Input: CSV with columns x,y (centers); parameters S,P, R, tol
- Graph: undirected edges between cells whose center distance ≈ 2R (+/- tol)
- Variables:
    x[i,k] ∈ {0,1}  -> cell i assigned to group k
    r[i,k] ∈ {0,1}  -> cell i is the root of group k  (Σ_i r[i,k] = 1; r[i,k] ≤ x[i,k])
    f[i,j,k] ∈ [0, P] (Int) -> flow on directed arc i→j for group k (only for adjacency edges)
    z1[k,i,j] ∈ {0,1} -> link between group k (cell i) and group k+1 (cell j), with (i,j) an edge
    z2[k,i,j] ∈ {0,1} -> link between group k (cell j) and group k+1 (cell i) (symmetric direction)

- Constraints:
    1) each group size: Σ_i x[i,k] = P
    2) each cell used at most once: Σ_k x[i,k] ≤ 1
    3) roots: Σ_i r[i,k] = 1 ; r[i,k] ≤ x[i,k]
    4) flow capacity: f[i,j,k] ≤ (P-1) * x[i,k]  and  f[i,j,k] ≤ (P-1) * x[j,k]
    5) flow conservation for connectivity:
          Σ_u f[u,i,k] - Σ_v f[i,v,k] = x[i,k] - P * r[i,k]      for all i,k
       (non-root selected nodes have net inflow 1; root has net inflow - (P-1))
    6) link activation:
          z1[k,i,j] ≤ x[i,k] ,  z1[k,i,j] ≤ x[j,k+1]
          z2[k,i,j] ≤ x[j,k] ,  z2[k,i,j] ≤ x[i,k+1]
       L_k = Σ_{(i,j)∈E} (z1[k,i,j] + z2[k,i,j])  for k=1..S-1
    7) thermal degree: for each cell i,
          Σ_{k} Σ_{j: (i,j)∈E} (z1[k,i,j] + z2[k,j,i]) ≤ 4
- Objective:
      maximize  Σ_k L_k  -  α * Σ_k |L_k - L_target|
   where L_target = min(2*P, maximum possible), α small (e.g., 0.1).

Optional: 'band' guidance (serpentina) can be added by restricting the cells
admissibili per gruppo k alle bande oblique (non incluso per semplicità).

USO:
  pip install ortools matplotlib numpy pandas
  python battery_layout_cpsat.py --csv out.csv --S 25 --P 9 --radius 9 --tol 0.8 --time_limit 60 --save fig.png
"""

import argparse
from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from ortools.sat.python import cp_model

import numpy.linalg as npl


import numpy.linalg as npl

def preview_pair_snake(coords, E, P, start_corner="TL", title=None):
    import matplotlib.pyplot as plt
    order, row_idx, rows = build_pair_snake_path(coords, E, start_corner)
    rows_sorted = sorted(set(row_idx))
    y_map = {r:i for i,r in enumerate(rows_sorted)}
    xs = np.arange(len(order))
    ys = np.array([y_map[row_idx[i]] for i in order], float)

    import matplotlib.colors as mcolors
    pal = list(mcolors.TABLEAU_COLORS.values())
    cols = [pal[(k//P) % len(pal)] for k in range(len(order))]

    fig, ax = plt.subplots(figsize=(14,6), dpi=120)
    ax.scatter(xs, ys, s=18, c=cols)
    for g in range(len(order)//P + 1):
        ax.axvline(g*P - .5, lw=.6, color='k', alpha=.25)

    # frecce tra blocchi
    G = len(order)//P
    for g in range(G-1):
        a,b = g*P, (g+1)*P
        c,d = (g+1)*P, (g+2)*P
        x0,y0 = xs[a:b].mean(), ys[a:b].mean()
        x1,y1 = xs[c:d].mean(), ys[c:d].mean()
        ax.annotate("", xy=(x1,y1), xytext=(x0,y0),
                    arrowprops=dict(arrowstyle="->", lw=1.3, color="black"))
        ax.text((x0+x1)/2, (y0+y1)/2, f"{g+1}",
                ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.2", fc="white"))

    ax.set_yticks(range(len(rows_sorted)))
    ax.set_yticklabels([f"r{r}" for r in rows_sorted])
    ax.grid(axis="y", alpha=.3)
    ax.set_xlabel("indice lungo pair-snake")
    ax.set_title(title or f"Serpentina a coppie — P={P}, start={start_corner}")
    plt.tight_layout(); plt.show()

def build_pair_snake_path(coords: np.ndarray, E, start_corner: str = "TL"):
    """
    Ritorna un ordine 'pair-snake' che attraversa le celle intercalando
    due righe alla volta: A0,B0,A1,B1, ... poi passa alla coppia di righe successiva
    invertendo la direzione (serpentina).
    """
    # assi + righe/colonne come prima
    X = coords - coords.mean(axis=0, keepdims=True)
    _, _, Vt = npl.svd(X, full_matrices=False)
    e1 = Vt[0] / npl.norm(Vt[0])
    e2 = np.array([-e1[1], e1[0]])

    t_row = coords @ e2
    t_col = coords @ e1
    r0 = t_row.min()
    # assegna indice di riga
    # (se vuoi ancorarti agli archi come prima, puoi sostituire con la stima del pitch)
    pitch = estimate_row_pitch_from_edges(coords, E, e1, e2)
    row_idx = np.round((t_row - r0) / pitch).astype(int)

    # celle per riga, ordinate per colonna
    from collections import defaultdict
    by_row = defaultdict(list)
    for i, r in enumerate(row_idx):
        by_row[int(r)].append(i)
    for r in by_row:
        by_row[r].sort(key=lambda i: t_col[i])

    rows = sorted(by_row.keys())
    if start_corner in ("TR", "BR"):     # parto da destra in alto/basso?
        rows = rows[::-1]                # inverti verso verticale

    # serpentina per coppie di righe
    order = []
    left_to_right_first = start_corner in ("TL", "BL")
    for p in range(0, len(rows), 2):
        rA = rows[p]
        rB = rows[p+1] if p+1 < len(rows) else None
        A = by_row[rA][:]
        B = by_row[rB][:] if rB is not None else []

        go_l2r = (p//2) % 2 == 0         # alterna direzione coppia-coppia
        if not left_to_right_first:
            go_l2r = not go_l2r
        if not go_l2r:
            A = A[::-1]; B = B[::-1]

        # interleaving A,B: A0,B0,A1,B1,...
        i = j = 0
        merged = []
        while i < len(A) or j < len(B):
            if i < len(A):
                merged.append(A[i]); i += 1
            if j < len(B):
                merged.append(B[j]); j += 1

        order.extend(merged)

    return order, row_idx, rows

def pair_snake_allowed_sets(coords, E, S, P, start_corner="TL", window_slack=2):
    """
    Ogni gruppo k può scegliere celle solo entro la finestra
    [k*P - w, (k+1)*P - 1 + w] rispetto all'ordine pair-snake.
    Se la finestra non contiene almeno P celle, viene allargata automaticamente.
    """
    order, row_idx, _ = build_pair_snake_path(coords, E, start_corner=start_corner)
    N = len(coords)
    pos = np.empty(N, dtype=int)
    pos[order] = np.arange(N)

    allowed = []
    for k in range(S):
        a = k * P
        b = (k + 1) * P - 1
        w = window_slack
        while True:
            lo = max(0, a - w)
            hi = min(N - 1, b + w)
            idx = [i for i in range(N) if lo <= pos[i] <= hi]
            if len(idx) >= P or (lo == 0 and hi == N - 1):
                allowed.append(set(idx))
                break
            w += 1
    return allowed, order, row_idx

def snake_allowed_sets(coords, E, S, P, start_corner="TL", window_slack=2):
    """
    Restituisce una lista 'allowed[k]' (insiemi di indici di celle) tale che
    il gruppo k può scegliere celle solo entro la finestra
    [k*P - window_slack, (k+1)*P-1 + window_slack] lungo la serpentina.

    Se la finestra non contiene almeno P celle (per buchi/irregolarità),
    viene allargata automaticamente finché possibile.
    """
    order, row_idx, rows_sorted = build_serpentine_path(coords, E, start_corner=start_corner)
    N = len(coords)
    pos = np.empty(N, dtype=int)
    pos[order] = np.arange(N)

    allowed = []
    for k in range(S):
        a = k * P
        b = (k + 1) * P - 1
        w = window_slack
        while True:
            lo = max(0, a - w)
            hi = min(N - 1, b + w)
            idx = [i for i in range(N) if lo <= pos[i] <= hi]
            if len(idx) >= P or (lo == 0 and hi == N - 1):
                allowed.append(set(idx))
                break
            w += 1  # auto-widen fino a capienza
    return allowed





def build_serpentine_path(coords: np.ndarray, E, start_corner: str = "BL"):
    """
    Restituisce:
      order: lista di indici di celle nell'ordine serpentina
      row_idx: vettore riga per cella
      rows_sorted: lista delle righe nell’ordine di percorrenza
    Parametri:
      start_corner in {"BL","BR","TL","TR"} per scegliere l’angolo di partenza.
    """
    e1, e2 = pca_axes(coords)
    pitch = estimate_row_pitch_from_edges(coords, E, e1, e2)
    t_row = coords @ e2
    t_col = coords @ e1
    r0 = t_row.min()
    row_idx = np.round((t_row - r0) / pitch).astype(int)

    # raggruppa per riga
    from collections import defaultdict
    by_row = defaultdict(list)
    for i, r in enumerate(row_idx):
        by_row[int(r)].append(i)
    rows = sorted(by_row.keys())

    # ordina righe dal basso o dall’alto
    if start_corner in ("BL","BR"):
        rows_sorted = rows                      # dal basso verso l’alto
    else:
        rows_sorted = list(reversed(rows))      # dall’alto verso il basso

    # in ogni riga ordina a destra o a sinistra in base all’angolo di partenza
    order = []
    left_to_right_first = start_corner in ("BL","TL")
    for k, r in enumerate(rows_sorted):
        idx = by_row[r]
        idx.sort(key=lambda i: t_col[i])        # crescente = sinistra→destra lungo e1
        go_l2r = (k % 2 == 0 and left_to_right_first) or (k % 2 == 1 and not left_to_right_first)
        if not go_l2r:
            idx.reverse()                       # destra→sinistra
        order.extend(idx)

    return order, row_idx, rows_sorted

def preview_serpentine_path(coords: np.ndarray, order, row_idx, P,
                            title="Serpentina (solo guida)"):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # normalizza y per righe orizzontali
    rows = sorted(set(row_idx))
    y_map = {r: i for i, r in enumerate(rows)}
    xs = np.arange(len(order))
    ys = np.array([y_map[row_idx[i]] for i in order], dtype=float)

    # colori per blocchi da P
    palette = list(mcolors.TABLEAU_COLORS.values())
    colors = [palette[(k // P) % len(palette)] for k in range(len(order))]

    fig, ax = plt.subplots(figsize=(14, 6), dpi=130)
    ax.scatter(xs, ys, s=18, c=colors, alpha=0.9, edgecolors="none")

    # separatori ogni P
    for g in range(len(order) // P + 1):
        xg = g * P - 0.5
        ax.axvline(xg, color="k", lw=0.6, alpha=0.25)

    # frecce tra centroidi dei blocchi (1→2→…)
    from math import floor
    G = floor(len(order) / P)
    for g in range(G - 1):
        a = g * P
        b = (g + 1) * P
        c = (g + 1) * P
        d = (g + 2) * P
        x0 = np.mean(xs[a:b]); y0 = np.mean(ys[a:b])
        x1 = np.mean(xs[c:d]); y1 = np.mean(ys[c:d])
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="black"))
        ax.text((x0+x1)/2, (y0+y1)/2, f"{g+1}", ha="center", va="center",
                fontsize=9, bbox=dict(boxstyle="circle,pad=0.2", fc="white"))

    ax.set_ylim(-0.7, len(rows)-0.3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"r{r}" for r in rows])
    ax.set_xlabel("indice lungo serpentina")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

def pca_axes(coords: np.ndarray):
    """Restituisce unit vectors (e1, e2) con e1 = asse di massima varianza."""
    X = coords - coords.mean(axis=0, keepdims=True)
    # SVD è robusta e più stabile numericamente della cov-eig
    U, s, Vt = npl.svd(X, full_matrices=False)
    e1 = Vt[0]  # direzione "lunga"
    e1 = e1 / npl.norm(e1)
    e2 = np.array([-e1[1], e1[0]])  # perpendicolare (verso arbitrario)
    return e1, e2

def estimate_row_pitch_from_edges(coords: np.ndarray, E, e1, e2):
    """Stima il passo tra righe (pitch) usando le differenze di proiezione
    sull'asse e2, ma SOLO su archi (i,j) del grafo e scartando i più piccoli."""
    t = coords @ e2
    dt = np.abs(t[[i for i,_ in E]] - t[[j for _,j in E]])
    if len(dt) == 0:
        return max(1.0, 1.732 * 0.5 * 2)  # fallback
    # Scarta il 60% più piccolo (tipicamente contatti "stessa riga")
    thr = np.quantile(dt, 0.60)
    cand = dt[dt >= thr]
    if len(cand) == 0:
        cand = dt
    pitch = float(np.median(cand))
    # un minimo di sanità
    if pitch < 1e-6:
        pitch = float(np.median(dt)) if np.median(dt) > 0 else 1.0
    return pitch




def build_graph(coords: np.ndarray, R: float, tol: float) -> Tuple[List[Tuple[int,int]], List[Tuple[int,int]]]:
    """Return undirected edges E and directed arcs A on adjacency distance 2R±tol."""
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=2*R + tol)
    E = []
    for i, j in pairs:
        d = float(np.linalg.norm(coords[i] - coords[j]))
        if abs(d - 2 * R) <= tol:
            E.append((i, j))
    A = []
    for (i, j) in E:
        A.append((i, j))
        A.append((j, i))
    return E, A


def solve_layout(coords: np.ndarray, S: int, P: int, R: float, tol: float,
                 time_limit: int = 60, alpha: float = 0.1, seed: int = 0, Lmin=1,
                 use_pair_snake: bool = True, degree_cap=4, start_corner: str = "TL",
                 window_slack: int = 3, enforce_degree: bool = False):
    N = len(coords)
    if S * P > N:
        raise SystemExit(f"Celle richieste {S*P} > disponibili {N}")

    E, A = build_graph(coords, R, tol)


    # ---------- pair-snake ----------
    if use_pair_snake:
        allowed_sets, snake_order, row_idx = pair_snake_allowed_sets(
            coords, E, S, P, start_corner=start_corner, window_slack=window_slack
        )
    else:
        allowed_sets = [set(range(len(coords))) for _ in range(S)]



    model = cp_model.CpModel()
    rng = np.random.default_rng(seed)

   

    # decision vars + blocco fuori finestra
    x, r = {}, {}
    for i in range(N):
        for k in range(S):
            xi = model.NewBoolVar(f"x[{i},{k}]")
            x[(i,k)] = xi
            if use_pair_snake and i not in allowed_sets[k]:
                model.Add(xi == 0)
            r[(i,k)] = model.NewBoolVar(f"r[{i},{k}]")



                    
    # flows
    f = {}      # f[i,j,k] integer 0..P-1
    cap = P - 1
    for (i,j) in A:
        for k in range(S):
            f[(i,j,k)] = model.NewIntVar(0, cap, f"f[{i}->{j},{k}]")

    # link vars between consecutive groups
    z1 = {}     # group k uses i, group k+1 uses j
    z2 = {}     # group k uses j, group k+1 uses i
    for k in range(S-1):
        for (i,j) in E:
            z1[(k,i,j)] = model.NewBoolVar(f"z1[{k},{i}-{j}]")
            z2[(k,i,j)] = model.NewBoolVar(f"z2[{k},{i}-{j}]")

    # group sizes
    for k in range(S):
        model.Add(sum(x[(i,k)] for i in range(N)) == P)

    # each cell at most one group
    for i in range(N):
        model.Add(sum(x[(i,k)] for k in range(S)) <= 1)

    # roots
    for k in range(S):
        model.Add(sum(r[(i,k)] for i in range(N)) == 1)
        for i in range(N):
            model.Add(r[(i,k)] <= x[(i,k)])

    # flow capacity (only along selected nodes)
    for (i,j) in A:
        for k in range(S):
            model.Add(f[(i,j,k)] <= cap * x[(i,k)])
            model.Add(f[(i,j,k)] <= cap * x[(j,k)])

    # flow conservation (connectivity)
    # Σ_in - Σ_out = x - P*r
    out_arcs = defaultdict(list)
    in_arcs = defaultdict(list)
    for (i,j) in A:
        out_arcs[(i)].append((i,j))
        in_arcs[(j)].append((i,j))

    for k in range(S):
        for i in range(N):
            model.Add(
                sum(f[(u,v,k)] for (u,v) in in_arcs[i])
                - sum(f[(u,v,k)] for (u,v) in out_arcs[i])
                == x[(i,k)] - P * r[(i,k)]
            )

    # link activation constraints and per-cell degree<=4
    # link activation constraints (senza vincolo di grado per default)
    L = []

    # solo se vuoi davvero limitare il grado, inizializza i contatori
    if enforce_degree:
        per_cell_degree = [model.NewIntVar(0, degree_cap, f"deg[{i}]") for i in range(N)]
        deg_expr = [0 for _ in range(N)]

    for k in range(S-1):
        Lk = model.NewIntVar(0, len(E), f"L[{k}]")
        link_terms = []
        for (i,j) in E:
            model.Add(z1[(k,i,j)] <= x[(i,k)])
            model.Add(z1[(k,i,j)] <= x[(j,k+1)])
            model.Add(z2[(k,i,j)] <= x[(j,k)])
            model.Add(z2[(k,i,j)] <= x[(i,k+1)])
            link_terms += [z1[(k,i,j)], z2[(k,i,j)]]

            if enforce_degree:
                deg_expr[i] = deg_expr[i] + z1[(k,i,j)] + z2[(k,i,j)]
                deg_expr[j] = deg_expr[j] + z1[(k,i,j)] + z2[(k,i,j)]

        model.Add(Lk == sum(link_terms))
        if Lmin > 0:
            model.Add(Lk >= Lmin)
        L.append(Lk)

    if enforce_degree:
        for i in range(N):
            model.Add(per_cell_degree[i] == deg_expr[i])
            model.Add(per_cell_degree[i] <= degree_cap)

    # objective: maximize links, keep them near target
    if len(L) > 0:
        target = min(2*P, max(0, int(np.mean([len(E)//S for _ in range(S)]))))
        # absolute deviation vars
        devs = []
        for k, Lk in enumerate(L):
            over = model.NewIntVar(0, 2*P, f"over[{k}]")
            under = model.NewIntVar(0, 2*P, f"under[{k}]")
            model.Add(Lk - target <= over)
            model.Add(target - Lk <= under)
            dev = model.NewIntVar(0, 2*P, f"dev[{k}]")
            model.Add(dev == over + under)
            devs.append(dev)
        # Scalar objective
        # maximize sum(L) - alpha * sum(dev)
        # CP-SAT maximizes; scale to integers
        ALPHA = int(alpha * 100)
        model.Maximize( 100 * sum(L) - ALPHA * sum(devs) )
    else:
        model.Maximize(0)

    # solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = 8
    solver.parameters.log_search_progress = True

    status = solver.Solve(model)
    return status, solver, x, r, L, z1, z2, E


def plot_solution(coords, R, S, x, z1, z2, E, L, solver,
                             title="", save=None, show_links=True, show_arrows=True):
    import matplotlib.colors as mcolors
    from matplotlib.patches import Circle
    import matplotlib.pyplot as plt

    palette = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    def color_for_group(k): return palette[k % len(palette)]

    N = len(coords)
    # gruppo assegnato per ogni cella
    assigned_group = [-1]*N
    for i in range(N):
        for k in range(S):
            if solver.BooleanValue(x[(i,k)]):
                assigned_group[i] = k
                break

    # centroidi dei gruppi
    centroids = np.zeros((S, 2), dtype=float)
    for k in range(S):
        idx = [i for i in range(N) if assigned_group[i] == k]
        if idx:
            centroids[k] = coords[idx].mean(axis=0)
        else:
            centroids[k] = np.array([np.nan, np.nan])

    fig, ax = plt.subplots(figsize=(12, 8), dpi=130)

    # disegna celle
    for i, (x0,y0) in enumerate(coords):
        gid = assigned_group[i]
        fc = color_for_group(gid) if gid >= 0 else "white"
        ax.add_patch(Circle((x0,y0), R, facecolor=fc, edgecolor="black", linewidth=0.6, zorder=2))

    # disegna contatti effettivi tra gruppi successivi (linee sottili grigie)
    if show_links:
        for k in range(S-1):
            for (i,j) in E:
                if solver.BooleanValue(z1[(k,i,j)]) or solver.BooleanValue(z2[(k,i,j)]):
                    xi, yi = coords[i]
                    xj, yj = coords[j]
                    ax.plot([xi, xj], [yi, yj], linewidth=1.0, alpha=0.5, color="black", zorder=3)

    # frecce con ordine (k -> k+1) e label con #link
    if show_arrows:
        for k in range(S-1):
            x0, y0 = centroids[k]
            x1, y1 = centroids[k+1]
            if not (np.isnan(x0) or np.isnan(x1)):
                # piccola freccia tra centroidi
                ax.annotate(
                    "", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", linewidth=1.5, color="black", shrinkA=10, shrinkB=10),
                    zorder=4
                )
                # etichetta ordine (1..S-1) e #link
                lk = int(solver.Value(L[k])) if L else 0
                xm, ym = (0.5*(x0+x1), 0.5*(y0+y1))
                ax.text(xm, ym, f"{k+1}\n({lk})", ha="center", va="center",
                        fontsize=9, bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec="black", alpha=0),
                        zorder=5)

    # mostra indice del gruppo sul centroide
    for k in range(S):
        xk, yk = centroids[k]
        if not (np.isnan(xk) or np.isnan(yk)):
            ax.text(xk, yk, f"G{k}", ha="center", va="center", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0), zorder=6)

    # limiti asse (evita canvas vuoto)
    xs = coords[:,0]; ys = coords[:,1]
    ax.set_xlim(xs.min()-2*R, xs.max()+2*R)
    ax.set_ylim(ys.max()+2*R, ys.min()-2*R)  # invertito per coerenza con il tuo codice
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title)
    plt.tight_layout()
    if save:
        plt.savefig(save, bbox_inches="tight", dpi=160)
    plt.show()



def main():

    csv = "new_version/out.csv"
    S=11
    P=6
    radius=9.0
    tol=2
    time_limit=120
    alpha=0.3
    degree_cap=6
    Lmin=1
    use_snake=False
    start_corner = "TL"
    window_slack=3
    enforce_degree=False

    

    df = pd.read_csv(csv)
    coords = df[['x','y']].to_numpy()

    E, A = build_graph(coords, radius, tol)

    # Anteprima serpentina (prima di lanciare il solver)
    #preview_pair_snake(coords, E, P, start_corner="TL")
    

    status, solver, x, r, L, z1, z2, E = solve_layout(
        coords, S, P, radius, tol,
        time_limit=time_limit, alpha=alpha, degree_cap=degree_cap, Lmin=Lmin,
        use_pair_snake=use_snake, start_corner=start_corner, window_slack=window_slack,
        enforce_degree=enforce_degree
    )

    status_name = {cp_model.OPTIMAL:"OPTIMAL", cp_model.FEASIBLE:"FEASIBLE",
                   cp_model.INFEASIBLE:"INFEASIBLE", cp_model.MODEL_INVALID:"MODEL_INVALID",
                   cp_model.UNKNOWN:"UNKNOWN"}.get(status, str(status))
    print("Solver status:", status_name)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        if L:
            links=[int(solver.Value(v)) for v in L]
            print("Links between successive groups:", links, "  avg:", sum(links)/len(links))
        plot_solution(
            coords, radius, S, x, z1, z2, E, L, solver,
            title=f"{S}S{P}P — CP-SAT ({status_name})",
            save=None, show_links=True, show_arrows=True
        )
    else:
        print("Nessuna soluzione trovata con i vincoli dati nel tempo limite.")


if __name__ == "__main__":
    main()
