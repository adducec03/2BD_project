
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


def solve_layout(coords: np.ndarray, S: int, P: int, R: float, tol: float, time_limit: int = 60,
                 alpha: float = 0.1, seed: int = 0, degree_cap=4, Lmin=1):
    N = len(coords)
    if S * P > N:
        raise SystemExit(f"Celle richieste {S*P} > disponibili {N}")

    E, A = build_graph(coords, R, tol)

    model = cp_model.CpModel()
    rng = np.random.default_rng(seed)

    # decision vars
    x = {}      # x[i,k]
    r = {}      # r[i,k]
    for i in range(N):
        for k in range(S):
            x[(i,k)] = model.NewBoolVar(f"x[{i},{k}]")
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
    L = []
    per_cell_degree = [model.NewIntVar(0, degree_cap, f"deg[{i}]") for i in range(N)]
    deg_expr = [0 for _ in range(N)]

    for k in range(S-1):
        # L_k
        Lk = model.NewIntVar(0, len(E), f"L[{k}]")
        link_terms = []
        for (i,j) in E:
            model.Add(z1[(k,i,j)] <= x[(i,k)])
            model.Add(z1[(k,i,j)] <= x[(j,k+1)])
            model.Add(z2[(k,i,j)] <= x[(j,k)])
            model.Add(z2[(k,i,j)] <= x[(i,k+1)])
            link_terms += [z1[(k,i,j)], z2[(k,i,j)]]
            # accumulate degree terms
            deg_expr[i] = deg_expr[i] + z1[(k,i,j)] + z2[(k,i,j)]
            deg_expr[j] = deg_expr[j] + z1[(k,i,j)] + z2[(k,i,j)]
        model.Add(Lk == sum(link_terms))
        if Lmin > 0:
            model.Add(Lk >= Lmin)
        L.append(Lk)

    # cap degree per cell
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
    return status, solver, x, r, L


def plot_solution(coords, R, S, x, solver, title="", save=None):
    import matplotlib.colors as mcolors
    from matplotlib.patches import Circle
    import matplotlib.pyplot as plt
    palette = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    def color_for_group(k): return palette[k % len(palette)]

    N = len(coords)
    assigned_group = [-1]*N
    for i in range(N):
        for k in range(S):
            if solver.BooleanValue(x[(i,k)]):
                assigned_group[i] = k
                break

    fig, ax = plt.subplots(figsize=(12, 8), dpi=130)
    for i, (x0,y0) in enumerate(coords):
        gid = assigned_group[i]
        fc = color_for_group(gid) if gid >= 0 else "white"
        ax.add_patch(Circle((x0,y0), R, facecolor=fc, edgecolor="black", linewidth=0.6))

    # >>> limiti espliciti (evita canvas "vuoto")
    xs = coords[:,0]; ys = coords[:,1]
    ax.set_xlim(xs.min()-2*R, xs.max()+2*R)
    ax.set_ylim(ys.max()+2*R, ys.min()-2*R)  # già invertito
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title)
    plt.tight_layout()
    if save:
        plt.savefig(save, bbox_inches="tight", dpi=160)
    plt.show()


def main():

    csv = "new_version/out.csv"
    S=15
    P=5
    radius=9.0
    tol=1
    time_limit=120
    alpha=0.1
    degree_cap=6
    Lmin=3

    

    df = pd.read_csv(csv)
    coords = df[['x','y']].to_numpy()

    status, solver, x, r, L = solve_layout(coords, S, P, radius, tol,
                                           time_limit=time_limit, alpha=alpha, degree_cap=degree_cap, Lmin=Lmin)

    status_name = {cp_model.OPTIMAL:"OPTIMAL", cp_model.FEASIBLE:"FEASIBLE",
                   cp_model.INFEASIBLE:"INFEASIBLE", cp_model.MODEL_INVALID:"MODEL_INVALID",
                   cp_model.UNKNOWN:"UNKNOWN"}.get(status, str(status))
    print("Solver status:", status_name)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        if L:
            links=[int(solver.Value(v)) for v in L]
            print("Links between successive groups:", links, "  avg:", sum(links)/len(links))
        plot_solution(coords, radius, S, x, solver,
                      title=f"{S}S{P}P — CP-SAT ({status_name})", save=None)
    else:
        print("Nessuna soluzione trovata con i vincoli dati nel tempo limite.")


if __name__ == "__main__":
    main()
