import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# ---------- Centroid utility -------------------------------------------------

def group_centroids(centres, part_of, S):
    """Centroid of each group (for tie-breaks and plotting)."""
    cent = np.zeros((S, 2), float)
    for g in range(S):
        idx = [i for i,p in part_of.items() if p == g]
        cent[g] = centres[idx].mean(axis=0)
    return cent

# ---------- Group graph construction ------------------------------------------

def _build_group_graph_contacts(G, centres, part_of, alpha=1.0, beta=0.05):
    """
    Return H: nodes=groups, edge iff groups touch.
    Edge attributes:
      w     = number of touching cell-cell pairs
      dist  = centroid distance
      score = w - beta * dist         (maximise score)
    """
    S = max(part_of.values()) + 1
    H = nx.Graph(); H.add_nodes_from(range(S))

    contact = {}
    for u, v in G.edges:
        gu, gv = part_of[u], part_of[v]
        if gu == gv: 
            continue
        a, b = sorted((gu, gv))
        contact[(a, b)] = contact.get((a, b), 0) + 1

    C = np.asarray(centres)
    cent = np.zeros((S, 2))
    for g in range(S):
        mem = [i for i, p in part_of.items() if p == g]
        cent[g] = C[mem].mean(axis=0)

    for (a, b), w in contact.items():
        d = float(np.linalg.norm(cent[a] - cent[b]))
        score = float(w) - beta * d
        H.add_edge(a, b, w=w, dist=d, score=score)

    return H

# ---------- OR-Tools Hamiltonian path computation -----------------------------

def series_order_ortools(G, centres, part_of,
                         beta=0.05, time_limit_s=5, seed=0):
    """
    Use OR-Tools Routing to compute a Hamiltonian path over groups:
    – uses only edges that are present in the group adjacency graph,
    – maximises contact score (touching pairs minus small distance penalty),
    – returns (order, edges) with edges always adjacent.
    """
    H = _build_group_graph_contacts(G, centres, part_of, beta=beta)
    S = H.number_of_nodes()
    if S == 0:
        return [], []

    if not nx.is_connected(H):
        raise RuntimeError("Group graph is not connected; no adjacency-only chain exists.")

    depot = S
    N     = S + 1

    INF = 10**6
    scores = {(u, v): H[u][v]['score'] for u, v in H.edges}
    max_w = max(H[u][v]['w'] for u, v in H.edges) if H.number_of_edges() else 1

    def arc_cost(i, j):
        if i == depot or j == depot:
            return 0
        if H.has_edge(i, j):
            s = H[i][j]['score']
            return int(round((max_w * 10) - (s * 10)))
        else:
            return INF

    manager = pywrapcp.RoutingIndexManager(N, 1, depot)
    routing = pywrapcp.RoutingModel(manager)

    def cost_cb(from_index, to_index):
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return arc_cost(i, j)

    transit_cb_index = routing.RegisterTransitCallback(cost_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_index)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.time_limit.FromSeconds(time_limit_s)
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    params.log_search = False

    try:
        params.random_seed = seed
    except Exception:
        pass

    solution = routing.SolveWithParameters(params)
    if solution is None:
        raise RuntimeError("OR-Tools could not find a feasible series path.")

    idx = routing.Start(0)
    seq = []
    while not routing.IsEnd(idx):
        node = manager.IndexToNode(idx)
        if node != depot:
            seq.append(node)
        idx = solution.Value(routing.NextVar(idx))

    for i in range(len(seq)-1):
        a, b = seq[i], seq[i+1]
        assert H.has_edge(a, b), f"Non-adjacent step produced by solver: {a}->{b}"

    edges = [(seq[i], seq[i+1]) for i in range(len(seq)-1)]
    return seq, edges

# ---------- Plotting ----------------------------------------------------------

def plot_series_order(poly, centres, part_of, S, order, group_color=None, R=9.0):
    """
    Plot groups (filled) and overlay the series order by drawing lines between group centroids.
    """
    fig, ax = plt.subplots(figsize=(7,7))
    ax.plot(*poly.exterior.xy, color='k', lw=2)

    if group_color is None:
        try:
            import colorcet as cc
            palette = [mpl.colors.to_rgb(c) for c in cc.glasbey]
        except Exception:
            palette = [cm.tab20(i%20) for i in range(S)]
        group_color = {g: palette[g % len(palette)] for g in range(S)}

    for i, (x, y) in enumerate(centres):
        g = part_of[i] if isinstance(part_of, dict) else part_of[i]
        ax.add_patch(plt.Circle((x, y), R, facecolor=group_color[g], edgecolor='k', lw=0.4))

    C = group_centroids(centres, part_of, S)
    xs = [C[g,0] for g in order]
    ys = [C[g,1] for g in order]
    ax.plot(xs, ys, '-o', color='black', lw=1.2, ms=3, alpha=0.85, zorder=3)

    for k, g in enumerate(order):
        ax.text(C[g,0], C[g,1], f"{k+1}",
                ha='center', va='center',
                fontsize=7, color='white', weight='bold',
                bbox=dict(boxstyle='circle,pad=0.2', fc='black', alpha=0.6),
                zorder=4)

    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(f"Series order (S={S})")
    plt.tight_layout(); plt.show()

def plot_series_order2(poly, centres, part_of, S, order, group_color=None, R=9.0):
    fig, ax = plt.subplots(figsize=(7,7))
    ax.plot(*poly.exterior.xy, color='k', lw=2)

    if group_color is None:
        try:
            import colorcet as cc
            palette = [mpl.colors.to_rgb(c) for c in cc.glasbey]
        except Exception:
            palette = [cm.tab20(i % 20) for i in range(S)]
        group_color = {g: palette[g % len(palette)] for g in range(S)}

    # --- draw cells (white for unassigned)
    for i, (x, y) in enumerate(centres):
        g = part_of.get(i, None)  # may be missing
        if g is None:
            face = (1, 1, 1, 1)
            edge = '0.7'
        else:
            face = group_color[g]
            edge = 'k'
        ax.add_patch(plt.Circle((x, y), R, facecolor=face, edgecolor=edge, lw=0.4))

    # group centroids & series path
    C = group_centroids(centres, part_of, S)   # uses only assigned cells
    xs = [C[g, 0] for g in order]
    ys = [C[g, 1] for g in order]
    ax.plot(xs, ys, '-o', color='black', lw=1.2, ms=3, alpha=0.85, zorder=3)

    for k, g in enumerate(order):
        ax.text(C[g,0], C[g,1], f"{k+1}",
                ha='center', va='center', fontsize=7, color='white', weight='bold',
                bbox=dict(boxstyle='circle,pad=0.2', fc='black', alpha=0.6), zorder=4)

    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(f"Series order (S={S})")
    plt.tight_layout(); plt.show()