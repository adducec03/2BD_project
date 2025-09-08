def build_contact_graph_anisotropic(centres, R=9.0, eps=0.2,
                                    w_row=5.0, w_diag=2.0):
    """
    Contact graph where same-row neighbors get stronger weights.
    This biases METIS to cut diagonals rather than rows → more regular blocks.
    """
    import numpy as np
    import networkx as nx

    N = len(centres)
    G = nx.Graph()
    G.add_nodes_from(range(N))

    thr = 2*R + eps
    # estimate row spacing and tolerance
    # For hex pack, vertical spacing ≈ sqrt(3)*R
    row_dy = np.sqrt(3) * R
    same_row_tol = 0.4 * R   # adjust if needed

    for i in range(N):
        xi, yi = centres[i]
        for j in range(i+1, N):
            xj, yj = centres[j]
            dx, dy = xj - xi, yj - yi
            d = np.hypot(dx, dy)
            if d <= thr:
                # bias weight for same-row neighbors
                if abs(dy) < same_row_tol:
                    w = w_row / max(d, 1e-6)     # heavier edge
                else:
                    w = w_diag / max(d, 1e-6)
                G.add_edge(i, j, weight=w)
    return G

def compute_group_interfaces(G, part_of, S):
    import numpy as np
    W = np.zeros((S,S), dtype=int)
    pair_edges = {}
    for u, v in G.edges():
        gu, gv = part_of[u], part_of[v]
        if gu == gv:
            continue
        a, b = (gu, gv) if gu < gv else (gv, gu)
        W[a,b]+=1; W[b,a]+=1
        pair_edges.setdefault((a,b), []).append((u,v))
    return W, pair_edges

import networkx as nx


def estimate_row_index(centres, R=9.0):
    """
    Assign each cell a row ID by scanning Y ascending and grouping
    consecutive points whose Y is within ~row_dy/2.
    """
    import numpy as np
    row_dy = np.sqrt(3) * R
    tol = 0.4 * R

    idx = np.argsort(centres[:,1])
    rows = -np.ones(len(centres), dtype=int)
    current_row = -1
    last_y = None

    for k in idx:
        y = centres[k, 1]
        if last_y is None or abs(y - last_y) > (row_dy - tol):
            current_row += 1
        rows[k] = current_row
        last_y = y
    return rows


import numpy as np
import networkx as nx
from collections import defaultdict

def compute_group_interfaces(G: nx.Graph, part_of: dict[int,int], S: int):
    """
    Return:
      W[g,h] = number of edges between group g and group h (symmetric, off-diagonal)
      pair_edges[(g,h)] = list of (u,v) edges (u∈g, v∈h).
    """
    W = np.zeros((S, S), dtype=int)
    pair_edges = defaultdict(list)

    for u, v in G.edges:
        gu, gv = part_of[u], part_of[v]
        if gu != gv:
            g, h = (gu, gv) if gu < gv else (gv, gu)
            W[g, h] += 1; W[h, g] += 1
            pair_edges[(g, h)].append((u, v) if gu < gv else (v, u))
    return W, pair_edges


def group_row_usage(part_of, rows, S):
    """
    Count how many cells per group lie in each row.
    Returns: dict g -> dict r -> count
    """
    ru = {g: defaultdict(int) for g in range(S)}
    for i, g in part_of.items():
        r = rows[i]
        ru[g][r] += 1
    return ru


def articulation_free(group_nodes, G):
    """Returns a set of articulation points in group's induced subgraph."""
    H = G.subgraph(group_nodes)
    return set(nx.articulation_points(H))

def refine_interfaces_balanced(G, part_of, S, target_min=2, target_max=5,
                               max_rounds=30, verbose=False):
    """
    Try to balance interfaces W[g,h] into [target_min, target_max] by 1↔1 swaps
    of boundary cells along interfaces while preserving contiguity and exact size.
    """
    changed = True
    rounds = 0

    while changed and rounds < max_rounds:
        changed = False
        rounds += 1

        # recompute boundaries
        W, _ = compute_group_interfaces(G, part_of, S)

        # find all pairs with W[g,h] out of band
        pairs = []
        for g in range(S):
            for h in range(g+1, S):
                w = W[g,h]
                if w < target_min or w > target_max:
                    pairs.append((w,g,h))
        # sort smaller first: fix low interfaces first
        pairs.sort(key=lambda x: x[0])

        if not pairs:
            break

        for w, g, h in pairs:
            # boundary candidates: nodes of g adjacent to h, and vice versa
            ngh_g = {u for u in G.nodes() if part_of[u]==g and
                     any(part_of[v]==h for v in G[u])}
            ngh_h = {u for u in G.nodes() if part_of[u]==h and
                     any(part_of[v]==g for v in G[u])}

            if not ngh_g or not ngh_h:
                continue

            # articulation sets to preserve connectivity
            cut_g = articulation_free([u for u in G if part_of[u]==g], G)
            cut_h = articulation_free([u for u in G if part_of[u]==h], G)

            best = None
            best_delta = 0

            for u in ngh_g:
                if u in cut_g:  # don't break g
                    continue
                # neighbors of u in each group
                deg_u_in_g  = sum(1 for v in G[u] if part_of[v]==g)
                deg_u_in_h  = sum(1 for v in G[u] if part_of[v]==h)

                for v in G[u]:
                    if part_of[v] != h:
                        continue
                    if v in cut_h:
                        continue
                    deg_v_in_h = sum(1 for w2 in G[v] if part_of[w2]==h)
                    deg_v_in_g = sum(1 for w2 in G[v] if part_of[w2]==g)

                    delta = (deg_u_in_h + deg_v_in_g) - (deg_u_in_g + deg_v_in_h)
                    # prefer swaps that push W[g,h] toward the middle of [min,max]
                    if w < target_min and delta <= 0:
                        # still consider if it increases adjacency even a bit
                        pass
                    if delta > best_delta:
                        best_delta = delta
                        best = (u, v)

            if best and best_delta > 0:
                u, v = best
                part_of[u] = h
                part_of[v] = g
                changed = True
                if verbose:
                    print(f"swap {u}(g→h) <-> {v}(h→g) on boundary {g}-{h} improves by {best_delta}")
                break  # re-run from scratch for fresh boundaries

    return part_of



def is_contiguous_after_swap(G, part_of, g, take_out, put_in):
    """
    Check contiguity for group g if we remove 'take_out' and add 'put_in'.
    We only test g's induced subgraph.
    """
    group_nodes = [u for u, gg in part_of.items() if gg == g and u != take_out]
    if put_in is not None:
        group_nodes.append(put_in)
    if not group_nodes:
        return False
    sub = G.subgraph(group_nodes)
    return nx.is_connected(sub)


def perimeter_of_group(G, part_of, g):
    """
    #edges from g to NOT g (proxy for boundary length)
    """
    boundary = 0
    members = [u for u, gg in part_of.items() if gg == g]
    in_g = set(members)
    for u in members:
        for v in G[u]:
            if v not in in_g:
                boundary += 1
    return boundary


def row_variance_penalty(ru_g):
    """
    penalize spread across many rows; lower is better.
    ru_g: dict row->count for this group
    """
    counts = list(ru_g.values())
    if not counts:
        return 0.0
    # variance of nonzero row counts
    import numpy as np
    return float(np.var(counts))


def score_partition(G, part_of, S, rows, centres,
                    lam_iface=1.0,
                    lam_perim=0.5,
                    lam_rowvar=0.5,
                    lam_rowspan=0.0,
                    lam_xspan=0.0):
    # contact interfaces
    W, _ = compute_group_interfaces(G, part_of, S)
    iface_score = np.sum(W) / 2.0  # undirected

    # perimeter compactness
    perim = sum(perimeter_of_group(G, part_of, g) for g in range(S))

    # row-usage balance
    ru = group_row_usage(part_of, rows, S)
    rowvar = sum(row_variance_penalty(ru[g]) for g in range(S))

    # NEW: spans (shape regularity)
    rspan = sum(row_span_penalty(rows, part_of, g) for g in range(S))
    xspan = sum(x_span_penalty(centres, part_of, g) for g in range(S))

    return (+lam_iface * iface_score
            - lam_perim * perim
            - lam_rowvar * rowvar
            - lam_rowspan * rspan
            - lam_xspan * xspan)


def refine_groups_regular(G, part_of_init, S, rows, centres,
                          lam_iface=0.3,
                          lam_perim=1.6,
                          lam_rowvar=2.0,
                          lam_rowspan=0.0,
                          lam_xspan=0.0,
                          max_rounds=80, verbose=False):
    """
    Local 1↔1 boundary swaps that keep groups contiguous and size P,
    and improve a composite score:
    - maximize adjacency between groups
    - minimize perimeters (more compact)
    - minimize per-group row variance (more regular)
    """


    def score(po):
        return score_partition(G, po, S, rows, centres,
                               lam_iface=lam_iface,
                               lam_perim=lam_perim,
                               lam_rowvar=lam_rowvar,
                               lam_rowspan=lam_rowspan,
                               lam_xspan=lam_xspan)
    
    
    part_of = dict(part_of_init)
    best_score = score(part_of)
    if verbose:
        print(f"[refine] start score = {best_score:.2f}")

    for it in range(max_rounds):
        improved = False

        # gather boundary edges per pair
        W, pair_edges = compute_group_interfaces(G, part_of, S)

        for (g, h), edges in pair_edges.items():
            # Candidates: nodes on the g–h interface
            g_nodes = set()
            h_nodes = set()
            for u, v in edges:
                g_nodes.add(u); h_nodes.add(v)
            # Try all small swaps (u∈g ↔ v∈h)
            for u in list(g_nodes):
                for v in list(h_nodes):
                    # Tentatively swap u↔v
                    old_g, old_h = part_of[u], part_of[v]
                    if old_g != g or old_h != h:
                        continue
                    part_of[u] = h
                    part_of[v] = g

                    # Check contiguity of both groups
                    cg = is_contiguous_after_swap(G, part_of, g, v, u)
                    ch = is_contiguous_after_swap(G, part_of, h, u, v)

                    if not (cg and ch):
                        # revert
                        part_of[u] = g
                        part_of[v] = h
                        continue

                    # Evaluate new score
                    sc = score(part_of)
                    if sc > best_score + 1e-6:
                        best_score = sc
                        improved = True
                        if verbose:
                            print(f"[refine] improve: swap {u}↔{v} on ({g},{h}), score={best_score:.2f}")
                    else:
                        # revert if not improving
                        part_of[u] = g
                        part_of[v] = h

        if not improved:
            if verbose:
                print(f"[refine] no improvement at iter {it}")
            break

    return part_of


def row_span_penalty(rows, part_of, g):
    """# of distinct rows covered by group g (larger → worse)."""
    rvals = [rows[u] for u, gg in part_of.items() if gg == g]
    if not rvals:
        return 0.0
    return float(max(rvals) - min(rvals) + 1)

def x_span_penalty(centres, part_of, g):
    """Horizontal extent of group g (larger → worse)."""
    xs = [centres[u, 0] for u, gg in part_of.items() if gg == g]
    if not xs:
        return 0.0
    return float(max(xs) - min(xs))