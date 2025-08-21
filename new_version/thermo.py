import numpy as np
import networkx as nx

def count_contacts(G, part_of, a, b):
    """Number of cell–cell edges in G that cross from group a to group b."""
    w = 0
    for u, v in G.edges:
        gu, gv = part_of[u], part_of[v]
        if (gu == a and gv == b) or (gu == b and gv == a):
            w += 1
    return w

def interface_table(G, part_of, order):
    """Return list of (a, b, w_ab) along the series order."""
    edges = []
    for i in range(len(order) - 1):
        a, b = order[i], order[i+1]
        w = count_contacts(G, part_of, a, b)
        edges.append((a, b, w))
    return edges

def simulate_series_electrothermal(
    G, part_of, order, P,
    I_pack=60.0,                 # A, peak or nominal
    r_cell=0.020,                # Ω per single cell (e.g. 20 mΩ)
    r_contact=0.0005,            # Ω per contact pair (e.g. 0.5 mΩ)
    r_extra=0.0,                 # Ω extra series in strap (if any)
    Rth_group=2.0,               # K/W per group to ambient (coarse)
    Rth_iface=0.7,               # K/W per interface/strap to ambient (coarse)
    Iperpair_max=10.0            # A, safe current per cell-cell pair (flagging)
):
    """
    Compute pack resistance, per-group & per-interface loss and steady ΔT.
    Returns dict with summary and detailed lists suitable for plotting/reporting.
    """
    S = max(part_of.values()) + 1

    # electrical: group resistances (P cells in parallel in each group)
    R_groups = [r_cell / float(P) for _ in range(S)]
    P_groups = [I_pack**2 * Rg for Rg in R_groups]
    dT_groups = [Pg * Rth_group for Pg in P_groups]

    # electrical: interface resistances from adjacency counts
    if len(order) != S:
        raise ValueError("Series order must be a permutation of 0..S-1")

    edges = interface_table(G, part_of, order)  # (a,b,w_ab)
    R_ifaces = []
    P_ifaces = []
    dT_ifaces = []
    overload  = []  # flags where I per contact exceeds spec
    for a, b, w in edges:
        if w <= 0:
            # No touching pairs at all – this should not happen if you used
            # an adjacency-hard solver. Set R very large and flag.
            R_ab = 1e6
            over = True
        else:
            R_ab = r_contact / float(w) + r_extra
            over = (I_pack / float(w)) > Iperpair_max
        R_ifaces.append(R_ab)
        P_ab = I_pack**2 * R_ab
        P_ifaces.append(P_ab)
        dT_ifaces.append(P_ab * Rth_iface)
        overload.append(over)

    R_pack = sum(R_groups) + sum(R_ifaces)
    V_drop = I_pack * R_pack
    P_loss = I_pack**2 * R_pack

    return {
        "R_pack": R_pack, "V_drop": V_drop, "P_loss": P_loss,
        "groups": {
            "order": order,
            "R": R_groups,
            "P": P_groups,
            "dT": dT_groups
        },
        "interfaces": {
            "edges": [(a, b) for (a, b, _) in edges],
            "w":     [w for (_, _, w) in edges],
            "R":     R_ifaces,
            "P":     P_ifaces,
            "dT":    dT_ifaces,
            "overload": overload
        }
    }

def group_contact_counts(G, part_of):
    """
    Return a dict {(a,b): w} with a<b giving the number of touching cell pairs
    between group a and group b.
    """
    counts = {}
    for u, v in G.edges:
        gu, gv = part_of[u], part_of[v]
        if gu == gv:
            continue
        a, b = (gu, gv) if gu < gv else (gv, gu)
        counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts

def print_all_group_contacts(counts, S):
    """
    Pretty-print, for each group g, the list of neighbour groups and
    the number of touching pairs with each.
    """
    adj = {g: [] for g in range(S)}
    for (a, b), w in counts.items():
        adj[a].append((b, w))
        adj[b].append((a, w))
    for g in range(S):
        nbrs = sorted(adj[g], key=lambda t: -t[1])  # highest w first
        s = ", ".join(f"{h}({w})" for h, w in nbrs)
        print(f"g {g:2d}: {s}")

def print_series_contacts(order, counts):
    """
    Print the number of touching pairs only for consecutive groups along 'order'.
    """
    print("\nSeries contacts (adjacent pairs along order):")
    for i in range(len(order) - 1):
        a, b = order[i], order[i+1]
        key = (a, b) if a < b else (b, a)
        w = counts.get(key, 0)
        print(f"{a:2d} -> {b:2d}: w = {w}")

# =====================================================================


import numpy as np
import networkx as nx

def group_contact_counts(G, part_of):
    """
    Count touching pairs between every pair of groups.
    Returns {(a,b): w} with a<b.
    """
    counts = {}
    for u, v in G.edges:
        gu, gv = part_of[u], part_of[v]
        if gu == gv:
            continue
        a, b = (gu, gv) if gu < gv else (gv, gu)
        counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts


def compute_series_thermal(G, centres, part_of, order, S, P,
                           I_pack=80.0,
                           r_cell_mohm=20.0,      # per-cell DC resistance (mΩ)
                           r_pair_mohm=0.50,      # ∼0.5 mΩ per touching pair of cells
                           Rtheta_group_KW=2.0,   # K/W group-to-ambient
                           Rtheta_iface_KW=0.7,   # K/W interface-to-ambient
                           Iperpair_max=10.0):    # (A) nominal limit per touching pair
    """
    Returns
    -------
    edges : list of dicts [{a,b,w,R_mohm,P_W,DeltaT_K,overload}, ...] for consecutive pairs along order
    groups: list of dicts [{g,R_mohm,P_W,DeltaT_K}, ...] for all groups
    totals: dict with R_total_mohm, V_drop_V, P_loss_W
    """
    # 1) contact counts
    counts = group_contact_counts(G, part_of)

    # 2) per-group resistance (parallel of P cells)
    #    R_group = R_cell/P (units mΩ)
    Rg_mohm = r_cell_mohm / float(P)

    # 3) edges along the series path
    edges = []
    for i in range(len(order)-1):
        a, b = order[i], order[i+1]
        key  = (a, b) if a < b else (b, a)
        w    = counts.get(key, 0)          # touching pairs
        # interface resistance: r_pair per touching pair in parallel
        R_iface_mohm = (r_pair_mohm / max(1, w))
        Pedge_W      = (I_pack**2) * (R_iface_mohm / 1000.0)
        dT_edge_K    = Rtheta_iface_KW * Pedge_W
        overload     = (w > 0 and (I_pack / w) > Iperpair_max)

        edges.append(dict(a=a, b=b, w=w,
                          R_mohm=R_iface_mohm,
                          P_W=Pedge_W,
                          DeltaT_K=dT_edge_K,
                          overload=overload))

    # 4) per-group heating (very simple lumped model)
    groups = []
    for g in range(S):
        Pg_W   = (I_pack**2) * (Rg_mohm / 1000.0)
        dTg_K  = Rtheta_group_KW * Pg_W
        groups.append(dict(g=g, R_mohm=Rg_mohm, P_W=Pg_W, DeltaT_K=dTg_K))

    # 5) totals
    R_total_mohm = S * Rg_mohm + sum(e["R_mohm"] for e in edges)
    V_drop_V     = I_pack * (R_total_mohm/1000.0)
    P_loss_W     = (I_pack**2) * (R_total_mohm/1000.0)

    totals = dict(R_total_mohm=R_total_mohm,
                  V_drop_V=V_drop_V,
                  P_loss_W=P_loss_W)

    return edges, groups, totals

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import numpy as np

def group_centroids(centres, part_of, S):
    cent = np.zeros((S, 2), float)
    for g in range(S):
        idx = [i for i,p in part_of.items() if p == g]
        cent[g] = centres[idx].mean(axis=0)
    return cent


def plot_series_thermal(poly, centres, part_of, S, order,
                        edges, groups, R=9.0,
                        cmap_groups='YlGnBu', cmap_edges='inferno',
                        annotate=True):
    """
    Draw cells coloured by group ΔT and a polyline through group centroids with
    segment colour proportional to interface ΔT. Segment thickness ∝ interface power.
    """
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(*poly.exterior.xy, color='k', lw=2)

    # --- map group ΔT → colour
    g_dT = np.array([g['DeltaT_K'] for g in groups])
    g_norm = mpl.colors.Normalize(vmin=0, vmax=max(1e-9, g_dT.max()))
    cmap_g = cm.get_cmap(cmap_groups)

    # draw cells
    for i, (x, y) in enumerate(centres):
        g = part_of[i]
        col = cmap_g(g_norm(groups[g]['DeltaT_K']))
        ax.add_patch(plt.Circle((x, y), R, facecolor=col, edgecolor='k', lw=0.25))

    # --- map edge ΔT → colour, width ~ power
    e_dT = np.array([e['DeltaT_K'] for e in edges])
    if len(e_dT) == 0:
        dTmax = 1.0
    else:
        dTmax = max(1e-9, e_dT.max())
    e_norm = mpl.colors.Normalize(vmin=0, vmax=dTmax)
    cmap_e = cm.get_cmap(cmap_edges)

    C = group_centroids(centres, part_of, S)
    for k, e in enumerate(edges):
        a, b = e['a'], e['b']
        xa, ya = C[a]
        xb, yb = C[b]
        col = cmap_e(e_norm(e['DeltaT_K']))
        # width proportional to power (cap it a bit for visibility)
        lw  = 0.5 + 4.0 * (e['P_W'] / max(1e-6, max(x['P_W'] for x in edges)))
        ax.plot([xa, xb], [ya, yb], color=col, lw=lw, alpha=0.9, zorder=4)
        if annotate:
            midx, midy = (xa+xb)/2, (ya+yb)/2
            ax.text(midx, midy, f"w={e['w']}\nP={e['P_W']:.2f}W",
                    ha='center', va='center', fontsize=7,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.75),
                    zorder=5, color='black')

    # --- colourbars
    sm_g = mpl.cm.ScalarMappable(norm=g_norm, cmap=cmap_g)
    sm_g.set_array([])
    cbar_g = fig.colorbar(sm_g, ax=ax, fraction=0.046, pad=0.03)
    cbar_g.set_label('ΔT group [K]')

    sm_e = mpl.cm.ScalarMappable(norm=e_norm, cmap=cmap_e)
    sm_e.set_array([])
    cbar_e = fig.colorbar(sm_e, ax=ax, fraction=0.046, pad=0.12)
    cbar_e.set_label('ΔT interface [K]')

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f"Series thermal map (S={S})")
    plt.tight_layout()
    plt.show()