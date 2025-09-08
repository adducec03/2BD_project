# nickel_plate.py
from __future__ import annotations
import numpy as np
import networkx as nx
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _union_discs(centres, idx, radius, res=32):
    """Union of discs of given radius centered at centres[idx]."""
    parts = [Point(*centres[i]).buffer(radius, resolution=res) for i in idx]
    if not parts:
        return None
    return unary_union(parts)

def _largest_component(geom):
    """Return largest polygon of a (Multi)Polygon."""
    if geom is None:
        return None
    geom = make_valid(geom)
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        # choose the largest by area
        areas = [p.area for p in geom.geoms]
        return geom.geoms[int(np.argmax(areas))]
    return None

# -----------------------------------------------------------------------------
# Plate generation
# -----------------------------------------------------------------------------

def make_group_plate(G: nx.Graph,
                     centres: np.ndarray,
                     part_of: dict[int,int] | list[int],
                     g: int,
                     R: float = 9.0,
                     land: float = 1.0,
                     gap: float = 1.8,
                     outline: Polygon | None = None,
                     res: int = 32) -> Polygon | None:
    """
    Build a single nickel-plate polygon for group g.

    Parameters
    ----------
    G        : contact graph over cells (edge if cells touch)
    centres  : (N,2) coordinates
    part_of  : node -> group
    g        : group id
    R        : cell radius (mm) (18650 ≈ 9.0 mm)
    land     : extra weld/cover land around each cell (mm)
    gap      : required clearance from other groups (mm)
    outline  : shapely Polygon (battery outline); if provided, we clip plate to it
    res      : circle discretization

    Returns
    -------
    plate polygon (largest component) or None.
    """
    if isinstance(part_of, list):
        part_of = {i: part_of[i] for i in range(len(part_of))}

    # --- group members
    mem = [i for i, p in part_of.items() if p == g]
    if not mem:
        return None

    # Raw blob of group discs (R + land)
    blob = _union_discs(centres, mem, R + land, res=res)

    # --- repulsion mask from adjacent foreign cells
    # find neighbours in contact graph that belong to other groups
    foreign = set()
    for i in mem:
        for j in G[i]:
            if part_of[j] != g:
                foreign.add(j)
    foreign = list(foreign)

    mask = _union_discs(centres, foreign, R + land + gap, res=res) if foreign else None

    # subtract mask to ensure gap to neighbours
    if mask is not None:
        plate = blob.difference(mask)
    else:
        plate = blob

    # clip to outline if provided
    if outline is not None:
        plate = plate.intersection(outline)

    return _largest_component(plate)

def make_all_plates(G, centres, part_of, S, R=9.0, land=1.2, gap=2.0, outline=None, res=48):
    plates = {}
    for g in range(S):
        plates[g] = make_group_plate_disjoint(G, centres, part_of, g,
                                              R=R, land=land, gap=gap,
                                              outline=outline, res=res)
    return plates

# -----------------------------------------------------------------------------
# Weld points
# -----------------------------------------------------------------------------

def weld_points_for_group(centres, part_of, g, R=9.0, offset=2.5):
    """
    Return two weld points per cell in group g.
    They are placed along the vector from cell to group-centroid, with ±offset.
    """
    if isinstance(part_of, list):
        part_of = {i: part_of[i] for i in range(len(part_of))}
    mem = [i for i,p in part_of.items() if p == g]
    if not mem:
        return []

    Xg = centres[mem]
    cg = Xg.mean(axis=0)

    weld_pts = []
    for i in mem:
        p = centres[i]
        v = cg - p
        n = np.linalg.norm(v)
        if n < 1e-6:
            # degenerate: pick any fixed direction
            v = np.array([1.0, 0.0])
        else:
            v = v / n
        # place two points perpendicular to v, at radius inside the cap
        t = np.array([-v[1], v[0]])
        w1 = p + t * offset
        w2 = p - t * offset
        weld_pts.append((w1, w2))
    return weld_pts

def weld_points_all(G, centres, part_of, S, R=9.0, offset=2.5):
    """dict {g -> [ (p1,p2), ... ]} two weld points per cell."""
    return {g: weld_points_for_group(centres, part_of, g, R, offset) for g in range(S)}



def plot_plates(poly, centres, part_of, S, plates, weld_pts=None, R=9.0, title="Nickel plates"):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(*poly.exterior.xy, color='k', lw=2)

    # draw cells faintly
    for i, (x,y) in enumerate(centres):
        ax.add_patch(plt.Circle((x, y), R, facecolor=(0.8,0.8,0.8,0.3), edgecolor='k', lw=0.3, zorder=1))

    # plates
    pal = [cm.tab20(i%20) for i in range(S)]
    for g in range(S):
        pg = plates.get(g)
        if pg is None or pg.is_empty:
            continue
        color = pal[g % len(pal)]
        xs, ys = pg.exterior.xy
        ax.fill(xs, ys, facecolor=color, alpha=0.45, edgecolor='k', lw=1.2, zorder=2)

    # weld points
    if weld_pts is not None:
        for g, pairs in weld_pts.items():
            for (p1, p2) in pairs:
                ax.plot([p1[0]], [p1[1]], 'k.', ms=3, zorder=3)
                ax.plot([p2[0]], [p2[1]], 'k.', ms=3, zorder=3)

    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title)
    plt.tight_layout(); plt.show()

def _union_discs(centres, idx, radius, res=32):
    parts = [Point(*centres[i]).buffer(radius, resolution=res) for i in idx]
    return unary_union(parts) if parts else None

def _largest_component(geom):
    if geom is None:
        return None
    geom = make_valid(geom)
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        return geom.geoms[int(np.argmax([p.area for p in geom.geoms]))]
    return None

def make_group_plate_disjoint(G, centres, part_of, g,
                              R=9.0, land=1.2, gap=2.0,
                              outline=None, res=48):
    """
    Build a nickel plate for group g that CANNOT overlap with plates of other groups.
    – plates are separated by ≥ gap (in mm),
    – the boundary keeps the same 'scalloped' look,
    – clipped to outline if provided.
    """
    if isinstance(part_of, list):
        part_of = {i: part_of[i] for i in range(len(part_of))}

    mem = [i for i,p in part_of.items() if p == g]
    others = [i for i,p in part_of.items() if p != g]
    if not mem:
        return None

    # union of this group's discs (welding land)
    blob_g   = _union_discs(centres, mem,    R + land,          res=res)
    # union of all other groups, dilated by half the corridor
    mask_not = _union_discs(centres, others, R + land + gap/2.0, res=res)

    plate = blob_g.difference(mask_not) if mask_not else blob_g
    if outline is not None:
        plate = plate.intersection(outline)

    # clean + keep the largest connected piece
    plate = _largest_component(plate)
    return plate


def smooth_plate(plate, r_open=0.8, r_close=0.4, gap=2.5, safety=0.15):
    """
    Round a nickel-plate polygon.

    r_open  : erosion then dilation (mm) – removes jaggedness, safe for clearance
    r_close : dilation then erosion (mm) – fillets outward corners
              (keep r_close <= gap/2 - safety)
    gap     : corridor enforced between plates in make_group_plate_disjoint
    safety  : final shrink to guarantee no re-touching after smoothing
    """
    if r_open > 0:
        plate = plate.buffer(-r_open, join_style=1, cap_style=1) \
                     .buffer(+r_open, join_style=1, cap_style=1)

    if r_close > 0:
        # keep r_close conservative with respect to the inter-group gap
        r_close = min(r_close, max(0.0, gap*0.5 - safety))
        if r_close > 0:
            plate = plate.buffer(+r_close, join_style=1, cap_style=1) \
                         .buffer(-r_close, join_style=1, cap_style=1)

    if safety > 0:
        plate = plate.buffer(-safety, join_style=1, cap_style=1)
    return plate