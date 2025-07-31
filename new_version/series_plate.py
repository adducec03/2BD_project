# series_plates.py
from __future__ import annotations
import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon, LineString, MultiLineString
from shapely.ops import unary_union
from typing import Iterable, Dict, Tuple, List
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import networkx as nx



def plot_series_plates_over_cells(poly, centres, part_of, S, bridges,
                                  R=9.0, cell_alpha=0.25, plate_fc="#DAA520"):
    """
    Draw cells (lightly) and overlay the series plates returned by make_series_bridge_plates.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(*poly.exterior.xy, color='k', lw=2)

    # light cells
    palette = [cm.tab20(i % 20) for i in range(S)]
    for i, (x, y) in enumerate(centres):
        g = part_of[i]
        ax.add_patch(plt.Circle((x, y), R, facecolor=palette[g], edgecolor='k', lw=0.3, alpha=cell_alpha))

    # series plates
    for (a, b), geom in bridges:
        for plate in _iter_polys(geom):
            xs, ys = plate.exterior.xy
            ax.fill(xs, ys, color=plate_fc, alpha=0.95, zorder=5, ec='k', lw=0.8)

    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(f"Series connection plates (S={S})")
    plt.tight_layout(); plt.show()

# --------------------------------------------------------------
# Helpers: per-group unions of cell discs
# --------------------------------------------------------------
def _group_cell_unions(centres: np.ndarray, part_of: Dict[int,int],
                       R: float, inset: float) -> Dict[int, Polygon|MultiPolygon]:
    """
    Return {g : union of (R-inset) discs for group g}.
    The small 'inset' ensures a bit of lip/clearance from cell edges.
    """
    discs = {}
    r = max(0.0, R - inset)
    for i, (x,y) in enumerate(centres):
        g = part_of[i]
        discs.setdefault(g, []).append(Point(x,y).buffer(r))
    unions = {g: unary_union(dlist) for g, dlist in discs.items()}
    return unions

def _iter_polys(geom):
    """Yield Polygon(s) from Polygon/MultiPolygon/empty."""
    if geom.is_empty:
        return
    if isinstance(geom, Polygon):
        yield geom
    elif isinstance(geom, MultiPolygon):
        for p in geom.geoms:
            yield p

# --------------------------------------------------------------
# Main: make series "band" plates between consecutive groups
# --------------------------------------------------------------
def make_series_bridge_plates(poly,
                              centres: np.ndarray,
                              part_of: Dict[int,int],
                              group_edges: Iterable[Tuple[int,int]],
                              R: float = 9.0,
                              band: float = 3.0,
                              inset: float = 0.6,
                              smooth: float = 1.2) -> List[Tuple[Tuple[int,int], Polygon|MultiPolygon]]:
    """
    Build **plates** (not lines) that connect each adjacent pair (a,b).

    For each pair:
        1) create the union of (R - inset) discs for group a and for group b,
        2) take their *overlap strip* (the interface),
        3) thicken it by 'band' on both sides (so it enters each group),
        4) keep only the portion inside (union_a ∪ union_b),
        5) optionally smooth with buffer(+smooth).buffer(-smooth).

    Returns: list [ ((a,b), geom), ... ] where geom is Polygon/MultiPolygon.
    """
    # Cache unions once
    unions = _group_cell_unions(centres, part_of, R=R, inset=inset)

    plates = []
    for a, b in group_edges:
        Ua = unions[a]; Ub = unions[b]

        # Area where the two group hulls actually "meet" (or nearly do)
        interface = Ua.buffer(0).intersection(Ub.buffer(0))
        if interface.is_empty:
            # If they do not overlap at all, take the *gap line* between them
            # and buffer it. (This happens when the two packs are very close but
            # not touching.)  Create a narrow corridor centered between them.
            corridor = Ua.buffer(band).intersection(Ub.buffer(band))
        else:
            # We do have an overlap region; use it as a centerline support.
            corridor = interface.buffer(band)

        # Keep the corridor only inside A∪B, so we never touch other groups
        plate = corridor.intersection(Ua.union(Ub))

        # Optional smoothing (nicely rounded edges)
        if smooth > 0:
            plate = plate.buffer(+smooth, join_style=1, cap_style=1) \
                         .buffer(-smooth, join_style=1, cap_style=1)

        # Clip to the battery outline (optional, if you have `poly`)
        if poly is not None and hasattr(poly, "intersection"):
            plate = plate.intersection(poly)

        if not plate.is_empty:
            plates.append(((a, b), plate))

    return plates

def _iter_polys(geom):
    if geom.is_empty:
        return
    if isinstance(geom, Polygon):
        yield geom
    elif isinstance(geom, MultiPolygon):
        for p in geom.geoms:
            yield p

def _expand_within_group(G: nx.Graph, seeds: set[int], allowed: set[int], rows: int) -> set[int]:
    """Multi-source BFS inside 'allowed' nodes for at most 'rows' steps."""
    if rows <= 0:
        return set(seeds)
    seen = set(seeds)
    frontier = set(seeds)
    for _ in range(rows):
        nxt = set()
        for v in frontier:
            for nb in G[v]:
                if nb in allowed and nb not in seen:
                    nxt.add(nb); seen.add(nb)
        if not nxt:
            break
        frontier = nxt
    return seen

def make_series_band_from_cells(G: nx.Graph,
                                centres: np.ndarray,
                                part_of: Dict[int, int],
                                pairs: Iterable[Tuple[int, int]],
                                R: float = 9.0,
                                rows: int = 2,          # how many cell rows to include from each side
                                inset: float = 0.6,     # shrink each disc → clearance to cell edges
                                smooth: float = 1.2,    # rounding radius
                                clip_poly: Polygon | None = None
                               ) -> List[Tuple[Tuple[int, int], Polygon | MultiPolygon]]:
    """
    Make **series plates** between consecutive groups.
    A plate is the union of discs of those cells that lie on the boundary between
    the two groups, grown 'rows' steps inside each group.
    """
    # cache members by group
    by_g: Dict[int, List[int]] = {}
    for i, g in part_of.items():
        by_g.setdefault(g, []).append(i)

    plates = []
    for a, b in pairs:
        A = set(by_g[a]); B = set(by_g[b])

        # boundary cells (have a neighbor in the other group)
        seed_a = {i for i in A if any(nb in B for nb in G[i])}
        seed_b = {j for j in B if any(nb in A for nb in G[j])}

        # expand inside each group
        band_a = _expand_within_group(G, seed_a, A, rows)
        band_b = _expand_within_group(G, seed_b, B, rows)

        chosen = list(band_a | band_b)
        r = max(0.001, R - inset)

        # union of discs
        geom = unary_union([Point(*centres[i]).buffer(r) for i in chosen])

        # optional rounding
        if smooth > 0:
            geom = geom.buffer(+smooth, join_style=1, cap_style=1) \
                       .buffer(-smooth, join_style=1, cap_style=1)

        if clip_poly is not None:
            geom = geom.intersection(clip_poly)

        if not geom.is_empty:
            plates.append(((a, b), geom))

    return plates

def plot_series_plates_over_cells(poly, centres, part_of, S, bridges,
                                  R=9.0, cell_alpha=0.25, plate_fc="#DAA520"):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(*poly.exterior.xy, color='k', lw=2)

    palette = [cm.tab20(i % 20) for i in range(S)]
    for i, (x, y) in enumerate(centres):
        g = part_of[i]
        ax.add_patch(plt.Circle((x, y), R, facecolor=palette[g],
                                edgecolor='k', lw=0.3, alpha=cell_alpha))

    for (a, b), geom in bridges:
        for poly in _iter_polys(geom):
            xs, ys = poly.exterior.xy
            ax.fill(xs, ys, color=plate_fc, alpha=0.95, zorder=5, ec='k', lw=0.8)

    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(f"Series connection plates (S={S})")
    plt.tight_layout(); plt.show()