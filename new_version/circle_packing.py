#!/usr/bin/env python3
"""
Circle-packing in an arbitrary polygonal outline.
– fixed radius = 9 mm (18650 cell seen from the top)
– boundary is supplied in the JSON field      input["disegnoJson"]["disegno"]["vertici"]

Road-map layers implemented so far
----------------------------------
0  Boundary as a Shapely polygon (+ fast “circle-inside” test)
1  Castillo-style hard constraints in the fitness
2  Hex-grid seeding                     (section 2 of the roadmap)
3  Global exploration with PSO          (section 3)
4  ***HOOK*** for GBO local compaction  (section 4 – not yet coded)
5  ***HOOK*** for electrical penalties  (section 5)

Author: <you>
For using place yourself in the 2BD_project directory and type the command:
    python new_version/circle_packing.py battery_construction/input.json new_version/out.csv
"""

import json
import sys
import random
from pathlib import Path

import numpy as np
import math
from shapely.geometry import Point, Polygon, LineString, MultiPolygon
from shapely.prepared import prep
from shapely.ops import unary_union
from shapely.affinity import rotate
from shapely import maximum_inscribed_circle
import pyswarms as ps
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from time import perf_counter


R = 9.0  # mm, radius of an 18650 seen from the top


##############################################################################
# ------------- 0. Parse command-line & input JSON --------------------------
##############################################################################

def _detect_scale_px(disegno: dict):
    """
    Rileva i fattori di conversione px→mm e px→cm.
    Priorità:
      1) scale_mm_per_px (mm/px)   → px_to_mm
      2) scale_cm_per_px (cm/px)   → px_to_cm
      3) median(misure_mm / misure_px)
      4) median(misure_cm / misure_px)
    Ritorna: (px_to_mm, px_to_cm) oppure (None, None) se non ricavabile.
    """
    def _to_float_list(seq):
        out = []
        for v in (seq or []):
            try:
                out.append(float(v))
            except Exception:
                pass
        return out

    # 1) scala esplicita mm/px
    s_mm = disegno.get("scale_mm_per_px", None)
    try:
        s_mm = float(s_mm) if s_mm is not None else None
    except Exception:
        s_mm = None
    if s_mm and s_mm > 0:
        return float(s_mm), float(s_mm) / 10.0  # px→mm, px→cm

    # 2) scala esplicita cm/px
    s_cm = disegno.get("scale_cm_per_px", None)
    try:
        s_cm = float(s_cm) if s_cm is not None else None
    except Exception:
        s_cm = None
    if s_cm and s_cm > 0:
        return float(s_cm) * 10.0, float(s_cm)

    # 3) misure_mm / misure_px
    mm = _to_float_list(disegno.get("misure_mm"))
    px = _to_float_list(disegno.get("misure_px"))
    if mm and px and len(mm) == len(px):
        pairs = [(m, p) for m, p in zip(mm, px) if p and float(p) > 0]
        if pairs:
            ratios = [m/p for m, p in pairs]  # mm/px
            med = float(np.median(ratios))
            return med, med / 10.0

    # 4) misure_cm / misure_px
    cm = _to_float_list(disegno.get("misure_cm"))
    px = _to_float_list(disegno.get("misure_px"))
    if cm and px and len(cm) == len(px):
        pairs = [(c, p) for c, p in zip(cm, px) if p and float(p) > 0]
        if pairs:
            ratios = [c/p for c, p in pairs]  # cm/px
            med = float(np.median(ratios))
            return med * 10.0, med

    return None, None


def load_boundary(json_path, to_units="mm", require_scale=False):
    """
    Carica il poligono e converte da px alle unità richieste.
    Supporta scale in mm/px o cm/px (o deduzione da misure_mm / misure_cm).
    """
    data = json.loads(open(json_path, "r").read())
    disegno = json.loads(data["disegnoJson"])["disegno"]

    verts_px = [(float(v["x"]), float(v["y"])) for v in disegno["vertici"]]

    px_to_mm, px_to_cm = _detect_scale_px(disegno)
    if px_to_mm is None or px_to_cm is None:
        if require_scale:
            raise ValueError("Scala assente: né scale_mm_per_px/scale_cm_per_px né misure_mm/cm con misure_px.")
        # fallback retro-compatibile: 1 px ≈ 1 mm
        px_to_mm = 1.0
        px_to_cm = 0.1

    # conversione
    if to_units == "mm":
        factor = px_to_mm
    elif to_units == "cm":
        factor = px_to_cm
    elif to_units == "m":
        factor = px_to_cm / 100.0
    else:
        raise ValueError("to_units deve essere 'mm', 'cm' o 'm'.")

    verts = [(x*factor, y*factor) for (x, y) in verts_px]
    poly = Polygon(verts).buffer(0)
    if not poly.is_valid:
        raise ValueError("Boundary polygon is invalid")

    meta = {
        "scale_cm_per_px": px_to_cm,   # ora è il valore effettivo cm/px
        "px_to_cm": px_to_cm,
        "px_to_mm": px_to_mm,
        "units": to_units,
    }
    return poly, prep(poly), meta


##############################################################################
# ------------- 1. Geometry helpers & feasibility checks --------------------
##############################################################################




def feasible_centre_region(poly: Polygon, centres: np.ndarray, R: float):
    """Admissible locations for a *new centre* (distance ≥R from boundary
       and ≥2R from existing centres).  Useful to report 'free area'."""
    inner = poly.buffer(-R).buffer(0)
    if inner.is_empty:
        return inner
    if len(centres):
        discs2R = [Point(x, y).buffer(2.0*R) for x, y in centres]
        forbidden = unary_union(discs2R)
        region = inner.difference(forbidden)
    else:
        region = inner
    return region.buffer(0)

def _clearance(poly: Polygon, centres: np.ndarray, R: float, pt: Point) -> float:
    """r(pt) = min(dist to true polygon boundary, dist to any cell surface)."""
    d_wall = pt.distance(poly.boundary)
    if len(centres):
        C = np.asarray(centres)
        dx = C[:,0] - pt.x
        dy = C[:,1] - pt.y
        d_cells = np.sqrt(dx*dx + dy*dy).min() - R
    else:
        d_cells = float('inf')
    return max(0.0, min(d_wall, d_cells))

def _hill_climb(seed: Point, step0: float, region: Polygon, f, tol=1e-3, itmax=80):
    p = Point(seed.x, seed.y)
    r = f(p)
    step = step0
    dirs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    for _ in range(itmax):
        improved = False
        for dx, dy in dirs:
            q = Point(p.x + dx*step, p.y + dy*step)
            if region.contains(q):
                rq = f(q)
                if rq > r:
                    p, r = q, rq
                    improved = True
        if not improved:
            step *= 0.5
            if step < tol:
                break
    return r, p

def largest_clearance_circle(poly: Polygon, centres: np.ndarray, R: float,
                             grid_density=220, seeds_per_component=6):
    """
    Global search:
      – If feasible-centre region F = poly.buffer(-R) is non-empty:
          search in every connected component of F.
      – Else (no admissible centre): fall back to searching in the whole polygon.

    Returns (radius_mm, centre_point).
    """
    F = poly.buffer(-R).buffer(0)
    search_geom = F if not F.is_empty else poly

    # Prepare list of components to visit
    comps = [search_geom] if search_geom.geom_type == "Polygon" else list(search_geom.geoms)

    best_r, best_p = 0.0, None

    # closure: clearance uses the *true polygon* and the cells
    def f(pt: Point): return _clearance(poly, centres, R, pt)

    for comp in comps:
        minx, miny, maxx, maxy = comp.bounds
        L = max(maxx - minx, maxy - miny)
        # grid spacing: denser of {L/ grid_density, R/3}
        step = max(L / float(grid_density), R / 3.0)
        xs = np.arange(minx, maxx + step, step)
        ys = np.arange(miny, maxy + step, step)

        # coarse scan to pick seeds
        cand = []
        for y in ys:
            for x in xs:
                p = Point(x, y)
                if comp.contains(p):
                    cand.append((f(p), p))
        if not cand:
            continue
        cand.sort(key=lambda t: t[0], reverse=True)
        seeds = [p for _, p in cand[:max(3, seeds_per_component)]]
        # always add the representative point (inside component)
        seeds.append(comp.representative_point())

        # refine each seed
        for p0 in seeds:
            r0, p_star = _hill_climb(p0, step, comp, f)
            if r0 > best_r:
                best_r, best_p = r0, p_star

    return best_r, best_p






def add_shapely_patch(ax, geom, facecolor=(0,1,0,0.7), edgecolor="#00000087",
                      zorder=0):
    """
    Draw a (Multi)Polygon 'geom' as one or more Matplotlib patches.
    Handles holes by drawing them as white-filled polygons on top.
    """
    if geom.is_empty:
        return
    if geom.geom_type == 'Polygon':
        ext = np.asarray(geom.exterior.coords)
        ax.add_patch(MplPolygon(ext, closed=True, facecolor=facecolor,
                                edgecolor=edgecolor, zorder=zorder))
        # holes
        for ring in geom.interiors:
            hole = np.asarray(ring.coords)
            ax.add_patch(MplPolygon(hole, closed=True, facecolor='white',
                                    edgecolor='none', zorder=zorder+0.01))
    elif geom.geom_type == 'MultiPolygon':
        for g in geom.geoms:
            add_shapely_patch(ax, g, facecolor=facecolor,
                              edgecolor=edgecolor, zorder=zorder)

def plot_phase(poly, centres, added_idx=None, title="",
               r=9.0, time_s=None,
               color_all="#377eb8",    # blue
               color_added="#e41a1c",  # red
               edgecolor="k",
               show_residual=True, show_lec=True,
               show_feasible=True, feasible_fc=(0,1,0,0.18)):
    """
    As before, plus:
      show_feasible : draw feasible centre region (poly eroded by r minus 2r-disks)
      feasible_fc   : RGBA of the green fill, default ~18% opaque.
    """
    fig, ax = plt.subplots(figsize=(6,6))

    # (optional) draw feasible region first (so cells are on top)
    if show_feasible:
        F = feasible_centre_region(poly, centres, r)
        add_shapely_patch(ax, F, facecolor=feasible_fc, edgecolor='none', zorder=0)

    # boundary on top of the green fill
    ax.plot(*poly.exterior.xy, lw=2, color='k', zorder=2)

    # residual+LEC for info box (unchanged)
    free_area = None; lec_rad = None; lec_diam = None; lec_c = None
    if show_residual or show_lec:
        # Free area to report is the feasible region area
        F = feasible_centre_region(poly, centres, r)
        free_area = F.area
        # LEC centre must be admissible, circle touches true polygon/cells
        lec_rad, lec_c = largest_clearance_circle(poly, centres, r)
        lec_diam = 2*lec_rad if lec_rad else 0.0

    # plot cells
    added_idx = np.array(added_idx) if added_idx is not None else np.array([], int)
    added_set = set(added_idx.tolist())
    for i, (x, y) in enumerate(centres):
        fc = color_added if i in added_set else color_all
        ax.add_patch(plt.Circle((x, y), r, facecolor=fc, edgecolor=edgecolor,
                                lw=0.5, alpha=0.9, zorder=3))

    # draw LEC
    if show_lec and lec_rad and lec_c is not None and lec_rad > 0.0:
        ax.add_patch(plt.Circle((lec_c.x, lec_c.y), lec_rad,
                                fill=False, edgecolor='green', lw=1.5,
                                linestyle='--', zorder=4))
        ax.plot(lec_c.x, lec_c.y, 'g+', ms=10, mew=2, zorder=4)

    # info box
    info = (f"Total = {len(centres)}"
            f"\nAdded this phase = {len(added_idx)}"
            + (f"\nTime = {time_s:.3f} s" if time_s is not None else "")
            + (f"\nFree area = {free_area:.1f} mm²" if free_area is not None else "")
            + (f"\nLEC radius = {lec_rad:.2f} mm\nLEC diameter = {lec_diam:.2f} mm"
               if lec_rad is not None else ""))
    ax.text(0.02, 0.98, info, transform=ax.transAxes, ha='left', va='top',
            fontsize=10, bbox=dict(boxstyle='round,pad=0.4',
            fc='white', ec='0.3', alpha=0.9), zorder=5)

    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title)
    plt.tight_layout(); plt.show()

##############################################################################
# ------------- 2. Hexagonal grid seeding -----------------------------------
##############################################################################

DX, DY = 2*R, math.sqrt(3)*R          # hex lattice periods

def longest_edge_angle(poly: Polygon) -> float:
    """Return angle (radians) of the longest edge of the polygon."""
    xs, ys = poly.exterior.xy
    pts    = list(zip(xs, ys))
    edges  = [(pts[i], pts[i+1]) for i in range(len(pts)-1)]
    (x1,y1), (x2,y2) = max(edges,
                           key=lambda e: math.hypot(e[1][0]-e[0][0],
                                                    e[1][1]-e[0][1]))
    return math.atan2(y2-y1, x2-x1)     # radians

def _hex_grid_in_rot(rot_poly, prep_rot, off_x, off_y, r=R):
    """Return centres inside rot_poly for one particular phase shift."""
    minx, miny, maxx, maxy = rot_poly.bounds
    centres = []
    y = miny + off_y
    row = 0
    while y <= maxy:
        x = minx + ((row % 2)*r) + off_x
        while x <= maxx:
            pt = Point(x, y)
            if prep_rot.contains(pt) and rot_poly.exterior.distance(pt) >= r:
                centres.append((x, y))
            x += DX
        y += DY
        row += 1
    return np.asarray(centres, float)

def oriented_hex_seed(poly: Polygon, r=R, n_phase=8):
    """
    1) rotate polygon so its longest edge is horizontal
    2) try n_phase×n_phase phase shifts (off_x, off_y) in [0,DX)×[0,DY/2)
    3) keep the densest seed, then rotate centres back
    """
    ang = longest_edge_angle(poly)                # radians
    rot_poly  = rotate(poly, -math.degrees(ang), origin='centroid')
    prep_rot  = prep(rot_poly)

    # sample offsets over one period (DY/2 because rows are staggered)
    phase_x = np.linspace(-DX, DX,  2*n_phase, endpoint=False)
    phase_y = np.linspace(-DY/2, DY/2, 2*n_phase, endpoint=False)

    best, best_cnt = None, -1
    for ox in phase_x:
        for oy in phase_y:
            C = _hex_grid_in_rot(rot_poly, prep_rot, ox, oy, r)
            if C.shape[0] > best_cnt:
                best, best_cnt = C, C.shape[0]

    # rotate centres back to the original frame
    cx, cy = poly.centroid.xy[0][0], poly.centroid.xy[1][0]
    c, s = math.cos(ang), math.sin(ang)
    rot_mat = np.array([[c, -s],
                        [s,  c]])
    centres_orig = ((best - [cx, cy]) @ rot_mat.T) + [cx, cy]
    return centres_orig




def best_hex_seed_two_angles(poly, n_phase=16):
    """Try the default orientation and a +30° rotated one, keep denser."""
    # --- 1  longest-edge orientation (what you already have)
    base = oriented_hex_seed(poly, n_phase=n_phase)

    # --- 2  +30° rotated outline ----------------------------------------
    ang_deg = 30.0
    poly_rot = rotate(poly, ang_deg, origin='centroid')     # shapely rotate
    alt_rot  = oriented_hex_seed(poly_rot, n_phase=n_phase) # build grid there

    # rotate the centres *back* by −30° (plain math, not shapely)
    theta = math.radians(-ang_deg)
    c, s  = math.cos(theta), math.sin(theta)
    rotM  = np.array([[c, -s],
                      [s,  c]])
    cx, cy = poly.centroid.xy[0][0], poly.centroid.xy[1][0]
    alt = ((alt_rot - [cx, cy]) @ rotM.T) + [cx, cy]

    return base if len(base) >= len(alt) else alt



##############################################################################
# ------------- 5. GBO ------------------------------------------------------
##############################################################################

def _local_grad_xy(xy, all_xy, r, poly):
    """∂/∂x,∂/∂y of overlap+boundary penalties for one point."""
    gx = gy = 0.0
    for xi,yi in all_xy:
        if xi == xy[0] and yi == xy[1]:                    # skip self
            continue
        dx = xy[0]-xi; dy = xy[1]-yi
        d2 = dx*dx+dy*dy
        if d2 < (2*r)**2-1e-6:
            # overlap gradient
            g = -2*((2*r)**2-d2)
            gx += g*dx; gy += g*dy
    # boundary gradient: push inward if too close
    p = Point(*xy)
    dist = poly.exterior.distance(p)
    if dist < r:
        # unit normal approximation: vector from closest boundary pt
        bx,by = poly.exterior.interpolate(
                 poly.exterior.project(p)).coords[0]
        nx,ny = (xy[0]-bx, xy[1]-by)
        norm   = math.hypot(nx,ny) or 1.0
        gx += -2*(r-dist)*nx/norm
        gy += -2*(r-dist)*ny/norm
    return gx, gy

def batch_bfgs_compact(centres, r, poly, n_pass=5):
    """
    centres : (N,2) numpy array
    batches : implicit → one hex row = one batch
    """
    # 1 – assign each circle to a 'row' by rounding its projection
    seed_ang = longest_edge_angle(poly)              # radians
    c, s      = math.cos(-seed_ang), math.sin(-seed_ang)
    rot       = np.array([[c, -s], [s,  c]])
    proj      = (centres @ rot.T)[:, 1]              # y-coordinate in that frame
    row_id    = np.round(proj / (r * math.sqrt(3))).astype(int)
    rows   = {rid:np.where(row_id==rid)[0] for rid in np.unique(row_id)}

    C = centres.copy()
    for _ in range(n_pass):
        for idx in rows.values():
            # small L-BFGS-B via SciPy or a simple gradient step
            for it in range(30):
                g = np.zeros((len(idx),2))
                for k,i in enumerate(idx):
                    g[k] = _local_grad_xy(C[i], C, r, poly)
                step = -0.02*g                          # fixed step size
                C[idx] += step
    return C

##############################################################################
# ------------- 6. Greedy Insert ---------------------------------------------
##############################################################################

# --------------------------------------------------------------------
# Greedy random insertion of additional circles
# --------------------------------------------------------------------
def greedy_insert(poly, centres, trials=1000, max_pass=6):
    """
    Adds extra circles by random sampling strictly INSIDE `poly`.
    Continues until one complete pass fails to add a circle, or
    `max_pass` passes have been performed.
    """
    ppoly = prep(poly)
    pts   = list(map(tuple, centres))           # mutable working list
    minx, miny, maxx, maxy = poly.bounds

    for _ in range(max_pass):
        added_this_pass = 0

        for _ in range(trials):
            # -- 1. rejection sample an interior point ------------------
            for __ in range(60):                # ≤ 60 attempts
                x = random.uniform(minx, maxx)
                y = random.uniform(miny, maxy)
                if ppoly.contains(Point(x, y)):
                    break                       # got an interior point
            else:
                continue                        # 60 failures → new trial

            # -- 2. clearance checks -----------------------------------
            d_cells = min(np.hypot(x-nx, y-ny) for nx, ny in pts)
            d_wall  = poly.exterior.distance(Point(x, y))

            if d_cells >= 2*R and d_wall >= R:  # fits!
                pts.append((x, y))
                added_this_pass += 1

        if added_this_pass == 0:                # nothing added → done
            break

    return np.asarray(pts, float)

def skeleton_insert(poly, centres, r=R, step=2.0):
    """
    Walk the medial axis (approx.) of the residual space and
    drop disks wherever there is 2R clearance.
    """
    # 1. residual pocket
    free     = poly.buffer(-r).buffer(0)
    hull     = unary_union([Point(x, y).buffer(r) for x, y in centres])
    pocket   = free.difference(hull)
    if pocket.is_empty:
        return centres                        # already saturated

    # 2. sample regularly on the pocket’s bounding box
    minx, miny, maxx, maxy = pocket.bounds
    grid_x = np.arange(minx, maxx + step, step)
    grid_y = np.arange(miny, maxy + step, step)

    pts     = list(map(tuple, centres))
    ppoly   = prep(poly)

    for y in grid_y:
        for x in grid_x:
            if not ppoly.contains(Point(x, y)):
                continue
            d_cells = min(np.hypot(x-nx, y-ny) for nx, ny in pts)
            d_wall  = poly.exterior.distance(Point(x, y))
            if d_cells >= 2*r and d_wall >= r:
                pts.append((x, y))

    return np.asarray(pts, float)

def largest_empty_circle(pocket):
    """Return (radius_mm, centre_point) or (0, None) if none."""
    if pocket.is_empty:
        return 0.0, None

    circ = maximum_inscribed_circle(pocket)

    if isinstance(circ, Polygon):          # “normal” case
        centre = circ.centroid
        radius = math.sqrt(circ.area / math.pi)
        return radius, centre

    elif isinstance(circ, Point):          # degenerated to a single point
        return 0.0, circ

    elif isinstance(circ, LineString):     # only a slit → radius = half width
        radius = circ.length / 2.0
        centre = circ.interpolate(0.5, normalized=True)
        return radius, centre

    else:                                  # should not happen
        return 0.0, None

##############################################################################
# ------------- 7. Driver ----------------------------------------------------
##############################################################################

    

def main(json_file: str, out_csv="centres.csv"):
    # 0) Load boundary
    poly, _ = load_boundary(Path(json_file))  # your JSON path

    # 1) Hex grid
    t0 = perf_counter()
    centres = best_hex_seed_two_angles(poly, n_phase=16)
    t1 = perf_counter()
    plot_phase(poly, centres, added_idx=np.arange(len(centres)),
            title="After hex grid", r=R, time_s=t1 - t0)

    # 2) First greedy
    prev = len(centres)
    t0 = perf_counter()
    centres = greedy_insert(poly, centres, trials=1000, max_pass=6)
    t1 = perf_counter()
    plot_phase(poly, centres, added_idx=np.arange(prev, len(centres)),
            title="After first greedy", r=R, time_s=t1 - t0)

    # 3) Local compaction (no new cells)
    t0 = perf_counter()
    centres = batch_bfgs_compact(centres, R, poly, n_pass=4)
    t1 = perf_counter()
    plot_phase(poly, centres, added_idx=[],
            title="After compaction", r=R, time_s=t1 - t0)

    # 4) Second greedy
    prev = len(centres)
    t0 = perf_counter()
    centres = greedy_insert(poly, centres, trials=500, max_pass=3)
    t1 = perf_counter()
    plot_phase(poly, centres, added_idx=np.arange(prev, len(centres)),
            title="After second greedy", r=R, time_s=t1 - t0)

    # 5) Skeleton insert
    prev = len(centres)
    t0 = perf_counter()
    centres = skeleton_insert(poly, centres, step=2.0)
    t1 = perf_counter()
    plot_phase(poly, centres, added_idx=np.arange(prev, len(centres)),
            title="After skeleton insert", r=R, time_s=t1 - t0)

    # Optional: Save final CSV
    np.savetxt(out_csv, centres, delimiter=",", header="x,y", comments="")
    print(f"Saved {centres.shape[0]} circle centres to {out_csv}")

    # ───▶  DIAGNOSTIC POCKET ANALYSIS  ◀─────────────────────────────────────
    free     = poly.buffer(-R).buffer(0)            # safe interior strip
    hull     = unary_union([Point(x, y).buffer(R) for x, y in centres])
    residual = free.difference(hull)

    print(f"Area that could still host a centre: {residual.area:.1f} mm²")

    # Optional: largest empty circle (Shapely≥2.0 only)
    try:
        from shapely import maximum_inscribed_circle

        rad, centre_pt = largest_empty_circle(residual)
        if rad > 0:
            print(f"Largest empty circle radius: {rad:.2f} mm "
                f"({2*rad:.2f} mm diameter)")
        else:
            print("Residual pocket too thin for even a tiny extra cell.")
    except ImportError:
        pass
    # ─────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: pack_cells.py <input.json> [out.csv]")
        sys.exit(1)
    json_in = sys.argv[1]
    csv_out = sys.argv[2] if len(sys.argv) > 2 else "centres.csv"
    main(json_in, csv_out)