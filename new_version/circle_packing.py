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
from shapely.geometry import Point, Polygon, LineString
from shapely.prepared import prep
from shapely.ops import unary_union
from shapely.affinity import rotate
from shapely import maximum_inscribed_circle
import pyswarms as ps
import matplotlib.pyplot as plt

##############################################################################
# ------------- 0. Parse command-line & input JSON --------------------------
##############################################################################

def load_boundary(json_path: Path):
    data = json.loads(json_path.read_text())
    disegno = json.loads(data["disegnoJson"])["disegno"]
    verts = [(v["x"], v["y"]) for v in disegno["vertici"]]
    poly   = Polygon(verts).buffer(0)       # tidy self-intersections
    if not poly.is_valid:                   # still crooked? abort early
        raise ValueError("Boundary polygon is invalid")
    return poly, prep(poly)                 # <- return both


##############################################################################
# ------------- 1. Geometry helpers & feasibility checks --------------------
##############################################################################

R = 9.0  # mm, radius of an 18650 seen from the top

def circle_fits(prep_poly, poly, x, y, r=R) -> bool:
    """True if the closed disk radius r centred at (x,y) is wholly inside."""
    pt = Point(x, y)
    return prep_poly.contains(pt) and poly.exterior.distance(pt) >= r

def plot_layout(poly, centres, r=R, title="Layout"):
    fig, ax = plt.subplots(figsize=(6,6))
    # boundary
    ax.plot(*poly.exterior.xy, lw=2)
    # circles
    for x, y in centres:
        ax.add_patch(plt.Circle((x, y), r, fill=False))
    ax.set_aspect('equal')
    ax.set_title(f"{title} ({len(centres)} cells)")
    plt.show()


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


def hex_seed(poly, prep_poly, r=R):
    minx, miny, maxx, maxy = poly.bounds
    dx, dy = 2*r, np.sqrt(3)*r
    centres = []
    row, y = 0, miny
    while y <= maxy:
        x = minx + (row % 2) * r
        while x <= maxx:
            if circle_fits(prep_poly, poly, x, y, r):
                centres.append([x, y])
            x += dx
        y += dy
        row += 1
    return np.asarray(centres, float)

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
# ------------- 3. Fitness function for PSO ---------------------------------
##############################################################################

def pairwise_sq_dists(mat):
    """Squared Euclidean distances for all unordered pairs (vectorised)."""
    diff = mat[:, None, :] - mat[None, :, :]
    # upper triangular (k>i) mask avoids double-count
    iu = np.triu_indices(mat.shape[0], k=1)
    return np.square(diff[iu]).sum(axis=1)

def fitness_single(x_flat, poly,
                   seed_flat=None,        # ← default = None
                   λ_overlap=1.0, λ_outside=1.0, λ_move=0.5):
    """
    x_flat : 1-D view [x1,y1,x2,y2,…]
    seed_flat : the original hex-grid positions (same length) or None
    """
    pts = x_flat.reshape(-1, 2)

    # ---------- overlap penalty ------------------------------------------
    d2 = pairwise_sq_dists(pts)
    P_overlap = np.clip((2*R)**2 - d2, 0, None).sum()

    # ---------- outside-polygon penalty ----------------------------------
    P_out = sum((max(0.0, R-poly.exterior.distance(Point(x,y)))**2)
                for x, y in pts)

    # ---------- don’t-move penalty ---------------------------------------
    if seed_flat is not None and λ_move > 0:
        P_move = λ_move * np.square(x_flat - seed_flat).sum()
    else:
        P_move = 0.0

    return λ_overlap*P_overlap + λ_outside*P_out + P_move


def make_pso_f(poly, seed_flat=None):
    """Return a PySwarms-compatible objective."""
    def f(X):
        return np.array([fitness_single(x, poly, seed_flat) for x in X])
    return f


##############################################################################
# ------------- 4. Simple PSO global pass -----------------------------------
##############################################################################

def pso_refine(init_centres, poly, iters=800, particles=50):
    n = init_centres.shape[0]
    dim = 2*n

    seed_flat = init_centres.flatten()

    minx, miny, maxx, maxy = map(float, poly.bounds)
    lb_pair = np.array([minx, miny])
    ub_pair = np.array([maxx, maxy])
    lb = np.tile(lb_pair, n)          # shape (dim,) float64
    ub = np.tile(ub_pair, n)

    # make absolutely sure every seed coord is inside [lb, ub]
    init = init_centres.copy()
    init[:, 0] = np.clip(init[:, 0], minx, maxx)
    init[:, 1] = np.clip(init[:, 1], miny, maxy)

    swarm = ps.single.GlobalBestPSO(
        n_particles = particles,
        dimensions  = dim,
        options     = dict(c1=1.5, c2=2.0, w=0.9),
        bounds      = (lb, ub),
        init_pos    = np.tile(init.flatten(), (particles, 1)),
    )
    cost, pos = swarm.optimize(make_pso_f(poly, seed_flat), iters=iters, n_processes=None)
    return pos.reshape(-1, 2)


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
    # --------------------------------------------------- load outline ----
    poly, _ = load_boundary(Path(json_file))

    # 1 ─ aligned hex grid (finer phase scan gives 1-3 extra cells)
    centres = best_hex_seed_two_angles(poly, n_phase=16)
    print("hex grid :", len(centres))

    # pso refine (deleted for now, see below)
    #centres1 = pso_refine(centres0, poly)
    #print("PSO done")

    # 2 ─ first greedy insertion (fills obvious edge gaps)
    centres = greedy_insert(poly, centres, trials=1000, max_pass=6)
    print("after greedy :", len(centres))

    # 3 ─ local compaction (Python re-implementation of Zhou’s batch-BFGS)
    centres = batch_bfgs_compact(centres, R, poly, n_pass=4)
    print("after compaction :", len(centres))

    # 4 ─ second greedy pass (micron pockets now opened by compactor)
    centres = greedy_insert(poly, centres, trials=1000, max_pass=3)
    print("final count :", len(centres))

    centres = skeleton_insert(poly, centres, step=2.0)
    print("after skeleton :", len(centres))

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

    # 5 ─ save + preview
    np.savetxt(out_csv, centres, delimiter=",", header="x,y", comments="")
    print(f"Saved {centres.shape[0]} circle centres to {out_csv}")
    plot_layout(poly, centres, title="Optimised layout")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: pack_cells.py <input.json> [out.csv]")
        sys.exit(1)
    json_in = sys.argv[1]
    csv_out = sys.argv[2] if len(sys.argv) > 2 else "centres.csv"
    main(json_in, csv_out)