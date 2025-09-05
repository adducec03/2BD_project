import numpy as np
from typing import Dict, List, Tuple
from math import isqrt
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.patches import Circle


# ------------------------------------------------------------
# helpers: turn a flat "cells" vector into a rows×cols matrix
# ------------------------------------------------------------
def cells_to_grid(cells: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """
    cells : array of length rows*cols containing the *cell ids* in row-major order.
    Returns a (rows, cols) array with the ids laid out as a grid.
    """
    assert len(cells) == rows*cols
    return cells.reshape(rows, cols)


# ------------------------------------------------------------
# PLOTTING
# ------------------------------------------------------------


# ------------------------------------------------------------


def plot_groups_blockwise(centres, part_of, poly,
                          R=9.0, S=None, title=None,
                          face_unassigned=(1,1,1,1), edge_unassigned='0.6',
                          show_labels=True,
                          label_unassigned=False,
                          fontsize=6):
    """
    centres : ndarray (N,2)
    part_of : dict {cell_id -> group_id}
              Use negative ids (e.g. -1) or missing keys for unassigned cells.
    poly    : shapely polygon
    """

    # ---- palette: EXCLUDE negative ids (treat them as unassigned)
    used_groups = sorted(g for g in set(part_of.values())
                         if isinstance(g, (int, np.integer)) and g >= 0)

    if S is None:
        S = max(len(used_groups), 1)
    try:
        import colorcet as cc
        base = [mpl.colors.to_rgb(c) for c in cc.glasbey[:max(S, len(used_groups))]]
    except Exception:
        base = [cm.tab20(i % 20) for i in range(max(S, len(used_groups)))]

    group_color = {g: base[g % len(base)] for g in used_groups}

    def _text_color(rgb):
        if isinstance(rgb, str):
            rgb = mpl.colors.to_rgb(rgb)
        L = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
        return 'black' if L >= 0.6 else 'white'

    # ---- plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(*poly.exterior.xy, color='k', lw=2)

    N = len(centres)
    assigned = 0
    for i, (x, y) in enumerate(centres):
        g = part_of.get(i, None)

        # Treat None or any negative id as unassigned  <<< KEY CHANGE
        if g is None or (isinstance(g, (int, np.integer)) and g < 0):
            fc = face_unassigned
            ec = edge_unassigned
            txt = (str(g) if label_unassigned and g is not None else "")
        else:
            assigned += 1
            fc = group_color.get(g, base[g % len(base)])
            ec = 'k'
            txt = str(g)

        ax.add_patch(Circle((x, y), R, facecolor=fc, edgecolor=ec, lw=0.4, zorder=2))
        if show_labels and txt != "":
            ax.text(x, y, txt, ha='center', va='center',
                    fontsize=fontsize, color=_text_color(fc), zorder=3)

    ax.set_aspect('equal'); ax.axis('off')
    title = title or "Blockwise grouping"
    ax.set_title(f"{title}   (assigned {assigned}/{N} cells)")
    plt.tight_layout(); plt.show()


# ------------------------------------------------------------
# 1) Build row structure from centres (no need to tell rows/cols)
# ------------------------------------------------------------
def build_rows_from_centres(centres, R, tol=None):
    """
    Return a list of rows; each row is a list of ORIGINAL indices,
    sorted left→right. Rows are determined by quantising y with a
    tolerance 'tol' (default = R*0.6).
    """
    if tol is None:
        tol = R*0.6
    idx = np.arange(len(centres))
    xs  = centres[:,0]
    ys  = centres[:,1]
    # quantise y into bands of height 'tol'
    y_sorted = np.argsort(ys)
    bands = []
    band = [ y_sorted[0] ]
    y0 = ys[y_sorted[0]]
    for k in y_sorted[1:]:
        if abs(ys[k]-y0) <= tol:
            band.append(k)
        else:
            bands.append(band)
            band = [k]
            y0   = ys[k]
    bands.append(band)
    # sort each band left→right
    rows = [list(sorted(b, key=lambda i: xs[i])) for b in bands]
    return rows

# ------------------------------------------------------------
# YOUR calc_parameters(y) (already implemented by you)
# here only a placeholder; keep the one you already have.
# ------------------------------------------------------------
def calc_parameters(y: int) -> Tuple[bool, Optional[int], Optional[int], Optional[int]]:
    """
    Dato y ≥ 0, trova tutte le terne (a,b,c) con a,b,c ≥ 0 tali che:
        y = (a+b)*c/2
        a - 1 <= b <= a
        a <= ceil(y/2)
        c <= a

    Restituisce la lista di soluzioni ordinate per:
        1) |a-c| crescente
        2) b crescente
        3) a crescente
    """
    if not isinstance(y, int):
        raise TypeError("y deve essere un intero")
    if y < 0:
        raise ValueError("y deve essere ≥ 0")

    if y == 0:
        return [(0, 0, 0)]

    n = 2 * y
    a_max = (y + 1) // 2  # ceil(y/2)

    sol_set = set()  # elimina i duplicati

    for c in range(1, isqrt(n) + 1):
        if n % c != 0:
            continue
        s = n // c

        # prova con (s, c)
        for a in (s // 2, (s + 1) // 2):
            b = s - a
            if a - 1 <= b <= a and a <= a_max and b >= 0 and c <= a:
                sol_set.add((a, b, c))

        # prova con (c, s) invertiti
        if s != c:
            c2, s2 = s, c
            for a in (s2 // 2, (s2 + 1) // 2):
                b = s2 - a
                if a - 1 <= b <= a and a <= a_max and b >= 0 and c2 <= a:
                    sol_set.add((a, b, c2))

    sol = sorted(sol_set, key=lambda t: (abs(t[0] - t[2]), t[1], t[0]))
    return sol

def verify_rls(r: int, l: int, s: int) -> Tuple[bool, Optional[int], Optional[int]]:
    """
    Verifica se esistono interi n,m tali che:
        r = n*l + m*s
    con vincoli:
        - m pari
        - n = m + 1 (quindi n dispari)

    Restituisce:
        (True, n, m) se esiste
        (False, None, None) altrimenti
    """
    if not all(isinstance(x, int) for x in (r, l, s)):
        raise TypeError("r, l, s devono essere interi")

    if l + s == 0:
        # Caso speciale: se l+s=0 ⇒ r deve essere = l
        if r == l:
            return (False, None, None)  # indeterminato, m qualsiasi (parità non gestita)
        else:
            return (False, None, None)

    num = r - l
    den = l + s
    if num % den != 0:
        return (False, None, None)

    m = num // den
    if m % 2 == 0:  # m pari
        n = m + 1  # automaticamente dispari
        return (True, n, m)

    return (False, None, None)

def verify_full_first_long_row(r: int, l: int, s: int) -> Tuple[bool, Optional[int], Optional[int]]:
    """
    Verifica se esistono interi m,n tali che:
        r = m*l + n*s
        |m - n| <= 1
    Restituisce (True, m, n) se esiste, altrimenti (False, None, None).
    Preferisce la soluzione con |m-n|=0, poi quelle con |m-n|=1.
    """
    if not all(isinstance(x, int) for x in (r, l, s)):
        raise TypeError("r, l, s devono essere interi")

    denom = l + s

    if denom != 0:
        # Caso |m-n|=0: m = n = k
        num0 = r
        if num0 % denom == 0:
            k = num0 // denom
            return True, k, k

        # Caso m = n+1: m = k+1, n = k
        num1 = r - l
        if num1 % denom == 0:
            k = num1 // denom
            return True, k + 1, k

        # Caso n = m+1: m = k, n = k+1
        num2 = r - s
        if num2 % denom == 0:
            k = num2 // denom
            return True, k, k + 1

        return False, None, None

    else:
        # denom = 0  => s = -l
        # r = m*l + n*(-l) = (m-n) * l
        # Possibilità:
        # - m=n  => r=0
        # - m=n+1 => r=l
        # - n=m+1 => r=s (= -l)
        if r == 0:
            return True, 0, 0  # m=n (|m-n|=0)
        if r == l:
            return True, 1, 0  # |m-n|=1
        if r == s:
            return True, 0, 1  # |m-n|=1
        return False, None, None

# ------------------------------------------------------------
# main grouping routine
# ------------------------------------------------------------
def group_xSyP_strict_reading(
    rows: List[List[int]],             # each row is a list of cell-ids (left→right)
    x: int,                            # number of series groups
    y: int                             # cells per group
) -> Tuple[Dict[int, int], List[List[int]]]:
    """
    Strict reading-order grouping that follows the pseudocode:

      • Start at top-left, read left→right within a row.
      • On an even row use (long, short) segments; on an odd row use (short, long).
      • If a segment does not fit in the remainder of the current row, move to the
        next row (left-most cell) and retry that same group.
      • Stop a group when it has exactly y cells.
      • After finishing a group, always move to the next row and start the next group
        under the row we just ended on.

    Returns
    -------
    part_of : {cell_id -> group_id}
    groups  : list of lists with the cell-ids for each group (length = x)
    """
    total = sum(len(r) for r in rows)
    if x * y > total:
        raise ValueError(f"Need {x*y} cells but only {total} exist")

    if(y<5):
        long=y
        short=0
        rpg=1
    else:
        configs=calc_parameters(y)
        long, short, rpg = configs[0]

    part_of: Dict[int, int]  = {}
    groups:  List[List[int]] = [[] for _ in range(x)]

    r = 0  # cursor: row and column within current row
    t = 0

    def row_len(rr: int) -> int:
        return len(rows[rr]) if 0 <= rr < len(rows) else 0
    
    def find_another_config(configs: List[Tuple[int, int, int]], target_c: int) -> Optional[Tuple[int, int, int]]:
        for t in configs:
            if t[2] == target_c:
                return t
        return None

    rem = row_len(r)
    gid = 0
    cid =0
    a=long
    b=short
    added_cells=0

    start_from_short=False
    make_all_rows_equal_size=False
    using_alternative=False

    minimum_row_size = row_len(0)

    n_rows=len(rows)+1
    

    if (row_len(0)!=row_len(1)):

        if (row_len(0)<row_len(1) and verify_rls(row_len(0),long,short)[0] and long!=short):
            print("start_from_short")
            start_from_short=True
        else:
            if(verify_full_first_long_row(row_len(0),long,short)[0]):
                    print("make_al_rows_equal_size")
                    make_all_rows_equal_size=True
                    minimum_row_size=min(row_len(0),row_len(1))
                    rem=minimum_row_size



    while added_cells<(x*y)-1:
        c=rpg
        n = 0

        while c > 0:
            
            # set the pattern for *this* row
            a = long if (r % 2 == 0) else short
            b = short if (r % 2 == 0) else long
            if(start_from_short):
                temp=a
                a=b
                b=temp

            print(f"row{r}, remaining cells for each block:{c}, a:{a}, b:{b}")

            # 1) try to place 'a' cells in this row
            if rem >= a and gid+n<x:
                print(f"still space to place a={a} cells (remaining space={rem})")
                j=0
                for j in range(a):
                    print(f"cid:{cid}, gid:{gid}, current group:{gid+n}")
                    part_of[cid] = gid+n
                    groups[gid+n].append(cid)
                    rem-=1
                    cid+=1
                    added_cells+=1
                    if added_cells >= x * y:
                        return part_of, groups
                n+=1
                print(f"{a} cell placed (rem:{rem}, n:{n}, cid:{cid})")
            else:
                if(rem < a):
                    print(f"not enough space, change row")
                elif(gid+n<x):
                    print (f"reached the groups limit current group:{gid+n}")
                    continue
                else:
                    print("unknow stopping")

                # not enough space → move to next row, same group
                if (make_all_rows_equal_size):
                    if (row_len(r)>minimum_row_size):
                        cid+=rem+1
                        r+=1
                        rem=row_len(r)
                    else:
                        cid+=rem
                        r+=1
                        rem=minimum_row_size
                else:
                    cid+=rem
                    r+=1
                    rem=row_len(r)
                c-=1
                t+=n
                n=0
                
                print(f"rpg:{rpg}, n_rows:{n_rows-1}, r:{r} n_rows-r-1:{n_rows-r-1}")
                print(f"rpg>n_rows-r-1:{rpg>n_rows-r-1}")
                print(f"is already using alterntive:{using_alternative}")
                print(f"c:{c}")
                print(f"rpg:{rpg}")
                print(f"enters in the alternative configuration if: {rpg>n_rows-r-1 and using_alternative==False and c==0}")
                if (c == 0 and rpg == n_rows-r-1):
                    continue
                if(rpg>n_rows-r-1 and using_alternative==False and c==0):
                    print(f"remaining_rows:{n_rows-r}")
                    target_c=n_rows-r-1
                    alt_config=None
                    found=False
                    if(c==0 and rpg==n_rows-r-1):
                        break
                    while(target_c>0 and found==False):
                        alt_config=find_another_config(configs,target_c)
                        print(f"configurazione alternativa:{alt_config}")    
                        if (alt_config is not None):
                            gid = t // rpg
                            t = 0
                            long=alt_config[0]
                            short=alt_config[1]
                            rpg=alt_config[2]
                            c=rpg
                            found=True
                            using_alternative=True
                        else:
                            target_c-=1
                    if(target_c==0):
                        raise Exception(f"Impossibile formare {x} gruppi da {y} celle")
                
                print(f"r:{r},rem:{rem},c:{c},t:{t},n{n})")
                continue
  
                      
            # 2) try to place 'b' cells in this SAME row
            if rem >= b  and gid+n<x:
                print(f"still space to place b={b} cells (remaining space={rem})")
                j=0
                for j in range(b):
                    print(f"cid:{cid}, gid:{gid}, current group:{gid+n}")
                    part_of[cid] = gid+n
                    groups[gid+n].append(cid)
                    rem-=1
                    cid+=1
                    added_cells+=1
                    if added_cells >= x * y:
                        return part_of, groups

                n+=1
                print(f"{b} cell placed (rem:{rem}, n:{n}, cid:{cid})")
            else:
                # not enough space → move to next row, same group
                if(rem < b):
                    print(f"not enough space, change row")
                elif(gid+n>=x):
                    print (f"reached the groups limit current group:{gid+n}")
                    continue
                else:
                    print("unknow stopping")

                if (make_all_rows_equal_size):
                    if (row_len(r)>minimum_row_size):
                        cid+=rem+1
                        r+=1
                        rem=row_len(r)
                    else:
                        cid+=rem
                        r+=1
                        rem=minimum_row_size
                else:
                    cid+=rem
                    r+=1
                    rem=row_len(r)
                c-=1
                t+=n
                n=0
                
                print(f"rpg:{rpg}, n_rows:{n_rows-1}, r:{r} n_rows-r-1:{n_rows-r-1}")
                print(f"rpg>n_rows-r-1:{rpg>n_rows-r+1}")
                print(f"is already using alterntive:{using_alternative}")
                print(f"c:{c}")
                print(f"rpg:{rpg}")
                print(f"enters in the alternative configuration if: {rpg>n_rows-r-1 and using_alternative==False and c==0}")

                if (c == 0 and rpg == n_rows-r-1):
                    continue
                if(rpg>n_rows-r-1 and using_alternative==False and c==0):
                    print(f"remaining_rows:{n_rows-r}")
                    target_c=n_rows-r-1
                    alt_config=None
                    found=False
                    while(target_c>0 and found==False):
                        alt_config=find_another_config(configs,target_c)
                        print(f"configurazione alternativa:{alt_config}")                        
                        if (alt_config is not None):
                            gid = t // rpg
                            t = 0
                            long=alt_config[0]
                            short=alt_config[1]
                            rpg=alt_config[2]
                            c=rpg
                            found=True
                            using_alternative=True
                        else:
                            target_c-=1

                    if(target_c==0):
                        raise Exception(f"Impossibile formare {x} gruppi da {y} celle")
                    

                
                print(f"(r:{r},rem:{rem},c:{c},t:{t},n{n})")
                #break

        # group complete: always start next group at the beginning of the
        # row BELOW the current row (as specified in your pseudocode).
        gid = int(t / rpg)
        print(f"change gid to {gid}")

    # safety check
    assert len(groups) == x
    #assert all(len(g) == y for g in groups), "some groups do not have exactly y cells"
    return part_of, groups

import numpy as np

def make_reading_rows(centres: np.ndarray, dy_tol=0.6):
    """
    Build rows (top -> bottom), each row left -> right.
    dy_tol is the 'same row' vertical tolerance in mm.
    """
    C = np.asarray(centres)
    # 1) sort by y (top first), then by x
    order = np.lexsort((C[:, 0], -C[:, 1]))
    C = C[order]

    rows, curr = [], [0]
    y0 = C[0, 1]
    for i in range(1, len(C)):
        if abs(C[i, 1] - y0) <= dy_tol:
            curr.append(i)
        else:
            # close previous row: sort by x (left->right)
            row = [int(order[j]) for j in curr]
            row.sort(key=lambda idx: centres[idx, 0])
            rows.append(row)
            # start new row
            curr = [i]
            y0 = C[i, 1]

    # last row
    row = [int(order[j]) for j in curr]
    row.sort(key=lambda idx: centres[idx, 0])
    rows.append(row)

    return rows




































import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import textwrap

RADIUS=9.0

# Reuse previous helpers: load_points, estimate_nn_distance, build_spatial_hash, neighbor_vectors,
# angle_histogram, find_primary_angle, unit_from_angle, angular_diff, estimate_spacing_along,
# map_to_lattice, build_occupancy, largest_rectangle_of_ones if they exist in the kernel.
# If not, redefine the minimal needed ones quickly (but they should exist from earlier).

def plot_result(result, radius=RADIUS, save_path=None):
    pts = result["points"]
    # compatibilità: prima prova 'selected_points', poi 'selected'
    sel = result.get("selected_points", result.get("selected"))
    if sel is None:
        raise KeyError("result non contiene né 'selected_points' né 'selected'.")

    corners = result["corners"]
    info = result["info"]

    fig, ax = plt.subplots(figsize=(8, 8))

    # tutti i cerchi (contorno)
    for x, y in pts:
        ax.add_patch(Circle((x, y), radius=radius, fill=False, linewidth=0.8))
    # cerchi dentro il parallelogramma (riempiti)
    for x, y in sel:
        ax.add_patch(Circle((x, y), radius=radius, fill=True, alpha=0.5))

    # contorno del parallelogramma
    ax.add_patch(Polygon(corners, fill=False, linewidth=2.0))

    # limiti assi per includere le patch
    xs, ys = pts[:, 0], pts[:, 1]
    pad = radius * 2.5
    ax.set_xlim(xs.min() - pad, xs.max() + pad)
    ax.set_ylim(ys.min() - pad, ys.max() + pad)

    ax.set_aspect('equal', 'box')
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"Parallelogramma ottimale: {info['rows']} righe × {info['cols']} colonne = {info['count']} cerchi")

    txt = textwrap.dedent(f"""
        angolo ≈ {info['angle_deg']:.1f}°
        passo u ≈ {info['du']:.3f}
        passo v ≈ {info['dv']:.3f}
        RMS residuo ≈ {info['residual_rms']:.3f}
        i ∈ [{info['i_range'][0]}, {info['i_range'][1]}], j ∈ [{info['j_range'][0]}, {info['j_range'][1]}]
    """).strip()
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va='top', ha='left', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.6))

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()

def map_to_lattice_snapped(points, a, b, grid_search_res=60):
    """
    Trova IJ interi e l'origine o tali che points ≈ o + i*a + j*b.
    Usa una ricerca su delta in [0,1)^2 per 'snapparsi' alla fase del reticolo.
    """
    B = np.column_stack([a, b])          # 2x2
    Binv = np.linalg.inv(B)
    C0 = (Binv @ points.T).T             # coordinate frazionarie grezze

    # ricerca grossolana su delta per minimizzare l'RMS dopo il rounding
    best = None
    grid = np.linspace(0.0, 1.0, grid_search_res, endpoint=False)
    for d0 in grid:
        for d1 in grid:
            delta = np.array([d0, d1])
            IJ_try = np.round(C0 - delta).astype(int)
            recon = (B @ IJ_try.T).T
            o_try = np.mean(points - recon, axis=0)
            rms = float(np.sqrt(np.mean(np.sum((points - (o_try + recon))**2, axis=1))))
            # criterio: RMS minimo -> migliore
            if (best is None) or (rms < best[0]):
                best = (rms, delta, IJ_try, o_try)

    rms, delta, IJ, o = best
    recon = (B @ IJ.T).T
    resid = np.linalg.norm(points - (o + recon), axis=1)
    return IJ, o, B, resid

def build_occupancy(IJ):
    I = IJ[:,0]; J = IJ[:,1]
    i_min, i_max = int(I.min()), int(I.max())
    j_min, j_max = int(J.min()), int(J.max())
    W = i_max - i_min + 1; H = j_max - j_min + 1
    M = np.zeros((H, W), dtype=np.uint8)
    for i,j in zip(I, J):
        M[j - j_min, i - i_min] = 1
    return M, (i_min, i_max, j_min, j_max)

def largest_rectangle_of_ones(M):
    H, W = M.shape
    heights = np.zeros(W, dtype=int)
    best = (0, 0, -1, 0, -1)
    for r in range(H):
        row = M[r]
        heights = np.where(row==1, heights+1, 0)
        stack = []
        for c in range(W + 1):
            curr_h = heights[c] if c < W else 0
            while stack and curr_h < heights[stack[-1]]:
                h = heights[stack.pop()]
                left = stack[-1] + 1 if stack else 0
                width = c - left
                area = h * width
                if area > best[0]:
                    j0 = r - h + 1; j1 = r; i0 = left; i1 = c - 1
                    best = (area, i0, i1, j0, j1)
            stack.append(c)
    return best

def load_points(csv_path):
    df = pd.read_csv(csv_path)
    lowered = {c: c.lower() for c in df.columns}
    for a,b in [('x','y'), ('cx','cy'), ('X','Y')]:
        if a in lowered.values() and b in lowered.values():
            ca = [k for k,v in lowered.items() if v==a][0]
            cb = [k for k,v in lowered.items() if v==b][0]
            return df[[ca, cb]].astype(float).values
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) < 2:
        raise ValueError("Non trovo due colonne numeriche (x, y) nel CSV.")
    return df[num_cols[:2]].astype(float).values

def estimate_nn_distance(points, sample_size=100):
    n = len(points)
    idx = np.random.choice(n, size=min(sample_size, n), replace=False)
    dmins = []
    P = points
    for i in idx:
        diffs = P - P[i]
        d2 = np.sum(diffs**2, axis=1)
        d2[i] = np.inf
        j = np.argmin(d2)
        dmins.append(math.sqrt(d2[j]))
    return float(np.median(dmins))

def build_spatial_hash(points, cell_size):
    keys = np.floor(points / cell_size).astype(int)
    buckets = {}
    for i, key in enumerate(map(tuple, keys)):
        buckets.setdefault(key, []).append(i)
    return buckets, keys

def neighbor_vectors(points, buckets, keys, cell_size, d_min, d_max):
    vecs = []
    n = len(points)
    P = points
    for i in range(n):
        kx, ky = keys[i]
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                neigh_key = (kx+dx, ky+dy)
                if neigh_key not in buckets: 
                    continue
                for j in buckets[neigh_key]:
                    if j == i: 
                        continue
                    v = P[j] - P[i]
                    d = math.hypot(v[0], v[1])
                    if d_min <= d <= d_max:
                        vecs.append(v)
    if len(vecs) == 0:
        return np.zeros((0,2))
    vecs_arr = np.array(vecs, float)
    ang = (np.degrees(np.arctan2(vecs_arr[:,1], vecs_arr[:,0])) + 180.0) % 180.0
    ln = np.round(np.linalg.norm(vecs_arr, axis=1), 3)
    angb = np.round(ang, 1)
    keep = []
    seen = set()
    for i,(a,l) in enumerate(zip(angb, ln)):
        key = (a,l)
        if key not in seen:
            seen.add(key)
            keep.append(i)
    return vecs_arr[keep]

def angle_histogram(vecs, bins=360, smooth=7):
    theta = (np.degrees(np.arctan2(vecs[:,1], vecs[:,0])) + 180.0) % 180.0
    hist, edges = np.histogram(theta, bins=bins, range=(0.0,180.0))
    if smooth > 1:
        k = np.ones(smooth, dtype=float)
        k /= k.sum()
        hist = np.convolve(np.r_[hist, hist, hist], k, mode="same")
        m = len(hist)//3
        hist = hist[m:2*m]
    centers = (edges[:-1] + edges[1:]) * 0.5
    return hist, centers

def find_primary_angle(hist, centers):
    i0 = int(np.argmax(hist))
    return float(centers[i0])

def unit_from_angle(deg):
    rad = math.radians(deg)
    return np.array([math.cos(rad), math.sin(rad)], float)

def angular_diff(a, b):
    d = abs((a-b+90.0) % 180.0 - 90.0)
    return d

def estimate_spacing_along(vecs, dir_angle, tol_deg=15.0):
    if len(vecs) == 0:
        return None
    u = unit_from_angle(dir_angle)
    theta = (np.degrees(np.arctan2(vecs[:,1], vecs[:,0])) + 180.0) % 180.0
    d1 = np.abs(np.vectorize(angular_diff)(theta, dir_angle))
    mask = d1 <= tol_deg
    if not np.any(mask):
        return None
    projs = np.abs(vecs[mask] @ u)
    return float(np.median(projs))

def rotate(vec, deg):
    rad = math.radians(deg)
    R = np.array([[math.cos(rad), -math.sin(rad)],
                  [math.sin(rad),  math.cos(rad)]])
    return (R @ vec)

def build_bases_aligned(points):
    d0 = estimate_nn_distance(points)
    buckets, keys = build_spatial_hash(points, cell_size=d0)
    vecs = neighbor_vectors(points, buckets, keys, cell_size=d0, d_min=0.6*d0, d_max=1.4*d0)
    if len(vecs) == 0:
        vecs = neighbor_vectors(points, buckets, keys, cell_size=d0, d_min=0.4*d0, d_max=1.8*d0)

    if len(vecs) >= 3:
        hist, centers = angle_histogram(vecs, bins=360, smooth=7)
        phi0 = find_primary_angle(hist, centers)
    else:
        pts = points - points.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(pts, full_matrices=False)
        v = Vt[0]
        phi0 = (np.degrees(np.arctan2(v[1], v[0])) + 180.0) % 180.0

    # tre assi della griglia esagonale (tutti separati di 60°)
    axes_deg = [(phi0 + k*60.0) % 180.0 for k in (0, 1, -1)]
    units = [unit_from_angle(ang) for ang in axes_deg]

    # stima passi lungo ogni asse
    def step(ang):
        s = estimate_spacing_along(vecs, ang)
        return s if s is not None else d0

    steps = [step(ang) for ang in axes_deg]
    # coppie candidate (60°/120°): (u0,u1), (u1,u2), (u2,u0)
    bases = [
        (steps[0]*units[0], steps[1]*units[1]),
        (steps[1]*units[1], steps[2]*units[2]),
        (steps[2]*units[2], steps[0]*units[0]),
    ]
    return bases

def evaluate_basis(points, a, b):
    IJ, o, B, resid = map_to_lattice_snapped(points, a, b)
    M, (i_min, i_max, j_min, j_max) = build_occupancy(IJ)
    area, i0, i1, j0, j1 = largest_rectangle_of_ones(M)
    return {
        "IJ": IJ, "o": o, "B": B, "resid": resid, "M": M,
        "bounds": (i_min, i_max, j_min, j_max), "rect": (area, i0, i1, j0, j1),
        "a": a, "b": b
    }


def find_best_parallelogram_aligned(points):
    candidates = build_bases_aligned(points)
    E_best = None
    for a, b in candidates:
        E = evaluate_basis(points, a, b)
        if (E_best is None) or (E["rect"][0] > E_best["rect"][0]):
            E_best = E

    IJ, o, B, resid = E_best["IJ"], E_best["o"], E_best["B"], E_best["resid"]
    a, b = E_best["a"], E_best["b"]
    (i_min, i_max, j_min, j_max) = E_best["bounds"]
    (area, i0, i1, j0, j1) = E_best["rect"]

    # indici del rettangolo in coordinate di reticolo
    i0_l = i_min + i0; i1_l = i_min + i1
    j0_l = j_min + j0; j1_l = j_min + j1

    # selezione dei punti dentro il rettangolo
    mask = (IJ[:,0] >= i0_l) & (IJ[:,0] <= i1_l) & (IJ[:,1] >= j0_l) & (IJ[:,1] <= j1_l)
    selected = points[mask]
    selected_IJ   = IJ[mask]
    selected_idx  = np.where(mask)[0]    

    # --- bordi a mezzo passo: allineati tra le righe/colonne ---
    P00 = o + (i0_l - 0.5) * a + (j0_l - 0.5) * b
    P10 = o + (i1_l + 0.5) * a + (j0_l - 0.5) * b
    P11 = o + (i1_l + 0.5) * a + (j1_l + 0.5) * b
    P01 = o + (i0_l - 0.5) * a + (j1_l + 0.5) * b
    corners = np.vstack([P00, P10, P11, P01])

    rows = (j1_l - j0_l + 1)
    cols = (i1_l - i0_l + 1)
    info = {
        "rows": int(rows),
        "cols": int(cols),
        "count": int(area),
        "du": float(np.linalg.norm(a)),
        "dv": float(np.linalg.norm(b)),
        "angle_deg": 60.0,
        "residual_rms": float(np.sqrt(np.mean(resid**2))),
        "i_range": (int(i0_l), int(i1_l)),
        "j_range": (int(j0_l), int(j1_l)),
    }

    return {
        "a": a, "b": b, "origin": o, "B": B,
        "IJ": IJ, "points": points,
        "selected_points": selected, "selected": selected,
        "selected_IJ": selected_IJ,
        "selected_idx": selected_idx,      
        "corners": corners, "info": info
    }


def rows_from_parallelogram(result):
    """
    Build rows (top→bottom) for the cells inside the parallelogram.
    Each row is a list of GLOBAL cell ids ordered left→right.
    """
    sel_IJ   = result["selected_IJ"]        # shape (M,2) → integer grid coords (i,j)
    sel_idx  = result["selected_idx"]       # shape (M,)  → global ids in original array
    i0, i1   = result["info"]["i_range"]
    j0, j1   = result["info"]["j_range"]

    rows = []
    # j runs from top row to bottom row; if you want bottom→top, reverse range
    for j in range(j0, j1 + 1):
        # positions of points that lie on row j
        pos   = np.where(sel_IJ[:, 1] == j)[0]
        if pos.size == 0:
            continue
        # sort by i (column index) to get left→right
        order = np.argsort(sel_IJ[pos, 0])
        row_global_ids = [int(sel_idx[p]) for p in pos[order]]
        rows.append(row_global_ids)

    return rows


def columns_as_rows(rows, centres, v=None):
    """
    Convert a list of equal-length rows into columns (used as rows).
    The order inside each new row is bottom→top by increasing projection
    on the v direction (if v is None, we use increasing y).
    """
    H = len(rows)           # number of original rows
    W = len(rows[0])        # cells per row (same for all)

    # projection direction used to order cells inside a column
    if v is not None:
        v = v / np.linalg.norm(v)
        proj = lambda idx: centres[idx] @ v
    else:
        proj = lambda idx: centres[idx, 1]  # y

    out = []
    for c in range(W):
        col = [rows[r][c] for r in range(H)]
        col.sort(key=proj)                  # bottom → top (or along v)
        out.append(col)
    return out

# Run on current CSV
#points = load_points("new_version/out.csv")
#result_aligned = find_best_parallelogram_aligned(points)

# Plot with the fixed plotter from previous cell
#plot_path3 = "/mnt/data/result_aligned.png"
#plot_result(result_aligned, radius=RADIUS)

#print("Parallelogramma allineato calcolato.")
#print("Grafico salvato in:", plot_path3)