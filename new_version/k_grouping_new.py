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
                          show_labels=True,               # <- draw group id text
                          label_unassigned=False,         # <- label empty cells?
                          fontsize=6):
    """
    centres   : ndarray (N,2)
    part_of   : dict {cell_id -> group_id} for the ASSIGNED cells only
    poly      : shapely polygon (only to draw the boundary)
    R         : cell radius
    S         : number of groups (optional, only to size the palette nicely)
    """

    # --- palette sized to the number of groups actually used
    used_groups = sorted(set(part_of.values()))
    if S is None:
        S = len(used_groups)
    try:
        import colorcet as cc
        base = [mpl.colors.to_rgb(c) for c in cc.glasbey[:max(S, len(used_groups))]]
    except Exception:
        base = [cm.tab20(i % 20) for i in range(max(S, len(used_groups)))]

    # consistent color for every group id
    group_color = {g: base[g % len(base)] for g in used_groups}

    def _text_color(rgb):
        """black on light colours, white on dark colours"""
        if isinstance(rgb, str):   # e.g. 'k'
            rgb = mpl.colors.to_rgb(rgb)
        L = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]   # perceived luminance
        return 'black' if L >= 0.6 else 'white'

    # --- plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(*poly.exterior.xy, color='k', lw=2)

    N = len(centres)
    assigned = 0
    for i, (x, y) in enumerate(centres):
        g = part_of.get(i, None)  # use .get so missing keys are “unassigned”
        if g is None:
            fc = face_unassigned
            ec = edge_unassigned
            txt = (str(g) if label_unassigned else "")
        else:
            assigned += 1
            fc = group_color[g]
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


