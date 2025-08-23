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
    Dato y ≥ 0, trova a,b,c ≥ 0 tali che:
        y = (a+b)*c/2
        a - 1 <= b <= a
        a <= ceil(y/2)
        c <= a
    minimizzando |a - c|.
    In caso di parità su |a-c|, minimizza b, poi a.

    Ritorna (ok, a, b, c). Se nessuna terna soddisfa i vincoli, ok=False.
    """
    if not isinstance(y, int):
        raise TypeError("y deve essere un intero")
    if y < 0:
        raise ValueError("y deve essere ≥ 0")

    if y == 0:
        return (True, 0, 0, 0)

    n = 2 * y
    a_max = (y + 1) // 2  # ceil(y/2)

    best = None  # (abs_diff, b, a, c)

    # esplora i divisori c di n
    for c in range(1, isqrt(n) + 1):
        if n % c != 0:
            continue
        s = n // c

        # prova con questa coppia (s, c)
        for a in (s // 2, (s + 1) // 2):
            b = s - a
            if a - 1 <= b <= a and a <= a_max and b >= 0 and c <= a:
                cand = (abs(a - c), b, a, c)
                if best is None or cand < best:
                    best = cand

        # prova con coppia invertita (c, s) se diversa
        if s != c:
            c2, s2 = s, c
            for a in (s2 // 2, (s2 + 1) // 2):
                b = s2 - a
                if a - 1 <= b <= a and a <= a_max and b >= 0 and c2 <= a:
                    cand2 = (abs(a - c2), b, a, c2)
                    if best is None or cand2 < best:
                        best = cand2

    if best is None:
        return (False, None, None, None)

    _, b, a, c = best
    return (a, b, c)

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

    long, short, rpg = calc_parameters(y)

    part_of: Dict[int, int]  = {}
    groups:  List[List[int]] = [[] for _ in range(x)]

    r = 0  # cursor: row and column within current row
    t = 0

    def row_len(rr: int) -> int:
        return len(rows[rr]) if 0 <= rr < len(rows) else 0

    def ensure_row(r: int) -> bool:
        return (r < len(rows)) and (len(rows[r]) > 0)

    rem = row_len(r)
    gid = 0
    cid =0
    a=long
    b=short

    while cid<(x*y)-1:
        c=rpg
        n = 0

        while c > 0:
            print(f"row{r}, remaining cells for each block:{c}, a:{a}, b:{b}")
            # set the pattern for *this* row
            a = long if (r % 2 == 0) else short
            b = short if (r % 2 == 0) else long

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
                n+=1
                print(f"{a} cell placed (rem:{rem}, n:{n}, cid:{cid})")
            else:
                if(rem < a):
                    print(f"not enough space, change row")
                elif(gid+n<x):
                    print (f"reached the groups limit current group:{gid+n}")
                # not enough space → move to next row, same group
                cid+=rem
                r += 1
                rem = row_len(r)
                c-=1
                t+=n
                n=0
                print(f"r:{r},rem:{rem},c:{c},t:{t},n{n})")
                #break
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
                n+=1
                print(f"{b} cell placed (rem:{rem}, n:{n}, cid:{cid})")
            else:
                # not enough space → move to next row, same group
                if(rem < b):
                    print(f"not enough space, change row")
                elif(gid+n>=x):
                    print (f"reached the groups limit current group:{gid+n}")
                cid+=rem
                r += 1
                rem = row_len(r)
                c-=1
                t+=n
                n=0
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


def group_xSyP_blockwise(
    rows: List[List[int]],   # row 0 = top row, each row is left→right list of cell ids
    x: int,                  # number of series groups
    y: int) -> Tuple[Dict[int, int], List[List[int]]]:
    """
    Fill groups in *blocks* across rows:

      • Let seg_len = long if (row is even) else short.
      • On a row, place that seg_len for groups gid, gid+1, … until it no longer fits.
      • These groups form the current *block* (n groups).
      • Then go down to next rows and keep feeding the *same* n groups
        exactly rpg−1 more row-segments each, always using seg_len of the
        current row parity.  If a row runs out of space for a group, drop
        to the next row and continue with the same group.
      • When all n groups of the block have received rpg rows each, they
        are complete.  Set gid += n and start a new block on the current row.
    """
    total = sum(len(r) for r in rows)
    if x * y > total:
        raise ValueError(f"Need {x*y} cells but only {total} exist")

    LONG, SHORT, RPG = calc_parameters(y)
    if RPG <= 0:
        raise ValueError("rpg must be ≥1")

    part_of: Dict[int, int]  = {}
    groups:  List[List[int]] = [[] for _ in range(x)]

    r, col = 0, 0  # reading cursor: row index, column index in that row

    def row_len(rr: int) -> int:
        return len(rows[rr]) if 0 <= rr < len(rows) else 0

    def seg_len_for_row(rr: int) -> int:
        return LONG if (rr % 2 == 0) else SHORT

    def need_more_rows():
        nonlocal r, col
        r += 1
        col = 0
        if r >= len(rows):
            raise RuntimeError("Ran out of rows while forming groups")

    def assign_segment(gid: int, length: int):
        """Take 'length' cells from the *current* row and give them to group gid.
           If they don't fit, go to the next row and try again (same gid)."""
        nonlocal r, col, part_of, groups

        while True:
            if col + length <= row_len(r):
                seg = rows[r][col: col + length]
                for cid in seg:
                    part_of[cid] = gid
                    groups[gid].append(cid)
                col += length
                return
            else:
                # go to the next row and keep trying for the same group
                need_more_rows()

    gid = 0  # next group id to be started

    while gid < x:
        # ---------- start a new block on the current row ----------
        if r >= len(rows):
            raise RuntimeError("Ran out of rows while starting a new block")

        seg = seg_len_for_row(r)
        col = min(col, row_len(r))  # just in case

        # a) build the list of groups that will be fed on this row
        block_groups: List[int] = []
        while gid + len(block_groups) < x and col + seg <= row_len(r):
            g = gid + len(block_groups)
            # give this row segment to g
            assign_segment(g, seg)
            block_groups.append(g)

        if not block_groups:
            # no space for even one group on this row → just move down
            need_more_rows()
            continue

        # b) feed the remaining rpg-1 rows to the same block, in order
        remain = [RPG - 1] * len(block_groups)
        # already gave 1 row to each in the block (the row we started on)
        while any(remain_j > 0 for remain_j in remain):
            # we’re somewhere (maybe same row if space remains)
            seg = seg_len_for_row(r)
            for j, g in enumerate(block_groups):
                if remain[j] == 0:
                    continue
                # try to assign this row’s segment to group g
                if col + seg > row_len(r):
                    # not enough space in this row → go to the next row
                    need_more_rows()
                    seg = seg_len_for_row(r)
                assign_segment(g, seg)
                remain[j] -= 1

        # c) the whole block is complete → advance gid
        gid += len(block_groups)

        # After a block we *stay* on the current row; if you want
        # the next block to always start at the next row, uncomment:
        # need_more_rows()

    return part_of, groups

def test2(
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

    long, short, rpg = calc_parameters(y)

    part_of: Dict[int, int]  = {}
    groups:  List[List[int]] = [[] for _ in range(x)]

    r, c = 0, 0  # cursor: row and column within current row
    i = 0
    t = 0

    def row_len(rr: int) -> int:
        return len(rows[rr]) if 0 <= rr < len(rows) else 0

    def ensure_row(r: int) -> bool:
        return (r < len(rows)) and (len(rows[r]) > 0)

    rem = row_len(r)
    gid=0

    while i<(x*y):
        c=rpg
        n = 0

        while c > 0:

            # set the pattern for *this* row
            a = long if (r % 2 == 0) else short
            b = short if (r % 2 == 0) else long

            # 1) try to place 'a' cells in this row
            if rem >= a:
                segA = rows[r][c:c+a]
                for cid in segA:
                    part_of[cid] = gid
                    groups[gid].append(cid)
                    rem-=1
                    cid+=1
                n+=1
            else:
                # not enough space → move to next row, same group
                r += 1
                rem = row_len(r)
                c-=1
                t+=n
                n=0
                break
  

            # 2) try to place 'b' cells in this SAME row
            if rem >= a:
                segA = rows[r][c:c+a]
                for cid in segA:
                    part_of[cid] = gid
                    groups[gid].append(cid)
                    rem-=1
                    cid+=1
                n+=1
            else:
                # not enough space → move to next row, same group
                r += 1
                rem = row_len(r)
                c-=1
                t+=n
                n=0
                break

        # group complete: always start next group at the beginning of the
        # row BELOW the current row (as specified in your pseudocode).
        gid += t / rpg

    # safety check
    assert len(groups) == x
    assert all(len(g) == y for g in groups), "some groups do not have exactly y cells"
    return part_of, groups


def group_xSyP_strict_reading2(
    rows: List[List[int]],             # each row is a list of cell-ids (left→right)
    x: int,                            # number of series groups
    y: int                             # cells per group
) -> Tuple[Dict[int, int], List[List[int]]]:

    total = sum(len(r) for r in rows)
    if x * y > total:
        raise ValueError(f"Need {x*y} cells but only {total} exist")

    long, short, rpg = calc_parameters(y)

    part_of: Dict[int, int]  = {}
    groups:  List[List[int]] = [[] for _ in range(x)]

    # r = current row, p = current column in that row
    r, p = 0, 0
    gid  = 0          # current group id
    t    = 0          # (kept only because you had it; no longer used to move gid)
    cid  = 0          # number of cells assigned so far (for the loop guard)

    def row_len(rr: int) -> int:
        return len(rows[rr]) if 0 <= rr < len(rows) else 0

    while gid < x and cid < x * y:
        c = rpg      # “rows per group” counter (not strictly needed, kept for you)
        n = 0        # segments placed in the current row (kept to preserve names)

        while len(groups[gid]) < y:    # fill exactly y cells in this group
            if r >= len(rows):

                raise RuntimeError("Ran out of rows while forming groups")

            rem = row_len(r) - p
            a = long if (r % 2 == 0) else short
            b = short if (r % 2 == 0) else long

            # ---- place 'a' in current row (or wrap to next row) ----------
            need = min(a, y - len(groups[gid]))  # do not exceed y
            if rem < need:
                r += 1; p = 0                    # next row, same group
                continue
            segA = rows[r][p:p+need]
            for cell in segA:
                part_of[cell] = gid
                groups[gid].append(cell)
                cid += 1
            p += need
            n += 1

            if len(groups[gid]) >= y:
                break

            # ---- place 'b' in current row (or wrap to next row) ----------
            rem = row_len(r) - p
            need = min(b, y - len(groups[gid]))
            if rem < need:
                r += 1; p = 0
                continue
            segB = rows[r][p:p+need]
            for cell in segB:
                part_of[cell] = gid
                groups[gid].append(cell)
                cid += 1
            p += need
            n += 1

        # the current group is complete (exactly y cells) → go to next group
        gid += 1
        r   += 1     # “start the next group under the row we just ended on”
        p    = 0
        t    = 0     # keep your variable but we do not use it to move gid

    # optional sanity checks
    assert len(groups) == x
    assert all(len(g) == y for g in groups), "some groups do not have exactly y cells"
    return part_of, groups