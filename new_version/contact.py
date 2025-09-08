# contacts2d.py
import json, math
from typing import Any, Dict, List, Tuple

def _pos_map(circles: List[Dict[str, Any]]) -> Dict[int, Tuple[float, float, float]]:
    out = {}
    for i, c in enumerate(circles):
        x, y = float(c["center"][0]), float(c["center"][1])
        out[i] = (x, y, float(c.get("radius", 0.0)))
    return out

def _series_uniform_pairs(series_links: List[Dict[str, Any]], S: int) -> Tuple[int, Dict[int, List[Tuple[int,int]]]]:
    """Stesso numero di link per ogni k: T = min_k |links_k|. Ordina per distanza (corta) prima di ritagliare."""
    by_k: Dict[int, List[Tuple[int,int]]] = {k: [] for k in range(max(0, S-1))}
    for l in (series_links or []):
        try:
            k = int(l["k"]); i = int(l["i"]); j = int(l["j"])
        except Exception:
            continue
        if 0 <= k < S-1 and i >= 0 and j >= 0:
            by_k[k].append((i, j))
    # ordina per distanza se possibile (serve pos)
    return by_k

def _uniform_select_sorted(by_k: Dict[int, List[Tuple[int,int]]], pos2d: Dict[int, Tuple[float,float,float]]) -> Tuple[int, Dict[int, List[Tuple[int,int]]]]:
    counts = []
    for k, lst in by_k.items():
        # ordina per distanza (più corti prima)
        lst_sorted = sorted(lst, key=lambda ij: (pos2d[ij[0]][0]-pos2d[ij[1]][0])**2 + (pos2d[ij[0]][1]-pos2d[ij[1]][1])**2)
        by_k[k] = lst_sorted
        counts.append(len(lst_sorted))
    T = min(counts) if counts else 0
    return T, {k: by_k[k][:T] for k in by_k.keys()}

def _edge_points(pi, ri, pj, rj, shrink_ratio=0.12):
    """Punti di contatto sulla circonferenza verso l’altra cella (leggermente 'rientrati')."""
    xi, yi, _ = pi; xj, yj, _ = pj
    vx, vy = (xj - xi), (yj - yi)
    d = math.hypot(vx, vy)
    if d < 1e-6:
        return (xi, yi), (xj, yj)
    ux, uy = vx/d, vy/d
    mi = max(0.0, ri * (1.0 - shrink_ratio))
    mj = max(0.0, rj * (1.0 - shrink_ratio))
    ai = (xi + ux*mi, yi + uy*mi)
    bj = (xj - ux*mj, yj - uy*mj)
    return ai, bj

def _nearest_pairs_in_group(members: List[int],
                            pos2d: Dict[int, Tuple[float,float,float]],
                            degree: int,
                            thresh_scale: float) -> List[Tuple[int,int]]:
    """Per ogni cella nel gruppo, collega i 'degree' più vicini entro soglia."""
    pairs = set()
    # raggio medio per soglia
    if not members:
        return []
    rs = [pos2d[i][2] for i in members]
    Rm = sum(rs)/len(rs) if rs else 0.0
    thresh2 = (2.0 * Rm * thresh_scale)**2
    for a in members:
        da = []
        ax, ay, _ = pos2d[a]
        for b in members:
            if b == a: continue
            bx, by, _ = pos2d[b]
            d2 = (ax-bx)**2 + (ay-by)**2
            if d2 <= thresh2:
                da.append((d2, b))
        da.sort(key=lambda t: t[0])
        for _, b in da[:degree]:
            u, v = (a, b) if a < b else (b, a)
            pairs.add((u, v))
    return sorted(pairs)

def attach_2d_contacts(layout_data: Dict[str, Any],
                       parallel_degree: int = 2,
                       parallel_thresh_scale: float = 1.20,
                       edge_shrink_ratio: float = 0.12) -> None:
    """
    Modifica IN-PLACE layout_data aggiungendo:
      - series_uniform_T
      - series_uniform_links (dict k -> [[i,j],...])
      - series_uniform_segments (lista segmenti con from/to)
      - parallel_links (lista [[i,j],...])
      - parallel_segments (lista segmenti con from/to e group)
    """
    circles: List[Dict[str, Any]] = layout_data.get("circles", [])
    gruppi: List[List[int]] = layout_data.get("gruppi", layout_data.get("groups", []))
    series_links: List[Dict[str, Any]] = layout_data.get("series_links", [])
    S = len(gruppi or [])
    pos2d = _pos_map(circles)

    # --- SERIE: stesso numero di link per ogni k ---
    by_k_raw = _series_uniform_pairs(series_links, S)
    T, by_k = _uniform_select_sorted(by_k_raw, pos2d)
    series_segments = []
    for k in range(max(0, S-1)):
        for (i, j) in by_k.get(k, []):
            ai, bj = _edge_points(pos2d[i], pos2d[i][2], pos2d[j], pos2d[j][2], shrink_ratio=edge_shrink_ratio)
            series_segments.append({
                "type": "series",
                "k": k, "i": i, "j": j,
                "from": [float(ai[0]), float(ai[1])],
                "to":   [float(bj[0]), float(bj[1])]
            })

    # --- PARALLELO: vicini dentro ciascun gruppo ---
    parallel_pairs_all = []
    parallel_segments = []
    for gidx, members in enumerate(gruppi or []):
        pairs = _nearest_pairs_in_group(members, pos2d, degree=parallel_degree, thresh_scale=parallel_thresh_scale)
        for (i, j) in pairs:
            parallel_pairs_all.append([i, j, gidx])
            ai, bj = _edge_points(pos2d[i], pos2d[i][2], pos2d[j], pos2d[j][2], shrink_ratio=edge_shrink_ratio)
            parallel_segments.append({
                "type": "parallel",
                "group": gidx, "i": i, "j": j,
                "from": [float(ai[0]), float(ai[1])],
                "to":   [float(bj[0]), float(bj[1])]
            })

    # --- scrivi nel layout ---
    layout_data["series_uniform_T"] = int(T)
    layout_data["series_uniform_links"] = {str(k): [[int(i), int(j)] for (i, j) in by_k.get(k, [])]
                                           for k in range(max(0, S-1))}
    layout_data["series_uniform_segments"] = series_segments
    layout_data["parallel_links"] = parallel_pairs_all
    layout_data["parallel_segments"] = parallel_segments


def add_contacts_to_file(path_in: str,
                         path_out: str = None,
                         parallel_degree: int = 2,
                         parallel_thresh_scale: float = 1.20,
                         edge_shrink_ratio: float = 0.12) -> str:
    """
    Legge un JSON variante, aggiunge i contatti 2D e salva.
    Ritorna il path del file scritto.
    """
    with open(path_in, "r") as f:
        doc = json.load(f)
    layout = doc.get("layout_data", doc)
    attach_2d_contacts(layout,
                       parallel_degree=parallel_degree,
                       parallel_thresh_scale=parallel_thresh_scale,
                       edge_shrink_ratio=edge_shrink_ratio)
    out = path_out or path_in
    with open(out, "w") as f:
        json.dump(doc, f, indent=2)
    return out