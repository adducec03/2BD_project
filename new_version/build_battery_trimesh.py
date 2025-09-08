# build_battery_trimesh.py
import json
import numpy as np
import trimesh
from typing import List, Dict, Any, Optional, Tuple
from trimesh import repair

# --------- util compatibili con varie versioni di trimesh ----------
def _iter_nodes_geometry(scene: trimesh.Scene):
    if scene.graph is None:
        return
    ng = getattr(scene.graph, "nodes_geometry", None)
    if ng is None:
        return
    try:
        res = ng() if callable(ng) else ng
    except TypeError:
        res = ng
    if isinstance(res, dict):
        for k, v in res.items():
            yield k, v
    elif isinstance(res, (list, tuple, set)):
        for pair in res:
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                yield pair[0], pair[1]

def _bake_parts(scene: trimesh.Scene, drop_names: Optional[List[str]] = None) -> List[Tuple[str, str, trimesh.Trimesh]]:
    out = []
    for node_name, geom_name in _iter_nodes_geometry(scene):
        geom = scene.geometry.get(geom_name)
        if geom is None:
            continue
        if drop_names and any(s.lower() in f"{node_name} {geom_name}".lower() for s in drop_names):
            continue
        T, _ = scene.graph.get(node_name)
        m = geom.copy()
        if T is not None:
            m.apply_transform(T)
        out.append((node_name or "", geom_name or "", m))
    if not out:
        # fallback: scena senza grafo
        out = [("root", "geom", scene.dump(concatenate=True))]
    return out

def _bbox_volume(mesh: trimesh.Trimesh) -> float:
    bmin, bmax = mesh.bounds
    ext = (bmax - bmin)
    return float(ext[0] * ext[1] * ext[2])

def _align_to_Z(mesh: trimesh.Trimesh, long_axis_on_model: str = "Y"):
    ax = long_axis_on_model.upper()
    if ax == "X":
        R = trimesh.transformations.rotation_matrix(np.pi/2, [0,1,0])
    elif ax == "Y":
        R = trimesh.transformations.rotation_matrix(-np.pi/2, [1,0,0])
    else:
        R = np.eye(4)
    mesh.apply_transform(R)

def _vertex_tint_rgba(mesh: trimesh.Trimesh, rgba: Tuple[int,int,int,int]):
    # aggiunge/aggiorna COLOR_0; i viewer moltiplicano texture * vertex colors
    n = len(mesh.vertices)
    col = np.tile(np.array(rgba, dtype=np.uint8), (n, 1))
    mesh.visual.vertex_colors = col

# tavolozza morbida
PALETTE = [
    (102,153,204,255), (204,153,102,255), (153,204,153,255),
    (204,102,102,255), (153,153,204,255), (102,204,204,255),
    (255,204,153,255), (204,153,204,255), (153,102,102,255),
    (102,102,153,255), (153,204,255,255), (204,255,204,255),
    (255,255,204,255), (204,204,255,255), (255,204,204,255),
    (204,255,255,255), (204,255,153,255), (255,229,204,255),
    (204,204,204,255), (102,153,153,255),
]

def convert_2d_to_3d(
    h_mm: float,
    input_json: str,
    output_glb: str,
    model_glb: str,
    xy_scale: float = 1.0,
    flip_groups_odd: bool = True,
    drop_nodes_with: Optional[List[str]] = None,   # <— lascia None per TENERE le scritte
    long_axis_on_model: str = "Y",
    colorize_groups: bool = True                   # <— colore per gruppo (sul body)
):
    # --------- leggi layout ----------
    with open(input_json, "r") as f:
        raw = json.load(f)
    layout = raw.get("layout_data", raw)
    circles: List[Dict[str, Any]] = layout.get("circles", [])
    gruppi: List[List[int]] = layout.get("gruppi", layout.get("groups", []))

    circle_to_group: Dict[int, int] = {}
    for gidx, g in enumerate(gruppi or []):
        for idx in g:
            circle_to_group[int(idx)] = gidx

    # --------- carica e prepara template (tutte le parti) ----------
    loaded = trimesh.load(model_glb)
    if isinstance(loaded, trimesh.Scene):
        parts = _bake_parts(loaded, drop_names=drop_nodes_with)
        meshes = [m for _, _, m in parts]
    else:
        meshes = [loaded]

    # allinea ogni parte allo Z come asse lungo
    for m in meshes:
        _align_to_Z(m, long_axis_on_model)

    # scegli il "body" (mesh col bbox più grande)
    body_idx = int(np.argmax([_bbox_volume(m) for m in meshes]))
    body = meshes[body_idx]

    # calcola scala per far combaciare l’altezza al valore richiesto
    bmin, bmax = body.bounds
    model_h = float((bmax - bmin)[2])
    s = (h_mm / model_h) if model_h > 0 else 1.0
    S = trimesh.transformations.scale_matrix(s)
    for m in meshes:
        m.apply_transform(S)

    # centra XY e porta il fondo a z=0 considerando TUTTE le parti
    all_min = np.min([m.bounds[0] for m in meshes], axis=0)
    all_max = np.max([m.bounds[1] for m in meshes], axis=0)
    cx, cy = (all_min[0] + all_max[0]) * 0.5, (all_min[1] + all_max[1]) * 0.5
    tz = -all_min[2]
    for m in meshes:
        m.apply_translation([-cx, -cy, tz])

    # facoltativo: sistema normali/inversioni (alcuni GLB hanno winding strano)
    for m in meshes:
        repair.fix_inversion(m)
        repair.fix_normals(m)

    # ricomponi un unico "template" mantenendo materiali/UV delle parti (per quanto possibile)
    template = trimesh.util.concatenate(meshes)

    # --------- istanzia SOLO le celle assegnate ----------
    instances: List[trimesh.Trimesh] = []
    center_point = np.array([0.0, 0.0, h_mm * 0.5])

    # insieme degli indici assegnati
    assigned = set(circle_to_group.keys())

    for i, c in enumerate(circles):
        # SKIP: cella non assegnata ad alcun gruppo
        if assigned and i not in assigned:
            continue

        x, y = float(c["center"][0]), float(c["center"][1])
        g = circle_to_group.get(i, 0)

        inst = template.copy()

        # colore per gruppo (tinta moltiplicativa, scritte/texture restano visibili)
        if colorize_groups:
            tint = PALETTE[g % len(PALETTE)]
            _vertex_tint_rgba(inst, tint)

        # flip gruppi dispari attorno al centro → nessuno “scalino” in Z
        if flip_groups_odd and (g % 2 == 1):
            Rflip = trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0], point=center_point)
            inst.apply_transform(Rflip)

        # posizionamento
        inst.apply_translation([x * xy_scale, y * xy_scale, 0.0])
        instances.append(inst)

    out_mesh = trimesh.util.concatenate(instances)
    repair.fix_inversion(out_mesh)
    repair.fix_normals(out_mesh)
    out_mesh.export(output_glb)
    print(f"✅ GLB esportato: {output_glb} | celle: {len(instances)}")