# pack3d.py
from typing import List, Dict, Any, Optional
import json, math
import numpy as np
from pygltflib import GLTF2, Scene, Node, Mesh, Material, \
    BufferView, Accessor, BYTE, UNSIGNED_BYTE, SHORT, UNSIGNED_SHORT, UNSIGNED_INT, FLOAT

# ----------------- utils quaternion -----------------
def quat_identity():
    return [0.0, 0.0, 0.0, 1.0]

def quat_180(axis: str):
    axis = axis.upper()
    if axis == "X": return [1.0, 0.0, 0.0, 0.0]
    if axis == "Y": return [0.0, 1.0, 0.0, 0.0]
    # Z default
    return [0.0, 0.0, 1.0, 0.0]


def _get_position_accessor_index(prim) -> Optional[int]:
    attr = getattr(prim, "attributes", None)
    if attr is None:
        return None
    # caso: dict
    if isinstance(attr, dict):
        return attr.get("POSITION", None)
    # caso: pygltflib.Attributes
    return getattr(attr, "POSITION", None)

# ----------------- accessor bounds (POSITION) -----------------
def _component_dtype(componentType: int):
    if componentType == FLOAT: return np.float32
    if componentType == UNSIGNED_INT: return np.uint32
    if componentType == UNSIGNED_SHORT: return np.uint16
    if componentType == SHORT: return np.int16
    if componentType == UNSIGNED_BYTE: return np.uint8
    if componentType == BYTE: return np.int8
    raise ValueError(f"Unsupported componentType {componentType}")

def _type_num_components(type_str: str) -> int:
    return {"SCALAR":1, "VEC2":2, "VEC3":3, "VEC4":4, "MAT2":4, "MAT3":9, "MAT4":16}[type_str]

def _recompute_position_bounds(gltf: GLTF2):
    blob = gltf.binary_blob()
    if blob is None:
        return
    for mesh in (gltf.meshes or []):
        for prim in (mesh.primitives or []):
            acc_idx = _get_position_accessor_index(prim)
            if acc_idx is None:
                continue
            acc: Accessor = gltf.accessors[acc_idx]

            # supporto standard: POSITION = VEC3 float32 in un bufferView
            if acc.bufferView is None or acc.type != "VEC3" or acc.componentType != FLOAT:
                continue

            bv: BufferView = gltf.bufferViews[acc.bufferView]
            base = (bv.byteOffset or 0) + (acc.byteOffset or 0)
            stride = bv.byteStride or 12  # 3 * float32
            count = acc.count

            # lettura robusta con stride
            if stride == 12:
                data = np.frombuffer(blob, dtype=np.float32,
                                     count=count*3, offset=base).reshape((count,3))
            else:
                data = np.empty((count,3), dtype=np.float32)
                for i in range(count):
                    off = base + i*stride
                    data[i,:] = np.frombuffer(blob[off:off+12], dtype=np.float32, count=3)

            mins = data.min(axis=0).astype(float).tolist()
            maxs = data.max(axis=0).astype(float).tolist()
            acc.min = mins
            acc.max = maxs

# ----------------- materiali -----------------
def _sanitize_materials(gltf: GLTF2):
    if not gltf.materials:
        return
    for m in gltf.materials:
        mode = getattr(m, "alphaMode", None)
        if mode is None or mode != "MASK":
            # alphaCutoff valido solo con MASK
            if hasattr(m, "alphaCutoff"):
                m.alphaCutoff = None

# ----------------- scelta mesh template -----------------
def _pick_template_mesh_index(gltf: GLTF2, exclude_substrings: List[str]) -> int:
    # prova: preferisci nodi con mesh e nome accettabile
    name_bad = lambda n: any(s.lower() in (n or "").lower() for s in exclude_substrings)
    if gltf.nodes:
        for n in gltf.nodes:
            if n.mesh is not None and not name_bad(getattr(n, "name", "")):
                return n.mesh
    # fallback: prima mesh
    if gltf.meshes and len(gltf.meshes) > 0:
        return 0
    raise RuntimeError("Nessuna mesh trovata nel GLB del modello cella.")

# ----------------- build -----------------
def build_glb_from_variant(
    variant_json_path: str,
    cell_model_path: str,
    out_glb_path: str,
    *,
    xy_scale: float = 1.0,
    model_ref_diameter_mm: float = 18.0,
    model_ref_height_mm: float = 65.0,
    flip_axis: str = "Z",
    invert_y: bool = False,
    template_strategy: str = "first",              # compat, non usato
    exclude_name_contains: Optional[List[str]] = None,
    sanitize_accessors: bool = True
):
    exclude = exclude_name_contains or []

    # ---- carica layout ----
    with open(variant_json_path, "r") as f:
        var = json.load(f)
    layout = var["layout_data"]
    circles = layout["circles"]
    gruppi = layout.get("gruppi", [])
    # mappa cella -> indice gruppo
    cell2grp: Dict[int,int] = {}
    for gidx, lst in enumerate(gruppi):
        for ci in lst:
            cell2grp[int(ci)] = gidx

    # altezza target in mm (nel tuo JSON è in "unità*10", es 650.0 per 18650)
    cell_height_mm = float(var["cell_used"]["height"]) / 10.0
    # diametro target = 2*radius (i cerchi hanno radius in unità layout, che deve combaciare con mm)
    # NB: usiamo il rapporto di scala per X/Y e per Z separato.
    # Se il GLB è già 18x65 in **mm**, lascia model_ref_* come 18/65 e xy_scale=1.0

    # ---- carica modello cella ----
    gltf = GLTF2().load_binary(cell_model_path)

    # prendi UNA mesh come template (niente duplicati di root)
    mesh_index = _pick_template_mesh_index(gltf, exclude)

    # svuota scene & nodes: ricreiamo la scena solo con le nostre istanze
    gltf.scenes = [Scene(nodes=[])]
    gltf.nodes = []

    # istanzia una sola node per cella
    scene_nodes: List[int] = []
    for idx, c in enumerate(circles):
        cx, cy = float(c["center"][0]), float(c["center"][1])
        r = float(c["radius"])
        target_diam_mm = 2.0 * r  # se il layout è in mm
        sx_sy = target_diam_mm / float(model_ref_diameter_mm)
        sz = cell_height_mm / float(model_ref_height_mm)

        tx = cx * xy_scale
        ty = (-cy if invert_y else cy) * xy_scale
        tz = 0.0

        # flip solo per gruppi dispari
        gidx = cell2grp.get(idx, 0)
        rot = quat_180(flip_axis) if (gidx % 2 == 1) else quat_identity()

        node = Node(
            mesh=mesh_index,
            translation=[tx, ty, tz],
            rotation=rot,
            scale=[sx_sy, sx_sy, sz],
            name=f"cell_{idx}_g{gidx}"
        )
        gltf.nodes.append(node)
        scene_nodes.append(len(gltf.nodes)-1)

    # collega i nodi creati alla scena 0
    gltf.scenes[0].nodes = scene_nodes
    gltf.scene = 0

    # pulizia materiali (alphaCutoff)
    _sanitize_materials(gltf)

    # bounds per POSITION
    if sanitize_accessors:
        _recompute_position_bounds(gltf)

    gltf.save_binary(out_glb_path)


    print("nodes:", len(gltf.nodes), "meshes:", len(gltf.meshes))