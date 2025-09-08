import datetime
import json
import os
import uuid
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import FileResponse

# === NUOVI IMPORT ===
import circle_packing as cp
import findBestConfiguration as fc
import battery_layout_cpsat_v2 as bl
from ortools.sat.python import cp_model

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CELLE_DISPONIBILI_PATH = os.path.join(BASE_DIR, "celle_disponibili.json")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------- helper: lettura parametri batteria -------------------------
def _read_battery_params(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Legge i campi attesi dal JSON d'ingresso (senza usare convert_file_json).
    Mantiene compatibilità con chiavi già usate in precedenza.
    """
    def _get(k, default=None):
        return raw.get(k, default)

    # Chiavi attese (come in versione precedente)
    total_voltage = int(_get("totalVoltage"))
    total_current = int(_get("totalBatteryCurrent")) / 1000.0  # mA -> A
    cell_voltage = int(_get("nominalVoltage"))
    cell_current = int(_get("chargingCurrent"))  # Ah attesi? prima era semplicemente int
    battery_type = _get("batteryType")           # es. "18650"
    if not battery_type or len(battery_type) < 4:
        raise HTTPException(status_code=400, detail="Campo 'batteryType' non valido.")

    # parsing come prima: "18650" -> diam=18 mm, h=65.0 mm
    battery_diameter = int(battery_type[:2])
    battery_height = int(battery_type[2:]) / 10.0

    return dict(
        total_voltage=total_voltage,
        total_current=total_current,
        cell_voltage=cell_voltage,
        cell_current=cell_current,
        battery_type=battery_type,
        battery_diameter=battery_diameter,
        battery_height=battery_height,
    )


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)

def _centroid(indices, centres):
    if not indices:
        return [0.0, 0.0]
    xs = [centres[i][0] for i in indices]
    ys = [centres[i][1] for i in indices]
    return [float(sum(xs)/len(xs)), float(sum(ys)/len(ys))]

def _build_serie_connections(links, groups, centres, S):
    """
    Ritorna S-1 segmenti {from:[x,y], to:[x,y]}.
    Se per un k non esiste link esplicito, usa il collegamento tra i centroidi dei gruppi k e k+1.
    """
    # raggruppa i link per k
    by_k = {k: [] for k in range(S-1)}
    for l in links:
        k = int(l["k"])
        if 0 <= k < S-1:
            by_k[k].append(l)

    segments = []
    for k in range(S-1):
        if by_k[k]:
            # prendi il primo link attivo tra gruppi k e k+1
            i = int(by_k[k][0]["i"])
            j = int(by_k[k][0]["j"])
            segments.append({
                "from": [float(centres[i][0]), float(centres[i][1])],
                "to":   [float(centres[j][0]), float(centres[j][1])]
            })
        else:
            # fallback: collega i centroidi dei gruppi
            a = _centroid(groups[k], centres)
            b = _centroid(groups[k+1], centres)
            segments.append({"from": a, "to": b})
    return segments

def _normalize_polygon(points):
    """Rende il poligono una lista di [x,y] (accetta anche [{x:..,y:..}])."""
    out = []
    for p in (points or []):
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            out.append([float(p[0]), float(p[1])])
        elif isinstance(p, dict) and "x" in p and "y" in p:
            out.append([float(p["x"]), float(p["y"])])
    return out


def _normalize_polygon_points(polygon_points, fallback_poly=None):
    """Rende il poligono una semplice lista di [x,y]. Gestisce:
       - lista di [x,y]
       - lista di dict {'x':..,'y':..}
       - lista di anelli [[ [x,y],... ], [hole...]] -> prende l'anello esterno
       Se non valido e c'è fallback_poly (di circle_packing), usa il suo bordo.
    """
    def norm_pt(p):
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            return [float(p[0]), float(p[1])]
        if isinstance(p, dict) and 'x' in p and 'y' in p:
            return [float(p['x']), float(p['y'])]
        return None

    out = []
    if isinstance(polygon_points, list) and polygon_points:
        # caso anelli annidati: [[...],[...]]
        if isinstance(polygon_points[0], (list, tuple)) and polygon_points and \
           len(polygon_points[0]) > 0 and isinstance(polygon_points[0][0], (list, tuple, dict)):
            ring = polygon_points[0]  # prendi l'anello esterno
            out = [norm_pt(pt) for pt in ring]
        else:
            out = [norm_pt(pt) for pt in polygon_points]
        out = [pt for pt in out if pt]
        if len(out) >= 3:
            return out

    # fallback dal poligono di circle_packing
    if fallback_poly is not None:
        try:
            coords = list(fallback_poly.exterior.coords)
            return [[float(x), float(y)] for (x, y) in coords]
        except Exception:
            pass
    return []

# ------------------------- helper: normalizza ritorno di fc -------------------------
def _unpack_fc_result(res) -> List[Dict[str, Any]]:
    """
    fc.calcola_configurazioni_migliori a volte è usata come:
        _, configs = fc.calcola_configurazioni_migliori(...)
    Qui normalizziamo a 'lista di configurazioni' oppure [].
    """
    if res is None:
        return []
    if isinstance(res, tuple) and len(res) == 2:
        # (qualcosa, configs)
        return res[1] or []
    if isinstance(res, list):
        return res
    # fallback prudente
    return []


# ------------------------- CIRCLE PACKING (nuovo) -------------------------
def _compute_centres_with_cp(json_file: str, R: float) -> List[Tuple[float, float]]:
    """
    Implementa esattamente la pipeline richiesta con circle_packing.
    """
    poly, _ = cp.load_boundary(Path(json_file))

    centres = cp.best_hex_seed_two_angles(poly, n_phase=16)
    print("hex grid :", len(centres))

    centres = cp.greedy_insert(poly, centres, trials=1000, max_pass=6)
    print("after greedy :", len(centres))

    centres = cp.batch_bfgs_compact(centres, R, poly, n_pass=4)
    print("after compaction :", len(centres))

    centres = cp.greedy_insert(poly, centres, trials=1000, max_pass=3)
    print("final count :", len(centres))

    prev = len(centres)
    centres = cp.skeleton_insert(poly, centres, step=2.0)
    if len(centres) != prev:
        print("skeleton insert :", len(centres))

    return [(float(x), float(y)) for (x, y) in centres]


# ------------------------- LOGICA CONFIGURAZIONI (nuova) -------------------------
def _trova_configurazioni(
    cell_voltage: int,
    cell_capacity_ah: float,
    total_voltage: int,
    total_current: float,
    max_cells: int,
    want_at_least: int = 5,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    1) con gli input originali: se trovi >=5 configurazioni, usa le prime 5
    2) se ne trovi <5:
       2.1) prova capacità ridotta finché non trovi almeno 1 adatta a max_cells -> STOP
       2.2) altrimenti cambia tipo cella scorrendo 'celle_disponibili.json',
            provando prima capacità originale poi ridotta; STOP alla prima valida.
    Ritorna (configs_selezionate, cell_info_usata)
    """
    # tentativo 1: cella originale
    res = fc.calcola_configurazioni_migliori(
        cell_voltage, cell_capacity_ah, total_voltage, total_current,
        celle_max=max_cells, top_k=want_at_least
    )
    configs = _unpack_fc_result(res)
    if len(configs) >= want_at_least:
        cell_info = dict(name=str(cell_voltage) + "V", diameter=None, height=None,
                         voltage=cell_voltage, capacity=cell_capacity_ah)
        return configs[:want_at_least], cell_info

    # tentativo 2: capacità ridotta (finché trovi una compatibile)
    cfg_tuple, nuova_cap = fc.trova_configurazione_con_capacita_ridotta(
        cell_voltage, cell_capacity_ah, total_voltage, total_current, max_cells
    )
    if cfg_tuple:
        found = _unpack_fc_result(cfg_tuple)
        if found:
            cell_info = dict(name=str(cell_voltage) + "V (cap ridotta)", diameter=None, height=None,
                             voltage=cell_voltage, capacity=nuova_cap)
            return found, cell_info

    # tentativo 3: celle alternative
    if not os.path.exists(CELLE_DISPONIBILI_PATH):
        raise HTTPException(
            status_code=400,
            detail=f"File celle_disponibili.json non trovato in {CELLE_DISPONIBILI_PATH}"
        )
    with open(CELLE_DISPONIBILI_PATH, "r") as f:
        celle_possibili = json.load(f)

    for cell in celle_possibili:
        diam = float(cell["diameter"])
        h = float(cell["height"]) / 10.0
        r = diam / 2.0

        # per il nuovo diametro, ricalcolo il packing ammissibile massimo (upper bound)
        # NB: qui assumo che l’outline sia lo stesso degli input correnti; per sicurezza
        #     ricalcolerai centres fuori, quando selezioni davvero questa cella.
        #     In questo step mi basta un bound: uso max_cells come fallback.
        # Se vuoi, puoi passare un poly pre-caricato e rifare cp qui; per semplicità resto così.

        # prova capacità originale
        res_alt = fc.calcola_configurazioni_migliori(
            cell["voltage"], cell["capacity"], total_voltage, total_current,
            celle_max=max_cells, top_k=1
        )
        lst_alt = _unpack_fc_result(res_alt)
        if lst_alt:
            cell_info = dict(
                name=cell.get("name") or cell.get("type") or "unknown",
                diameter=diam, height=h, voltage=cell["voltage"],
                capacity=cell["capacity"]
            )
            return lst_alt, cell_info

        # prova capacità ridotta
        cfg_tuple_alt, nuova_cap_alt = fc.trova_configurazione_con_capacita_ridotta(
            cell["voltage"], cell["capacity"], total_voltage, total_current, max_cells
        )
        if cfg_tuple_alt:
            found_alt = _unpack_fc_result(cfg_tuple_alt)
            if found_alt:
                cell_info = dict(
                    name=(cell.get("name") or cell.get("type") or "unknown") + " (cap ridotta)",
                    diameter=diam, height=h, voltage=cell["voltage"],
                    capacity=nuova_cap_alt
                )
                return found_alt, cell_info

    # Se ancora nulla…
    return [], {}


# ------------------------- estrazione soluzione CP-SAT -------------------------
def _estrai_gruppi_e_link(
    solver: cp_model.CpSolver,
    x_vars: Dict[Tuple[int, int], Any],
    z1_vars: Dict[Tuple[int, int, int], Any],
    z2_vars: Dict[Tuple[int, int, int], Any],
    E: List[Tuple[int, int]],
    S: int,
) -> Tuple[List[List[int]], List[Dict[str, Any]]]:
    """
    Gruppi: lista di liste con gli indici delle celle per ciascun gruppo k.
    Link: lista di dizionari con (k, i, j) per i link attivi tra gruppi consecutivi.
    """
    # gruppi
    all_i = sorted({i for (i, _) in x_vars.keys()})
    groups = [[] for _ in range(S)]
    for i in all_i:
        for k in range(S):
            if solver.BooleanValue(x_vars[(i, k)]):
                groups[k].append(i)
                break

    # link attivi
    links = []
    for k in range(S - 1):
        for (i, j) in E:
            if solver.BooleanValue(z1_vars[(k, i, j)]) or solver.BooleanValue(z2_vars[(k, i, j)]):
                links.append(dict(k=k, i=i, j=j))
    return groups, links


# ------------------------- pipeline principale -------------------------
def elabora_dati(input_file_path: str):
    # 0) leggo l’intero JSON di input (outline + parametri batteria)
    with open(input_file_path, "r") as f:
        incoming = json.load(f)
        polygon_points = incoming.get("polygon") or incoming.get("vertices") or []
        polygon_points = _normalize_polygon(polygon_points) 

    # outline file: uso direttamente lo stesso JSON come boundary per cp
    json_outline_path = input_file_path

    # carico anche il poligono "geometrico" per eventuale fallback
    poly, _ = cp.load_boundary(Path(json_outline_path))

    # normalizzo i punti del poligono in formato semplice [[x,y],...]
    polygon_points_norm = _normalize_polygon_points(
        incoming.get("polygon") or incoming.get("vertices") or [],
        fallback_poly=poly
    )

    # parametri batteria
    bp = _read_battery_params(incoming)
    cell_voltage = bp["cell_voltage"]
    cell_current = bp["cell_current"]     # Ah
    total_voltage = bp["total_voltage"]
    total_current = bp["total_current"]   # A
    battery_type = bp["battery_type"]
    battery_diameter = bp["battery_diameter"]
    battery_height = bp["battery_height"]




    # 1) CIRCLE PACKING
    R = battery_diameter / 2.0
    centres = _compute_centres_with_cp(json_outline_path, R)
    max_cells = len(centres)
    print("⭕️ numero massimo di celle:", max_cells)





    # 2) CONFIGURAZIONI (nuova logica)
    configs, cell_info = _trova_configurazioni(
        cell_voltage=cell_voltage,
        cell_capacity_ah=cell_current,
        total_voltage=total_voltage,
        total_current=total_current,
        max_cells=max_cells,
        want_at_least=5,
    )
    if not configs:
        raise HTTPException(status_code=400, detail="❌ Nessuna configurazione valida trovata.")

    # Se ho cambiato cella in fallback, devo rifare il packing con il nuovo diametro
    if cell_info.get("diameter"):
        R_alt = float(cell_info["diameter"]) / 2.0
        centres = _compute_centres_with_cp(json_outline_path, R_alt)
        max_cells = len(centres)
        print("⭕️ (alt) numero massimo di celle:", max_cells)
        R = R_alt  # aggiorno raggio reale usato in seguito






    # 3) LAYOUT CP-SAT per ciascuna configurazione selezionata
    coords = np.array(centres, dtype=float)
    output_varianti = []

    for idx, config in enumerate(configs):
        S = int(config["S"])
        P = int(config["P"])
        used_cells = S * P
        if used_cells > max_cells:
            # Skippa config non fattibile rispetto alle celle massime realmente packate
            print(f"⚠️ Skip config {S}S{P}P: richiede {used_cells} > {max_cells} disponibili.")
            continue

        # Parametri solver (tieni i tuoi default preferiti)
        time_budget = 30
        tol = 2.0
        degree_cap = 6
        enforce_degree = False
        target_T = 2 * P
        use_hole_penality = False

        status, solver, x, r, L, z1, z2, E, T = bl.auto_tune_and_solve(
            coords, S, P, R, tol,
            time_budget=time_budget,
            target_T=target_T,
            degree_cap=degree_cap,
            enforce_degree=enforce_degree,
            profiles=("fast", "fast", "quality"),
            seeds=(0, 1, 2, 3),
            workers=6,
            use_hole_penality=use_hole_penality
        )

        status_name = {
            cp_model.OPTIMAL: "OPTIMAL",
            cp_model.FEASIBLE: "FEASIBLE",
            cp_model.INFEASIBLE: "INFEASIBLE",
            cp_model.MODEL_INVALID: "MODEL_INVALID",
            cp_model.UNKNOWN: "UNKNOWN"
        }.get(status, str(status))
        print(f"[{idx}] Solver status: {status_name}")

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            print(f"❌ Nessuna soluzione CP-SAT per {S}S{P}P entro il budget.")
            continue

        links_per_k = [int(solver.Value(v)) for v in L]
        tval = solver.Value(T) if T is not None else None
        print(f"[{idx}] min_k Lk = {tval}   sum(L) = {sum(links_per_k)}   per-k: {links_per_k}")

        # estraggo gruppi e link attivi
        groups, links = _estrai_gruppi_e_link(solver, x, z1, z2, E, S)

                # --- costruisci i segmenti "serie_connections" come coppie di coordinate (from/to)
        serie_connections = _build_serie_connections(links, groups, centres, S)

        # --- normalizza tipi/scale come richiesto dal front-end
        # diameter intero (es. 18), height in "unità *10" (es. 650.0 per 18650)
        diam_out = int(round((cell_info.get("diameter") or battery_diameter)))
        height_out = float((cell_info.get("height") or battery_height) * 10.0)

        voltage_cell_out = int(round(cell_info.get("voltage", cell_voltage)))
        capacity_mAh_out = int(round((cell_info.get("capacity", cell_current)) * 1000.0))

        # battery_parameters completi (usa incoming se ha campi extra, altrimenti fallback sensati)
        battery_id = incoming.get("id")
        total_voltage_out = float(total_voltage)
        total_battery_current_mA_out = float(total_current * 1000.0)

        rated_capacity_mAh = _safe_float(incoming.get("ratedCapacity", capacity_mAh_out))
        max_discharge_A = _safe_float(incoming.get("maxDischarge", 0.0))
        nominal_voltage_V = _safe_float(incoming.get("nominalVoltage", cell_voltage))
        max_charge_voltage_V = _safe_float(incoming.get("maxChargeVoltage", 0.0))
        charging_current_A = _safe_float(incoming.get("chargingCurrent", cell_current))

        # output elettrico come interi
        v_tot_out = int(round(config.get("tensione_effettiva", 0)))
        ah_tot_out = int(round(config.get("capacita_effettiva", 0) * 1000.0))

        timestamp = datetime.datetime.utcnow().isoformat() + "Z"
        variant = {
            "timestamp": timestamp,
            "battery_parameters": {
                "id": battery_id,
                "batteryType": battery_type,
                "totalVoltage": total_voltage_out,
                "totalBatteryCurrent": total_battery_current_mA_out,
                "ratedCapacity": rated_capacity_mAh,
                "maxDischarge": max_discharge_A,
                "nominalVoltage": nominal_voltage_V,
                "maxChargeVoltage": max_charge_voltage_V,
                "chargingCurrent": charging_current_A
            },
            "cell_used": {
                "name": (cell_info.get("name") or battery_type),
                "diameter": diam_out,
                "height": height_out,
                "voltage": voltage_cell_out,
                "capacity": capacity_mAh_out
            },
            "data_output": {
                "S": S,
                "P": P,
                "v_tot": v_tot_out,
                "ah_tot": ah_tot_out,
                "used_cells": used_cells
            },
            "layout_data": {
                "polygon": polygon_points_norm,  # << aggiunto per lo schema del sito
                "circles": [
                    {"center": [float(x_), float(y_)], "radius": float(R)}
                    for (x_, y_) in centres
                ],
                "gruppi": groups,                 # << rename: prima "groups"
                "serie_connections": serie_connections,  # << coordinate from/to, NON indici
                "series_links": links   
            }
        }

        variant_json = os.path.join(OUTPUT_DIR, f"fullOutput_{idx}.json")
        with open(variant_json, "w") as f:
            json.dump(variant, f, indent=2)

        output_varianti.append({"json": variant_json})

    if not output_varianti:
        raise HTTPException(status_code=400, detail="❌ Nessuna variante valida generata (CP-SAT fallito).")

    # ZIP finale
    zip_name = os.path.join(OUTPUT_DIR, "output_varianti.zip")
    with zipfile.ZipFile(zip_name, "w") as zipf:
        for v in output_varianti:
            zipf.write(v["json"], arcname=os.path.basename(v["json"]))


# ------------------------- endpoint FastAPI -------------------------
# uvicorn main:app --host 0.0.0.0 --port 5050
@app.post("/genera-progetti")
async def process_files(file: UploadFile = File(...)):
    print(f"📥 Ricevuto file: {file.filename}")
    input_file_path = "input.json"
    with open(input_file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    elabora_dati(input_file_path)

    zip_path = os.path.abspath(os.path.join(OUTPUT_DIR, "output_varianti.zip"))
    print(f"📦 Inviando file ZIP: {zip_path}")
    return FileResponse(path=zip_path, media_type='application/zip', filename="output_varianti.zip")