import datetime
import circlePacking2 as cp2
import assignPolarityWithLinking as ap
#import findBestConfiguration as fc
import findBestConfiguration2 as fc
import convert_file_json as cf
import construction3D as td
import json
import os
from fastapi import FastAPI, HTTPException, Request
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import shutil
import uuid
import zipfile

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
polygon_file_path = os.path.join(BASE_DIR, "polygon.json")
polygon_with_circles_file_path = os.path.join(BASE_DIR, "polygon_with_circles.json")
polygon_with_circles_and_connections_file_path = os.path.join(BASE_DIR, "polygon_with_circles_and_connections.json")
battery3D_file_path = os.path.join(BASE_DIR, "battery3D.glb")
model_battery_file_path = os.path.join(BASE_DIR, "model_battery.glb")


def elabora_dati(input_file_path):
    # 1. Converti file json
    polygon_file_path, battery_data = cf.estrai_poligono_e_dati(input_file_path)

    total_voltage = int(battery_data.get("totalVoltage"))
    total_current = int(battery_data.get("totalBatteryCurrent")) / 1000
    cell_voltage = int(battery_data.get("nominalVoltage"))
    cell_current = int(battery_data.get("chargingCurrent"))
    battery_type = battery_data.get("batteryType")
    battery_diameter = int(battery_type[:2]) if battery_type else None
    battery_height = int(battery_type[2:]) / 10

    # 2. Circle packing
    with open(polygon_file_path, "r") as f:
        polygon_data = json.load(f)
    vertices = polygon_data["vertices"] if isinstance(polygon_data, dict) else polygon_data

    poly = cp2.Polygon(vertices)
    circle_radius = battery_diameter / 2
    best_centers, best_angle = cp2.find_best_packing(poly, circle_radius)
    max_cells = len(best_centers)

    # 3. Calcola le 5 migliori configurazioni
    configurazioni = fc.calcola_configurazioni_migliori(
        cell_voltage, cell_current,
        total_voltage, total_current,
        celle_max=max_cells, top_k=5
    )

    if not configurazioni:
        raise HTTPException(
            status_code=400,
            detail="❌ Nessuna configurazione valida trovata. Rivedi i parametri della batteria o la forma del contenitore."
        )

    output_varianti = []

    for i, config in enumerate(configurazioni):
        S = config["S"]
        P = config["P"]
        used_cells = S * P

        # Salva il layout con polarità e collegamenti
        export_data = {
            "polygon": vertices,
            "circles": [
                {"center": [x, y], "radius": circle_radius} for x, y in best_centers
            ]
        }
        layout_path = f"polygon_with_circles_{i}.json"
        with open(layout_path, "w") as f:
            json.dump(export_data, f, indent=2)

        with open(layout_path, "r") as f:
            data = json.load(f)

        centers = [tuple(c["center"]) for c in data["circles"]]
        radius = data["circles"][0]["radius"] * 4
        gruppi = ap.trova_gruppi_con_raggio_adattivo(centers, radius, S, P)
        connessioni = ap.crea_collegamenti_serie_ottimizzati(gruppi, centers, radius)

        layout_data = {
            "polygon": data["polygon"],
            "circles": data["circles"],
            "gruppi": gruppi,
            "serie_connections": connessioni
        }

        # Specifiche batteria
        data_output = {
            "S": S,
            "P": P,
            "v_tot": config["tensione_effettiva"],
            "ah_tot": config["capacita_effettiva"] * 1000,
            "used_cells": used_cells
        }

        timestamp = datetime.datetime.utcnow().isoformat() + "Z"
        merged_data = {
            "timestamp": timestamp,
            "battery_parameters": battery_data,
            "data_output": data_output,
            "layout_data": layout_data
        }

        variant_json = f"fullOutput_{i}.json"
        with open(variant_json, "w") as f:
            json.dump(merged_data, f, indent=2)

        # 3D model
        variant_glb = f"battery3D_{i}.glb"
        model_glb = "model_battery.glb"
        td.convert_2d_to_3d(battery_height,
                            polygon_with_circles_and_connections_file_path,
                            variant_glb,
                            model_glb)

        output_varianti.append({
            "json": variant_json,
            "glb": variant_glb
        })

    # Comprimiamo in un .zip finale tutte le varianti
    with zipfile.ZipFile("output_varianti.zip", 'w') as zipf:
        for variant in output_varianti:
            zipf.write(variant["json"], arcname=os.path.basename(variant["json"]))
            zipf.write(variant["glb"], arcname=os.path.basename(variant["glb"]))



# comando per avviare il server locale:
# uvicorn main:app --host 0.0.0.0 --port 5050
@app.post("/genera-progetti")
async def process_files(file: UploadFile = File(...)):
    print(f"📥 Ricevuto file: {file.filename}")
    input_id = uuid.uuid4().hex
    input_file_path = "input.json"
    output_json_path = "fullOutput.json"
    output_glb_path = "battery3D.glb"
    zip_path = "output_varianti.zip"

    # Salva il file JSON ricevuto
    with open(input_file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Elabora i dati e li comprime(questa funzione genera anche il file .glb e .json)
    elabora_dati(input_file_path)

    # Comprime tutti i full_output_{i}.json generati
    #with zipfile.ZipFile(zip_path, 'w') as zipf:
    #    for filename in os.listdir("."):
    #        if filename.startswith("full_output_") and filename.endswith(".json"):
    #            zipf.write(filename, arcname=filename)




        #with zipfile.ZipFile(zip_path, 'w') as zipf:
        #zipf.write(output_json_path, arcname="output.json")
        #zipf.write(output_glb_path, arcname="model.glb")

    print(f"📦 Inviando file ZIP: {zip_path}")
    return FileResponse(path=os.path.abspath(zip_path), media_type='application/zip', filename="output_varianti.zip")

