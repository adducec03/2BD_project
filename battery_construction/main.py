import datetime
import circlePacking2 as cp2
import assignPolarityWithLinking as ap
import findBestConfiguration as fc
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
    total_current = int(battery_data.get("totalBatteryCurrent"))/1000
    cell_voltage = int(battery_data.get("nominalVoltage"))
    cell_current = int(battery_data.get("chargingCurrent"))
    battery_type = battery_data.get("batteryType")
    battery_diameter = int(battery_type[:2]) if battery_type else None
    battery_height = int(battery_type[2:]) / 10



    # 2. Circle packing
    with open(polygon_file_path, "r") as f:
        vertices = json.load(f)
    x, y = zip(*vertices)
    poly = cp2.Polygon(vertices)
    circle_radius = battery_diameter/2
    best_centers, best_angle = cp2.find_best_packing(poly, circle_radius)
    max_cells = len(best_centers)
    export_data = {
        "polygon": vertices,
        "circles": [
            {"center": [x, y], "radius": circle_radius}
            for x, y in best_centers
        ]
    }
    with open(polygon_with_circles_file_path, "w") as f:
        json.dump(export_data, f, indent=2)


    # 3. Calcola configurazione (per il momento consiglia solo una configurazione bilanciata)
    config = fc.calcola_disposizione(cell_voltage, cell_current, total_voltage, total_current)
    S = config.get("S")
    P = config.get("P")
    v_tot = config.get("tensione_effettiva")
    ah_tot = config.get("corrente_effettiva")

    #controllo sul numero di celle massimo
    if S * P > max_cells:
        raise HTTPException(
            status_code=400,
            detail="Numero di celle richiesto troppo alto per la forma fornita."
        )

    # 4. Assegna le polarità alle celle e crea i collegamenti
    used_cells=S*P
    with open(polygon_with_circles_file_path, "r") as f:
        data = json.load(f)
    polygon = data["polygon"]
    circles = data["circles"]
    radius = circles[0]["radius"]*4
    centers = [tuple(c["center"]) for c in circles]
    gruppi = ap.trova_gruppi_con_raggio_adattivo(centers, radius, S, P)
    centroidi = ap.calcola_centroidi_gruppi(centers, gruppi)
    connessioni = ap.crea_collegamenti_serie_ottimizzati(gruppi,centers,radius)
    layout_data = {
        "polygon": polygon,
        "circles": circles,
        "gruppi": gruppi,
        "serie_connections": connessioni
    }
    with open(polygon_with_circles_and_connections_file_path, "w") as f:
        json.dump(layout_data, f, indent=2)

    # 5. Crea il file di output con le specifiche della batteria
    data_output = {
        "S": S,
        "P": P,
        "v_tot": v_tot, # Tensione totale della batteria
        "ah_tot": ah_tot*1000, # Capacità totale della batteria (mAh)
        "used_cells": used_cells, # Celle utilizzate
    }

    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    merged_data = {
        "timestamp": timestamp,
        "battery_parameters": battery_data,
        "data_output": data_output,
        "layout_data": layout_data
    }

    with open("fullOutput.json", "w") as f:
        json.dump(merged_data, f, indent=2)

    # 6. Converti da 2D a 3D
    td.convert_2d_to_3d(battery_height,
                        polygon_with_circles_and_connections_file_path,
                        battery3D_file_path,
                        model_battery_file_path)




@app.post("/genera-progetti")
async def process_files(file: UploadFile = File(...)):
    print(f"📥 Ricevuto file: {file.filename}")
    input_id = uuid.uuid4().hex
    input_file_path = "input.json"
    output_json_path = "fullOutput.json"
    output_glb_path = "battery3D.glb"
    zip_path = "output.zip"

    # Salva il file JSON ricevuto
    with open(input_file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Elabora i dati (questa funzione genera anche il file .glb e .json)
    elabora_dati(input_file_path)

    # Comprimiamo i due file da restituire in un .zip
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(output_json_path, arcname="output.json")
        #zipf.write(output_glb_path, arcname="model.glb")

    return FileResponse(zip_path, media_type='application/zip', filename="output_package.zip")

#if __name__ == "__main__":
#
#    input_file = "input.json"
#    input_file_path = os.path.join(BASE_DIR, input_file)
#
#
#    process_files = app.post("/process")(process_files)
#    if(process_files==None):
#        input_file_path = os.path.join(BASE_DIR, input_file)
#        elabora_dati(input_file_path)


 

    