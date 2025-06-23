import json

def estrai_poligono_e_dati(percorso_file):
    # Carica il file JSON principale
    with open(percorso_file, 'r') as f:
        data = json.load(f)

    battery_id = data.get("progettoId", "unknown")

    # Estrai parametri batteria
    battery_data = {
        "id": battery_id,
        "batteryType": data.get("batteryType"),
        "totalVoltage": data.get("totalVoltage"),
        "totalBatteryCurrent": data.get("totalBatteryCurrent"),
        "ratedCapacity": data.get("ratedCapacity"),
        "maxDischarge": data.get("maxDischarge"),
        "nominalVoltage": data.get("nominalVoltage"),
        "maxChargeVoltage": data.get("maxChargeVoltage"),
        "chargingCurrent": data.get("chargingCurrent")
    }

    # Parsing del campo disegnoJson
    disegno_str = data.get("disegnoJson")
    if not disegno_str:
        raise ValueError("Campo 'disegnoJson' mancante.")

    try:
        disegno = json.loads(disegno_str)
        lines = disegno["disegno"]["lines"]
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError("Errore nel parsing di 'disegnoJson' o nei dati interni.") from e

    # Costruzione del poligono usando sia x1,y1 che x2,y2 delle linee, evitando duplicati consecutivi
    polygon = []
    for line in lines:
        pt1 = [line["x1"], line["y1"]]
        pt2 = [line["x2"], line["y2"]]
        if not polygon or polygon[-1] != pt1:
            polygon.append(pt1)
        if polygon[-1] != pt2:
            polygon.append(pt2)

    # Chiudi il poligono se non già chiuso
    if polygon[0] != polygon[-1]:
        polygon.append(polygon[0])

    # Salva il poligono
    polygon_filename = f"polygon_{battery_id}.json"
    with open(polygon_filename, 'w') as f:
        json.dump(polygon, f, indent=2)

    return polygon_filename, battery_data

if __name__ == "__main__":

    input_file_path = "prova.json"
    polygon_file_path, battery_data = estrai_poligono_e_dati(input_file_path)

    print(f"Polygon file created: {polygon_file_path}")
    print(f"Battery data extracted: {battery_data}")

