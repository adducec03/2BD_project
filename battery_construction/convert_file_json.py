import json

def estrai_poligono_e_dati(percorso_file):
    with open(percorso_file, 'r') as f:
        data = json.load(f)

    battery_id = data.get("progettoId", "unknown")

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

    disegno_str = data.get("disegnoJson")
    if not disegno_str:
        raise ValueError("Campo 'disegnoJson' mancante.")

    try:
        disegno = json.loads(disegno_str)
        vertici = disegno["disegno"]["vertici"]
        polygon = [[v["x"], v["y"]] for v in vertici]
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError("Errore nel parsing di 'disegnoJson' o nei dati interni.") from e

    if polygon[0] != polygon[-1]:
        polygon.append(polygon[0])

    polygon_filename = "polygon.json"
    with open(polygon_filename, 'w') as f:
        json.dump(polygon, f, indent=2)

    return polygon_filename, battery_data