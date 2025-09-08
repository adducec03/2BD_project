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
        lines = disegno["disegno"]["lines"]
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError("Errore nel parsing di 'disegnoJson'.") from e

    # Costruisci un grafo di connessioni punto → [successori]
    from collections import defaultdict, deque

    connessioni = defaultdict(list)
    for line in lines:
        pt1 = (line["x1"], line["y1"])
        pt2 = (line["x2"], line["y2"])
        connessioni[pt1].append(pt2)
        connessioni[pt2].append(pt1)  # bidirezionale

    # Trova un punto di partenza (uno qualsiasi)
    start = next(iter(connessioni))
    visited = set()
    polygon = []

    # DFS per seguire i punti connessi in sequenza
    def dfs(p, prev=None):
        polygon.append(list(p))
        visited.add(p)
        for neighbor in connessioni[p]:
            if neighbor != prev and neighbor not in visited:
                dfs(neighbor, p)

    dfs(start)

    # Chiudi il poligono se non è già chiuso
    if polygon[0] != polygon[-1]:
        polygon.append(polygon[0])

    # Salva su file
    polygon_filename = f"polygon_{battery_id}.json"
    with open(polygon_filename, 'w') as f:
        json.dump(polygon, f, indent=2)

    return polygon_filename, battery_data