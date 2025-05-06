import json

# === CONFIGURAZIONE ===
input_file = 'json_valeria/1.json'
output_file = 'polygon.json'

# === FUNZIONE DI CONVERSIONE ===
def estrai_poligono_da_json(percorso_file):
    with open(percorso_file, 'r') as f:
        data = json.load(f)
    
    lines = data.get("lines", [])
    if not lines:
        raise ValueError("Nessuna linea trovata nel file JSON.")

    # Costruisci il poligono come lista di coordinate
    polygon = [[line["x1"], line["y1"]] for line in lines]
    polygon.append([lines[-1]["x2"], lines[-1]["y2"]])  # chiusura del poligono

    return polygon

# === SALVATAGGIO SU FILE ===
def salva_poligono_su_file(polygon, percorso_output):
    with open(percorso_output, 'w') as f:
        json.dump(polygon, f, indent=2)

# === ESECUZIONE ===
if __name__ == "__main__":
    try:
        poligono = estrai_poligono_da_json(input_file)
        salva_poligono_su_file(poligono, output_file)
        print(f"Poligono salvato in '{output_file}' con successo.")
    except Exception as e:
        print(f"Errore: {e}")