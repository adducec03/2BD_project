import circlePacking as cp
import assignPolarityWithLinking as ap
import findBestConfiguration as fc
import convert_file_json as cf
import json
import os

input_file = '../json_valeria/4.json'
battery_diameter = 18


if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    polygon_file_path = os.path.join(BASE_DIR, "polygon.json")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    polygon_with_circles_file_path = os.path.join(BASE_DIR, "polygon_with_circles.json")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    polygon_with_circles_and_connections_file_path = os.path.join(BASE_DIR, "polygon_with_circles_and_connections.json")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    input_file_path = os.path.join(BASE_DIR, input_file)


    # 1. Converti file json
    try:
        poligono = cf.estrai_poligono_da_json(input_file_path)
        cf.salva_poligono_su_file(poligono, polygon_file_path)
        print(f"Poligono salvato in polygon.json con successo.")
    except Exception as e:
        print(f"Errore: {e}")


    # 2. Circle packing
    with open(polygon_file_path, "r") as f:
        vertices = json.load(f)
    x, y = zip(*vertices)
    poly = cp.Polygon(vertices)
    circle_radius = battery_diameter
    best_centers, best_angle = cp.find_best_rotation(poly, circle_radius)
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
    config = fc.configurazione_bilanciata(max_cells)


    # 4. Assegna le polarità alle celle e crea i collegamenti
    S, P, v_tot, ah_tot = config
    used_cells=S*P
    with open(polygon_with_circles_file_path, "r") as f:
        data = json.load(f)
    polygon = data["polygon"]
    circles = data["circles"]
    radius = circles[0]["radius"]
    centers = [tuple(c["center"]) for c in circles]
    gruppi = ap.trova_gruppi_con_raggio_adattivo(centers, radius, S, P)
    centroidi = ap.calcola_centroidi_gruppi(centers, gruppi)
    connessioni = ap.crea_collegamenti_serie_ottimizzati(gruppi,centers,radius)
    data_output = {
        "polygon": polygon,
        "circles": circles,
        "gruppi": gruppi,
        "serie_connections": connessioni
    }
    with open(polygon_with_circles_and_connections_file_path, "w") as f:
        json.dump(data_output, f, indent=2)
    print("✅ File salvato: polygon_with_circles_and_connections.json")
    with open(polygon_with_circles_and_connections_file_path, "r") as f:
        data_loaded = json.load(f)
    ap.plot_batteria_con_collegamenti(data_loaded, radius,S,P,ah_tot,v_tot,used_cells)

    