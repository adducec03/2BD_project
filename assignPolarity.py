import json
import matplotlib.pyplot as plt
from shapely.geometry import Polygon


def assign_polarities(input_json, output_json, S, P):
    # Step 1: Carica il JSON
    with open(input_json, 'r') as f:
        data = json.load(f)

    circles = data['circles']

    if len(circles) < S * P:
        raise ValueError(f"Non ci sono abbastanza celle ({len(circles)}) per la configurazione richiesta {S}S{P}P.")

    # Step 2: Estrai e ordina i centri
    centers = [(c['center'][0], c['center'][1]) for c in circles]
    radius = circles[0]['radius']  # Assumiamo stesso raggio per tutte

    # Ordina: da sinistra a destra, poi dal basso verso l'alto
    ordered = sorted(centers, key=lambda p: (p[0], p[1]))

    # Step 3: Raggruppa in S gruppi da P
    usable = ordered[:S * P]
    groups = [usable[i * P:(i + 1) * P] for i in range(S)]

    # Step 4: Assegna polarità alternata per serie
    labeled_cells = []
    for i, group in enumerate(groups):
        polarity = '+' if i % 2 == 0 else '-'
        for cx, cy in group:
            labeled_cells.append({
                "center": [cx, cy],
                "radius": radius,
                "polarity": polarity
            })

    # Step 5: Esporta nuovo JSON
    output = {
        "polygon": data.get("polygon", []),
        "circles": labeled_cells
    }

    with open(output_json, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"✅ Polarità assegnate e salvate in '{output_json}'")

    # Step 6: Disegna risultato
    plot_labeled_cells(output)


def plot_labeled_cells(data):
    fig, ax = plt.subplots()
    
    # Disegna il poligono
    if "polygon" in data and data["polygon"]:
        poly = Polygon(data["polygon"])
        x, y = poly.exterior.xy
        ax.plot(x, y, color='black', linewidth=2)

    # Disegna le celle
    for cell in data["circles"]:
        cx, cy = cell["center"]
        r = cell["radius"]
        polarity = cell["polarity"]

        color = 'red' if polarity == '+' else 'blue'
        face_color = '#ccf2ff' if polarity == '+' else '#e0e0ff'

        # Cerchio
        circle = plt.Circle((cx, cy), r, edgecolor='black', facecolor=face_color, alpha=0.8)
        ax.add_patch(circle)

        # Simbolo + o -
        ax.text(cx, cy, polarity, color=color, fontsize=12, weight='bold', ha='center', va='center')

    ax.set_aspect('equal')
    ax.set_title("Configurazione Celle con Polarità")
    plt.show()


# Esempio di utilizzo
if __name__ == "__main__":
    assign_polarities(
        input_json="polygon_with_circles.json",
        output_json="labeled_cells.json",
        S=9,
        P=5
    )