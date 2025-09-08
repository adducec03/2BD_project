import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

def visualizza_batteria_da_json(path_file):
    # === 1. Carica i dati ===
    with open(path_file, 'r') as f:
        data = json.load(f)

    layout = data["layout_data"]
    polygon = layout["polygon"]
    circles = layout["circles"]
    gruppi = layout.get("gruppi", [])
    connessioni = layout.get("serie_connections", [])

    # === 2. Prepara il grafico ===
    fig, ax = plt.subplots(figsize=(10, 6))

    # === 3. Disegna il poligono ===
    poly_patch = patches.Polygon(polygon, closed=True, edgecolor='black', facecolor='lightblue', alpha=0.2)
    ax.add_patch(poly_patch)

    # === 4. Colori casuali per ogni gruppo ===
    group_colors = {}
    for i, gruppo in enumerate(gruppi):
        color = [random.random() for _ in range(3)]
        group_colors[i] = color
        for idx in gruppo:
            center = circles[idx]["center"]
            radius = circles[idx]["radius"]
            circle = patches.Circle(center, radius, color=color, alpha=0.6, edgecolor='black')
            ax.add_patch(circle)

    # === 5. Disegna le connessioni serie ===
    for conn in connessioni:
        x_values = [conn["from"][0], conn["to"][0]]
        y_values = [conn["from"][1], conn["to"][1]]
        ax.plot(x_values, y_values, 'r--', linewidth=2, label="Connessione serie")

    # === 6. Formatta il grafico ===
    ax.set_title("Visualizzazione configurazione batteria")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.axis("equal")
    ax.grid(True)

    plt.show()

if __name__ == "__main__":
    # Esempio di utilizzo
    visualizza_batteria_da_json("fullOutput_4.json")
