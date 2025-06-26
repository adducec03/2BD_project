import json
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import os

import json
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import os
import matplotlib.cm as cm
import numpy as np

def plot_battery_layout(json_path):
    if not os.path.exists(json_path):
        print(f"❌ File non trovato: {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    layout = data.get("layout_data", {})
    circles = layout.get("circles", [])
    polygon = layout.get("polygon", [])
    series_connections = layout.get("serie_connections", [])
    gruppi = layout.get("gruppi", [])

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect('equal')

    # Disegna il poligono
    if polygon:
        ax.add_patch(Polygon(polygon, closed=True, fill=False, edgecolor='black', linewidth=1.5, label='Polygon'))

    # Colori per i gruppi
    cmap = cm.get_cmap('tab20', len(gruppi))  # tab20 = 20 colori diversi

    for group_idx, group in enumerate(gruppi):
        color = cmap(group_idx)
        for idx in group:
            if idx < len(circles):  # protezione da errori di indice
                c = circles[idx]
                ax.add_patch(Circle(c["center"], c["radius"], fill=True, edgecolor='black', facecolor=color, alpha=0.8))

    # Disegna le connessioni
    for conn in series_connections:
        p1, p2 = conn["from"], conn["to"]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r--', linewidth=2, label='Connection' if 'Connection' not in ax.get_legend_handles_labels()[1] else "")

    plt.title("Battery Layout with Colored Groups")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():

    plot_battery_layout("fullOutput.json")

if __name__ == "__main__":
    main()