import json
import matplotlib.pyplot as plt
from shapely.geometry import Point
import numpy as np
from sklearn.cluster import DBSCAN


def calcola_configurazione_batteria(v_cella, ah_cella, v_target, ah_target, celle_disponibili):
    s = round(v_target / v_cella)
    p = round(ah_target / ah_cella)
    totale = s * p

    print(f"Configurazione suggerita: {s}S{p}P")
    print(f"Celle totali richieste: {totale}")

    if totale <= celle_disponibili:
        print("Le celle entrano nella forma disponibile.")
    else:
        print("⚠️ Troppe celle. Considera di abbassare la capacità o cambiare disposizione.")
    return s, p, totale

def assign_from_grid(centers, radius, S, P, tolerance=0.3):
    centers = np.array(centers)

    # Clusterizza per Y (altezza) per ottenere le righe
    clustering = DBSCAN(eps=radius * 1.5, min_samples=P, metric=lambda a, b: abs(a[1] - b[1]))
    labels = clustering.fit_predict(centers)

    rows_dict = {}
    for i, label in enumerate(labels):
        if label == -1:
            continue
        rows_dict.setdefault(label, []).append(centers[i])

    rows = list(rows_dict.values())

    # Ordina le celle in ogni riga da sinistra a destra
    for row in rows:
        row.sort(key=lambda p: p[0])

    rows.sort(key=lambda row: np.mean([p[1] for p in row]))  # ordina le righe dall'alto in basso

    labeled = []
    series_idx = 0

    for row in rows:
        i = 0
        while i + P <= len(row) and series_idx < S:
            group = row[i:i + P]
            polarity = '+' if series_idx % 2 == 0 else '-'
            for cx, cy in group:
                labeled.append({
                    "center": [cx, cy],
                    "radius": radius,
                    "polarity": polarity
                })
            series_idx += 1
            i += P

        if series_idx >= S:
            break

    if series_idx < S:
        raise ValueError(f"Non sono riuscito a formare {S} gruppi in serie. Trovati solo {series_idx} gruppi.")
    
    return labeled
    # Ordina per Y per trovare le righe
    centers = sorted(centers, key=lambda p: p[1])
    rows = []
    current_row = []
    last_y = None

    for cx, cy in centers:
        if last_y is None or abs(cy - last_y) < tolerance:
            current_row.append((cx, cy))
        else:
            rows.append(current_row)
            current_row = [(cx, cy)]
        last_y = cy
    if current_row:
        rows.append(current_row)

    # Ordina per X ogni riga
    for row in rows:
        row.sort(key=lambda p: p[0])

    # Appiattisci la griglia
    grid = [cell for row in rows for cell in row]

    if len(grid) < S * P:
        raise ValueError(f"Non ci sono abbastanza celle ({len(grid)}) per {S}S{P}P")

    labeled = []
    used = 0
    series_idx = 0
    for row in rows:
        if len(row) < P:
            continue
        if series_idx >= S:
            break

        group = row[:P]
        polarity = '+' if series_idx % 2 == 0 else '-'

        for cx, cy in group:
            labeled.append({
                "center": [cx, cy],
                "radius": radius,
                "polarity": polarity
            })
            used += 1
        series_idx += 1

    if used < S * P:
        raise ValueError(f"Non sono riuscito a comporre tutti i gruppi! Solo {used} celle assegnate.")
    
    return labeled

def plot_labeled_battery(polygon, cells):
    fig, ax = plt.subplots()
    # Disegna il contorno
    x, y = zip(*polygon)
    ax.plot(x, y, color='black')

    # Disegna le celle
    for cell in cells:
        cx, cy = cell["center"]
        r = cell["radius"]
        polarity = cell["polarity"]
        circle = plt.Circle((cx, cy), r, edgecolor='blue', facecolor='lightblue', alpha=0.6)
        ax.add_patch(circle)
        ax.text(cx, cy, polarity, ha='center', va='center', fontsize=10, weight='bold', color='black')

    ax.set_aspect('equal')
    plt.title("Disposizione batterie con polarità")
    plt.show()

# MAIN PROGRAM
if __name__ == "__main__":
    # STEP 1: Calcola configurazione
    S, P, total = calcola_configurazione_batteria(
        v_cella=3.6,
        ah_cella=2.5,
        v_target=37,
        ah_target=18,
        celle_disponibili=76
    )

    # STEP 2: Carica i dati dal JSON
    with open("polygon_with_circles.json", "r") as f:
        data = json.load(f)
    
    polygon = data["polygon"]
    circles = data["circles"]
    radius = circles[0]["radius"]
    centers = [tuple(c["center"]) for c in circles]

    # STEP 3: Assegna polarità
    labeled_cells = assign_from_grid(centers, radius, S, P)

    # STEP 4: Visualizza
    plot_labeled_battery(polygon, labeled_cells)