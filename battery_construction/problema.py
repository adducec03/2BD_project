import json
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import networkx as nx

def calcola_configurazione_batteria(v_cella, ah_cella, v_target, ah_target, celle_disponibili):
    s = round(v_target / v_cella)
    p = round(ah_target / ah_cella)
    totale = s * p

    print(f"Configurazione suggerita: {s}S{p}P")
    print(f"Celle totali richieste: {totale}")

    if totale <= celle_disponibili:
        print("✅ Le celle entrano nella forma disponibile.")
    else:
        print("⚠️ Troppe celle. Considera di abbassare la capacità o cambiare disposizione.")
    return s, p, totale

def costruisci_grafo_adiacenza(centers, distanza_massima):
    G = nx.Graph()
    for i, c1 in enumerate(centers):
        G.add_node(i, center=c1)
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dist = np.linalg.norm(np.array(centers[i]) - np.array(centers[j]))
            if dist <= distanza_massima:
                G.add_edge(i, j)
    return G

def estrai_gruppi_connessi(G, S, P):
    """Trova S gruppi distinti da P celle connesse usando BFS."""
    gruppi = []
    usate = set()

    for componente in nx.connected_components(G):
        nodi = list(componente)
        subG = G.subgraph(nodi)

        for nodo in nodi:
            if nodo in usate:
                continue

            visitati = set()
            gruppo = []
            coda = deque([nodo])

            while coda and len(gruppo) < P:
                current = coda.popleft()
                if current in visitati or current in usate:
                    continue
                visitati.add(current)
                gruppo.append(current)
                for vicino in G.neighbors(current):
                    if vicino not in visitati and vicino not in usate:
                        coda.append(vicino)

            if len(gruppo) == P:
                gruppi.append(gruppo)
                usate.update(gruppo)
                if len(gruppi) == S:
                    return gruppi
    raise ValueError(f"Impossibile formare {S} gruppi di {P} celle adiacenti.")

def trova_gruppi_con_raggio_adattivo(centers, radius, S, P, raggio_iniziale=2.0, raggio_massimo=6.0, passo=0.5):
    for moltiplicatore in np.arange(raggio_iniziale, raggio_massimo + passo, passo):
        distanza = moltiplicatore * radius
        print(f"Tentativo con distanza massima: {distanza:.2f}")
        G = costruisci_grafo_adiacenza(centers, distanza)
        try:
            gruppi = estrai_gruppi_connessi(G, S, P)
            print(f"✅ Gruppi trovati con distanza {distanza:.2f}")
            return gruppi
        except ValueError as e:
            print(f"❌ {e}")
    raise ValueError(f"Impossibile formare {S} gruppi di {P} celle adiacenti anche aumentando il raggio.")

def plot_labeled_battery(polygon, centers, radius, gruppi):
    fig, ax = plt.subplots(figsize=(12, 8))
    x, y = zip(*polygon)
    ax.plot(x, y, color='black', linewidth=2)

    colors = plt.cm.get_cmap('tab20', len(gruppi))
    for idx, gruppo in enumerate(gruppi):
        polarity = '+' if idx % 2 == 0 else '-'
        for n in gruppo:
            cx, cy = centers[n]
            circle = plt.Circle((cx, cy), radius, edgecolor='black', facecolor=colors(idx), alpha=0.6)
            ax.add_patch(circle)
            ax.text(cx, cy, f"{polarity}\nG{idx}", ha='center', va='center', fontsize=8, weight='bold')
    ax.set_aspect('equal')
    plt.grid(True, alpha=0.3)
    plt.title("Disposizione batterie con polarità e gruppi")
    plt.show()

if __name__ == "__main__":
    # 1. Calcola configurazione
    S, P, total = calcola_configurazione_batteria(
        v_cella=3.6,
        ah_cella=2.5,
        v_target=28.8,
        ah_target=47.50,
        celle_disponibili=152
    )

    # 2. Carica dati
    with open("polygon_with_circles.json", "r") as f:
        data = json.load(f)
    
    polygon = data["polygon"]
    circles = data["circles"]
    radius = circles[0]["radius"]
    centers = [tuple(c["center"]) for c in circles]

    # 3. Crea grafo e trova gruppi
    gruppi = trova_gruppi_con_raggio_adattivo(centers, radius, S, P)

    # 4. Visualizza
    plot_labeled_battery(polygon, centers, radius, gruppi)