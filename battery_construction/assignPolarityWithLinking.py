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

def plot_batteria_con_collegamenti(data, radius, S, P, ah_tot, v_tot, cells_tot):
    polygon = data["polygon"]
    circles = data["circles"]
    gruppi = data["gruppi"]
    connessioni = data["serie_connections"]

    centers = [tuple(c["center"]) for c in circles]

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

    # Disegna collegamenti tra centroidi
    for conn in connessioni:
        x_vals = [conn["from"][0], conn["to"][0]]
        y_vals = [conn["from"][1], conn["to"][1]]
        ax.plot(x_vals, y_vals, color='red', linestyle='--', linewidth=2)

    ax.set_aspect('equal')
    plt.grid(True, alpha=0.3)
    if(P==None or S==None or ah_tot==None or v_tot==None or cells_tot==None):
        print("Disposizione celle in serie con collegmenti")
    else:
        plt.title(f"Disposizione celle in {S}s{P}p\nCapacità:{ah_tot}A\nVoltaggio:{v_tot}V\nNumero celle totali:{cells_tot}")
    plt.show()

def calcola_centroidi_gruppi(centers, gruppi):
    centroidi = []
    for gruppo in gruppi:
        coords = np.array([centers[i] for i in gruppo])
        centroid = coords.mean(axis=0)
        centroidi.append(centroid.tolist())
    return centroidi

def crea_collegamenti_serie_ottimizzati(gruppi, centers, radius):
    # Costruisci il grafo dei gruppi
    G = nx.Graph()
    num_gruppi = len(gruppi)
    
    polarita = ['+' if i % 2 == 0 else '-' for i in range(num_gruppi)]
    centroidi = calcola_centroidi_gruppi(centers, gruppi)
    
    # Crea archi tra gruppi con polarità opposte e molte celle adiacenti
    for i in range(num_gruppi):
        for j in range(i + 1, num_gruppi):
            if polarita[i] == polarita[j]:
                continue  # Serve alternanza
            contatti = conta_celle_adiacenti(gruppi[i], gruppi[j], centers, soglia_distanza=2.5 * radius)
            if contatti > 0:
                distanza = np.linalg.norm(np.array(centroidi[i]) - np.array(centroidi[j]))
                # Pesiamo l'arco con la distanza penalizzata e il numero di contatti
                peso = distanza - 0.5 * contatti
                G.add_edge(i, j, weight=peso)

    # Trova cammino ottimo usando greedy alternando polarità
    connessioni = []
    visitati = set()
    nodo_corrente = 0
    visitati.add(nodo_corrente)

    while len(visitati) < num_gruppi:
        candidati = [
            (vicino, G[nodo_corrente][vicino]['weight'])
            for vicino in G.neighbors(nodo_corrente)
            if vicino not in visitati and polarita[vicino] != polarita[nodo_corrente]
        ]
        if not candidati:
            break  # Cammino interrotto
        prossimo, _ = min(candidati, key=lambda x: x[1])
        connessioni.append({
            "from": centroidi[nodo_corrente],
            "to": centroidi[prossimo]
        })
        nodo_corrente = prossimo
        visitati.add(nodo_corrente)

    return connessioni

def conta_celle_adiacenti(gruppo1, gruppo2, centers, soglia_distanza):
    count = 0
    for i in gruppo1:
        for j in gruppo2:
            dist = np.linalg.norm(np.array(centers[i]) - np.array(centers[j]))
            if dist <= soglia_distanza:
                count += 1
    return count

if __name__ == "__main__":
    # 1. Calcola configurazione
    S, P, total = calcola_configurazione_batteria(
        v_cella=3.6,
        ah_cella=2.5,
        v_target=47,
        ah_target=22,
        celle_disponibili=125
    )

    # 2. Carica dati originali
    with open("polygon_with_circles.json", "r") as f:
        data = json.load(f)
    
    polygon = data["polygon"]
    circles = data["circles"]
    radius = circles[0]["radius"]
    centers = [tuple(c["center"]) for c in circles]

    # 3. Trova gruppi
    gruppi = trova_gruppi_con_raggio_adattivo(centers, radius, S, P)

    # 4. Calcola centroidi e collegamenti
    centroidi = calcola_centroidi_gruppi(centers, gruppi)
    connessioni = crea_collegamenti_serie_ottimizzati(gruppi,centers,radius)

    # 5. Salva nuovo file con connessioni
    data_output = {
        "polygon": polygon,
        "circles": circles,
        "gruppi": gruppi,
        "serie_connections": connessioni
    }

    with open("polygon_with_circles_and_connections.json", "w") as f:
        json.dump(data_output, f, indent=2)

    print("✅ File salvato: polygon_with_circles_and_connections.json")

    # 6. Visualizza il file salvato
    with open("polygon_with_circles_and_connections.json", "r") as f:
        data_loaded = json.load(f)

    plot_batteria_con_collegamenti(data_loaded, radius)