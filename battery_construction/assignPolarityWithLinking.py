import json
import numpy as np
from collections import deque
import networkx as nx



#Crea un grafo non orientato dove ogni nodo è una cella e viene aggiunto un arco tra
#due celle se la loro distanza è minore o uduale alla distanza massima
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





#trova tutte le componenti connesse del grafo G
#Per ogni comopnenete analizza i singoli nodi della componente
# Se non è stato gia segnata come usata esegue una bfs per travare un gruppo da P celle adiacenti
def estrai_gruppi_connessi(G, S, P):
    gruppi = []
    usate = set()

    for componente in nx.connected_components(G):
        nodi = list(componente)

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





# varia il raggio di connesisone massimo da usare . Per ogni calore costruisce un grafo di adiacenza 
# e prova a estrarre i gruppi. Se non riesce aumenta il raggio e riprova.
def trova_gruppi_con_raggio_adattivo(centers, radius, S, P, raggio_iniziale=2.0, raggio_massimo=20.0, passo=0.5):
    for moltiplicatore in np.arange(raggio_iniziale, raggio_massimo + passo, passo):
        distanza = moltiplicatore * radius
        #print(f"Tentativo con distanza massima: {distanza:.2f}")
        G = costruisci_grafo_adiacenza(centers, distanza)
        try:
            gruppi = estrai_gruppi_connessi(G, S, P)
            print(f"✅ Gruppi trovati con distanza {distanza:.2f}")
            return gruppi
        except ValueError:
            continue
    

def verifica_gruppi_connessi(gruppi, centers, distanza_verifica):
    nuovi_gruppi = []
    for gruppo in gruppi:
        G = nx.Graph()
        for i in gruppo:
            G.add_node(i)
        for i in gruppo:
            for j in gruppo:
                if i < j:
                    dist = np.linalg.norm(np.array(centers[i]) - np.array(centers[j]))
                    if dist <= distanza_verifica:
                        G.add_edge(i, j)
        componenti = list(nx.connected_components(G))

        if len(componenti) > 1:
            print(f"⚠️ Gruppo spezzato: {len(componenti)} componenti")

        for comp in componenti:
            nuovi_gruppi.append(list(comp))
    return nuovi_gruppi



def riassegna_gruppi_piccoli(gruppi, centers, P, distanza_massima):
    gruppi_finali = []
    buffer = []

    for gruppo in gruppi:
        if len(gruppo) == P:
            gruppi_finali.append(gruppo)
        else:
            buffer.append(gruppo)

    for piccolo in buffer:
        centro_piccolo = np.mean([centers[i] for i in piccolo], axis=0)
        min_dist = float('inf')
        gruppo_target = None
        for g in gruppi_finali:
            if len(g) >= P:
                continue
            centro_g = np.mean([centers[i] for i in g], axis=0)
            dist = np.linalg.norm(centro_piccolo - centro_g)
            if dist < min_dist:
                min_dist = dist
                gruppo_target = g
        if gruppo_target and len(gruppo_target) + len(piccolo) <= P:
            gruppo_target.extend(piccolo)
    return gruppi_finali




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


