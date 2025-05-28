import json
import math
import itertools
import networkx as nx
from pathlib import Path
from typing import Union


def load_circles(json_path: Union[str, Path], key: str = "circles"):
    """
    Carica dal file .json tutti i cerchi del dizionario `key`.
    Ogni cerchio deve avere almeno 'x' e 'y'. Se 'id' manca
    viene usato l'indice nell'array.
    """
    with open(json_path, "r") as fh:
        data = json.load(fh)

    circles = data[key]
    nodes = []
    for idx, c in enumerate(circles):
        node_id = c.get("id", idx)     # id esplicito oppure indice
        nodes.append(
            {"id": node_id,
             "x":  float(c["x"]),
             "y":  float(c["y"])}
        )
    return nodes

def circles_to_graph(circles, r: float = 18.0, tol: float = 1e-2):
    """
    Converte la lista di cerchi (diz.) in un grafo semplice non pesato.
    
    Parametri
    ---------
    circles : list[dict] – oggetti con 'id', 'x', 'y'
    r       : float      – raggio comune (default 18)
    tol     : float      – tolleranza assoluta sul confronto di distanza
    
    Ritorna
    -------
    networkx.Graph
    """
    target = 2 * r                 # diametro atteso (36)
    target2 = target * target      # confrontiamo i quadrati

    G = nx.Graph()
    for c in circles:
        G.add_node(c["id"], **c)   # i cerchi vengono salvati come attributi
    
    # Esplora tutte le coppie non ordinate
    for a, b in itertools.combinations(circles, 2):
        dx = a["x"] - b["x"]
        dy = a["y"] - b["y"]
        dist2 = dx*dx + dy*dy
        if math.isclose(dist2, target2, abs_tol=2*target*tol):
            G.add_edge(a["id"], b["id"])
    return G

# ------------------------------------------------------------
# ESEMPIO RAPIDO
# ------------------------------------------------------------
if __name__ == "__main__":

    json_file = "my_circles.json"
    circles = load_circles(json_file)
    G = circles_to_graph(circles, r=18.0, tol=1e-2)
    components = dense_connected_partition(G, S=5, P=25)
    print(components)

    #print(f"Nodi  : {G.number_of_nodes()}")
    #print(f"Archi : {G.number_of_edges()}")
    # opzionale: salva in GML o GraphML per ispezione con Gephi
    # nx.write_graphml(G, json_file.with_suffix('.graphml'))