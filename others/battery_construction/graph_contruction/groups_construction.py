import itertools
import json
import math
from pathlib import Path
from typing import Union

import networkx as nx
from pulp import (
    LpBinary,
    LpMaximize,
    LpProblem,
    LpStatus,
    LpVariable,
    lpSum,
)


# ------------------------------------------------------------------
# 1)  LETTURA JSON  -------------------------------------------------
# ------------------------------------------------------------------
def load_circles(json_path: Union[str, Path], key: str = "circles"):
    """
    Ritorna una lista di dict  {id, x, y}
    Accetta cerchi in una delle due forme:
        {"center": [x, y], "radius": ...}
        {"x": x, "y": y, "radius": ...}
    Se manca 'id' usa l'indice nell'array.
    """
    json_path = Path(json_path)
    if not json_path.is_absolute():
        json_path = Path(__file__).resolve().parent / json_path

    if not json_path.exists():
        raise FileNotFoundError(f"File non trovato: {json_path}")

    with json_path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    circles_raw = data[key]
    nodes = []
    for idx, c in enumerate(circles_raw):
        if "center" in c and len(c["center"]) == 2:
            x, y = c["center"]
        elif "x" in c and "y" in c:
            x, y = c["x"], c["y"]
        else:
            raise ValueError(
                "Ogni cerchio deve avere 'center': [x, y] oppure le chiavi 'x' e 'y'"
            )

        nodes.append(
            {
                "id": c.get("id", idx),
                "x": float(x),
                "y": float(y),
            }
        )
#    if len(circles_raw) != 125:
#        raise ValueError("Numero di cerchi non valido (125 attesi)")
#    else:
#        print(f"Caricati {len(nodes)} cerchi da {json_path}")
    return nodes


# ------------------------------------------------------------------
# 2)  GRAFO DAI CERCHI  --------------------------------------------
# ------------------------------------------------------------------
def circles_to_graph(circles, r: float = 18.0, tol: float = 1e-2):
    """
    Converte i cerchi in un grafo semplice non pesato:
    due nodi sono adiacenti se dist(centri) ≅ 2*r (36).
    """
    target2 = (2 * r) ** 2

    G = nx.Graph()
    for c in circles:
        G.add_node(c["id"], **c)

    for a, b in itertools.combinations(circles, 2):
        dx = a["x"] - b["x"]
        dy = a["y"] - b["y"]
        dist2 = dx * dx + dy * dy
        if math.isclose(dist2, target2, abs_tol=2 * (2 * r) * tol):
            G.add_edge(a["id"], b["id"])
    return G


# ------------------------------------------------------------------
# 3)  ILP PER LE S COMPONENTI  --------------------------------------
# ------------------------------------------------------------------
def dense_connected_partition(G: nx.Graph, S: int, P: int):
    prob = LpProblem("DenseConnectedPartition", LpMaximize)

    # Variabili nodo e arco
    x = {
        (v, c): LpVariable(f"x_{v}_{c}", 0, 1, LpBinary)
        for v in G.nodes
        for c in range(S)
    }
    y = {
        (u, v, c): LpVariable(f"y_{u}_{v}_{c}", 0, 1, LpBinary)
        for u, v in G.edges
        for c in range(S)
    }

    # ---  flusso su ENTRAMBE le orientazioni  ------------------ #
    f = {
        (u, v, c): LpVariable(f"f_{u}_{v}_{c}", 0)
        for u, v in G.edges
        for c in range(S)
    }
    f.update(
        {
            (v, u, c): LpVariable(f"f_{v}_{u}_{c}", 0)
            for u, v in G.edges
            for c in range(S)
        }
    )
    # ------------------------------------------------------------ #

    # 1. P nodi per componente
    for c in range(S):
        prob += lpSum(x[v, c] for v in G.nodes) == P

    # 2. ogni nodo in ESATTAMENTE una componente
    for v in G.nodes:
        prob += lpSum(x[v, c] for c in range(S)) == 1

    # 3. coerenza y ≤ x
    for (u, v) in G.edges:
        for c in range(S):
            prob += y[u, v, c] <= x[u, c]
            prob += y[u, v, c] <= x[v, c]

    # 4. connettività (flusso di Kirchhoff)
    roots = list(G.nodes)[:S]  # radici fisse
    for c, r in enumerate(roots):
        for v in G.nodes:
            in_flow = lpSum(f[u, v, c] for u in G.neighbors(v))
            out_flow = lpSum(f[v, u, c] for u in G.neighbors(v))
            prob += out_flow - in_flow == (P - 1 if v == r else -1) * x[v, c]

        for (u, v) in G.edges:
            prob += f[u, v, c] <= (P - 1) * y[u, v, c]
            prob += f[v, u, c] <= (P - 1) * y[u, v, c]  # stesso y per l’arco opposto

    # 5. obiettivo
    prob += lpSum(y[u, v, c] for (u, v) in G.edges for c in range(S))

    prob.solve()
    if LpStatus[prob.status] != "Optimal":
        raise RuntimeError("Soluzione non ottimale trovata")

    return [
        [v for v in G.nodes if x[v, c].value() > 0.5] for c in range(S)
    ]


# ------------------------------------------------------------------
# 4)  MAIN  ---------------------------------------------------------
# ------------------------------------------------------------------
if __name__ == "__main__":
    circles = load_circles("my_circles.json")  # stesso path dello script
    G = circles_to_graph(circles, r=18.0, tol=1e-2)

    # esempio: 5 componenti da 25 nodi -> 125 nodi in totale
    comps = dense_connected_partition(G, S=5, P=25)
    print("Componenti trovate:")
    for i, comp in enumerate(comps, 1):
        print(f"  C{i}: {len(comp)} nodi – {comp[:8]}{' …' if len(comp) > 8 else ''}")