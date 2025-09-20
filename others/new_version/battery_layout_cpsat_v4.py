from typing import List, Tuple
import numpy as np
import pandas as pd
import numpy.linalg as npl
from ortools.sat.python import cp_model
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient














#######################################################
#-----------------------------------------------------#
################# HELPERS FUNCTIONS ###################
#-----------------------------------------------------#
#######################################################

def pca_axes(coords: np.ndarray):
    """Restituisce unit vectors (e1, e2) con e1 = asse di massima varianza."""
    X = coords - coords.mean(axis=0, keepdims=True)
    # SVD è robusta e più stabile numericamente della cov-eig
    U, s, Vt = npl.svd(X, full_matrices=False)
    e1 = Vt[0]  # direzione "lunga"
    e1 = e1 / npl.norm(e1)
    e2 = np.array([-e1[1], e1[0]])  # perpendicolare (verso arbitrario)
    return e1, e2

def estimate_row_pitch_from_edges(coords: np.ndarray, E, e1, e2):
    """Stima il passo tra righe (pitch) usando le differenze di proiezione
    sull'asse e2, ma SOLO su archi (i,j) del grafo e scartando i più piccoli."""
    t = coords @ e2
    dt = np.abs(t[[i for i,_ in E]] - t[[j for _,j in E]])
    if len(dt) == 0:
        return max(1.0, 1.732 * 0.5 * 2)  # fallback
    # Scarta il 60% più piccolo (tipicamente contatti "stessa riga")
    thr = np.quantile(dt, 0.60)
    cand = dt[dt >= thr]
    if len(cand) == 0:
        cand = dt
    pitch = float(np.median(cand))
    # un minimo di sanità
    if pitch < 1e-6:
        pitch = float(np.median(dt)) if np.median(dt) > 0 else 1.0
    return pitch




def build_graph(coords: np.ndarray, R: float, tol: float) -> Tuple[List[Tuple[int,int]], List[Tuple[int,int]]]:
    """Return undirected edges E and directed arcs A on adjacency distance 2R±tol."""
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=2*R + tol)
    E = []
    for i, j in pairs:
        d = float(np.linalg.norm(coords[i] - coords[j]))
        if abs(d - 2 * R) <= tol:
            E.append((i, j))
    A = []
    for (i, j) in E:
        A.append((i, j))
        A.append((j, i))
    return E, A


def build_acyclic_arcs(coords: np.ndarray, E, orient_axis: np.ndarray | None = None):
    """
    Orienta gli archi non direzionali E in modo aciclico lungo 'orient_axis'.
    Se orient_axis è None, usa e1 (asse di massima varianza).
    Restituisce:
      A_dir   : lista di archi diretti (u,v)
      incoming: dict i -> lista di archi (u,i) entranti
    """
    if orient_axis is None:
        e1, _ = pca_axes(coords)
        axis = e1
    else:
        axis = orient_axis / np.linalg.norm(orient_axis)

    s = coords @ axis  # proiezione scalare lungo l’asse scelto

    def orient(i, j):
        if s[i] < s[j]:
            return (i, j)
        if s[i] > s[j]:
            return (j, i)
        # tie-break stabile per evitare cicli quando s[i] == s[j]
        return (i, j) if i < j else (j, i)

    A_dir = [orient(i, j) for (i, j) in E]
    incoming = {i: [] for i in range(len(coords))}
    for (u, v) in A_dir:
        incoming[v].append((u, v))
    return A_dir, incoming

def is_quadrangular_by_rows(row_counts: np.ndarray) -> bool:
    """
    True se:
      - il numero di celle per riga varia al più di 1 (max-min <= 1)
      - righe adiacenti differiscono al più di 1
    """
    rc = np.asarray(row_counts, dtype=int)
    if rc.size < 2:
        return False
    if (rc.max() - rc.min()) > 1:
        return False
    diffs = np.abs(np.diff(rc))
    return int(diffs.max(initial=0)) <= 1

def _infer_rows_and_positions_hex(coords, E):
    """
    Rileva le righe di una griglia esagonale e, per ogni riga,
    l'ordine orizzontale delle celle (posizione/rango nella riga).

    Ritorna:
      row_of[i]      : indice riga della cella i   (0..R-1)
      pos_in_row[i]  : posizione nella riga (0..len(riga)-1)
      rows           : lista degli indici riga presenti (0..R-1)
      row_counts     : np.array con #celle per riga
      C_complete     : numero di 'colonne complete' (min(row_counts))
    """
    e1, e2 = pca_axes(coords)  # e1: orizzontale (asse lungo), e2: verticale
    pitch_r = estimate_row_pitch_from_edges(coords, E, e1, e2)

    t_row = coords @ e2       # coordinata "verticale" → per raggruppare righe
    t_col = coords @ e1       # coordinata "orizzontale" → per ordinare nella riga

    r0 = t_row.min()
    row_of = np.round((t_row - r0) / max(1e-9, pitch_r)).astype(int)

    # raggruppa per riga e ordina orizzontalmente
    from collections import defaultdict
    by_row = defaultdict(list)
    for i, r in enumerate(row_of):
        by_row[int(r)].append(i)
    rows = sorted(by_row.keys())

    pos_in_row = np.empty(len(coords), dtype=int)
    for r in rows:
        # ordina da sinistra a destra lungo e1
        by_row[r].sort(key=lambda i: t_col[i])
        for p, i in enumerate(by_row[r]):
            pos_in_row[i] = p

    row_counts = np.array([len(by_row[r]) for r in rows], dtype=int)
    C_complete = int(row_counts.min()) if len(row_counts) else 0
    return row_of, pos_in_row, rows, row_counts, C_complete


















#######################################################
#-----------------------------------------------------#
################# MODEL CONSTRUCTION ##################
#-----------------------------------------------------#
#######################################################

def add_row_block_stripes(model, x, S, P,
                          row_of, pos_in_row, rows, row_counts):
    """
    Se un gruppo k usa la riga r, deve prendere esattamente P celle
    contigue (un blocco) in quella riga.
    - y[r,k] = 1 se la riga r è assegnata al gruppo k (al più una riga per gruppo, al più un gruppo per riga)
    - b[r,k,c0] = 1 se il blocco di lunghezza P inizia alla posizione c0 (0..Lr-P)
    - x[(i,k)] può essere 1 solo se la sua posizione c è coperta dal blocco scelto
    """
    N = len(row_of)
    R = len(rows)

    # y[r,k] = riga r assegnata al gruppo k
    y = {(r,k): model.NewBoolVar(f"row_assign[{r},{k}]") for r in range(R) for k in range(S)}

    # 1) ogni gruppo sceglie esattamente una riga
    for k in range(S):
        model.Add(sum(y[(r,k)] for r in range(R)) == 1)

    # 2) ogni riga al più un gruppo (se vuoi imporre copertura perfetta quando S==R, cambia in ==1)
    for r in range(R):
        model.Add(sum(y[(r,k)] for k in range(S)) <= 1)

    # 3) Prepara mapping riga→lista (ordinata) di (pos, cell_index)
    by_row = {r: [] for r in range(R)}
    for i in range(N):
        rr = int(row_of[i])
        if rr in by_row:
            by_row[rr].append((int(pos_in_row[i]), i))
    for r in range(R):
        by_row[r].sort(key=lambda t: t[0])  # ordina per posizione crescente

    # 4) Contiguità: blocco di P posizioni consecutive
    b = {}  # b[(r,k,c0)]
    for r in range(R):
        Lr = row_counts[r]
        max_start = max(0, Lr - P)
        for k in range(S):
            # somma(x_i_k su riga r) = P * y[r,k]
            idxs_r = [i for (_c, i) in by_row[r]]
            model.Add(sum(x[(i,k)] for i in idxs_r) == P * y[(r,k)])

            # variabili "start" del blocco
            starts = []
            for c0 in range(max_start + 1):
                b[(r,k,c0)] = model.NewBoolVar(f"blkstart[{r},{k},{c0}]")
                starts.append(b[(r,k,c0)])
            # scegli esattamente 1 start se la riga è usata, 0 altrimenti
            if starts:
                model.Add(sum(starts) == y[(r,k)])
            else:
                # riga troppo corta per P → forza y[r,k]=0
                model.Add(y[(r,k)] == 0)

            # x(i,k) consentito solo se la sua posizione c cade nel blocco scelto
            # x[i,k] ≤ Σ_{c0: c0 ≤ c ≤ c0+P-1} b[r,k,c0]
            for (c, i) in by_row[r]:
                cover_starts = []
                for c0 in range(max_start + 1):
                    if c0 <= c <= c0 + P - 1:
                        cover_starts.append(b[(r,k,c0)])
                if cover_starts:
                    model.Add(x[(i,k)] <= sum(cover_starts))
                else:
                    # se nessun blocco può coprire c, allora x[i,k] deve essere 0
                    model.Add(x[(i,k)] == 0)



def add_hex_column_block_stripes(model, x, S, P,
                                 row_of, pos_in_row, rows, C_complete,
                                 force_consecutive_cols=False):
    """
    Colonne per griglia esagonale con blocco verticale contiguo:
      – gruppo k sceglie una colonna c (tra 0..C_complete-1)
      – sceglie uno start r0 (0..R-P)
      – prende esattamente 1 cella per ciascuna riga r ∈ [r0, r0+P-1] in quella colonna.
    Precondizioni: C_complete >= S, len(rows) (=R) >= P

    Se force_consecutive_cols=True impone col[k] = col0 + k (gruppi su colonne adiacenti).
    """
    N = len(row_of)
    R = len(rows)
    assert C_complete >= S and R >= P

    # Mappa (r,c) -> lista di celle (di solito 1)
    I_rc = {(r, c): [] for r in range(R) for c in range(C_complete)}
    for i in range(N):
        r = int(row_of[i]); c = int(pos_in_row[i])
        if 0 <= r < R and 0 <= c < C_complete:
            I_rc[(r, c)].append(i)

    # y[c,k] = colonna scelta dal gruppo k
    y = {(c,k): model.NewBoolVar(f"hexcol[{c},{k}]")
         for c in range(C_complete) for k in range(S)}
    for k in range(S):
        model.Add(sum(y[(c,k)] for c in range(C_complete)) == 1)
    for c in range(C_complete):
        model.Add(sum(y[(c,k)] for k in range(S)) <= 1)

    # opzionale: colonne consecutive e in ordine
    if force_consecutive_cols:
        col0 = model.NewIntVar(0, C_complete - S, "col0")
        for k in range(S):
            ck = model.NewIntVar(0, C_complete - 1, f"col[{k}]")
            model.Add(ck == col0 + k)
            model.Add(ck == sum(c * y[(c,k)] for c in range(C_complete)))

    # b[c,k,r0] = blocco verticale che parte da r0 per il gruppo k nella colonna c
    b = {}
    for c in range(C_complete):
        for k in range(S):
            max_start = R - P
            starts = []
            for r0 in range(max_start + 1):
                b[(c,k,r0)] = model.NewBoolVar(f"vblk[{c},{k},{r0}]")
                starts.append(b[(c,k,r0)])
            # scegli esattamente 1 start se la colonna è usata
            if starts:
                model.Add(sum(starts) == y[(c,k)])
            else:
                # se R < P non si entra qui perché abbiamo assert R >= P
                pass

            # vincoli cell-by-cell: per ogni riga r, se è dentro il blocco, prendi 1 cella in (r,c)
            for r in range(R):
                idxs = I_rc[(r, c)]
                # copertura verticale: r è "attivo" se r ∈ [r0, r0+P-1] per lo start scelto
                cover = []
                for r0 in range(max_start + 1):
                    if r0 <= r <= r0 + P - 1:
                        cover.append(b[(c,k,r0)])
                if idxs and cover:
                    # somma x sulla riga r e colonna c = y[c,k] * is_row_used
                    model.Add(sum(x[(i,k)] for i in idxs) == sum(cover))
                else:
                    # nessuna cella in (r,c) oppure r non può cadere in alcun blocco → deve essere 0
                    if idxs:
                        model.Add(sum(x[(i,k)] for i in idxs) == 0)

    # Nota: il vincolo Σ_i x[(i,k)] == P (già nel modello) è compatibile:
    # per ciascun gruppo k, esattamente P righe risultano "coperte" (una cella per riga) → totale P.



def add_stripe_constraints(model, x, S, P,
                           row_of, col_of,
                           mode,               # "columns" oppure "rows"
                           n_rows, n_cols,
                           must_cover_all_stripes=False):
    """
    Impone che ogni gruppo k usi ESATTAMENTE P celle da UNA sola colonna (mode="columns")
    oppure da UNA sola riga (mode="rows").
    Se S < n_cols (o n_rows), il solver sceglie quali colonne (o righe) usare.
    """
    N = len(col_of) if mode == "columns" else len(row_of)

    if mode == "columns":
        # y[c,k] = 1 se la colonna c è assegnata al gruppo k
        y = {(c,k): model.NewBoolVar(f"col_assign[{c},{k}]") for c in range(n_cols) for k in range(S)}
        # a ogni gruppo esattamente una colonna
        for k in range(S):
            model.Add(sum(y[(c,k)] for c in range(n_cols)) == 1)
        # ogni colonna al più un gruppo (==1 se vuoi coprire tutte le colonne quando S==n_cols)
        for c in range(n_cols):
            if must_cover_all_stripes and S == n_cols:
                model.Add(sum(y[(c,k)] for k in range(S)) == 1)
            else:
                model.Add(sum(y[(c,k)] for k in range(S)) <= 1)
        # celle vincolate: le celle della colonna c possono andare solo al gruppo k collegato a c
        idx_by_col = {c: [i for i in range(N) if col_of[i] == c] for c in range(n_cols)}
        for c in range(n_cols):
            for k in range(S):
                # somma celle della colonna c nel gruppo k = P * y[c,k]
                model.Add(sum(x[(i,k)] for i in idx_by_col[c]) == P * y[(c,k)])
        # tutte le altre x[(i,k)] fuori dalla colonna assegnata sono implicitamente 0 grazie ai vincoli sopra

    elif mode == "rows":
        # y[r,k] = 1 se la riga r è assegnata al gruppo k
        y = {(r,k): model.NewBoolVar(f"row_assign[{r},{k}]") for r in range(n_rows) for k in range(S)}
        for k in range(S):
            model.Add(sum(y[(r,k)] for r in range(n_rows)) == 1)
        for r in range(n_rows):
            if must_cover_all_stripes and S == n_rows:
                model.Add(sum(y[(r,k)] for k in range(S)) == 1)
            else:
                model.Add(sum(y[(r,k)] for k in range(S)) <= 1)
        idx_by_row = {r: [i for i in range(N) if row_of[i] == r] for r in range(n_rows)}
        for r in range(n_rows):
            for k in range(S):
                model.Add(sum(x[(i,k)] for i in idx_by_row[r]) == P * y[(r,k)])
    else:
        raise ValueError("mode deve essere 'columns' o 'rows'")
    
    
def add_unused_component_forest(
    model,
    N,
    A_dir,           # archi diretti aciclici (da build_acyclic_arcs)
    incoming,        # incoming[v] = lista archi (u->v)
    use,             # dict: use[i] ∈ {0,1} già nel tuo modello
    boundary_mask,   # np.array di 0/1: 1 = cella di bordo
    mode="penalize", # "penalize" | "forbid"
):
    """
    Costruisce una foresta sulle celle NON assegnate U[i] = 1 - use[i].
    - Se mode="forbid": vieta componenti interne (nessuna radice ammessa dentro).
    - Se mode="penalize": consente radici interne ma le segnala per la penalità.

    Ritorna:
      vars: dict con chiavi:
        "U":  dict[i] -> IntVar {0,1}
        "yH": dict[(u,v)] -> BoolVar, archi della foresta sui non-assegnati
        "rH": dict[i] -> BoolVar, radici di componente sui non-assegnati
        "n_holes": IntVar (solo in mode="penalize"), #componenti interne
    """
    # U[i] = 1 - use[i]
    U = {i: model.NewIntVar(0, 1, f"U[{i}]") for i in range(N)}
    for i in range(N):
        model.Add(U[i] == 1 - use[i])

    # Archi sulla foresta dei non-assegnati
    yH = {}
    for (u, v) in A_dir:
        y = model.NewBoolVar(f"yH[{u}->{v}]")
        # attivo solo se entrambi i nodi sono non-assegnati
        model.Add(y <= U[u])
        model.Add(y <= U[v])
        yH[(u, v)] = y

    # Radici (una per componente): preferibilmente sul bordo
    rH = {i: model.NewBoolVar(f"rH[{i}]") for i in range(N)}
    for i in range(N):
        model.Add(rH[i] <= U[i])  # puoi essere root solo se non-assegnato
        if mode == "forbid":
            # vieta radici interne: obbliga ogni componente a toccare il bordo
            if int(boundary_mask[i]) == 0:
                model.Add(rH[i] == 0)

    # Bilancio per ogni nodo: somma incoming + root = U[i]
    for i in range(N):
        model.Add(sum(yH[(u, i)] for (u, _v) in incoming[i]) + rH[i] == U[i])

    # In una foresta: #archi = #nodi - #radici
    model.Add(sum(yH.values()) == sum(U[i] for i in range(N)) - sum(rH[i] for i in range(N)))

    out = {"U": U, "yH": yH, "rH": rH}

    if mode == "penalize":
        # Conta quante radici sono in celle INTERNE (cioè componenti senza bordo)
        int_roots = []
        for i in range(N):
            if int(boundary_mask[i]) == 0:
                si = model.NewBoolVar(f"intRoot[{i}]")
                model.Add(si == rH[i])      # s[i] = rH[i] se non di bordo
                int_roots.append(si)
        n_holes = model.NewIntVar(0, N, "n_hole_components")
        if int_roots:
            model.Add(n_holes == sum(int_roots))
        else:
            model.Add(n_holes == 0)
        out["n_holes"] = n_holes

    return out

















#######################################################
#-----------------------------------------------------#
##################### SOLVING #########################
#-----------------------------------------------------#
#######################################################

def make_profile(profile_name: str, time_s: float, seed: int, workers: int):
    p = dict(
        max_time_in_seconds=float(time_s),
        num_search_workers=int(workers),
        random_seed=int(seed),
        randomize_search=True,
        search_branching=cp_model.PORTFOLIO_SEARCH,
        log_search_progress=True,
    )
    if profile_name == "fast":
        p.update(dict(cp_model_presolve=True, cp_model_probing_level=0, linearization_level=0))
    elif profile_name == "quality":
        p.update(dict(cp_model_presolve=True, cp_model_probing_level=1, linearization_level=1))
    elif profile_name == "aggressive":
        p.update(dict(cp_model_presolve=True, cp_model_probing_level=2, linearization_level=2))
    return p

def auto_tune_and_solve(coords, S, P, R, tol,
                        time_budget=60,
                        target_T=None,     # metti <= 2*P, altrimenti None
                        degree_cap=6,
                        enforce_degree=False,
                        profiles=("fast","fast","quality"),
                        seeds=(0,1,2,3),
                        workers=6,
                        use_hole_penality=False, stripe_mode="auto"):
    combos = []
    for t in range(max(len(profiles), len(seeds))):
        combos.append( (profiles[t % len(profiles)], seeds[t % len(seeds)]) )

    per_run = max(5, time_budget // max(1, len(combos)))
    best_pack, best_T, best_sumL = None, -1, -1

    # cap target_T a 2*P (limite realistico)
    if target_T is not None:
        target_T = min(target_T, 2*P)

    for (prof, seed) in combos:
        cp_params = make_profile(prof, per_run, seed, workers)

        status, solver, x, r, L, z1, z2, E, T = solve_layout(
            coords, S, P, R, tol,
            time_limit=per_run, alpha=0.0, Lmin=1,
            degree_cap=degree_cap, enforce_degree=enforce_degree,
            cp_params=cp_params, use_hole_penality=False,
            stripe_mode=stripe_mode
        )

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            tval = solver.Value(T) if T is not None else 0
            sumL = sum(int(solver.Value(Lk)) for Lk in L)
            if (tval > best_T) or (tval == best_T and sumL > best_sumL):
                best_pack = (status, solver, x, r, L, z1, z2, E, T)
                best_T, best_sumL = tval, sumL
            if target_T is not None and tval >= target_T:
                break

    # se nessuna run è FEASIBLE/OPTIMAL, ritorna l’ultima (best_pack resta None)
    if best_pack is not None:
        return best_pack
    else:
        return status, solver, x, r, L, z1, z2, E, T






def solve_layout(coords: np.ndarray, S: int, P: int, R: float, tol: float,
                 time_limit: int = 60, alpha: float = 0.1, seed: int = 0, Lmin=1,
                 degree_cap=4, enforce_degree: bool = False,
                 cp_params: dict | None = None, use_hole_penality=False, stripe_mode="auto"):
    N = len(coords)
    if S * P > N:
        raise SystemExit(f"Celle richieste {S*P} > disponibili {N}")

    # --- grafo non diretto + assi ---
    E, _ = build_graph(coords, R, tol)
    e1, e2 = pca_axes(coords)

    # --- euristica righe/colonne ---
    row_of_h, pos_in_row_h, rows_h, row_counts_h, C_complete_h = _infer_rows_and_positions_hex(coords, E)
    rect_like = is_quadrangular_by_rows(row_counts_h)

    # adesso consentiamo colonne se: forma "rettangolare", R>=P, C_complete>=S
    can_hex_cols = (stripe_mode in ("auto","columns")) and rect_like \
                and (len(rows_h) >= P) and (C_complete_h >= S)


    # --- colonne esagonali SOLO se: forma rettangolare, P == #righe, e ci sono almeno S colonne complete
    

    orient_axis = e2 if can_hex_cols else e1
    A, incoming = build_acyclic_arcs(coords, E, orient_axis=orient_axis)

    # (DEBUG) quanti nodi hanno 0 archi entranti lungo l'asse scelto?
    deg_in = np.array([len(incoming[v]) for v in range(N)], int)
    print("nodi con deg_in=0:", int((deg_in == 0).sum()))

    # --- vicinati, ecc. ---
    neighbors = {i: [] for i in range(N)}
    for (i, j) in E:
        neighbors[i].append(j); neighbors[j].append(i)

    # (consigliato) bordo = grado < max_deg osservato
    max_deg = max((len(neighbors[i]) for i in range(N)), default=0)
    boundary_mask = np.array([1 if len(neighbors[i]) < max_deg else 0 for i in range(N)], dtype=int)

    # === model & variabili ===
    model = cp_model.CpModel()

    # x, r
    x, r = {}, {}
    for i in range(N):
        for k in range(S):
            x[(i,k)] = model.NewBoolVar(f"x[{i},{k}]")
            r[(i,k)] = model.NewBoolVar(f"r[{i},{k}]")

    # use[i] = Σ_k x[i,k]
    use = {}
    for i in range(N):
        use[i] = model.NewIntVar(0, 1, f"use[{i}]")
        model.Add(use[i] == sum(x[(i,k)] for k in range(S)))

    # --- exp[i] = cella esposta (bordo oppure ha almeno un vicino non usato) ---
    exp, outN = {}, {}
    for i in range(N):
        e = model.NewBoolVar(f"exp[{i}]")
        if boundary_mask[i] == 1:
            model.Add(e == 1)
        else:
            if neighbors[i]:
                oi = model.NewBoolVar(f"outN[{i}]")
                for j in neighbors[i]:
                    model.Add(oi >= 1 - use[j])                # se un vicino è vuoto → oi può essere 1
                model.Add(oi <= sum(1 - use[j] for j in neighbors[i]))  # upper bound
                model.Add(e == oi)
                outN[i] = oi
            else:
                model.Add(e == 1)  # cella isolata: esposta
        exp[i] = e

    # --- esposizione gruppi agli estremi della catena (k=0 e k=S-1) ---
    end_groups = [0] if S == 1 else [0, S-1]
    g_exposed = {}
    for k_end in end_groups:
        eik = []
        for i in range(N):
            ei = model.NewBoolVar(f"exp_in_grp[{k_end},{i}]")
            model.Add(ei <= x[(i, k_end)])
            model.Add(ei <= exp[i])
            model.Add(ei >= x[(i, k_end)] + exp[i] - 1)  # AND
            eik.append(ei)
        gk = model.NewBoolVar(f"group_exposed[{k_end}]")
        if eik:
            model.AddMaxEquality(gk, eik)  # OR sugli ei
        else:
            model.Add(gk == 0)
        g_exposed[k_end] = gk

    # (1) gruppi da P celle, (2) ogni cella al più una volta, (3) una root per gruppo
    for k in range(S):
        model.Add(sum(x[(i,k)] for i in range(N)) == P)
        model.Add(sum(r[(i,k)] for i in range(N)) == 1)
        for i in range(N):
            model.Add(r[(i,k)] <= x[(i,k)])
    for i in range(N):
        model.Add(sum(x[(i,k)] for k in range(S)) <= 1)

    # --- STRIPE GUIDANCE (dopo che esistono model e x!) ---
    stripe_applied = False
    if can_hex_cols:
        add_hex_column_block_stripes(
            model, x, S, P,
            row_of=row_of_h, pos_in_row=pos_in_row_h,
            rows=rows_h, C_complete=C_complete_h,
            force_consecutive_cols=True   # o False se vuoi libertà
        )
        stripe_applied = True

    # Fallback a RIGHE solo se forma rettangolare e ci sono almeno S righe con >= P celle
    if (not stripe_applied) and (stripe_mode in ("auto","rows")):
        rows_geP = int((row_counts_h >= P).sum())
        if rect_like and rows_geP >= S:
            add_row_block_stripes(
                model, x, S, P,
                row_of=row_of_h,
                pos_in_row=pos_in_row_h,
                rows=rows_h,
                row_counts=row_counts_h
            )
            stripe_applied = True

    # --- connettività: parent-edge su A (già orientato correttamente) ---
    y = {}
    for (i, j) in A:
        for k in range(S):
            y[(i, j, k)] = model.NewBoolVar(f"y[{i}->{j},{k}]")
            model.Add(y[(i, j, k)] <= x[(i, k)])
            model.Add(y[(i, j, k)] <= x[(j, k)])

    # per ogni nodo selezionato: o root o esattamente 1 genitore
    for k in range(S):
        for i in range(N):
            model.Add(sum(y[(u, i, k)] for (u, _v) in incoming[i]) + r[(i, k)] == x[(i, k)])

    # albero per gruppo: #archi = #nodi - 1
    for k in range(S):
        model.Add(sum(y[(i, j, k)] for (i, j) in A) == sum(x[(i, k)] for i in range(N)) - 1)


    # --- Single-cell holes (opzionale): definisci sempre holes per evitare NameError ---
    holes = []   # sarà una lista di BoolVar; se non usata resta vuota


    # ------ NUOVO: buco multi-cella (componenti interne dei NON-assegnati) ------
    HOLE_MODE = "penalize"   # "forbid" per vietarli del tutto
    hole_vars = add_unused_component_forest(
        model,
        N=N,
        A_dir=A,             # DAG che hai già costruito con build_acyclic_arcs
        incoming=incoming,
        use=use,
        boundary_mask=boundary_mask,
        mode=HOLE_MODE
    )


    # --- (5) Link tra gruppi consecutivi (z1/z2) su TUTTI gli archi E ---
    L = []
    z1, z2 = {}, {}
    for k in range(S-1):
        Lk = model.NewIntVar(0, max(1, 2*len(E)), f"L[{k}]")
        link_terms = []
        for (i, j) in E:
            z1[(k,i,j)] = model.NewBoolVar(f"z1[{k},{i}-{j}]")  # i in k, j in k+1
            z2[(k,i,j)] = model.NewBoolVar(f"z2[{k},{i}-{j}]")  # j in k, i in k+1
            model.Add(z1[(k,i,j)] <= x[(i,k)])
            model.Add(z1[(k,i,j)] <= x[(j,k+1)])
            model.Add(z2[(k,i,j)] <= x[(j,k)])
            model.Add(z2[(k,i,j)] <= x[(i,k+1)])
            link_terms += [z1[(k,i,j)], z2[(k,i,j)]]
        model.Add(Lk == sum(link_terms)) if link_terms else model.Add(Lk == 0)
        if Lmin > 0:
            model.Add(Lk >= Lmin)
        L.append(Lk)

    # (OPZ) limite di grado – off per default
    if enforce_degree:
        per_cell_degree = [model.NewIntVar(0, degree_cap, f"deg[{i}]") for i in range(N)]
        deg_expr = [0 for _ in range(N)]
        for k in range(S-1):
            for (i, j) in E:
                deg_expr[i] = deg_expr[i] + z1[(k,i,j)] + z2[(k,i,j)]
                deg_expr[j] = deg_expr[j] + z1[(k,i,j)] + z2[(k,i,j)]
        for i in range(N):
            model.Add(per_cell_degree[i] == deg_expr[i])
            model.Add(per_cell_degree[i] <= degree_cap)



    # --- Obiettivo: copertura, minimo T, somma contatti, penalità ---
    T = None
    if L:
        # b[k] = 1 se Lk >= 1
        b = []
        for k, Lk in enumerate(L):
            bk = model.NewBoolVar(f"connected[{k}]")
            model.Add(Lk >= bk)
            b.append(bk)

        # T = min_k Lk
        T = model.NewIntVar(0, 2 * P, "T_min_links")
        for Lk in L:
            model.Add(T <= Lk)

        # pesi
        wCover, wT, wSum = 10_000, 1_000, 100
        wHole1     = 150    # micro-buchi (se attivati)
        wHoleComp  = 600    # componenti interne non-assegnate
        wEndExpose = 2_000  # estremi non esposti

        # base
        obj = wCover * sum(b) + wT * T + wSum * sum(L)

        # penalità estremi non esposti
        for k_end in end_groups:
            obj -= wEndExpose * (1 - g_exposed[k_end])

        # penalità componenti interne dei NON-assegnati (se hai chiamato add_unused_component_forest)
        hole_comp_term = hole_vars.get("n_holes") if 'hole_vars' in locals() else None
        if hole_comp_term is not None:
            obj -= wHoleComp * hole_comp_term

        # penalità micro-buchi singoli (se attivata)
        if use_hole_penality and 'holes' in locals() and holes:
            obj -= wHole1 * sum(holes)

        model.Maximize(obj)
    else:
        model.Maximize(0)

    # --- Solver settings + override (auto-tuning) ---
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = -1       # usa tutti i core
    solver.parameters.cp_model_presolve = True
    solver.parameters.cp_model_probing_level = 2
    solver.parameters.linearization_level = 2
    solver.parameters.search_branching = cp_model.PORTFOLIO_SEARCH
    solver.parameters.symmetry_level = 2
    solver.parameters.randomize_search = True
    solver.parameters.log_search_progress = True
    if cp_params:
        for k, v in cp_params.items():
            setattr(solver.parameters, k, v)

    status = solver.Solve(model)
    return status, solver, x, r, L, z1, z2, E, T




















#######################################################
#-----------------------------------------------------#
################### VISUALIZATION #####################
#-----------------------------------------------------#
#######################################################

def plot_solution(coords, R, S, x, z1, z2, E, L, solver,
                             title="", save=None, show_links=True, show_arrows=True):
    import matplotlib.colors as mcolors
    from matplotlib.patches import Circle
    import matplotlib.pyplot as plt

    palette = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    def color_for_group(k): return palette[k % len(palette)]

    N = len(coords)
    # gruppo assegnato per ogni cella
    assigned_group = [-1]*N
    for i in range(N):
        for k in range(S):
            if solver.BooleanValue(x[(i,k)]):
                assigned_group[i] = k
                break

    # centroidi dei gruppi
    centroids = np.zeros((S, 2), dtype=float)
    for k in range(S):
        idx = [i for i in range(N) if assigned_group[i] == k]
        if idx:
            centroids[k] = coords[idx].mean(axis=0)
        else:
            centroids[k] = np.array([np.nan, np.nan])

    fig, ax = plt.subplots(figsize=(12, 8), dpi=130)

    # disegna celle
    for i, (x0,y0) in enumerate(coords):
        gid = assigned_group[i]
        fc = color_for_group(gid) if gid >= 0 else "white"
        ax.add_patch(Circle((x0,y0), R, facecolor=fc, edgecolor="black", linewidth=0.6, zorder=2))

    # disegna contatti effettivi tra gruppi successivi (linee sottili grigie)
    if show_links:
        for k in range(S-1):
            for (i,j) in E:
                if solver.BooleanValue(z1[(k,i,j)]) or solver.BooleanValue(z2[(k,i,j)]):
                    xi, yi = coords[i]
                    xj, yj = coords[j]
                    ax.plot([xi, xj], [yi, yj], linewidth=1.0, alpha=0.5, color="black", zorder=3)

    # frecce con ordine (k -> k+1) e label con #link
    if show_arrows:
        for k in range(S-1):
            x0, y0 = centroids[k]
            x1, y1 = centroids[k+1]
            if not (np.isnan(x0) or np.isnan(x1)):
                # piccola freccia tra centroidi
                ax.annotate(
                    "", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", linewidth=1.5, color="black", shrinkA=10, shrinkB=10),
                    zorder=4
                )
                # etichetta ordine (1..S-1) e #link
                lk = int(solver.Value(L[k])) if L else 0
                xm, ym = (0.5*(x0+x1), 0.5*(y0+y1))
                ax.text(xm, ym, f"{k+1}\n({lk})", ha="center", va="center",
                        fontsize=9, bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec="black", alpha=0),
                        zorder=5)

    # mostra indice del gruppo sul centroide
    for k in range(S):
        xk, yk = centroids[k]
        if not (np.isnan(xk) or np.isnan(yk)):
            ax.text(xk, yk, f"G{k}", ha="center", va="center", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0), zorder=6)

    # limiti asse (evita canvas vuoto)
    xs = coords[:,0]; ys = coords[:,1]
    ax.set_xlim(xs.min()-2*R, xs.max()+2*R)
    ax.set_ylim(ys.max()+2*R, ys.min()-2*R)  # invertito per coerenza con il tuo codice
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title)
    plt.tight_layout()
    if save:
        plt.savefig(save, bbox_inches="tight", dpi=160)
    plt.show()




def _edge_labels(ax, poly: Polygon, offset_mm: float,
                 fontsize=12, box_fc="white", zorder=20):
    """
    Scrive le quote (mm) su tutti i lati dell'anello esterno.
    Il poligono è atteso in mm e con asse Y verso l'alto.
    """
    # Forza orientazione CCW dell’anello esterno
    ext = np.asarray(orient(poly, sign=1.0).exterior.coords)

    for (x1, y1), (x2, y2) in zip(ext[:-1], ext[1:]):
        dx, dy = (x2 - x1), (y2 - y1)
        L = float(np.hypot(dx, dy))
        if L <= 1e-9:
            continue
        # punto medio e normale esterna (per CCW → (dy, -dx))
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        nx, ny = dy / L, -dx / L
        ax.text(mx + offset_mm * nx,
                my + offset_mm * ny,
                f"{L:.1f} mm",
                ha="center", va="center",
                fontsize=fontsize,
                bbox=dict(boxstyle="round,pad=0.25",
                          fc=box_fc, ec="0.3", alpha=0.96),
                zorder=zorder)




def plot_solution2(coords, R, S, x, z1, z2, E, L, solver,
                  title=None,
                  save=None,
                  show_links=True,
                  show_arrows=True,
                  boundary=None,             # None | path JSON | shapely.Polygon
                  show_dimensions=True,
                  dim_offset_R=1.4,
                  flip_vertical=True,
                  fs_dim=12, fs_group=16, fs_arrow=13,  # font più grandi di default
                  dpi=260):
    import numpy as np
    import matplotlib.colors as mcolors
    from matplotlib.patches import Circle
    from shapely.geometry import Polygon
    from shapely.affinity import scale as shp_scale
    from pathlib import Path
    import matplotlib.pyplot as plt
    import circle_packing as cp

    palette = list(mcolors.TABLEAU_COLORS.values())
    def color_for_group(k): return palette[k % len(palette)]

    coords = np.asarray(coords, float)
    N = len(coords)

    # ---- boundary (in mm) ----
    poly = None
    if boundary is not None:
        if isinstance(boundary, (str, Path)):
            poly, _prep, _meta = cp.load_boundary(Path(boundary),
                                                  to_units="mm",
                                                  require_scale=False,
                                                  flip_y=True)
        elif isinstance(boundary, Polygon):
            poly = boundary
        else:
            raise TypeError("boundary deve essere None, un path JSON o un shapely.Polygon")

    # ---- bbox per eventuale flip ----
    xmin = coords[:,0].min(); xmax = coords[:,0].max()
    ymin = coords[:,1].min(); ymax = coords[:,1].max()
    if poly is not None:
        bxmin, bymin, bxmax, bymax = poly.bounds
        xmin, xmax = min(xmin, bxmin), max(xmax, bxmax)
        ymin, ymax = min(ymin, bymin), max(ymax, bymax)
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0

    # ---- coordinate di display (flip opzionale) ----
    coords_disp = coords.copy()
    poly_disp = poly
    if flip_vertical:
        coords_disp[:,1] = 2*cy - coords_disp[:,1]
        if poly is not None:
            poly_disp = shp_scale(poly, xfact=1.0, yfact=-1.0, origin=(cx, cy))

    # ---- assegnazioni + “cella più centrale” per ogni gruppo ----
    assigned_group = [-1]*N
    for i in range(N):
        for k in range(S):
            if solver.BooleanValue(x[(i,k)]):
                assigned_group[i] = k
                break

    centroids = np.full((S,2), np.nan)
    label_pos = np.full((S,2), np.nan)   # centro della cella su cui scrivere il numero
    label_idx = [-1]*S
    for k in range(S):
        idx = [i for i in range(N) if assigned_group[i] == k]
        if not idx:
            continue
        pts = coords_disp[idx]
        c = pts.mean(axis=0)
        centroids[k] = c
        j = idx[np.argmin(((pts[:,0]-c[0])**2 + (pts[:,1]-c[1])**2))]
        label_pos[k] = coords_disp[j]
        label_idx[k] = j

    # ---- figura ----
    fig, ax = plt.subplots(figsize=(12, 4.3), dpi=dpi)

    # 1) contorno
    if poly_disp is not None:
        xB, yB = poly_disp.exterior.xy
        ax.plot(xB, yB, color="black", lw=2.0, zorder=5)

    # 2) celle
    for i, (x0, y0) in enumerate(coords_disp):
        gid = assigned_group[i]
        fc = color_for_group(gid) if gid >= 0 else "white"
        ax.add_patch(Circle((x0, y0), R, facecolor=fc,
                            edgecolor="black", linewidth=0.8, zorder=10))

    # 3) link sottili fra gruppi adiacenti (opzionale)
    if show_links:
        for k in range(S-1):
            for (i, j) in E:
                if solver.BooleanValue(z1[(k,i,j)]) or solver.BooleanValue(z2[(k,i,j)]):
                    xi, yi = coords_disp[i]; xj, yj = coords_disp[j]
                    ax.plot([xi, xj], [yi, yj], lw=1.1, alpha=0.50, color="black", zorder=12)

    # 4) frecce TRA LE CELLE ETICHETTATE (niente testo, niente “(6)”)
    if show_arrows:
        for k in range(S-1):
            x0, y0 = label_pos[k]
            x1, y1 = label_pos[k+1]
            if not (np.isnan(x0) or np.isnan(x1)):
                ax.annotate("",
                            xy=(x1, y1), xytext=(x0, y0),
                            arrowprops=dict(arrowstyle="->",
                                            linewidth=1.8, color="black",
                                            shrinkA=R*0.35, shrinkB=R*0.35),
                            zorder=18)

    # 5) NUMERO DEL GRUPPO — rettangolo nero smussato, semi-trasp., testo bianco bold
    for k in range(S):
        px, py = label_pos[k]
        if not (np.isnan(px) or np.isnan(py)):
            ax.text(px, py, f"{k}",
                    ha="center", va="center",
                    fontsize=fs_group, color="white", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.18",
                              fc=(0,0,0,0.70), ec="black", lw=0.9),
                    zorder=19)

    # 6) quote in mm
    if poly_disp is not None and show_dimensions:
        _edge_labels(ax, poly_disp, offset_mm=float(dim_offset_R)*float(R),
                     fontsize=fs_dim)

    # 7) layout pulito
    xs, ys = coords_disp[:,0], coords_disp[:,1]
    xmin_p, xmax_p = xs.min()-2*R, xs.max()+2*R
    ymin_p, ymax_p = ys.min()-2*R, ys.max()+2*R
    if poly_disp is not None:
        bxmin, bymin, bxmax, bymax = poly_disp.bounds
        xmin_p, ymin_p = min(xmin_p, bxmin), min(ymin_p, bymin)
        xmax_p, ymax_p = max(xmax_p, bxmax), max(ymax_p, bymax)

    ax.set_xlim(xmin_p, xmax_p)
    ax.set_ylim(ymin_p, ymax_p)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    if title:  # per default è None → nessun titolo
        ax.set_title(title, fontsize=fs_group, pad=6)

    plt.tight_layout()
    if save:
        plt.savefig(save, bbox_inches="tight", dpi=300)
    plt.show()




















#######################################################
#-----------------------------------------------------#
################# MAIN ENTRY POINT ####################
#-----------------------------------------------------#
#######################################################



def main():



    csv = "out.csv"
    S = 10
    P =5
    radius = 9.0
    tol = 2
    time_budget = 30
    degree_cap = 6
    enforce_degree = False
    target_T=2*P
    use_hole_penality=False

    df = pd.read_csv(csv)
    coords = df[['x','y']].to_numpy()

    # tuning senza snake
    pack = auto_tune_and_solve(
        coords, S, P, radius, tol,
        time_budget=time_budget,
        target_T=target_T,             
        degree_cap=degree_cap,
        enforce_degree=enforce_degree,
        profiles=("fast","fast","quality"),
        seeds=(0,1,2,3),
        workers=6,
        use_hole_penality=use_hole_penality,
        stripe_mode="auto"
    )

    status, solver, x, r, L, z1, z2, E, T = pack
    status_name = {cp_model.OPTIMAL:"OPTIMAL", cp_model.FEASIBLE:"FEASIBLE",
                    cp_model.INFEASIBLE:"INFEASIBLE", cp_model.MODEL_INVALID:"MODEL_INVALID",
                    cp_model.UNKNOWN:"UNKNOWN"}.get(status, str(status))
    print("Solver status:", status_name)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        links = [int(solver.Value(v)) for v in L]
        tval = solver.Value(T) if T is not None else None
        print(f"min_k Lk = {tval}   sum(L) = {sum(links)}   per-k: {links}")
        plot_solution(
            coords, radius, S, x, z1, z2, E, L, solver,
            title=f"{S}S{P}P — CP-SAT ({status_name})",
            save=None, show_links=True, show_arrows=True
        )
    else:
        print("Nessuna soluzione trovata entro il budget.")


if __name__ == "__main__":
    main()