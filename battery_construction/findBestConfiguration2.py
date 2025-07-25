import math
import circlePacking2 as cp2
import json

#prova a cercare una configurazione con capacità ridotta se le specifiche iniziali non sono soddisfacibili
def trova_configurazione_con_capacita_ridotta(cell_v, cell_c, target_v, target_c, max_cells, step=0.5, min_c=2.0):
    current = target_c
    while current >= min_c:
        configs = calcola_configurazioni_migliori(
            cell_v, cell_c, target_v, current, celle_max=max_cells, top_k=5
        )
        if configs:
            return configs, current  # Successo
        current -= step  # Prova con capacità più bassa
    return None, None  # Fallimento


#prova a generare configurazioni valide con celle diverse se quelle iniziali non sono valide
def prova_altre_celle(json_path, target_v, target_c, max_polygon, top_k=5):
    # Legge le celle disponibili da un file JSON
    with open(json_path, "r") as f:
        celle_possibili = json.load(f)

    # Per ogni tipo di cella, prova a generare configurazioni valide
    for cell in celle_possibili:
        circle_radius = cell["diameter"] / 2
        best_centers, _ = cp2.find_best_packing(max_polygon, circle_radius)
        max_cells = len(best_centers)

        configs = calcola_configurazioni_migliori(
            cell["voltage"], cell["capacity"],
            target_v, target_c,
            celle_max=max_cells, top_k=top_k
        )

        if configs:
            return configs, cell, best_centers  # appena ne trova una valida, ritorna

    return None, None, None  # Nessuna configurazione valida


# -----------------------------------------
# Funzione obiettivo multi-criterio
# - scarto_v: deviazione % dalla tensione target
# - scarto_c: deviazione % dalla capacità target
# - scarto_celle: penalità se si usano troppe celle rispetto all’ideale
# I pesi sono bilanciati per dare più importanza a tensione e capacità e infine allo scarto di celle.
# score = 0.5 * scarto_v + 0.4 * scarto_c + 0.1 * max(scarto_celle, 0)
# -----------------------------------------

def calcola_configurazioni_migliori(cell_voltage, cell_current,
                                     target_voltage, target_capacity,
                                     celle_max=150, top_k=5):
    risultati = []

    # Intervalli plausibili per S e P
    # estende gli intervalli di +- 20% per S r +-50% per P 
    S_min = max(1, math.floor(target_voltage / cell_voltage * 0.8))
    S_max = math.ceil(target_voltage / cell_voltage * 1.2)
    P_min = max(1, math.floor(target_capacity / cell_current * 0.8))
    P_max = math.ceil(target_capacity / cell_current * 1.5)


    # Prova tutte le configurazioni possibili all’interno degli intervalli calcolati
    for S in range(S_min, S_max + 1):
        for P in range(P_min, P_max + 1):
            celle_totali = S * P
            if celle_totali > celle_max:
                continue  # scarta configurazione se il numero di celle necessarie supera il numero massimo di celle

            # CAlcola i valori di tensione e capacita effettivi della batteria
            V_eff = S * cell_voltage
            C_eff = P * cell_current

            # Scarti relativi (deviazione percentuale rispetto al target)
            scarto_v = abs(V_eff - target_voltage) / target_voltage
            scarto_c = abs(C_eff - target_capacity) / target_capacity
            scarto_celle = (celle_totali - (target_voltage * target_capacity) / (cell_voltage * cell_current)) / celle_max

            # Funzione obiettivo composita (peso su tensione, capacità, e celle)
            score = 0.5 * scarto_v + 0.4 * scarto_c + 0.1 * max(scarto_celle, 0)

            risultati.append({
                "S": S,
                "P": P,
                "celle_totali": celle_totali,
                "tensione_effettiva": V_eff,
                "capacita_effettiva": C_eff,
                "score": score
            })

    # Se non ci sono risultati ritorna una lista vuota
    if not risultati:
        return []

    # Ordina per score crescente e restituisci i migliori
    risultati.sort(key=lambda x: x["score"])
    return risultati[:top_k]


if __name__ == "__main__":
    # --------------------------
    # Parametri di input
    # --------------------------
    cell_voltage = 3.6         # V
    cell_current = 2.5         # Ah
    target_voltage = 48.0      # V
    target_capacity = 30.0     # Ah
    celle_max = 150            # massimo consentito

    top = calcola_configurazioni_migliori(
        cell_voltage, cell_current,
        target_voltage, target_capacity,
        celle_max=celle_max, top_k=5
    )

    print("📋 Migliori 5 configurazioni:")
    for i, cfg in enumerate(top, 1):
        print(f"{i}.  {cfg['S']}S × {cfg['P']}P  → "
              f"{cfg['celle_totali']} celle  | "
              f"{cfg['tensione_effettiva']:.1f} V, "
              f"{cfg['capacita_effettiva']:.1f} Ah, "
              f"score: {cfg['score']:.3f}")