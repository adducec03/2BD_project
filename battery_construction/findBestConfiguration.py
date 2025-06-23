import math

def calcola_disposizione(tensione_cella, corrente_cella, tensione_batteria, corrente_batteria):
    if any(x <= 0 for x in [tensione_cella, corrente_cella, tensione_batteria, corrente_batteria]):
        raise ValueError("Tutti i valori devono essere positivi.")

    # Celle in serie (S): determinano la tensione
    S = math.ceil(tensione_batteria / tensione_cella)

    # Celle in parallelo (P): determinano la corrente (capacità totale)
    P = math.ceil(corrente_batteria / corrente_cella)

    tensione_totale_effettiva = S * tensione_cella
    corrente_totale_effettiva = P * corrente_cella
    celle_totali = S * P

    return {
        "S": S,
        "P": P,
        "celle_totali": celle_totali,
        "tensione_effettiva": tensione_totale_effettiva,
        "corrente_effettiva": corrente_totale_effettiva
    }
