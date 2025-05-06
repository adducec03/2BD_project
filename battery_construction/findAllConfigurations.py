def trova_tutte_configurazioni(celle_totali, v_cella=3.6, ah_cella=2.5):
    configurazioni = []

    for s in range(1, celle_totali + 1):
        if celle_totali % s != 0:
            continue  # solo configurazioni intere
        p = celle_totali // s
        v_tot = s * v_cella
        ah_tot = p * ah_cella
        configurazioni.append((s, p, v_tot, ah_tot))

    if not configurazioni:
        raise ValueError("Nessuna configurazione valida trovata")

    # Massima tensione
    max_tensione = max(configurazioni, key=lambda x: x[2])

    # Massima capacità
    max_capacita = max(configurazioni, key=lambda x: x[3])

    # Bilanciata (tensione ≈ capacità in termini assoluti)
    bilanciata = min(configurazioni, key=lambda x: abs(x[2] - x[3]))

    return {
        "massima_tensione": max_tensione,
        "massima_capacita": max_capacita,
        "bilanciata": bilanciata
    }

def stampa_configurazioni(configs):
    for nome, (s, p, v, ah) in configs.items():
        label = {
            "massima_tensione": "Massima Tensione",
            "massima_capacita": "Massima Capacità",
            "bilanciata": "Bilanciata"
        }[nome]
        print(f"{label}: {s}S{p}P  →  {v:.2f} V, {ah:.2f} Ah")

# Esempio d'uso
if __name__ == "__main__":
    celle_totali = 132  # puoi cambiarlo
    configs = trova_tutte_configurazioni(celle_totali)
    stampa_configurazioni(configs)
    print(type(configs))