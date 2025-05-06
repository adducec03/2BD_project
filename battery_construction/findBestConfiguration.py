def configurazione_bilanciata(celle_totali, v_cella=3.6, ah_cella=2.5):
    migliore = None
    differenza_minima = float("inf")

    for s in range(1, celle_totali + 1):
        if celle_totali % s != 0:
            continue
        p = celle_totali // s

        v_tot = s * v_cella
        ah_tot = p * ah_cella

        rapporto = v_tot / ah_tot
        differenza = abs(rapporto - 1)  # vogliamo V/Ah ≈ 1

        if differenza < differenza_minima:
            differenza_minima = differenza
            migliore = (s, p, v_tot, ah_tot)

    if migliore is None:
        raise ValueError("Nessuna configurazione bilanciata trovata.")
    
    s, p, v_tot, ah_tot = migliore
    print(f"⚖️ Configurazione bilanciata trovata: {s}S{p}P → {v_tot:.2f} V, {ah_tot:.2f} Ah (V/Ah = {v_tot/ah_tot:.2f})")
    return migliore

# Esempio d'uso
if __name__ == "__main__":
    celle = 94  # prova con 94 celle
    configurazione_bilanciata(celle)