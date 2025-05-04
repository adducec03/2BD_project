def calcola_configurazione_batteria(v_cella, ah_cella, v_target, ah_target, celle_disponibili):
    s = round(v_target / v_cella)
    p = round(ah_target / ah_cella)
    totale = s * p

    print(f"Configurazione suggerita: {s}S{p}P")
    print(f"Celle totali richieste: {totale}")

    if totale <= celle_disponibili:
        print("Le celle entrano nella forma disponibile.")
    else:
        print("Troppe celle. Considera di abbassare la capacità o cambiare disposizione.")
    return s, p, totale

# Esempio
calcola_configurazione_batteria(
    v_cella=3.6,
    ah_cella=2.5,
    v_target=36,
    ah_target=22,
    celle_disponibili=94
)