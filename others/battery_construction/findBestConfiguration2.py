import math
import circlePacking2 as cp2
import json
import csv
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

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


def salva_csv(risultati, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["S","P","N_tot","V_eff","C_eff","delta_V","delta_C","delta_N","score","feasible"])
        for r in risultati:
            S,P = r["S"], r["P"]
            V_eff, C_eff = r["tensione_effettiva"], r["capacita_effettiva"]
            N_tot = r["celle_totali"]
            delta_V = abs(V_eff - target_voltage)/target_voltage
            delta_C = abs(C_eff - target_capacity)/target_capacity
            N_ideal = (target_voltage*target_capacity)/(cell_voltage*cell_current)
            delta_N = max(0.0, (N_tot - N_ideal)/celle_max)
            feasible = int(N_tot <= celle_max)  # estendi se hai altri vincoli hard
            w.writerow([S,P,N_tot,V_eff,C_eff,delta_V,delta_C,delta_N,r["score"],feasible])


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
    eta_s = 0.2
    eta_p = 0.5

    S_min = max(1, math.floor(target_voltage / cell_voltage * (1 - eta_s)))
    S_max = math.ceil(target_voltage / cell_voltage * (1 + eta_s))
    P_min = max(1, math.floor(target_capacity / cell_current * (1 - eta_p)))
    P_max = math.ceil(target_capacity / cell_current * (1 + eta_p))


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
    return risultati, risultati[:top_k]

# -----------------------------------------
# PLOT 1: Heatmap S×P (punteggio) + frontiera SP=Nmax
# -----------------------------------------
def plot_heatmap_sp(risultati, N_max, out_path="figs/heatmap_sp"):
    S_vals = sorted({r["S"] for r in risultati})
    P_vals = sorted({r["P"] for r in risultati})
    S2i = {s:i for i,s in enumerate(S_vals)}
    P2i = {p:i for i,p in enumerate(P_vals)}

    Z = np.full((len(P_vals), len(S_vals)), np.nan)  # righe=P, colonne=S
    for r in risultati:
        Z[P2i[r["P"]], S2i[r["S"]]] = r["score"]

    fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
    im = ax.imshow(
        Z, origin="lower", cmap="viridis", aspect="auto",
        extent=[min(S_vals)-0.5, max(S_vals)+0.5, min(P_vals)-0.5, max(P_vals)+0.5]
    )
    cbar = fig.colorbar(im, ax=ax, label="punteggio f (↓ meglio)")

    # Frontiera SP = Nmax
    S_line = np.linspace(min(S_vals), max(S_vals), 400)
    P_line = N_max / S_line
    ax.plot(S_line, P_line, "k--", lw=1, label=r"$SP=N_{\max}$")

    ax.set_xlabel("S (celle in serie)")
    ax.set_ylabel("P (celle in parallelo)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="upper right")
    ax.set_title("Mappa S×P (score) con frontiera SP = Nmax")

    pdf = out_path + ".pdf"; png = out_path + ".png"
    fig.savefig(pdf); fig.savefig(png, dpi=300)
    plt.close(fig)
    return pdf, png

# -----------------------------------------
# PLOT 2: Scatter (Veff, Ceff) con target e top-k
# -----------------------------------------
def plot_scatter_vc(risultati, top, target_voltage, target_capacity, out_path="figs/scatter_vc"):
    Xe = [r["tensione_effettiva"] for r in risultati]
    Ye = [r["capacita_effettiva"] * 1000 for r in risultati]  # <-- conversione in mAh

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax.scatter(Xe, Ye, s=14, c="0.7", label="ammissibili")

    xt = [t["tensione_effettiva"] for t in top]
    yt = [t["capacita_effettiva"] * 1000 for t in top]  # <-- idem qui
    ax.scatter(xt, yt, s=36, c="crimson", label="top-k", zorder=3)

    # annotazioni compatte
    for t in top:
        ax.annotate(f'{t["S"]}S×{t["P"]}P',
                    xy=(t["tensione_effettiva"], t["capacita_effettiva"] * 1000),
                    xytext=(5,5), textcoords="offset points", fontsize=9)

    ax.scatter([target_voltage], [target_capacity * 1000],  # <-- anche il target
               s=60, c="k", marker="X", label="target")

    ax.set_xlabel(r"$V_{\mathrm{eff}}$ [V]")
    ax.set_ylabel(r"$C_{\mathrm{eff}}$ [mAh]")  # <-- etichetta aggiornata
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    ax.set_title(r"Soluzioni nel piano ($V_{\mathrm{eff}}$, $C_{\mathrm{eff}}$)")

    pdf = out_path + ".pdf"; png = out_path + ".png"
    fig.savefig(pdf); fig.savefig(png, dpi=300)
    plt.close(fig)
    return pdf, png



if __name__ == "__main__":
    # --------------------------
    # Parametri di input
    # --------------------------
    cell_voltage = 3.6         # V
    cell_current = 2.5         # Ah
    target_voltage = 48.0     # V
    target_capacity = 30.0     # Ah
    celle_max = 200            # massimo consentito

    risultati, top = calcola_configurazioni_migliori(
        cell_voltage, cell_current,
        target_voltage, target_capacity,
        celle_max=celle_max, top_k=5
    )

    salva_csv(risultati, "results_sp_full.csv")
    salva_csv(top,       "results_sp_topk.csv")

    
    # Cartella figure
    os.makedirs("figs", exist_ok=True)

    # Plot
    pdf1, png1 = plot_heatmap_sp(risultati, celle_max, out_path="figs/heatmap_sp")
    pdf2, png2 = plot_scatter_vc(risultati, top, target_voltage, target_capacity, out_path="figs/scatter_vc")

    # Report su console
    print("📊 Figure salvate:")
    print(" -", pdf1)
    print(" -", pdf2)

    print("\n📋 Migliori 5 configurazioni:")
    for i, cfg in enumerate(top, 1):
        print(f"{i}.  {cfg['S']}S × {cfg['P']}P  → "
              f"{cfg['celle_totali']} celle  | "
              f"{cfg['tensione_effettiva']:.1f} V, "
              f"{cfg['capacita_effettiva']:.1f} Ah, "
              f"score: {cfg['score']:.3f}")