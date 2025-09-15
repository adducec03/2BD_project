# batch_test_from_csv_no_cli.py
import os, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # niente finestre
import matplotlib.pyplot as plt

import battery_layout_cpsat_v2 as bc
from ortools.sat.python import cp_model


# ===================== PARAMETRI =====================
CSV_PATH         = "others/new_version/out.csv"   # CSV con colonne x,y
OUT_DIR          = "others/new_version/test/test"                         # cartella output (PNG+TXT)

S_MIN, S_MAX     = 2, None                        # se None → fino a N
P_MIN, P_MAX     = 2, None

RADIUS_MM        = 9.0                            # None → stimato da dati (NN/2)
TOL_MM           = 5.0                            # None → 0.25*R

TIME_BUDGET_PER  = 30                             # sec per configurazione
DEGREE_CAP       = 6
ENFORCE_DEGREE   = False
SEEDS            = (0, 1, 2)
WORKERS          = 6
STRIPE_MODE      = "auto"                         # "auto" | "rows" | "columns" | "none"
MAX_CONFIGS      = None                           # None = tutte le (S,P) con S*P ≤ N
SAVE_PNG         = True                           # salva i PNG dei layout
# =====================================================


def status_name(code):
    return {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
        cp_model.UNKNOWN: "UNKNOWN",
    }.get(code, str(code))


def estimate_radius_from_coords(coords: np.ndarray) -> float:
    """Stima R come mediana della distanza al nearest neighbor diviso 2."""
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    d, _ = tree.query(coords, k=2)     # k=1 è (0), k=2 è il vero NN
    nn = d[:, 1]
    r_est = float(np.median(nn) / 2.0)
    return max(1e-6, r_est)


# ----------------- GRAFICI DI QUALITÀ (AGGIUNTA) -----------------
def _parse_results(results_path):
    """
    Legge results.txt. Ritorna una lista di dict:
    {S,P,status,sumL,minL,avgL,ratio_min,ratio_avg,used_cells,wall_time_s}
    """
    rows = []
    if not os.path.isfile(results_path):
        return rows
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            # Formato:
            # S,P,status,sum_links,min_links,avg_links,ratio_min,ratio_avg,used_cells,wall_time_s
            try:
                S = int(parts[0]); P = int(parts[1]); status = parts[2]
                rec = dict(S=S, P=P, status=status,
                           sumL=None, minL=None, avgL=None,
                           ratio_min=None, ratio_avg=None,
                           used_cells=None, wall_time_s=None)
                if status in ("FEASIBLE","OPTIMAL"):
                    rec["sumL"] = int(float(parts[3]))
                    rec["minL"] = int(float(parts[4]))
                    rec["avgL"] = float(parts[5])
                    rec["ratio_min"] = float(parts[6])
                    rec["ratio_avg"] = float(parts[7])
                    rec["used_cells"] = int(float(parts[8]))
                    rec["wall_time_s"] = float(parts[9])
                else:
                    # INFEASIBLE riga: S,P,INFEASIBLE,,,,,,used_cells,wall_time_s
                    rec["used_cells"] = int(float(parts[-2])) if len(parts) >= 2 else None
                    rec["wall_time_s"] = float(parts[-1]) if len(parts) >= 1 else None
                rows.append(rec)
            except Exception:
                # linea malformata → skip
                continue
    return rows


def make_quality_plots(results_path, out_dir="others/new_version/test/test"):
    import math
    os.makedirs(out_dir, exist_ok=True)
    data = _parse_results(results_path)
    if not data:
        print("Nessun dato in results.txt: salto i grafici.")
        return

    import pandas as pd
    df = pd.DataFrame(data)

    # helper: salvataggio sicuro senza mostrare
    def _save(fig, name):
        path = os.path.join(out_dir, name)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)

    # status numerico
    status_code = {"INFEASIBLE": 0, "FEASIBLE": 1, "OPTIMAL": 2}
    df["status_code"] = df["status"].map(status_code).fillna(np.nan)

    # normalizzazioni e denominatori
    df["den_min"] = 2.0 * df["P"].clip(lower=1)                # per minL, avgL
    df["den_sum"] = (2.0 * df["P"] * (df["S"] - 1).clip(lower=1))
    df["q_min"]   = np.where(df["status"].isin(["FEASIBLE","OPTIMAL"]),
                             np.clip(df["minL"] / df["den_min"], 0, 1), np.nan)
    df["q_avg"]   = np.where(df["status"].isin(["FEASIBLE","OPTIMAL"]),
                             np.clip(df["avgL"] / df["den_min"], 0, 1), np.nan)
    df["sum_norm"]= np.where(df["status"].isin(["FEASIBLE","OPTIMAL"]),
                             np.clip(df["sumL"] / df["den_sum"], 0, 1), np.nan)

    Smax, Pmax = int(df["S"].max()), int(df["P"].max())

    # ---------- HEATMAPS ----------
    def heat(values_series, title, fname, vmin=0, vmax=1, cmap="magma", log_colorbar=False):
        grid = np.full((Smax + 1, Pmax + 1), np.nan, float)
        for _, r in df.iterrows():
            s, p = int(r["S"]), int(r["P"])
            v = r[values_series] if isinstance(values_series, str) else values_series(r)
            grid[s, p] = v
        fig, ax = plt.subplots(figsize=(10, 6))
        if log_colorbar:
            from matplotlib.colors import LogNorm
            # aggiungi epsilon per evitare zero in log
            eps = np.nanmin(grid[grid > 0]) if np.any(grid > 0) else 1e-3
            im = ax.imshow(grid.T, origin="lower", aspect="auto",
                           norm=LogNorm(vmin=max(eps, 1e-3), vmax=np.nanmax(grid)),
                           cmap=cmap)
            cbar = fig.colorbar(im, ax=ax)
        else:
            im = ax.imshow(grid.T, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
            cbar = fig.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(title)
        ax.set_xlabel("S (serie)"); ax.set_ylabel("P (parallelo)")
        ax.set_title(title)
        _save(fig, fname)

    # stato
    heat("status_code", "Status (0=INF,1=FEA,2=OPT)", "status_heatmap.png", vmin=0, vmax=2, cmap="viridis")
    # qualità
    heat("q_min",   "q_min = minL / 2P",            "quality_heatmap.png",      0, 1, "magma")
    heat("q_avg",   "q_avg = avgL / 2P",            "quality_avg_heatmap.png",  0, 1, "plasma")
    heat("sum_norm","sumL / ((S-1)·2P)",            "sum_norm_heatmap.png",     0, 1, "inferno")
    # tempi
    heat("wall_time_s", "Wall time [s] (log)",      "runtime_heatmap.png",      cmap="viridis", log_colorbar=True)

    # ---------- STACKED BAR: conteggi per P e per S ----------
    def stacked_counts(group_key, fname, title):
        counts = (df.groupby([group_key, "status"])
                    .size().unstack(fill_value=0)
                    .reindex(columns=["INFEASIBLE","FEASIBLE","OPTIMAL"], fill_value=0))
        xs = counts.index.values
        INF = counts.get("INFEASIBLE", pd.Series(0, index=xs))
        FEA = counts.get("FEASIBLE",   pd.Series(0, index=xs))
        OPT = counts.get("OPTIMAL",    pd.Series(0, index=xs))
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(xs, INF, label="INF")
        ax.bar(xs, FEA, bottom=INF, label="FEA")
        ax.bar(xs, OPT, bottom=INF+FEA, label="OPT")
        ax.set_xlabel(group_key)
        ax.set_ylabel("# configurazioni")
        ax.set_title(title)
        ax.legend()
        _save(fig, fname)

    stacked_counts("P", "status_by_P.png", "Stato per P (stacked)")
    stacked_counts("S", "status_by_S.png", "Stato per S (stacked)")

    # ---------- BOXPLOT: q_min e q_avg per P e per S ----------
    def boxplot_by(key, metric, fname, title, ylim=(0,1)):
        sub = df[df["status"].isin(["FEASIBLE","OPTIMAL"]) & df[metric].notna()]
        if sub.empty:
            return
        groups = []
        labels = []
        for k in sorted(sub[key].unique()):
            vals = sub.loc[sub[key]==k, metric].values
            if len(vals) > 0:
                groups.append(vals); labels.append(int(k))
        if not groups:
            return
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.boxplot(groups, labels=labels, showmeans=True)
        ax.set_xlabel(key); ax.set_ylabel(metric)
        if ylim: ax.set_ylim(*ylim)
        ax.set_title(title)
        _save(fig, fname)

    boxplot_by("P", "q_min", "box_qmin_by_P.png", "q_min per P (boxplot)")
    boxplot_by("S", "q_min", "box_qmin_by_S.png", "q_min per S (boxplot)")
    boxplot_by("P", "q_avg", "box_qavg_by_P.png", "q_avg per P (boxplot)")
    boxplot_by("S", "q_avg", "box_qavg_by_S.png", "q_avg per S (boxplot)")

    # ---------- CURVE MEDIE ±1σ ----------
    def mean_curve(key, metric, fname, title, ylim=(0,1)):
        sub = df[df["status"].isin(["FEASIBLE","OPTIMAL"]) & df[metric].notna()]
        if sub.empty: return
        agg = sub.groupby(key)[metric].agg(["mean","std"]).reset_index()
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(agg[key], agg["mean"], marker="o")
        if np.any(~np.isnan(agg["std"].values)):
            ax.fill_between(agg[key], agg["mean"]-agg["std"], agg["mean"]+agg["std"], alpha=0.2)
        ax.set_xlabel(key); ax.set_ylabel(metric)
        if ylim: ax.set_ylim(*ylim)
        ax.set_title(title)
        _save(fig, fname)

    mean_curve("P", "q_min",    "curve_qmin_vs_P.png",   "q_min medio vs P (±1σ)")
    mean_curve("P", "sum_norm", "curve_sumNorm_vs_P.png","sumL_norm medio vs P (±1σ)")
    mean_curve("S", "q_min",    "curve_qmin_vs_S.png",   "q_min medio vs S (±1σ)", ylim=(0,1))
    # tempi
    def mean_curve_time(key, fname, title):
        sub = df[df["wall_time_s"].notna()]
        if sub.empty: return
        agg = sub.groupby(key)["wall_time_s"].agg(["mean","std"]).reset_index()
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(agg[key], agg["mean"], marker="o")
        if np.any(~np.isnan(agg["std"].values)):
            ax.fill_between(agg[key], np.maximum(0, agg["mean"]-agg["std"]), agg["mean"]+agg["std"], alpha=0.2)
        ax.set_xlabel(key); ax.set_ylabel("Wall time [s]")
        ax.set_title(title)
        _save(fig, fname)

    mean_curve_time("P", "curve_time_vs_P.png", "Wall time medio vs P (±1σ)")
    mean_curve_time("S", "curve_time_vs_S.png", "Wall time medio vs S (±1σ)")

    # ---------- SCATTER ----------
    # sum_norm vs q_min, colorato per stato
    sub = df[df["q_min"].notna() & df["sum_norm"].notna()]
    if not sub.empty:
        colors = sub["status"].map({"OPTIMAL":"tab:green","FEASIBLE":"tab:blue","INFEASIBLE":"tab:red"}).fillna("gray")
        fig, ax = plt.subplots(figsize=(6,6))
        ax.scatter(sub["q_min"], sub["sum_norm"], s=18, c=colors)
        ax.set_xlabel("q_min = minL/2P"); ax.set_ylabel("sumL / ((S-1)·2P)")
        ax.set_title("Qualità globale vs minima")
        ax.grid(alpha=.3)
        _save(fig, "scatter_sumNorm_vs_qmin.png")

    # tempo vs celle usate
    sub = df[df["wall_time_s"].notna() & df["used_cells"].notna()]
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(8,4))
        ax.scatter(sub["used_cells"], sub["wall_time_s"], s=14, alpha=.8)
        ax.set_xlabel("S·P (celle usate)"); ax.set_ylabel("Wall time [s]")
        ax.set_title("Tempo di risoluzione vs celle usate")
        ax.grid(alpha=.3)
        _save(fig, "runtime_vs_cells.png")

    # ---------- ISTOGRAMMI ----------
    # distribuzione q_min
    sub = df[df["q_min"].notna()]
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(8,4))
        ax.hist(sub["q_min"], bins=20, range=(0,1), alpha=0.9)
        ax.set_xlabel("q_min"); ax.set_ylabel("conteggio")
        ax.set_title("Distribuzione q_min")
        _save(fig, "hist_qmin.png")

    # distribuzione tempi (log x)
    sub = df[df["wall_time_s"].notna() & (df["wall_time_s"] > 0)]
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(8,4))
        ax.hist(sub["wall_time_s"], bins=20, alpha=0.9)
        ax.set_xscale("log")
        ax.set_xlabel("Wall time [s] (log)"); ax.set_ylabel("conteggio")
        ax.set_title("Distribuzione tempi")
        _save(fig, "hist_time_log.png")

    # ---------- CONTEGGI GLOBALI ----------
    n_opt = int((df["status"]=="OPTIMAL").sum())
    n_fea = int((df["status"]=="FEASIBLE").sum())
    n_inf = int((df["status"]=="INFEASIBLE").sum())
    fig, ax = plt.subplots(figsize=(5,5))
    ax.bar(["OPT","FEA","INF"], [n_opt, n_fea, n_inf])
    for i, v in enumerate([n_opt, n_fea, n_inf]):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set_title("Conteggi soluzione")
    _save(fig, "status_counts.png")

    print("Grafici di qualità salvati in:", out_dir)
# ----------------------------------------------------


def run():
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- carica coords dal CSV ---
    df = pd.read_csv(CSV_PATH)
    coords = df[["x", "y"]].to_numpy(dtype=float)
    N = len(coords)
    if N < 2:
        raise SystemExit("Il file CSV deve contenere almeno 2 punti (colonne x,y).")

    # --- raggio e tolleranza ---
    radius_mm = RADIUS_MM if RADIUS_MM is not None else estimate_radius_from_coords(coords)
    tol_mm = (0.25 * radius_mm) if TOL_MM is None else float(TOL_MM)

    print(f"[info] N={N}  R={radius_mm:.3f} mm  tol={tol_mm:.3f} mm")

    # --- range S,P ---
    Smax = S_MAX or N
    Pmax = P_MAX or N

    results_txt = os.path.join(OUT_DIR, "results.txt")
    summary_txt = os.path.join(OUT_DIR, "summary.txt")

    tested = feasible = optimal = 0
    ratios_min, ratios_avg = [], []
    best_cfg = None  # (score, S, P, min_links, sum_links, status)

    with open(results_txt, "w", encoding="utf-8") as f:
        f.write("# S,P,status,sum_links,min_links,avg_links,ratio_min,ratio_avg,used_cells,wall_time_s\n")

        for S in range(S_MIN, Smax + 1):
            for P in range(P_MIN, Pmax + 1):
                if S * P > N:
                    continue
                if MAX_CONFIGS is not None and tested >= MAX_CONFIGS:
                    break
                tested += 1

                try:
                    pack = bc.auto_tune_and_solve(
                        coords, S, P, radius_mm, tol_mm,
                        time_budget=TIME_BUDGET_PER,
                        target_T=min(2 * P, 2 * P),
                        degree_cap=DEGREE_CAP,
                        enforce_degree=ENFORCE_DEGREE,
                        profiles=("fast", "fast", "quality"),
                        seeds=SEEDS,
                        workers=WORKERS,
                        use_hole_penality=False,
                        stripe_mode=STRIPE_MODE
                    )
                except SystemExit:
                    f.write(f"{S},{P},INFEASIBLE,,,,,,{S*P},0.0\n")
                    continue

                status, solver, x, r, L, z1, z2, E, T = pack
                sname = status_name(status)

                if sname in ("OPTIMAL", "FEASIBLE"):
                    feasible += 1
                    if sname == "OPTIMAL":
                        optimal += 1

                    links = [int(solver.Value(v)) for v in L] if L else []
                    sumL  = int(sum(links)) if links else 0
                    minL  = int(min(links)) if links else 0
                    avgL  = float(sumL / max(1, len(links))) if links else 0.0

                    ratio_min = (minL / float(2 * P)) if P > 0 else 0.0
                    ratio_avg = (avgL / float(2 * P)) if P > 0 else 0.0

                    ratios_min.append(ratio_min)
                    ratios_avg.append(ratio_avg)

                    # score: privilegia alto minL, poi sumL
                    score = (minL, sumL)
                    if (best_cfg is None) or (score > best_cfg[0]):
                        best_cfg = (score, S, P, minL, sumL, sname)

                    if SAVE_PNG:
                        png = os.path.join(OUT_DIR, f"layout_S{S}_P{P}.png")
                        bc.plot_solution(
                            coords, radius_mm, S, x, z1, z2, E, L, solver,
                            title=f"{S}S{P}P — {sname} (sumL={sumL}, minL={minL})",
                            save=png, show_links=True, show_arrows=True
                        )
                        plt.close("all")

                    f.write(f"{S},{P},{sname},{sumL},{minL},{avgL:.2f},{ratio_min:.3f},"
                            f"{ratio_avg:.3f},{S*P},{solver.WallTime():.3f}\n")
                else:
                    f.write(f"{S},{P},INFEASIBLE,,,,,,{S*P},0.0\n")

            if MAX_CONFIGS is not None and tested >= MAX_CONFIGS:
                break

    # --- metriche aggregate (+ salvataggio summary) ---
    total = tested
    feas_rate = feasible / total if total else 0.0
    opt_rate  = (optimal / feasible) if feasible else 0.0
    mean_ratio_min = float(np.mean(ratios_min)) if ratios_min else 0.0
    mean_ratio_avg = float(np.mean(ratios_avg)) if ratios_avg else 0.0
    std_ratio_min  = float(np.std(ratios_min))  if ratios_min else 0.0

    with open(summary_txt, "w", encoding="utf-8") as g:
        g.write("=== Batch summary ===\n")
        g.write(f"CSV: {CSV_PATH}\n")
        g.write(f"N celle: {N}\n")
        g.write(f"Configs testate: {total}\n")
        g.write(f"Feasible: {feasible}  ({feas_rate:.1%})\n")
        g.write(f"Optimal:  {optimal}   ({opt_rate:.1%} dei feasible)\n")
        g.write(f"Mean ratio_min (minL/2P): {mean_ratio_min:.3f}\n")
        g.write(f"Mean ratio_avg (avgL/2P): {mean_ratio_avg:.3f}\n")
        g.write(f"Std  ratio_min:          {std_ratio_min:.3f}\n")
        if best_cfg:
            _score, bS, bP, bMin, bSum, bStat = best_cfg
            g.write(f"Miglior config (per minL poi sumL): S={bS}, P={bP}, "
                    f"minL={bMin}, sumL={bSum}, status={bStat}\n")

    # --- SALVA GRAFICI DI QUALITÀ (nuovo) ---
    make_quality_plots(results_txt, out_dir=OUT_DIR)

    print(f"Fatto.\n- Log: {results_txt}\n- Summary: {summary_txt}\n- PNG: {OUT_DIR}/layout_S*_P*.png")


if __name__ == "__main__":
    run()