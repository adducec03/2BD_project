# batch_test.py
import os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend non interattivo per salvare PNG senza finestre

import battery_layout_cpsat_v2 as bc
from ortools.sat.python import cp_model


def hex_rect_coords(rows=6, cols=12, R=9.0):
    """Rettangolo esagonale: righe sfalsate, distanza tra centri = 2R."""
    coords = []
    h = math.sqrt(3.0) * R         # passo verticale
    for r in range(rows):
        y = r * h
        xoff = (r % 2) * R         # sfalsamento orizzontale alternato
        for c in range(cols):
            x = 2.0 * R * c + xoff
            coords.append((x, y))
    return np.asarray(coords, dtype=float)


def run_all_tests(
    rows=6,
    cols=12,
    R=9.0,
    tol=2.0,
    time_budget_per=30,         # sec per configurazione (aumenta se vuoi più qualità)
    out_dir="test",
    stripe_mode="auto",
    seeds=(0,1,2),
    workers=6
):
    os.makedirs(out_dir, exist_ok=True)
    coords = hex_rect_coords(rows, cols, R)
    N = rows * cols
    results_path = os.path.join(out_dir, "results.txt")

    def status_name(code):
        return {
            cp_model.OPTIMAL: "OPTIMAL",
            cp_model.FEASIBLE: "FEASIBLE",
            cp_model.INFEASIBLE: "INFEASIBLE",
            cp_model.MODEL_INVALID: "MODEL_INVALID",
            cp_model.UNKNOWN: "UNKNOWN",
        }.get(code, str(code))

    with open(results_path, "w", encoding="utf-8") as f:
        f.write("# S,P,status,sum_links,min_links,per_k\n")
        for S in range(3, N+1):
            for P in range(3, N+1):
                if S * P > N:
                    continue  # non puoi assegnare più celle di quante ne hai

                try:
                    pack = bc.auto_tune_and_solve(
                        coords, S, P, R, tol,
                        time_budget=time_budget_per,
                        target_T=min(2*P, 2*P),   # safe cap
                        degree_cap=6,
                        enforce_degree=False,
                        profiles=("fast","fast","quality"),
                        seeds=seeds,
                        workers=workers,
                        use_hole_penality=False,
                        stripe_mode=stripe_mode
                    )
                except SystemExit as e:
                    # es. se S*P > N (non dovrebbe capitare per via del continue)
                    f.write(f"{S},{P},INFEASIBLE\n")
                    continue

                status, solver, x, r, L, z1, z2, E, T = pack
                sname = status_name(status)

                if sname in ("OPTIMAL", "FEASIBLE"):
                    links = [int(solver.Value(v)) for v in L] if L else []
                    sumL  = sum(links) if links else 0
                    minL  = min(links) if links else 0

                    # salva PNG
                    png = os.path.join(out_dir, f"layout_{S}S{P}P.png")
                    bc.plot_solution(
                        coords, R, S, x, z1, z2, E, L, solver,
                        title=f"{S}S{P}P — {sname} (sumL={sumL}, minL={minL})",
                        save=png, show_links=True, show_arrows=True
                    )

                    # log su txt
                    f.write(f"{S},{P},{sname},{sumL},{minL},{links}\n")
                else:
                    f.write(f"{S},{P},INFEASIBLE\n")

    print(f"Fatto. PNG in '{out_dir}', risultati in '{results_path}'.")
    return results_path


if __name__ == "__main__":
    run_all_tests()