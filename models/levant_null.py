"""
Climate-only null with farmer onset.

    dF/dt = r_F * F * (1 - F / (K_F * C(t)))
    for t < A_START: A = 0
    for t >= A_START: dA/dt = r_A * A * (1 - A / (K_A * C(t)))

Free parameters: r_F, K_F, r_A, K_A, A_START, A_seed = 6
Fit by grid search.
"""

import numpy as np
from latest_levant_model import load_climate, DT, T_START, T_END, N_STEPS


def run_null_v2(r_F, K_F, r_A, K_A, A_START, A_SEED,
                climate_path="levant_climate.csv"):
    climate = load_climate(climate_path)
    F = 1.0
    A = 0.0
    t_arr, F_arr, A_arr = [], [], []
    for s in range(N_STEPS):
        t = T_START - s * DT
        C = climate(t)
        K_F_eff = max(K_F * C, 1e-6)
        K_A_eff = max(K_A * C, 1e-6)
        if t > A_START:        # older than onset: no farmers
            A = 0.0
        else:                   # younger than onset: farmers grow
            if A < A_SEED:
                A = A_SEED
            A = A + r_A * A * (1 - A / K_A_eff) * DT
            A = max(A, 0.0)
        F = F + r_F * F * (1 - F / K_F_eff) * DT
        F = max(F, 0.0)
        t_arr.append(t)
        F_arr.append(F)
        A_arr.append(A)
    return np.array(t_arr), np.array(F_arr), np.array(A_arr)


def fit_null_v2(emp, climate_path="levant_climate.csv", verbose=True):
    r_F_grid = [0.0001, 0.0002, 0.0005, 0.001]
    K_F_grid = [200.0, 500.0, 2000.0, 5000.0]
    r_A_grid = [0.0002, 0.0005, 0.001, 0.002]
    K_A_grid = [100000.0, 300000.0, 500000.0, 1000000.0]
    A_START_grid = [12500.0, 12000.0, 11500.0, 11000.0]
    A_SEED_grid = [1.0, 10.0, 100.0]

    elm_n = emp["forager_raw"] / emp["forager_raw"].max()
    efm_n = emp["farmer_raw"] / emp["farmer_raw"].max()
    ex = emp["x"]

    best = None
    n_evals = 0
    for rF in r_F_grid:
        for KF in K_F_grid:
            for rA in r_A_grid:
                for KA in K_A_grid:
                    for A_START in A_START_grid:
                        for A_SEED in A_SEED_grid:
                            t, F, A = run_null_v2(
                                rF, KF, rA, KA, A_START, A_SEED,
                                climate_path)
                            Fn = F / max(F.max(), 1e-9)
                            An = A / max(A.max(), 1e-9)
                            Fn_i = np.interp(ex, t[::-1], Fn[::-1])
                            An_i = np.interp(ex, t[::-1], An[::-1])
                            mse_f = float(np.mean((Fn_i - elm_n) ** 2))
                            mse_a = float(np.mean((An_i - efm_n) ** 2))
                            total = mse_f + mse_a
                            n_evals += 1
                            if best is None or total < best["mse_total"]:
                                best = dict(
                                    r_F=rF, K_F=KF, r_A=rA, K_A=KA,
                                    A_START=A_START, A_SEED=A_SEED,
                                    mse_f=mse_f, mse_a=mse_a,
                                    mse_total=total)
                                if verbose:
                                    print(f"  [{n_evals}] "
                                          f"r_F={rF} K_F={KF} "
                                          f"r_A={rA} K_A={KA} "
                                          f"A_START={A_START} "
                                          f"A_SEED={A_SEED}  "
                                          f"MSE={total:.4f}")
    return best, run_null_v2(best["r_F"], best["K_F"],
                             best["r_A"], best["K_A"],
                             best["A_START"], best["A_SEED"],
                             climate_path)