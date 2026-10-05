"""
Sensitivity sweep of the four free parameters.

For each parameter, runs the grid model across a range of values,
computes MSE and crossing year, and produces a 2x2 figure with one
panel per parameter.

Usage:
    python run_sensitivity.py <climate_csv>
    python run_sensitivity.py levant_climate.csv
    python run_sensitivity.py levant_climate2.csv
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "models"))

_DATA_DIR = _REPO_ROOT / "data"
_SPD_DIR = _REPO_ROOT / "xronos SPDs"
_FIG_DIR = _REPO_ROOT / "figures"

from latest_levant_model import Model, T_START, T_END


def load_empirical():
    f = np.load(_SPD_DIR / "levant_forager_spd.npz")
    a = np.load(_SPD_DIR / "levant_farmer_spd.npz")
    return {
        "x": f["age"],
        "forager": f["median"],
        "farmer": a["median"],
        "forager_raw": f["raw"] if "raw" in f.files else f["median"],
        "farmer_raw": a["raw"] if "raw" in a.files else a["median"],
    }


def evaluate(h, emp):
    t = np.array(h["t"])
    F = np.array(h["F"])
    A = np.array(h["A"])
    share = np.array(h["farmer_share"])

    Fn = F / F.max() if F.max() > 0 else F
    An = A / A.max() if A.max() > 0 else A
    elm_n = emp["forager_raw"] / emp["forager_raw"].max()
    efm_n = emp["farmer_raw"] / emp["farmer_raw"].max()
    ex = emp["x"]

    Fn_i = np.interp(ex, t[::-1], Fn[::-1])
    An_i = np.interp(ex, t[::-1], An[::-1])
    mse_f = float(np.mean((Fn_i - elm_n) ** 2))
    mse_a = float(np.mean((An_i - efm_n) ** 2))

    denom = emp["forager_raw"] + emp["farmer_raw"]
    share_emp = np.where(denom > 1e-6,
                         emp["farmer_raw"] / denom, np.nan)
    valid = ~np.isnan(share_emp)
    share_i = np.interp(ex, t[::-1], share[::-1])
    mse_share = float(np.mean(
        (share_i[valid] - share_emp[valid]) ** 2))

    above = np.where(share >= 0.5)[0]
    cross_year = float(t[above[0]]) if len(above) > 0 else np.nan

    return dict(mse_f=mse_f, mse_a=mse_a, mse_total=mse_f + mse_a,
                mse_share=mse_share, cross_year=cross_year)


def sweep(param_name, values, emp, climate_path):
    rows = []
    for v in values:
        m = Model(seed=42, climate_path=climate_path, **{param_name: v})
        h = m.run()
        e = evaluate(h, emp)
        e[param_name] = v
        rows.append(e)
        print(f"  {param_name}={v:<8}  "
              f"MSE={e['mse_total']:.4f}  "
              f"share={e['mse_share']:.4f}  "
              f"cross={e['cross_year']:.0f}")
    return rows


def main():
    climate_path = sys.argv[1] if len(sys.argv) > 1 else str(_DATA_DIR / "levant_climate2.csv")
    tag = Path(climate_path).stem
    print("=" * 70)
    print(f"SENSITIVITY SWEEP — {climate_path}")
    print("=" * 70)

    emp = load_empirical()

    grids = {
        "TAU_D": [200.0, 400.0, 600.0, 900.0, 1400.0],
        "K_CONV": [0.01, 0.02, 0.03, 0.05, 0.10],
        "ALPHA_SHORT": [1.0, 2.0, 5.0, 10.0, 20.0],
        "SOIL_DEGR": [0.00035, 0.00040, 0.00050, 0.00060, 0.00070, 0.00080],
        
    }

    results = {}
    for name, values in grids.items():
        print(f"\nSweeping {name}")
        results[name] = sweep(name, values, emp, climate_path)

    # --- Figure ---
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    for ax, (name, values) in zip(axs.flat, grids.items()):
        rows = results[name]
        xs = [r[name] for r in rows]
        mse = [r["mse_total"] for r in rows]
        share_mse = [r["mse_share"] for r in rows]
        cross = [r["cross_year"] for r in rows]

        ax.plot(xs, mse, "o-", color="firebrick", lw=2,
                label="Total MSE")
        ax.plot(xs, share_mse, "s--", color="purple", lw=1.5,
                label="Share MSE")
        ax.set_xlabel(name)
        ax.set_ylabel("MSE", color="firebrick")
        ax.tick_params(axis="y", labelcolor="firebrick")
        ax.grid(alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(xs, cross, "^:", color="steelblue", lw=1.5,
                 label="Crossing year (BP)")
        ax2.set_ylabel("Farmer-share crossing (BP)",
                       color="steelblue")
        ax2.tick_params(axis="y", labelcolor="steelblue")
        ax2.invert_yaxis()

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  loc="best", fontsize=8)
        ax.set_title(f"Sensitivity: {name}  ({tag})")

    plt.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    out = _FIG_DIR / f"sensitivity_{tag}.png"
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"\nSaved {out}")

    # --- Summary table ---
    print(f"\nSummary:")
    print(f"{'Parameter':<14s} {'MSE min':>10s} {'MSE max':>10s} "
          f"{'Cross min':>10s} {'Cross max':>10s}")
    print("-" * 58)
    for name in grids:
        rows = results[name]
        mses = [r["mse_total"] for r in rows]
        crosses = [r["cross_year"] for r in rows
                   if not np.isnan(r["cross_year"])]
        c_min = f"{min(crosses):.0f}" if crosses else "nan"
        c_max = f"{max(crosses):.0f}" if crosses else "nan"
        print(f"{name:<14s} {min(mses):>10.4f} {max(mses):>10.4f} "
              f"{c_min:>10s} {c_max:>10s}")


if __name__ == "__main__":
    main()