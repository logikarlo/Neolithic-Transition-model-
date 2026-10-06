"""
Small ensemble: N realizations with Ornstein-Uhlenbeck noise on log C(t).

The noise is a mean-reverting random walk. Stationary standard deviation
is ~15% in log C, which is enough to produce spread in the crossing year
without making the transition fire at random times.

Usage:
    python run_ensemble.py [n_seeds] [climate_csv]
    python run_ensemble.py 10 levant_climate.csv
"""



import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "models"))

_DATA_DIR = _REPO_ROOT / "data"
_SPD_DIR = _REPO_ROOT / "xronos_SPDs"
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


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    climate_path = sys.argv[2] if len(sys.argv) > 2 else str(_DATA_DIR / "levant_climate.csv")
    tag = Path(climate_path).stem

    print("=" * 72)
    print(f"GRID ENSEMBLE ({n_seeds} realizations) — {climate_path}")
    print("=" * 72)
    print(f"OU noise: sigma = 0.005, tau = 2000 yr")

    emp = load_empirical()
    ex = emp["x"]
    elm_n = emp["forager_raw"] / emp["forager_raw"].max()
    efm_n = emp["farmer_raw"] / emp["farmer_raw"].max()

    denom_emp = emp["forager_raw"] + emp["farmer_raw"]
    share_emp = np.where(denom_emp > 1e-6,
                         emp["farmer_raw"] / denom_emp, np.nan)
    valid = ~np.isnan(share_emp)

    t_ref = None
    F_all, A_all, share_all = [], [], []
    ms_f, ms_a, ms_s = [], [], []
    cross_yr = []
    first_conv = []

    for seed in range(n_seeds):
        print(f"  seed {seed}...", end=" ", flush=True)
        m = Model(seed=seed, climate_path=climate_path, noise_sigma=0.005)
        h = m.run()
        t = np.array(h["t"])
        F = np.array(h["F"])
        A = np.array(h["A"])
        share = np.array(h["farmer_share"])
        if t_ref is None:
            t_ref = t

        Fn = F / F.max() if F.max() > 0 else F
        An = A / A.max() if A.max() > 0 else A
        Fn_i = np.interp(ex, t[::-1], Fn[::-1])
        An_i = np.interp(ex, t[::-1], An[::-1])
        ms_f.append(float(np.mean((Fn_i - elm_n) ** 2)))
        ms_a.append(float(np.mean((An_i - efm_n) ** 2)))

        share_i = np.interp(ex, t[::-1], share[::-1])
        ms_s.append(float(np.mean((share_i[valid] - share_emp[valid]) ** 2)))

        above = np.where(share >= 0.5)[0]
        cross_yr.append(float(t[above[0]]) if len(above) > 0
                        else float("nan"))

        first_conv.append(m.first_conversion_bp
                          if m.first_conversion_bp is not None
                          else float("nan"))

        F_all.append(F)
        A_all.append(A)
        share_all.append(share)
        print(f"cross={cross_yr[-1]:.0f}")

    F_arr = np.array(F_all)
    A_arr = np.array(A_all)
    share_arr = np.array(share_all)

    print(f"\nMSE across {n_seeds} realizations:")
    print(f"  Forager:  {np.mean(ms_f):.4f} +/- {np.std(ms_f):.4f}")
    print(f"  Farmer:   {np.mean(ms_a):.4f} +/- {np.std(ms_a):.4f}")
    print(f"  Share:    {np.mean(ms_s):.4f} +/- {np.std(ms_s):.4f}")
    print(f"  Crossing year: "
          f"{np.nanmean(cross_yr):.0f} +/- {np.nanstd(cross_yr):.0f} BP "
          f"(range {np.nanmin(cross_yr):.0f}–{np.nanmax(cross_yr):.0f})")
    print(f"  First conversion: "
          f"{np.nanmean(first_conv):.0f} +/- "
          f"{np.nanstd(first_conv):.0f} BP")

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    ax = axs[0, 0]
    ax.fill_between(t_ref,
                    np.percentile(F_arr, 5, axis=0),
                    np.percentile(F_arr, 95, axis=0),
                    color="steelblue", alpha=0.25)
    ax.plot(t_ref, np.median(F_arr, axis=0), color="steelblue",
            lw=2, label="Foragers")
    ax.fill_between(t_ref,
                    np.percentile(A_arr, 5, axis=0),
                    np.percentile(A_arr, 95, axis=0),
                    color="firebrick", alpha=0.25)
    ax.plot(t_ref, np.median(A_arr, axis=0), color="firebrick",
            lw=2, label="Farmers")
    ax.set_xlim(T_START, T_END)
    ax.set_xlabel("Calibrated age (BP)")
    ax.set_ylabel("Population")
    ax.set_title(f"Ensemble populations ({n_seeds} seeds)")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axs[0, 1]
    ax.fill_between(t_ref,
                    np.percentile(share_arr, 5, axis=0),
                    np.percentile(share_arr, 95, axis=0),
                    color="purple", alpha=0.25)
    ax.plot(t_ref, np.median(share_arr, axis=0), color="purple",
            lw=2, label="Farmer share")
    ax.axhline(0.5, color="grey", ls=":", alpha=0.6)
    ax.set_xlim(T_START, T_END)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Calibrated age (BP)")
    ax.set_ylabel("Farmer share")
    ax.set_title(f"Ensemble farmer share")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axs[1, 0]
    ax.hist([c for c in cross_yr if not np.isnan(c)],
            bins=max(3, n_seeds // 3),
            color="firebrick", alpha=0.7)
    ax.set_xlabel("Farmer share crossing (BP)")
    ax.set_ylabel("Count")
    ax.set_title(f"Crossing distribution "
                 f"(std = {np.nanstd(cross_yr):.0f} yr)")
    ax.grid(alpha=0.3, axis="y")

    ax = axs[1, 1]
    ax.scatter(range(n_seeds), ms_f, color="steelblue",
               label="Forager MSE")
    ax.scatter(range(n_seeds), ms_a, color="firebrick",
               label="Farmer MSE")
    ax.scatter(range(n_seeds), ms_s, color="purple",
               label="Share MSE", alpha=0.7)
    ax.set_xlabel("Seed")
    ax.set_ylabel("MSE")
    ax.set_title("Per-seed MSE")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    plt.savefig(_FIG_DIR / f"grid_ensemble_{tag}.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
