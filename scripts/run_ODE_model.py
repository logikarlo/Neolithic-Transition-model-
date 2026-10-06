"""
Main ODE comparison: model vs empirical SPD.

Usage:
    python run_model.py <climate_csv>
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

from model_ode import Model, T_START, T_END, DOMESTICATION_START_BP


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
    climate_path = sys.argv[1] if len(sys.argv) > 1 else str(_DATA_DIR / "levant_climate.csv")
    tag = Path(climate_path).stem
    print("=" * 70)
    print(f"SINGLE-POPULATION ODE — {climate_path}")
    print("=" * 70)

    m = Model(seed=42, climate_path=climate_path)
    print(f"PRE-CHECK C(15000) = {m.climate(15000.0):.4f}")
    print(f"PRE-CHECK C(13000) = {m.climate(13000.0):.4f}")
    print()

    h = m.run(verbose=True)

    t = np.array(h["t"])
    F = np.array(h["F"])
    A = np.array(h["A"])
    H = np.array(h["H"])
    d = np.array(h["d"])
    share = np.array(h["share"])
    conv_in = np.array(h["conv_in_situ"])
    conv_cu = np.array(h["conv_cultural"])
    c_for = np.array(h["c_for"])
    c_farm = np.array(h["c_farm"])
    P_dom = np.array(h["P_dom"])
    T_r = np.array(h["T_ratio"])

    emp = load_empirical()
    ex = emp["x"]

    Fn = F / F.max() if F.max() > 0 else F
    An = A / A.max() if A.max() > 0 else A
    elm_n = emp["forager"] / emp["forager"].max()
    efm_n = emp["farmer"] / emp["farmer"].max()

    Fn_i = np.interp(ex, t[::-1], Fn[::-1])
    An_i = np.interp(ex, t[::-1], An[::-1])
    mse_f = float(np.mean((Fn_i - elm_n) ** 2))
    mse_a = float(np.mean((An_i - efm_n) ** 2))

    denom_emp = emp["forager_raw"] + emp["farmer_raw"]
    share_emp = np.where(denom_emp > 1e-6,
                         emp["farmer_raw"] / denom_emp, np.nan)
    share_i = np.interp(ex, t[::-1], share[::-1])
    valid = ~np.isnan(share_emp)
    mse_share = float(np.mean((share_i[valid] - share_emp[valid]) ** 2))

    print()
    print(f"{'Age BP':>8s} {'F':>8s} {'A':>8s} {'H':>7s} "
          f"{'d':>7s} {'share':>7s} {'T':>7s}")
    print("-" * 60)
    for age in (22000, 20000, 18000, 16000, 15000, 14000, 13000,
                12500, 12000, 11500, 11000, 10000, 9000, 8000):
        i = int(np.argmin(np.abs(t - age)))
        print(f"{age:>8d} {F[i]:>8.2f} {A[i]:>8.2f} "
              f"{H[i]:>7.3f} {d[i]:>7.3f} {share[i]:>7.3f} "
              f"{T_r[i]:>7.3f}")

    print()
    print(f"MSE forager: {mse_f:.4f}, farmer: {mse_a:.4f}, "
          f"total: {mse_f + mse_a:.4f}")
    print(f"MSE farmer share: {mse_share:.4f}")
    if m.first_conversion_bp:
        print(f"First conversion: {m.first_conversion_bp:.0f} BP")

    fig, axs = plt.subplots(2, 3, figsize=(20, 11))

    ax = axs[0, 0]
    ax.plot(ex, emp["forager_raw"], "steelblue", lw=2, label="Foragers")
    ax.plot(ex, emp["farmer_raw"], "firebrick", lw=2, label="Farmers")
    ax.set_xlim(T_START, T_END)
    ax.set_xlabel("Calibrated age (BP)")
    ax.set_ylabel("Raw SPD amplitude")
    ax.set_title("1. Empirical population")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axs[0, 1]
    ax.plot(t, F, "steelblue", lw=2, label="Foragers")
    ax.plot(t, A, "firebrick", lw=2, label="Farmers")
    ax.axvline(DOMESTICATION_START_BP, color="grey", ls=":",
               label=f"Gate ({DOMESTICATION_START_BP:.0f} BP)")
    ax.set_xlim(T_START, T_END)
    ax.set_xlabel("Calibrated age (BP)")
    ax.set_ylabel("Population")
    ax.set_title("2. Model population")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axs[0, 2]
    ax.plot(t, H, "olive", lw=2, label="H(t)")
    ax.plot(t, d, "darkgreen", lw=2, label="d(t)")
    ax.axhline(0.5, color="red", ls=":", label="d_min")
    ax.set_xlim(T_START, T_END)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Calibrated age (BP)")
    ax.set_ylabel("Habitat / Dependence")
    ax.set_title("3. Habitat and dependence")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axs[1, 0]
    C_plot = np.array([m.climate(ti) for ti in t])
    ax.plot(t, C_plot, "grey", lw=2, label="C(t)")
    ax.plot(t, P_dom, "darkgoldenrod", lw=2, ls="--", label="P(t)")
    ax.plot(t, T_r, "purple", lw=2, ls="-.", label="T(t)")
    ax.set_xlim(T_START, T_END)
    ax.set_ylim(0, 1.6)
    ax.set_xlabel("Calibrated age (BP)")
    ax.set_ylabel("Value")
    ax.set_title("4. Climate, domestication, cost")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axs[1, 1]
    ax.plot(ex, elm_n, "steelblue", lw=1.4, ls="--", alpha=0.6,
            label="Foragers (emp.)")
    ax.plot(ex, efm_n, "firebrick", lw=1.4, ls="--", alpha=0.6,
            label="Farmers (emp.)")
    ax.plot(t, Fn, "steelblue", lw=2.2, label="Foragers (model)")
    ax.plot(t, An, "firebrick", lw=2.2, label="Farmers (model)")
    ax.set_xlim(T_START, T_END)
    ax.set_xlabel("Calibrated age (BP)")
    ax.set_ylabel("Normalised")
    ax.set_title("5. Model vs empirical (normalised)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axs[1, 2]
    ax.plot(t, share, "darkred", lw=2.2, label="Farmer share (model)")
    ax.plot(ex, share_emp, "purple", lw=1.4, ls="--", alpha=0.6,
            label="Farmer share (emp.)")
    ax.axhline(0.5, color="grey", ls=":", alpha=0.6)
    ax.set_xlim(T_START, T_END)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Calibrated age (BP)")
    ax.set_ylabel("Farmer share")
    ax.set_title("6. Farmer share")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    plt.savefig(_FIG_DIR / f"ode_{tag}.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
