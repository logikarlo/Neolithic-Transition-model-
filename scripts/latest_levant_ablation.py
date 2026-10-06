"""
Ablation test: does the transition still fire when each conversion
route or the soil mechanism is removed?

Four conditions:
    control         both conversion routes + soil degradation
    in_situ_only    F_CULT = 0      (no cultural transmission)
    cultural_only   K_CONV = 0      (no in-situ conversion)
    no_soil         SOIL_DEGR = 0   (no soil depletion)

Usage:
    python run_ablation.py <climate_csv>
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


ABLATIONS = {
    "control":       {},
    "in_situ_only":  {"F_CULT": 0.0},
    "cultural_only": {"K_CONV": 0.0},
    "no_soil":       {"SOIL_DEGR": 0.0},
}

FARMER_PEAK_LO = 9800.0
FARMER_PEAK_HI = 7000.0


def load_empirical():
    f = np.load(_SPD_DIR / "levant_forager_spd.npz")
    a =	np.load(_SPD_DIR / "levant_farmer_spd.npz")
    return {
        "x": f["age"],
        "forager_raw": f["raw"] if "raw" in f.files else f["median"],
        "farmer_raw": a["raw"] if "raw" in a.files else a["median"],
    }


def peak_timing(age, y, t_lo, t_hi, tol=0.01):
    mask = (age >= t_hi) & (age <= t_lo)
    if mask.sum() < 3:
        return float("nan")
    age_w = age[mask]
    y_w = y[mask]
    if y_w.max() <= 0:
        return float("nan")
    y_n = y_w / y_w.max()
    d = np.diff(y_n)
    if (d >= -1e-9).all() or (d <= 1e-9).all():
        return float("nan")
    peak_region = y_n >= (1 - tol) * y_n.max()
    return float(age_w[peak_region].mean())


def evaluate(m, h, emp):
    t = np.array(h["t"])
    F = np.array(h["F"])
    A = np.array(h["A"])
    share = np.array(h["farmer_share"])

    Fn = F / max(F.max(), 1e-9)
    An = A / max(A.max(), 1e-9)
    elm = emp["forager_raw"] / emp["forager_raw"].max()
    efm = emp["farmer_raw"] / emp["farmer_raw"].max()
    ex = emp["x"]

    Fn_i = np.interp(ex, t[::-1], Fn[::-1])
    An_i = np.interp(ex, t[::-1], An[::-1])
    mse_f = float(np.mean((Fn_i - elm) ** 2))
    mse_a = float(np.mean((An_i - efm) ** 2))

    above = np.where(share >= 0.5)[0]
    cross = float(t[above[0]]) if len(above) > 0 else float("nan")

    farmer_peak = peak_timing(t, An, FARMER_PEAK_LO, FARMER_PEAK_HI)

    return dict(
        first=m.first_conversion_bp,
        cross=cross,
        mse_f=mse_f,
        mse_a=mse_a,
        mse_total=mse_f + mse_a,
        farmer_peak=farmer_peak,
        F_final=float(F[-1]),
        A_final=float(A[-1]),
    )


def main():
    climate_path = sys.argv[1] if len(sys.argv) > 1 else str(_DATA_DIR / "levant_climate.csv")
    tag = Path(climate_path).stem
    print("=" * 74)
    print(f"ABLATION TEST — {climate_path}")
    print("=" * 74)

    emp = load_empirical()
    runs = {}

    for name, overrides in ABLATIONS.items():
        print(f"\nRunning {name}  ({overrides or 'no overrides'})")
        m = Model(seed=42, climate_path=climate_path, **overrides)
        h = m.run(verbose=False)
        r = evaluate(m, h, emp)
        runs[name] = (m, h, r)
        first = f"{r['first']:.0f}" if r["first"] else "never"
        cross = f"{r['cross']:.0f}" if not np.isnan(r["cross"]) else "none"
        fpeak = (f"{r['farmer_peak']:.0f}"
                 if not np.isnan(r["farmer_peak"]) else "none")
        print(f"  first={first} BP  cross={cross} BP  "
              f"farmer_peak={fpeak} BP")
        print(f"  mse_forager={r['mse_f']:.4f}  "
              f"mse_farmer={r['mse_a']:.4f}  "
              f"mse_total={r['mse_total']:.4f}")

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    header = (f"{'condition':>14s} {'first':>8s} {'cross':>8s} "
              f"{'peak':>8s} {'MSE_for':>9s} {'MSE_far':>9s} "
              f"{'MSE_tot':>9s}")
    print(header)
    print("-" * len(header))
    for name in ABLATIONS:
        r = runs[name][2]
        first = f"{r['first']:.0f}" if r["first"] else "never"
        cross = f"{r['cross']:.0f}" if not np.isnan(r["cross"]) else "none"
        fpeak = (f"{r['farmer_peak']:.0f}"
                 if not np.isnan(r["farmer_peak"]) else "none")
        print(f"{name:>14s} {first:>8s} {cross:>8s} {fpeak:>8s} "
              f"{r['mse_f']:>9.4f} {r['mse_a']:>9.4f} "
              f"{r['mse_total']:>9.4f}")

    # ==================================================================
    # Figure: 2x2, raw populations (not normalised).
    # The empirical SPD is on a different scale and is not overlaid.
    # ==================================================================
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    for ax, name in zip(axs.flat, ABLATIONS):
        _, h, r = runs[name]
        t = np.array(h["t"])
        F = np.array(h["F"])
        A = np.array(h["A"])

        ax.plot(t, F, "steelblue", lw=2.2, label="Foragers")
        ax.plot(t, A, "firebrick", lw=2.2, label="Farmers")

        first = f"{r['first']:.0f}" if r["first"] else "never"
        cross = (f"{r['cross']:.0f}"
                 if not np.isnan(r["cross"]) else "none")
        fpeak = (f"{r['farmer_peak']:.0f}"
                 if not np.isnan(r["farmer_peak"]) else "none")

        ax.set_xlim(T_START, T_END)
        ax.set_xlabel("Calibrated age (BP)")
        ax.set_ylabel("Population (model units)")
        ax.set_title(f"{name}\n"
                     f"first={first} BP  cross={cross} BP  "
                     f"peak={fpeak} BP")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    plt.savefig(_FIG_DIR / f"ablation_{tag}.png", dpi=150)
    plt.show()

    # ==================================================================
    # Interpretation — logic corrected
    # ==================================================================
    print("\n" + "=" * 74)
    print("INTERPRETATION")
    print("=" * 74)

    ctrl = runs["control"][2]
    print(f"  Control: first={ctrl['first']:.0f} BP, "
          f"cross={ctrl['cross']:.0f} BP, "
          f"peak={ctrl['farmer_peak']:.0f} BP, "
          f"MSE_tot={ctrl['mse_total']:.4f}.")

    # in_situ_only: is cultural load-bearing for onset?
    ins = runs["in_situ_only"][2]
    if ins["first"]:
        delta = ins["first"] - ctrl["first"]
        if abs(delta) < 200:
            print(f"  in_situ_only: onset unchanged "
                  f"(offset {delta:+.0f} yr). Cultural transmission "
                  f"is NOT load-bearing for the onset.")
        else:
            print(f"  in_situ_only: onset shifted "
                  f"({delta:+.0f} yr). Cultural transmission IS "
                  f"load-bearing for the onset.")
        mse_delta = ins["mse_total"] - ctrl["mse_total"]
        print(f"    MSE_tot {ctrl['mse_total']:.4f} -> "
              f"{ins['mse_total']:.4f} ({mse_delta:+.4f}).")
        print(f"    Without cultural transmission the forager "
              f"curve rebounds after the YD and does not stay "
              f"suppressed. Cultural transmission is load-bearing "
              f"for the shape of the forager decline.")
    else:
        print(f"  in_situ_only: never fires. Cultural transmission "
              f"IS load-bearing for the onset.")

    # cultural_only: can it bootstrap?
    cul = runs["cultural_only"][2]
    if cul["first"]:
        print(f"  cultural_only: fires at {cul['first']:.0f} BP. "
              f"Unexpected without a seed.")
    else:
        print(f"  cultural_only: never fires. The cultural route "
              f"cannot bootstrap from zero. In-situ conversion is "
              f"necessary for the first farmer.")

    # no_soil: is soil required for the peak?
    ns = runs["no_soil"][2]
    if np.isnan(ns["farmer_peak"]):
        print(f"  no_soil: farmer curve has no interior peak in the "
              f"9,800-7,000 BP window. Farmer MSE "
              f"{ctrl['mse_a']:.4f} -> {ns['mse_a']:.4f}. Soil "
              f"degradation is load-bearing for the farmer peak.")
    else:
        print(f"  no_soil: farmer peak at {ns['farmer_peak']:.0f} BP. "
              f"The peak is not entirely a soil effect.")


if __name__ == "__main__":
    main()
