"""
Cereal model vs climate-only null.

Windows:
    Forager diagnostic:  13,000 -> 11,000 BP
        The Younger Dryas shortfall and the immediate post-YD decline.
        This is where the cereal mechanism is supposed to fire and
        where the null model has no corresponding structure.

    Farmer rise:         12,500 -> 9,800 BP
        The full empirical farmer rise.

    Farmer peak+decline:  9,800 -> 7,000 BP
        The empirical farmer peak and decline. The diagnostic window
        for the farmer mechanism: the null model rises monotonically
        through it while the empirical curve declines.

Console output:
    - Empirical peak check (both populations)
    - Forager MSE: full and YD window
    - Farmer MSE: full, rise, peak+decline
    - Direction correlation for both populations in their own window
    - Peak timing for both populations
    - Interpretation summary

Figure: 2 x 4 layout.
    Row 1 (forager):
        empirical | null | cereal | YD-window zoom
    Row 2 (farmer):
        empirical | null | cereal | peak-window zoom

Usage:
    python run_levant_null.py <climate_csv>
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

from levant_null import fit_null_v2
from latest_levant_model import Model, T_START, T_END


# Forager diagnostic window (YD shortfall and immediate decline)
FORAGER_DIR_LO = 13000.0
FORAGER_DIR_HI = 11000.0

# Farmer windows
FARMER_RISE_LO = 12500.0
FARMER_RISE_HI = 9800.0
FARMER_PEAK_LO = 9800.0
FARMER_PEAK_HI = 7000.0

# Younger Dryas shading for the full-curve forager panels
YD_LO = 12900.0
YD_HI = 11700.0


def load_empirical():
    f = np.load(_SPD_DIR / "levant_forager_spd.npz")
    a =	np.load(_SPD_DIR / "levant_farmer_spd.npz")
    return {
        "x": f["age"],
        "forager": f["median"],
        "farmer": a["median"],
        "forager_raw": f["raw"] if "raw" in f.files else f["median"],
        "farmer_raw": a["raw"] if "raw" in a.files else a["median"],
    }


def mse_in_window(t, Y, emp_key, emp, t_lo, t_hi):
    """Normalised MSE of Y against emp[emp_key] within [t_hi, t_lo]."""
    ex = emp["x"]
    mask = (ex >= t_hi) & (ex <= t_lo)
    if not mask.any():
        return float("nan")
    Yn = Y / max(Y.max(), 1e-9)
    emp_y = emp[emp_key]
    emp_n = emp_y / emp_y.max()
    Y_i = np.interp(ex[mask], t[::-1], Yn[::-1])
    return float(np.mean((Y_i - emp_n[mask]) ** 2))


def direction_correlation(t, Y, emp_key, emp, t_lo, t_hi):
    """Level correlation of model and empirical over the window."""
    ex = emp["x"]
    mask = (ex >= t_hi) & (ex <= t_lo)
    if mask.sum() < 3:
        return float("nan")
    ex_w = ex[mask]
    emp_y = emp[emp_key][mask]
    if emp_y.max() <= 0:
        return float("nan")
    emp_n = emp_y / emp_y.max()
    Yn = Y / max(Y.max(), 1e-9)
    Y_i = np.interp(ex_w, t[::-1], Yn[::-1])
    if Y_i.std() < 1e-9 or emp_n.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(Y_i, emp_n)[0, 1])


def peak_timing(age, y, t_lo, t_hi, tol=0.01):
    """
    Centroid of the top region of y within [t_hi, t_lo].
    t_lo = older bound (larger BP), t_hi = younger bound (smaller BP).
    Returns (peak_year, peak_value). Returns (nan, value) if monotonic.
    """
    mask = (age >= t_hi) & (age <= t_lo)
    if mask.sum() < 3:
        return float("nan"), float("nan")
    age_w = age[mask]
    y_w = y[mask]
    if y_w.max() <= 0:
        return float("nan"), 0.0
    y_n = y_w / y_w.max()
    d = np.diff(y_n)
    if (d >= -1e-9).all() or (d <= 1e-9).all():
        return float("nan"), float(y_n.max())
    peak_region = y_n >= (1 - tol) * y_n.max()
    return float(age_w[peak_region].mean()), float(y_n.max())


def s(x):
    return f"{x:.0f}" if not np.isnan(x) else "none"


def main():
    climate_path = sys.argv[1] if len(sys.argv) > 1 else str(_DATA_DIR / "levant_climate.csv")
    tag = Path(climate_path).stem
    print("=" * 76)
    print(f"CEREAL vs NULL — {climate_path}")
    print("=" * 76)

    emp = load_empirical()

    print("\nFitting null model...")
    best, (t_null, F_null, A_null) = fit_null_v2(
        emp, climate_path=climate_path, verbose=False)

    print(f"\nBest null parameters:")
    print(f"  r_F={best['r_F']:.5f}  K_F={best['K_F']:.0f}  "
          f"r_A={best['r_A']:.5f}  K_A={best['K_A']:.0f}  "
          f"A_START={best['A_START']:.0f}")

    m = Model(seed=42, climate_path=climate_path)
    h = m.run()
    t_c = np.array(h["t"])
    F_c = np.array(h["F"])
    A_c = np.array(h["A"])

    # ---------- Empirical peak check ----------
    emp_F_pt_full, emp_F_val = peak_timing(
        emp["x"], emp["forager_raw"], T_START, T_END)
    emp_A_pt_full, emp_A_val = peak_timing(
        emp["x"], emp["farmer_raw"], T_START, T_END)
    emp_A_pt_peak, _ = peak_timing(
        emp["x"], emp["farmer_raw"], FARMER_PEAK_LO, FARMER_PEAK_HI)
    print(f"\nEmpirical peak check:")
    print(f"  forager_raw    full-range peak = {s(emp_F_pt_full)} BP "
          f"(value = {emp_F_val:.3f})")
    print(f"  farmer_raw     full-range peak = {s(emp_A_pt_full)} BP "
          f"(value = {emp_A_val:.3f})")
    print(f"  farmer_raw     peak-window     = {s(emp_A_pt_peak)} BP")

    # ---------- MSE ----------
    # Forager: full + YD window
    null_F_full = mse_in_window(t_null, F_null, "forager_raw", emp,
                                T_START, T_END)
    null_F_yd = mse_in_window(t_null, F_null, "forager_raw", emp,
                              FORAGER_DIR_LO, FORAGER_DIR_HI)
    cereal_F_full = mse_in_window(t_c, F_c, "forager_raw", emp,
                                  T_START, T_END)
    cereal_F_yd = mse_in_window(t_c, F_c, "forager_raw", emp,
                                FORAGER_DIR_LO, FORAGER_DIR_HI)

    # Farmer: full, rise, peak+decline
    null_A_full = mse_in_window(t_null, A_null, "farmer_raw", emp,
                                T_START, T_END)
    null_A_rise = mse_in_window(t_null, A_null, "farmer_raw", emp,
                                FARMER_RISE_LO, FARMER_RISE_HI)
    null_A_peak = mse_in_window(t_null, A_null, "farmer_raw", emp,
                                FARMER_PEAK_LO, FARMER_PEAK_HI)
    cereal_A_full = mse_in_window(t_c, A_c, "farmer_raw", emp,
                                  T_START, T_END)
    cereal_A_rise = mse_in_window(t_c, A_c, "farmer_raw", emp,
                                  FARMER_RISE_LO, FARMER_RISE_HI)
    cereal_A_peak = mse_in_window(t_c, A_c, "farmer_raw", emp,
                                  FARMER_PEAK_LO, FARMER_PEAK_HI)

    # ---------- Direction correlation ----------
    null_F_dir = direction_correlation(t_null, F_null, "forager_raw",
                                       emp, FORAGER_DIR_LO, FORAGER_DIR_HI)
    cereal_F_dir = direction_correlation(t_c, F_c, "forager_raw",
                                         emp, FORAGER_DIR_LO, FORAGER_DIR_HI)
    null_A_dir = direction_correlation(t_null, A_null, "farmer_raw",
                                       emp, FARMER_PEAK_LO, FARMER_PEAK_HI)
    cereal_A_dir = direction_correlation(t_c, A_c, "farmer_raw",
                                         emp, FARMER_PEAK_LO, FARMER_PEAK_HI)

    # ---------- Peak timing ----------
    null_F_pt, _ = peak_timing(t_null, F_null / max(F_null.max(), 1e-9),
                               T_START, T_END)
    cereal_F_pt, _ = peak_timing(t_c, F_c / max(F_c.max(), 1e-9),
                                 T_START, T_END)
    null_A_pt, _ = peak_timing(t_null, A_null / max(A_null.max(), 1e-9),
                               FARMER_PEAK_LO, FARMER_PEAK_HI)
    cereal_A_pt, _ = peak_timing(t_c, A_c / max(A_c.max(), 1e-9),
                                 FARMER_PEAK_LO, FARMER_PEAK_HI)

    # ---------- Print tables ----------
    print(f"\n" + "-" * 76)
    print(f"FORAGER MSE  (vs empirical forager SPD)")
    print(f"  windows: full = {T_END:.0f}-{T_START:.0f} BP,  "
          f"YD = {FORAGER_DIR_HI:.0f}-{FORAGER_DIR_LO:.0f} BP")
    print("-" * 76)
    print(f"{'':14s} {'Full':>12s} {'YD window':>12s}")
    print(f"{'Null':14s} {null_F_full:>12.4f} {null_F_yd:>12.4f}")
    print(f"{'Cereal':14s} {cereal_F_full:>12.4f} {cereal_F_yd:>12.4f}")
    print(f"{'Ratio (c/n)':14s} "
          f"{cereal_F_full/null_F_full:>12.3f} "
          f"{cereal_F_yd/null_F_yd:>12.3f}")

    print(f"\n" + "-" * 76)
    print(f"FARMER MSE  (vs empirical farmer SPD)")
    print(f"  windows: full = {T_END:.0f}-{T_START:.0f} BP,  "
          f"rise = {FARMER_RISE_HI:.0f}-{FARMER_RISE_LO:.0f} BP,  "
          f"peak+dec = {FARMER_PEAK_HI:.0f}-{FARMER_PEAK_LO:.0f} BP")
    print("-" * 76)
    print(f"{'':14s} {'Full':>12s} {'Rise':>12s} {'Peak+dec':>12s}")
    print(f"{'Null':14s} {null_A_full:>12.4f} {null_A_rise:>12.4f} "
          f"{null_A_peak:>12.4f}")
    print(f"{'Cereal':14s} {cereal_A_full:>12.4f} {cereal_A_rise:>12.4f} "
          f"{cereal_A_peak:>12.4f}")
    print(f"{'Ratio (c/n)':14s} "
          f"{cereal_A_full/null_A_full:>12.3f} "
          f"{cereal_A_rise/null_A_rise:>12.3f} "
          f"{cereal_A_peak/null_A_peak:>12.3f}")

    print(f"\n" + "-" * 76)
    print(f"DIRECTION CORRELATION  (peak/decline or diagnostic window)")
    print("-" * 76)
    print(f"  Forager ({FORAGER_DIR_HI:.0f}-{FORAGER_DIR_LO:.0f} BP):  "
          f"null = {null_F_dir:+.3f}   cereal = {cereal_F_dir:+.3f}")
    print(f"  Farmer  ({FARMER_PEAK_HI:.0f}-{FARMER_PEAK_LO:.0f} BP):  "
          f"null = {null_A_dir:+.3f}   cereal = {cereal_A_dir:+.3f}")

    print(f"\n" + "-" * 76)
    print(f"PEAK TIMING")
    print("-" * 76)
    print(f"{'':14s} {'Forager':>14s} {'Farmer':>14s}")
    print(f"{'Empirical':14s} {s(emp_F_pt_full):>14s} {s(emp_A_pt_peak):>14s}")
    print(f"{'Null':14s} {s(null_F_pt):>14s} {s(null_A_pt):>14s}")
    print(f"{'Cereal':14s} {s(cereal_F_pt):>14s} {s(cereal_A_pt):>14s}")

    print(f"\n" + "=" * 76)
    print(f"INTERPRETATION")
    print("=" * 76)
    for_ratio = cereal_F_full / null_F_full if null_F_full > 0 else np.inf
    if for_ratio < 1:
        print(f"  Forager: cereal full MSE is {1/for_ratio:.1f}x better "
              f"than null.")
    print(f"  Forager YD window: null dir = {null_F_dir:+.2f}, "
          f"cereal dir = {cereal_F_dir:+.2f}.")
    if null_A_dir < -0.3 and cereal_A_dir > 0.3:
        print(f"  Farmer peak window: null {null_A_dir:+.2f} "
              f"(wrong direction), cereal {cereal_A_dir:+.2f} "
              f"(right direction).")
    elif cereal_A_dir > null_A_dir + 0.3:
        print(f"  Farmer peak window: cereal is closer to right "
              f"direction (null {null_A_dir:+.2f}, "
              f"cereal {cereal_A_dir:+.2f}).")
    if not np.isnan(emp_A_pt_peak) and not np.isnan(cereal_A_pt):
        print(f"  Farmer peak offset: "
              f"{abs(emp_A_pt_peak - cereal_A_pt):.0f} years.")

    # ==================================================================
    # Figure: 2 x 4
    # ==================================================================
    fig, axs = plt.subplots(2, 4, figsize=(24, 11))

    ex = emp["x"]
    efm = emp["forager_raw"] / emp["forager_raw"].max()
    eam = emp["farmer_raw"] / emp["farmer_raw"].max()
    Fn_null = F_null / max(F_null.max(), 1e-9)
    An_null = A_null / max(A_null.max(), 1e-9)
    Fn_c = F_c / max(F_c.max(), 1e-9)
    An_c = A_c / max(A_c.max(), 1e-9)

    # ---------- ROW 1: FORAGER ----------

    ax = axs[0, 0]
    ax.plot(ex, efm, "steelblue", lw=2.0, label="Foragers (empirical)")
    ax.axvspan(YD_HI, YD_LO, color="grey", alpha=0.15,
               label="Younger Dryas")
    ax.set_xlim(T_START, T_END)
    ax.set_xlabel("Calibrated age (BP)")
    ax.set_ylabel("Normalised")
    ax.set_title("Empirical forager SPD")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    ax = axs[0, 1]
    ax.plot(ex, efm, "steelblue", lw=1.4, ls="--", alpha=0.6,
            label="Foragers (emp.)")
    ax.plot(t_null, Fn_null, "steelblue", lw=2.2,
            label="Foragers (null)")
    ax.set_xlim(T_START, T_END)
    ax.set_xlabel("Calibrated age (BP)")
    ax.set_ylabel("Normalised")
    ax.set_title(f"Null forager\nfull MSE = {null_F_full:.4f}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    ax = axs[0, 2]
    ax.plot(ex, efm, "steelblue", lw=1.4, ls="--", alpha=0.6,
            label="Foragers (emp.)")
    ax.plot(t_c, Fn_c, "steelblue", lw=2.2,
            label="Foragers (cereal)")
    ax.set_xlim(T_START, T_END)
    ax.set_xlabel("Calibrated age (BP)")
    ax.set_ylabel("Normalised")
    ax.set_title(f"Cereal forager\nfull MSE = {cereal_F_full:.4f}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    ax = axs[0, 3]
    mask_f = (ex >= FORAGER_DIR_HI) & (ex <= FORAGER_DIR_LO)
    ax.plot(ex[mask_f], efm[mask_f], "steelblue", lw=2.5,
            label="Foragers (emp.)")
    Fn_null_i = np.interp(ex, t_null[::-1], Fn_null[::-1])
    Fn_c_i = np.interp(ex, t_c[::-1], Fn_c[::-1])
    ax.plot(ex[mask_f], Fn_null_i[mask_f], "steelblue", lw=1.8,
            ls="--", alpha=0.8, label="Foragers (null)")
    ax.plot(ex[mask_f], Fn_c_i[mask_f], "steelblue", lw=1.8,
            ls=":", alpha=1.0, label="Foragers (cereal)")
    ax.set_xlim(FORAGER_DIR_LO, FORAGER_DIR_HI)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Calibrated age (BP)")
    ax.set_ylabel("Normalised")
    ax.set_title(f"Forager YD window {FORAGER_DIR_HI:.0f}-{FORAGER_DIR_LO:.0f} BP\n"
                 f"dir: null {null_F_dir:+.2f}   cereal {cereal_F_dir:+.2f}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")

    # ---------- ROW 2: FARMER ----------

    ax = axs[1, 0]
    ax.plot(ex, eam, "firebrick", lw=2.0, label="Farmers (empirical)")
    if not np.isnan(emp_A_pt_peak):
        ax.axvline(emp_A_pt_peak, color="black", ls=":", alpha=0.5,
                   label=f"Peak {emp_A_pt_peak:.0f} BP")
    ax.axvspan(FARMER_RISE_HI, FARMER_RISE_LO, color="steelblue",
               alpha=0.10, label="Rise window")
    ax.axvspan(FARMER_PEAK_HI, FARMER_PEAK_LO, color="firebrick",
               alpha=0.10, label="Peak+dec window")
    ax.set_xlim(T_START, T_END)
    ax.set_xlabel("Calibrated age (BP)")
    ax.set_ylabel("Normalised")
    ax.set_title("Empirical farmer SPD")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")

    ax = axs[1, 1]
    ax.plot(ex, eam, "firebrick", lw=1.4, ls="--", alpha=0.6,
            label="Farmers (emp.)")
    ax.plot(t_null, An_null, "firebrick", lw=2.2,
            label="Farmers (null)")
    ax.axvspan(FARMER_RISE_HI, FARMER_RISE_LO, color="steelblue",
               alpha=0.10)
    ax.axvspan(FARMER_PEAK_HI, FARMER_PEAK_LO, color="firebrick",
               alpha=0.10)
    ax.set_xlim(T_START, T_END)
    ax.set_xlabel("Calibrated age (BP)")
    ax.set_ylabel("Normalised")
    ax.set_title(f"Null farmer\nfull MSE = {null_A_full:.4f}   "
                 f"peak dir = {null_A_dir:+.2f}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    ax = axs[1, 2]
    ax.plot(ex, eam, "firebrick", lw=1.4, ls="--", alpha=0.6,
            label="Farmers (emp.)")
    ax.plot(t_c, An_c, "firebrick", lw=2.2,
            label="Farmers (cereal)")
    ax.axvspan(FARMER_RISE_HI, FARMER_RISE_LO, color="steelblue",
               alpha=0.10)
    ax.axvspan(FARMER_PEAK_HI, FARMER_PEAK_LO, color="firebrick",
               alpha=0.10)
    ax.set_xlim(T_START, T_END)
    ax.set_xlabel("Calibrated age (BP)")
    ax.set_ylabel("Normalised")
    ax.set_title(f"Cereal farmer\nfull MSE = {cereal_A_full:.4f}   "
                 f"peak dir = {cereal_A_dir:+.2f}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    ax = axs[1, 3]
    mask_a = (ex >= FARMER_PEAK_HI) & (ex <= FARMER_PEAK_LO)
    ax.plot(ex[mask_a], eam[mask_a], "firebrick", lw=2.5,
            label="Farmers (emp.)")
    An_null_i = np.interp(ex, t_null[::-1], An_null[::-1])
    An_c_i = np.interp(ex, t_c[::-1], An_c[::-1])
    ax.plot(ex[mask_a], An_null_i[mask_a], "firebrick", lw=1.8,
            ls="--", alpha=0.8, label="Farmers (null)")
    ax.plot(ex[mask_a], An_c_i[mask_a], "firebrick", lw=1.8,
            ls=":", alpha=1.0, label="Farmers (cereal)")
    if not np.isnan(cereal_A_pt):
        ax.axvline(cereal_A_pt, color="black", ls=":", alpha=0.4)
    ax.set_xlim(FARMER_PEAK_LO, FARMER_PEAK_HI)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Calibrated age (BP)")
    ax.set_ylabel("Normalised")
    ax.set_title(f"Farmer peak window "
                 f"{FARMER_PEAK_HI:.0f}-{FARMER_PEAK_LO:.0f} BP\n"
                 f"dir: null {null_A_dir:+.2f}   cereal {cereal_A_dir:+.2f}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    plt.savefig(_FIG_DIR / f"figures/null_both_{tag}.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()