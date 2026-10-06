"""
Gate-removal test: does the mechanism predict the onset of cultivation
on its own, or does the gate do the work?

Runs the model with DOMESTICATION_START_BP = T_START (23,000 BP) so
farming is viable from the start. Reports the first conversion age.

If it fires before 12,500 BP, the mechanism does not self-constrain;
the gate is doing the timing work.

Usage:
    python run_no_gate.py <climate_csv>
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

import model_grid
from model_grid import Model, T_START, T_END, DOMESTICATION_START_BP


def load_empirical():
    f = np.load(_SPD_DIR / "levant_forager_spd.npz")
    a = np.load(_SPD_DIR / "levant_farmer_spd.npz")
    return {
        "x": f["age"],
        "farmer_raw": a["raw"] if "raw" in a.files else a["median"],
        "forager_raw": f["raw"] if "raw" in f.files else f["median"],
    }


def run_ungated(climate_path):
    """
    Run the model with the domestication gate removed.

    Tries two override paths in order:
      1. Constructor argument DOMESTICATION_START_BP=T_START
         (works if the model supports the kwarg).
      2. Module-level constant mutation of
         model_grid.DOMESTICATION_START_BP
         (works if the model reads the constant at call time).
    """
    try:
        m = Model(seed=42, climate_path=climate_path,
                  DOMESTICATION_START_BP=T_START)
        h = m.run()
        return m, h
    except TypeError:
        original = model_grid.DOMESTICATION_START_BP
        model_grid.DOMESTICATION_START_BP = T_START
        try:
            m = Model(seed=42, climate_path=climate_path)
            h = m.run()
        finally:
            model_grid.DOMESTICATION_START_BP = original
        return m, h


def main():
    climate_path = sys.argv[1] if len(sys.argv) > 1 else str(_DATA_DIR / "levant_climate.csv")
    tag = Path(climate_path).stem
    print("=" * 70)
    print(f"GATE-REMOVAL TEST — {climate_path}")
    print("=" * 70)

    # Gated run (baseline)
    m_gated = Model(seed=42, climate_path=climate_path)
    h_gated = m_gated.run()
    t_g = np.array(h_gated["t"])
    A_g = np.array(h_gated["A"])

    # Ungated run
    m_ungated, h_ungated = run_ungated(climate_path)
    t_u = np.array(h_ungated["t"])
    A_u = np.array(h_ungated["A"])

    gated_onset = m_gated.first_conversion_bp
    ungated_onset = m_ungated.first_conversion_bp

    if gated_onset:
        print(f"\nGated first conversion:   {gated_onset:.0f} BP")
    else:
        print(f"\nGated first conversion:   never fired")
    if ungated_onset:
        print(f"Ungated first conversion: {ungated_onset:.0f} BP")
    else:
        print(f"Ungated first conversion: never fired")
    if gated_onset and ungated_onset:
        delta = gated_onset - ungated_onset
        print(f"Difference: {delta:.0f} years")
        if abs(delta) < 1e-6:
            print()
            print("WARNING: gated and ungated onsets are identical.")
            print("The override did not take. The model is still reading")
            print("the module-level DOMESTICATION_START_BP constant and")
            print("does not accept a constructor kwarg for it.")
            print("Patch the model to accept DOMESTICATION_START_BP as a")
            print("constructor kwarg (add it to DEFAULTS and use")
            print("self.DOMESTICATION_START_BP inside step()).")

    A_g_n = A_g / max(A_g.max(), 1e-9)
    A_u_n = A_u / max(A_u.max(), 1e-9)
    emp = load_empirical()
    eam_n = emp["farmer_raw"] / emp["farmer_raw"].max()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(t_g, A_g_n, "firebrick", lw=2.2,
            label=f"Gated ({gated_onset:.0f} BP)" if gated_onset
            else "Gated")
    ax.plot(t_u, A_u_n, "darkorange", lw=2.2, ls="--",
            label=f"Ungated ({ungated_onset:.0f} BP)" if ungated_onset
            else "Ungated")
    ax.plot(emp["x"], eam_n, "purple", lw=1.4, ls=":", alpha=0.6,
            label="Empirical farmer SPD")
    ax.axvline(DOMESTICATION_START_BP, color="grey", ls=":", alpha=0.6,
               label=f"Archaeological gate "
                     f"({DOMESTICATION_START_BP:.0f} BP)")
    ax.set_xlim(T_START, T_END)
    ax.set_xlabel("Calibrated age (BP)")
    ax.set_ylabel("Normalised farmer population")
    ax.set_title(f"Gate-removal test — {tag}")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    plt.savefig(_FIG_DIR / f"no_gate_{tag}.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
