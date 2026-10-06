"""
XRONOS forager-farmer SPDs for the Levant and northern Levant,
with grid-cell research-intensity correction.

Window: 23,000-7,300 BP.
Cyprus excluded.
Bounding box: lat 28-40 N, lon 30-42 E.

Four-panel figure:
    Panel 1: raw SPDs, absolute amplitudes, shared y-axis
    Panel 2: grid-cell corrected SPDs, overlaid on raw (dashed)
    Panel 3: farmer share of the dated record
    Panel 4: thinned date counts by 1000-year bin (histogram)

Usage:
    python xronos_SPD_levant.py
"""

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


# =====================================================================
# Configuration
# =====================================================================

XRONOS_FILE = "data_2025-02-11.csv"

LAT_MIN, LAT_MAX = 28.0, 40.0
LON_MIN, LON_MAX = 30.0, 42.0

CYPRUS_LAT_MIN, CYPRUS_LAT_MAX = 34.5, 35.7
CYPRUS_LON_MIN, CYPRUS_LON_MAX = 32.2, 34.6

CALC_MIN_BP = 7300
CALC_MAX_BP = 23000
DISPLAY_MIN_BP = 7300
DISPLAY_MAX_BP = 23000

BIN_SIZE = 200
BANDWIDTH = 60.0
N_BOOTSTRAP = 500
BOOTSTRAP_SEED = 42

FARMING_HORIZON_BP = 11650

CELL_SIZE = 1.0
MIN_DATES_PER_CELL = 5

TABLE_AGES = [22000, 20000, 18000, 16000, 15000, 14000,
              13000, 12500, 12000, 11500, 11000, 10500,
              10000, 9500, 9000, 8500, 8000, 7500]


# =====================================================================
# Classification patterns
# =====================================================================

FORAGER_PATTERNS = [
    "kebaran", "geometric kebaran",
    "nizzanan", "nebekian", "masraqan", "masragan",
    "mushabian", "ramonian",
    "natufian", "natuf",
    "harifian", "harif",
    "epipalaeolithic", "epipaleolithic", "epi-palaeolithic",
    "late palaeolithic", "late paleolithic",
    "zarzian",
]

FARMER_PATTERNS = [
    "pre-pottery neolithic", "pre-pottery", "prepottery",
    "ppna", "ppnb", "ppnc",
    "khiamian", "sultanian", "mureybetian", "aswadian",
    "neolithic",
    "pottery neolithic", "yarmukian",
    "jericho ix", "wadi rabah", "lodian",
    "halaf", "samarra", "hassuna",
    "chalcolithic", "copper age",
]

PRE_HOLOCENE_PATTERNS = [
    "upper palaeolithic", "upper paleolithic",
    "late upper palaeolithic", "late upper paleolithic",
    "ahmarian", "late ahmarian",
    "aurignacian", "gravettian", "solutrean", "magdalenian",
    "mousterian", "middle palaeolithic", "middle paleolithic",
    "lower palaeolithic", "lower paleolithic",
]


def matches_any(value, patterns):
    if not isinstance(value, str):
        return False
    s = value.lower()
    return any(p in s for p in patterns)


def in_cyprus(lat, lon):
    return (CYPRUS_LAT_MIN <= lat <= CYPRUS_LAT_MAX and
            CYPRUS_LON_MIN <= lon <= CYPRUS_LON_MAX)


# =====================================================================
# Calibration
# =====================================================================

_CALIB_LOOKUP = None
CURVE_URL = ("https://raw.githubusercontent.com/Maarten14C/rintcal/"
             "master/inst/extdata/3Col_intcal20.14C")


def build_calib_lookup():
    global _CALIB_LOOKUP
    if _CALIB_LOOKUP is not None:
        return _CALIB_LOOKUP

    print("Downloading IntCal20...")
    with urllib.request.urlopen(CURVE_URL, timeout=30) as resp:
        text = resp.read().decode("utf-8", errors="ignore")

    tokens = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        for part in line.replace(",", " ").split():
            try:
                tokens.append(float(part))
            except ValueError:
                continue

    n = len(tokens) // 3
    arr = np.array(tokens[:n * 3]).reshape(n, 3)
    cal_bp, c14, _ = arr[:, 0], arr[:, 1], arr[:, 2]

    lookup = {}
    for target in range(0, 55001):
        lo, hi = target - 30.0, target + 30.0
        mask = (c14 >= lo) & (c14 <= hi)
        if not mask.any():
            idx = int(np.argmin(np.abs(c14 - target)))
            lookup[target] = (float(cal_bp[idx]), 30.0)
        else:
            cr = cal_bp[mask]
            mid = 0.5 * (float(cr.min()) + float(cr.max()))
            half = max(0.5 * (float(cr.max()) - float(cr.min())), 30.0)
            lookup[target] = (mid, half)
    _CALIB_LOOKUP = lookup
    print(f"  Built lookup table ({len(lookup)} entries).")
    return lookup


def calibrate_array(ages, errs, lookup):
    cal_ages = np.full(len(ages), np.nan)
    cal_sds = np.full(len(ages), np.nan)
    for i in range(len(ages)):
        a, s = ages[i], errs[i]
        if np.isnan(a) or np.isnan(s):
            continue
        ti = max(0, min(55000, int(round(a))))
        mid, half_nom = lookup[ti]
        cal_ages[i] = mid
        cal_sds[i] = max(half_nom * (s / 30.0), s)
    return cal_ages, cal_sds


# =====================================================================
# Thinning and SPD
# =====================================================================

def bin_and_thin(df, bin_size=BIN_SIZE):
    if len(df) == 0:
        return df.copy()
    df = df.copy().reset_index(drop=True)
    df["_bin"] = (df["CalAge"] // bin_size).astype(int)
    site = df["site"].astype("object").copy()
    mask = site.isna()
    if mask.any():
        n = int(mask.sum())
        site.loc[mask] = [f"__m{i}__" for i in range(n)]
    df["_site_key"] = site.astype(str)
    shuffled = df.sample(frac=1.0, random_state=0)
    thinned = shuffled.drop_duplicates(
        subset=["_bin", "_site_key"], keep="first")
    return (thinned.drop(columns=["_bin", "_site_key"])
                   .reset_index(drop=True))


def build_spd(ages, sigmas, x_range, bandwidth=BANDWIDTH):
    if len(ages) == 0:
        return np.zeros_like(x_range, dtype=float)
    bin_width = float(x_range[1] - x_range[0])
    edges = np.concatenate([
        x_range - bin_width / 2.0,
        [x_range[-1] + bin_width / 2.0],
    ])
    hist, _ = np.histogram(ages, bins=edges)
    sigmas_eff = np.maximum(np.asarray(sigmas, dtype=float), bandwidth)
    sigma_bins = max(float(np.mean(sigmas_eff)) / bin_width, 1.0)
    return gaussian_filter1d(hist.astype(float), sigma=sigma_bins,
                              mode="nearest")


def spd_with_ci(ages, sigmas, x_range,
                n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    if len(ages) < 5:
        return None
    ages = np.asarray(ages, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    raw = build_spd(ages, sigmas, x_range)
    point_max = raw.max() if raw.max() > 0 else 1.0

    rng = np.random.default_rng(seed)
    n = len(ages)
    boot = np.zeros((n_bootstrap, len(x_range)), dtype=float)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot[b, :] = build_spd(ages[idx], sigmas[idx], x_range)
    boot /= point_max

    med = np.median(boot, axis=0)
    lo = np.percentile(boot, 5, axis=0)
    hi = np.percentile(boot, 95, axis=0)
    med_max = med.max() if med.max() > 0 else 1.0
    return (x_range, med / med_max, lo / med_max, hi / med_max, raw)


# =====================================================================
# Grid-cell research-intensity correction
# =====================================================================

def grid_cell_spd(df, x_range, cell_size=CELL_SIZE,
                  min_dates=MIN_DATES_PER_CELL):
    if len(df) == 0:
        return x_range, np.zeros_like(x_range, dtype=float), 0

    df = df.copy()
    df["_cell_lat"] = (df["lat"] // cell_size) * cell_size
    df["_cell_lon"] = (df["lng"] // cell_size) * cell_size
    df["_cell"] = (df["_cell_lat"].astype(str) + "_" +
                    df["_cell_lon"].astype(str))

    curves = []
    for cell, sub in df.groupby("_cell"):
        if len(sub) < min_dates:
            continue
        spd = build_spd(sub["CalAge"].values,
                        sub["CalSD"].values, x_range)
        if spd.max() > 0:
            curves.append(spd / spd.max())

    if not curves:
        return x_range, np.zeros_like(x_range, dtype=float), 0

    curves = np.array(curves)
    mean = curves.mean(axis=0)
    if mean.max() > 0:
        mean = mean / mean.max()
    return x_range, mean, len(curves)


# =====================================================================
# Tables
# =====================================================================

def raw_density_table(x, forager_raw, farmer_raw):
    print()
    print("=" * 82)
    print("RAW DATED-RECORD DENSITY AT KEY AGES")
    print("(un-normalised SPD amplitude, in units of dated sites "
          "per 10-yr bin, smoothed)")
    print("=" * 82)
    print(f"{'Age (BP)':>10s} {'Foragers':>12s} {'Farmers':>12s} "
          f"{'Total':>12s} {'Farmer share':>16s} {'A/F':>10s}")
    print("-" * 82)

    rows = []
    for age in TABLE_AGES:
        idx = int(np.argmin(np.abs(x - age)))
        f_val = float(forager_raw[idx])
        a_val = float(farmer_raw[idx])
        total = f_val + a_val
        share = a_val / total if total > 1e-9 else np.nan
        ratio = a_val / f_val if f_val > 1e-9 else np.inf
        share_str = (f"{share:>16.3f}" if not np.isnan(share)
                     else "n/a".rjust(16))
        ratio_str = (f"{ratio:>10.2f}" if np.isfinite(ratio)
                     else "inf".rjust(10))
        print(f"{age:>10d} {f_val:>12.3f} {a_val:>12.3f} "
              f"{total:>12.3f} {share_str} {ratio_str}")
        rows.append({
            "age_bp": age,
            "forager_raw": f_val,
            "farmer_raw": a_val,
            "total_raw": total,
            "farmer_share": share,
            "a_over_f": ratio,
        })
    return pd.DataFrame(rows)


def find_crossover(x, forager_raw, farmer_raw):
    """
    Return the oldest age at which farmer density exceeds
    forager density. x is ascending from young (7300) to old
    (23000), so we walk backward from the oldest index.
    """
    for i in range(len(x) - 1, -1, -1):
        if farmer_raw[i] > forager_raw[i] and forager_raw[i] > 1e-6:
            return x[i]
    return None


# =====================================================================
# Main
# =====================================================================

def main():
    p = Path(XRONOS_FILE)
    if not p.exists():
        print(f"File not found: {p}")
        print("Download from: https://zenodo.org/records/14850157")
        return

    print(f"Loading {p}...")
    df = pd.read_csv(p, low_memory=False)
    print(f"Rows: {len(df)}\n")

    df = df.dropna(subset=["lat", "lng", "bp", "std"])
    df = df[(df["lat"] >= LAT_MIN) & (df["lat"] <= LAT_MAX) &
            (df["lng"] >= LON_MIN) & (df["lng"] <= LON_MAX)]
    print(f"After bbox: {len(df)}")

    n_before = len(df)
    df = df[~df.apply(
        lambda r: in_cyprus(r["lat"], r["lng"]), axis=1)]
    print(f"After removing Cyprus: {len(df)} "
          f"(removed {n_before - len(df)})")

    lookup = build_calib_lookup()
    ca, cs = calibrate_array(df["bp"].astype(float).values,
                              df["std"].astype(float).values,
                              lookup)
    df["CalAge"] = ca
    df["CalSD"] = cs
    df = df.dropna(subset=["CalAge", "CalSD"])
    df = df[(df["CalAge"] >= CALC_MIN_BP) &
            (df["CalAge"] <= CALC_MAX_BP)]
    print(f"In calc range {CALC_MIN_BP}-{CALC_MAX_BP} cal BP: "
          f"{len(df)}")

    if len(df) < 20:
        print("Too few dates.")
        return

    # ---- Classify ----
    combined = (
        df["periods"].fillna("").astype(str) + " " +
        df["typochronological_units"].fillna("").astype(str)
    ).str.lower()

    df["_is_forager"] = combined.apply(
        lambda s: matches_any(s, FORAGER_PATTERNS))
    df["_is_farmer"] = combined.apply(
        lambda s: matches_any(s, FARMER_PATTERNS))
    df["_is_pre_holocene"] = combined.apply(
        lambda s: matches_any(s, PRE_HOLOCENE_PATTERNS))

    both = df["_is_forager"] & df["_is_farmer"]
    n_both = int(both.sum())
    df.loc[both, "_is_farmer"] = False

    df["_is_forager"] = df["_is_forager"] & ~df["_is_pre_holocene"]

    too_old = df["_is_farmer"] & (df["CalAge"] > FARMING_HORIZON_BP)
    n_dropped = int(too_old.sum())
    df.loc[too_old, "_is_farmer"] = False

    n_forager = int(df["_is_forager"].sum())
    n_farmer = int(df["_is_farmer"].sum())
    n_unclass = len(df) - n_forager - n_farmer

    print()
    print("=" * 72)
    print("CLASSIFICATION")
    print("=" * 72)
    print(f"Foragers:      {n_forager}")
    print(f"Farmers:       {n_farmer}")
    print(f"Unclassified:  {n_unclass} "
          f"({100.0 * n_unclass / max(len(df), 1):.1f}%)")
    print(f"  ({n_both} rows matched both; resolved to forager)")
    print(f"  ({n_dropped} farmer rows dropped as pre-PPNA)")

    if n_forager == 0 or n_farmer == 0:
        print("\nOne class is empty.")
        return

    # ---- Geographic breakdown ----
    print()
    print("=" * 72)
    print("GEOGRAPHIC BREAKDOWN")
    print("=" * 72)
    if "country" in df.columns:
        for label, sub in [("Foragers", df[df["_is_forager"]]),
                            ("Farmers", df[df["_is_farmer"]])]:
            if len(sub) == 0:
                continue
            counts = sub["country"].value_counts().head(10)
            print(f"{label} top countries:")
            for k, v in counts.items():
                print(f"  {str(k):30s} {v:5d}")

    # ---- Subsets ----
    foragers = df[df["_is_forager"]].copy()
    farmers = df[df["_is_farmer"]].copy()

    # ---- Thin ----
    print()
    print("=" * 72)
    print("THINNING")
    print("=" * 72)
    f_thin = bin_and_thin(foragers)
    a_thin = bin_and_thin(farmers)
    print(f"Foragers:   {len(foragers)} -> {len(f_thin)}")
    print(f"Farmers:    {len(farmers)} -> {len(a_thin)}")

    # ---- SPDs ----
    x = np.arange(CALC_MIN_BP, CALC_MAX_BP + 1, 10)

    print()
    print("=" * 72)
    print("BUILDING SPDs")
    print("=" * 72)

    f_result = spd_with_ci(f_thin["CalAge"].values,
                            f_thin["CalSD"].values, x)
    a_result = spd_with_ci(a_thin["CalAge"].values,
                            a_thin["CalSD"].values, x)

    if f_result is not None:
        f_raw = f_result[4]
        peak = x[np.argmax(f_raw)]
        print(f"Forager SPD:   n={len(f_thin)}, "
              f"peak={peak} cal BP, "
              f"peak raw amplitude={f_raw.max():.2f}")
    if a_result is not None:
        a_raw = a_result[4]
        peak = x[np.argmax(a_raw)]
        print(f"Farmer SPD:    n={len(a_thin)}, "
              f"peak={peak} cal BP, "
              f"peak raw amplitude={a_raw.max():.2f}")

    cross_raw = None
    if f_result is not None and a_result is not None:
        f_raw = f_result[4]
        a_raw = a_result[4]
        cross_raw = find_crossover(x, f_raw, a_raw)
        if cross_raw is not None:
            print(f"Raw crossover: {cross_raw:.0f} cal BP")

    if f_result is not None and a_result is not None:
        raw_df = raw_density_table(x, f_result[4], a_result[4])
        raw_df.to_csv("levant_raw_density.csv", index=False)

    # ---- Grid-cell corrected SPDs ----
    print()
    print("=" * 72)
    print(f"GRID-CELL CORRECTION (cell size = {CELL_SIZE} deg, "
          f"min dates/cell = {MIN_DATES_PER_CELL})")
    print("=" * 72)

    _, f_grid, n_cells_f = grid_cell_spd(f_thin, x)
    _, a_grid, n_cells_a = grid_cell_spd(a_thin, x)
    print(f"Forager: {n_cells_f} cells used")
    print(f"Farmer:  {n_cells_a} cells used")
    if n_cells_f > 0:
        peak = x[np.argmax(f_grid)]
        print(f"Grid-corrected forager peak: {peak} cal BP")
    if n_cells_a > 0:
        peak = x[np.argmax(a_grid)]
        print(f"Grid-corrected farmer peak:  {peak} cal BP")

    cross_grid = None
    if n_cells_f > 0 and n_cells_a > 0:
        cross_grid = find_crossover(x, f_grid, a_grid)
        if cross_grid is not None:
            print(f"Grid-corrected crossover:    {cross_grid:.0f} cal BP")

    # ---- Dates per 1000-year bin ----
    print()
    print("=" * 72)
    print("THINNED DATES PER 1000-YEAR BIN")
    print("=" * 72)
    bins_1000 = np.arange(CALC_MIN_BP, CALC_MAX_BP + 1000, 1000)
    for label, sub in [("Foragers", f_thin),
                       ("Farmers", a_thin)]:
        counts, _ = np.histogram(sub["CalAge"], bins=bins_1000)
        centers = bins_1000[:-1] + 500
        line = " ".join(
            f"{int(c)}:{int(n)}" for c, n in zip(centers, counts))
        print(f"{label:12s} {line}")

    # ================================================================
    # Plot: four panels
    # ================================================================
    fig, axs = plt.subplots(2, 2, figsize=(16, 11))

    # Panel 1 (top-left): raw SPDs, absolute amplitudes
    ax = axs[0, 0]
    if f_result is not None:
        f_raw = f_result[4]
        peak = x[np.argmax(f_raw)]
        ax.plot(x, f_raw, color="steelblue", linewidth=2.2,
                label=f"Foragers (peak {peak} BP, "
                      f"n={len(f_thin)})")
    if a_result is not None:
        a_raw = a_result[4]
        peak = x[np.argmax(a_raw)]
        ax.plot(x, a_raw, color="firebrick", linewidth=2.2,
                label=f"Farmers (peak {peak} BP, "
                      f"n={len(a_thin)})")
    ax.axvline(FARMING_HORIZON_BP, color="grey", linestyle=":",
               alpha=0.6,
               label=f"Farming horizon {FARMING_HORIZON_BP} BP")
    ax.set_xlabel("Calibrated age (BP)")
    ax.set_ylabel("Raw SPD amplitude (dated sites per 10-yr bin)")
    ax.set_title("Raw dated-record density (shared y-axis)")
    ax.set_xlim(DISPLAY_MAX_BP, DISPLAY_MIN_BP)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    # Panel 2 (top-right): grid-cell corrected SPDs overlaid on raw
    ax = axs[0, 1]
    # First plot raw as dashed reference
    if f_result is not None:
        f_raw = f_result[4]
        f_raw_norm = f_raw / f_raw.max() if f_raw.max() > 0 else f_raw
        ax.plot(x, f_raw_norm, color="steelblue", linewidth=1.4,
                linestyle="--", alpha=0.6,
                label="Foragers (raw)")
    if a_result is not None:
        a_raw = a_result[4]
        a_raw_norm = a_raw / a_raw.max() if a_raw.max() > 0 else a_raw
        ax.plot(x, a_raw_norm, color="firebrick", linewidth=1.4,
                linestyle="--", alpha=0.6,
                label="Farmers (raw)")
    # Then plot corrected as solid
    if n_cells_f > 0:
        peak = x[np.argmax(f_grid)]
        ax.plot(x, f_grid, color="steelblue", linewidth=2.4,
                label=f"Foragers (corrected, peak {peak} BP, "
                      f"{n_cells_f} cells)")
    if n_cells_a > 0:
        peak = x[np.argmax(a_grid)]
        ax.plot(x, a_grid, color="firebrick", linewidth=2.4,
                label=f"Farmers (corrected, peak {peak} BP, "
                      f"{n_cells_a} cells)")
    ax.axvline(FARMING_HORIZON_BP, color="grey", linestyle=":",
               alpha=0.6)
    ax.set_xlabel("Calibrated age (BP)")
    ax.set_ylabel("Normalised SPD")
    ax.set_title(f"Grid-cell correction — raw (dashed) vs "
                 f"corrected (solid), {CELL_SIZE}° cells, "
                 f"min {MIN_DATES_PER_CELL}")
    ax.set_xlim(DISPLAY_MAX_BP, DISPLAY_MIN_BP)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

    # Panel 3 (bottom-left): farmer share of the dated record
    ax = axs[1, 0]
    if f_result is not None and a_result is not None:
        f_raw = f_result[4]
        a_raw = a_result[4]
        denom = f_raw + a_raw
        share = np.where(denom > 1e-6, a_raw / denom, np.nan)
        ax.plot(x, share, color="purple", linewidth=2.2,
                label="Farmer share of dated record")
        ax.axhline(0.5, color="grey", linestyle=":", alpha=0.6,
                   label="50% (equal)")
        if cross_raw is not None:
            ax.axvline(cross_raw, color="grey", linestyle=":",
                       alpha=0.6,
                       label=f"Crossover {cross_raw:.0f} BP")
    ax.axvline(FARMING_HORIZON_BP, color="grey", linestyle=":",
               alpha=0.6)
    ax.set_xlabel("Calibrated age (BP)")
    ax.set_ylabel("Farmer share of dated record")
    ax.set_title("Relative population proportions")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(DISPLAY_MAX_BP, DISPLAY_MIN_BP)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    # Panel 4 (bottom-right): thinned date counts per 1000-year bin
    ax = axs[1, 1]
    if len(f_thin) > 0:
        ax.hist(f_thin["CalAge"], bins=bins_1000,
                color="steelblue", alpha=0.7,
                label=f"Foragers (n={len(f_thin)})")
    if len(a_thin) > 0:
        ax.hist(a_thin["CalAge"], bins=bins_1000,
                color="firebrick", alpha=0.6,
                label=f"Farmers (n={len(a_thin)})")
    ax.axvline(FARMING_HORIZON_BP, color="grey", linestyle=":",
               alpha=0.6)
    ax.set_xlabel("Calibrated age (BP)")
    ax.set_ylabel("Number of dates")
    ax.set_title("Thinned date counts by 1000-year bin")
    ax.set_xlim(DISPLAY_MAX_BP, DISPLAY_MIN_BP)
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    plt.savefig("_FIG_DIR /levant_forager_farmer_spd.png", dpi=150)
    plt.show()

    # ---- Save ----
    foragers.to_csv("levant_foragers.csv", index=False)
    farmers.to_csv("levant_farmers.csv", index=False)

    if f_result is not None:
        xr, med, lo, hi, raw = f_result
        np.savez("levant_forager_spd.npz",
                 age=xr, median=med, lo=lo, hi=hi, raw=raw)
    if a_result is not None:
        xr, med, lo, hi, raw = a_result
        np.savez("levant_farmer_spd.npz",
                 age=xr, median=med, lo=lo, hi=hi, raw=raw)
    np.savez("levant_grid_corrected.npz",
             age=x, forager=f_grid, farmer=a_grid,
             n_cells_forager=n_cells_f, n_cells_farmer=n_cells_a)

    print()
    print("Saved:")
    print("  figures/levant_forager_farmer_spd.png")
    print("  levant_foragers.csv")
    print("  levant_farmers.csv")
    print("  levant_raw_density.csv")
    print("  levant_forager_spd.npz")
    print("  levant_farmer_spd.npz")
    print("  levant_grid_corrected.npz")


if __name__ == "__main__":
    main()
