#!/usr/bin/env python3
"""
Download and process the Bar-Matthews et al. (2003) Soreq Cave d18O
record into a climate multiplier C(t) for the cereal-availability
model.

The 2003 record stores age in thousands of years (ka BP). The parser
converts to years by multiplying by 1000. The data columns are:

    age_calkaBP   d18O_vpdb   d13C_vpdb

Anchors for the linear mapping from d18O to C are hardcoded by
default:

    d18o_lgm    = -3.0 per mil   (LGM, dry)
    d18o_modern = -5.5 per mil   (early Holocene, wet)

These are mapped to C_lgm = 0.33 and C_modern = 1.0. The values can
be overridden on the command line, or auto-derived from the record
with --auto-anchors.

Reference:
    Bar-Matthews, M., Ayalon, A., Gilmour, M., Matthews, A.,
    Hawkesworth, C.J. (2003). Sea-land oxygen isotopic relationships
    from planktonic foraminifera and speleothems in the Eastern
    Mediterranean region and their application for paleorainfall
    during interglacial intervals. Geochimica et Cosmochimica Acta
    67, 3181-3199.

Output: levant_climate.csv with columns cal_bp, C
"""

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

URL = (
    "https://www.ncei.noaa.gov/pub/data/paleo/speleothem/"
    "israel/soreq_2003-noaa.txt"
)
OUT_CSV = "levant_climate.csv"


def download(url: str, dest: Path) -> Path:
    print(f"Downloading {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = resp.read()
    except Exception as e:
        print(f"Download failed: {e}")
        sys.exit(1)
    dest.write_bytes(data)
    print(f"Saved {len(data)} bytes to {dest}")
    return dest


def parse_noaa(path: Path) -> pd.DataFrame:
    """
    Parse the Soreq 2003 NOAA template file.

    Data lines are whitespace-delimited with three columns:
        age_calkaBP  d18O_vpdb  d13C_vpdb

    Age is in thousands of years (ka BP). Convert to years by
    multiplying by 1000.
    """
    ages, d18os, d13cs = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 3:
                continue
            try:
                age_ka = float(parts[0])
                d18o = float(parts[1])
                d13c = float(parts[2])
            except ValueError:
                continue
            ages.append(age_ka * 1000.0)   # ka BP -> years BP
            d18os.append(d18o)
            d13cs.append(d13c)

    if not ages:
        print("No data parsed. Check file format.")
        sys.exit(1)

    df = pd.DataFrame({"age": ages, "d18O": d18os, "d13C": d13cs})
    df = df.sort_values("age").reset_index(drop=True)
    print(
        f"Parsed {len(df)} records, age range "
        f"{df['age'].min():.0f}-{df['age'].max():.0f} yr BP"
    )
    print(
        f"d18O range {df['d18O'].min():.2f} to "
        f"{df['d18O'].max():.2f} per mil"
    )
    return df


def smooth(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, center=True, min_periods=1).mean()


def auto_anchors(df: pd.DataFrame, t_start: float, t_end: float) -> tuple:
    """
    Derive LGM and Holocene anchors from the record itself.

    LGM anchor:    median d18O over 19-23 ka BP
    Holocene anchor: median d18O over 7-10 ka BP
    """
    lgm = df[(df["age"] >= 19000) & (df["age"] <= 23000)]["d18O"]
    hol = df[(df["age"] >= 7000) & (df["age"] <= 10000)]["d18O"]

    if len(lgm) == 0 or len(hol) == 0:
        print("Cannot derive anchors: insufficient data in "
              "reference windows.")
        sys.exit(1)

    d18o_lgm = float(lgm.median())
    d18o_modern = float(hol.median())

    print(f"\nAuto-derived anchors:")
    print(f"  LGM (19-23 ka BP):      d18O = {d18o_lgm:.2f} "
          f"(n = {len(lgm)})")
    print(f"  Holocene (7-10 ka BP):  d18O = {d18o_modern:.2f} "
          f"(n = {len(hol)})")
    print(f"  Difference:             {d18o_lgm - d18o_modern:.2f} per mil")
    return d18o_lgm, d18o_modern


def d18o_to_C(
    d18o: np.ndarray,
    d18o_modern: float,
    d18o_lgm: float,
    C_modern: float,
    C_lgm: float,
    clip_min: float = 0.15,
    clip_max: float = 1.5,
) -> np.ndarray:
    """
    Linear map from d18O to precipitation multiplier C.

    More negative d18O = wetter = higher C.
    Anchored at two reference points:
        d18o_lgm    -> C_lgm
        d18o_modern -> C_modern
    """
    t = (d18o - d18o_lgm) / (d18o_modern - d18o_lgm)
    C = C_lgm + t * (C_modern - C_lgm)
    C = np.clip(C, clip_min, clip_max)
    return C


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=URL)
    ap.add_argument("--out", default=OUT_CSV)
    ap.add_argument("--cache", default="soreq_2003-noaa.txt")
    ap.add_argument("--smooth-years", type=float, default=500.0)
    ap.add_argument("--auto-anchors", action="store_true",
                    help="Derive anchors from the record instead of "
                         "using the hardcoded defaults")
    ap.add_argument("--d18o-modern", type=float, default=-5.5,
                    help="Modern reference d18O (per mil VPDB)")
    ap.add_argument("--d18o-lgm", type=float, default=-3.0,
                    help="LGM reference d18O (per mil VPDB)")
    ap.add_argument("--C-modern", type=float, default=1.0)
    ap.add_argument("--C-lgm", type=float, default=0.33)
    ap.add_argument("--t-start", type=float, default=23000.0)
    ap.add_argument("--t-end", type=float, default=7300.0)
    ap.add_argument("--dt", type=float, default=25.0)
    args = ap.parse_args()

    raw_path = Path(args.cache)
    if not raw_path.exists():
        download(args.url, raw_path)
    else:
        print(f"Using cached file {raw_path}")

    df = parse_noaa(raw_path)

    if args.auto_anchors:
        auto_lgm, auto_modern = auto_anchors(
            df, args.t_start, args.t_end)
        d18o_lgm = auto_lgm
        d18o_modern = auto_modern
    else:
        d18o_lgm = args.d18o_lgm
        d18o_modern = args.d18o_modern
        print(f"\nUsing hardcoded anchors:")
        print(f"  LGM:      d18O = {d18o_lgm:.2f} per mil")
        print(f"  Holocene: d18O = {d18o_modern:.2f} per mil")

    # Restrict to window
    mask = (df["age"] >= args.t_end) & (df["age"] <= args.t_start)
    df = df[mask].copy()
    if df.empty:
        print("No data in requested time window.")
        sys.exit(1)
    print(
        f"\nWindow {args.t_end:.0f}-{args.t_start:.0f} cal BP: "
        f"{len(df)} records"
    )

    # Smooth
    if len(df) > 1:
        median_dt = float(np.median(np.diff(df["age"].values)))
        win_samples = max(1, int(round(args.smooth_years / median_dt)))
    else:
        win_samples = 1
    df["d18O_smooth"] = smooth(df["d18O"], win_samples)
    print(
        f"Smoothing window: {args.smooth_years:.0f} yr "
        f"({win_samples} samples)"
    )

    # Interpolate to a regular grid
    t_grid = np.arange(args.t_end, args.t_start + args.dt, args.dt)
    d18o_grid = np.interp(
        t_grid, df["age"].values, df["d18O_smooth"].values
    )

    C_grid = d18o_to_C(
        d18o_grid, d18o_modern, d18o_lgm,
        args.C_modern, args.C_lgm,
    )

    out = pd.DataFrame({"cal_bp": t_grid, "C": C_grid})
    out = out.sort_values("cal_bp", ascending=False).reset_index(drop=True)

    out.to_csv(args.out, index=False)
    print(f"\nWrote {args.out} ({len(out)} rows)")
    print(f"C range: {out['C'].min():.3f} to {out['C'].max():.3f}")
    print(f"Mean C: {out['C'].mean():.3f}")

    print("\nKey time slices:")
    print(f"{'Age BP':>10s} {'d18O':>8s} {'C':>8s}")
    for age in [
        22000, 20000, 18000, 16000, 15000, 14000, 13000,
        12500, 12000, 11500, 11000, 10000, 9000, 8000, 7300,
    ]:
        i = int(np.argmin(np.abs(t_grid - age)))
        print(f"{age:>10d} {d18o_grid[i]:>8.2f} {C_grid[i]:>8.3f}")


if __name__ == "__main__":
    main()