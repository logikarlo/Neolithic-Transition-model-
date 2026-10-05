"""
Grid ABM of the Levantine Neolithic transition.

40 x 40 grid of cells, each carrying forager population F(i,j), farmer
population A(i,j), forager dependence d(i,j), and soil quality S(i,j).

Habitat H(i,j,t) is derived from a spatial precipitation baseline
multiplied by the climate multiplier C(t):

    P(i,j,t) = P_base(i,j) * C(t)
    H(i,j,t) = habitat_suitability(P(i,j,t))

Precipitation baseline: Gaussian centered on the Galilee (33.5N,
35.5E, sigma = 2.5 deg), peak 800 mm/yr. Captures the first-order
Levantine precipitation gradient (wet coast, dry interior).

Conversion to farming has two routes:
    (1) In-situ (cost-mediated): a dependent forager in a cell where
        farming has become cheaper switches to farming.
    (2) Cultural (contact-mediated, well-mixed): farmers convert
        foragers at rate F_CULT * A * F / (A + F), following Fort
        (2012). Well-mixed across the grid.

Farming is viable only for ages younger than DOMESTICATION_START_BP
(i.e., after 13,000 cal BP in calendar time).

Free parameters (5):
    TAU_D       dependence timescale
    K_CONV      in-situ conversion rate
    ALPHA_SHORT shortfall multiplier
    SOIL_DEGR   soil degradation rate
    F_CULT      cultural transmission rate

Definitional (fixed, not free):
    D_MIN = 0.5     foragers are dependent if cereals >50% of diet
    GAMMA = 1.0     no bias in cultural transmission
    DOMESTICATION_START_BP = 13000 cal BP

Climate noise: optional Ornstein-Uhlenbeck process on log C(t).

    log_C_perturb(t + dt) = log_C_perturb(t) * (1 - dt / tau)
                          + sigma * sqrt(dt) * N(0, 1)
    C_noisy(t) = C_det(t) * exp(log_C_perturb(t))

with tau = 2000 yr and sigma = 0.005 giving stationary std ~ 15% in C.

Usage:
    from latest_levant_model import Model
    m = Model(seed=42, climate_path="levant_climate.csv")
    h = m.run()
"""

import numpy as np
import pandas as pd

GRID = 40
DT = 25.0
T_START = 23000.0
T_END = 7000.0
N_STEPS = int((T_START - T_END) / DT)

LAT_RANGE = (31.0, 36.0)
LON_RANGE = (34.0, 39.0)
P_MAX = 800.0
P_SIGMA = 2.5
P_LAT_C = 33.5
P_LON_C = 35.5

BARLEY_MIN = 250.0
EMMER_MIN = 350.0
EINKORN_MIN = 400.0
WIDTH = 50.0

K_F = 100.0
K_A = 500.0
R_F = 0.0006
R_A = 0.0012

D_MIN = 0.5
GAMMA = 1.0
F_CULT = 0.001

DOMESTICATION_MIDPOINT_BP = 10050.0
DOMESTICATION_RATE = 0.0014
T_WILD = 1.5
T_DOMESTICATED = 1.0
DOMESTICATION_START_BP = 13000.0

DEFAULTS = dict(
    TAU_D=600.0,
    K_CONV=0.03,
    ALPHA_SHORT=5.0,
    SOIL_DEGR=0.0004,
    DOMESTICATION_START_BP=13000.0,
    F_CULT=0.001,
)

NOISE_TAU = 2000.0
NOISE_SIGMA = 0.005
NOISE_CLIP = 0.5


def load_climate(path="levant_climate.csv"):
    df = pd.read_csv(path)
    t = df["cal_bp"].values.astype(float)
    C = df["C"].values.astype(float)
    order = np.argsort(t)
    t, C = t[order], C[order]
    return lambda t_bp: float(np.interp(t_bp, t, C))


def build_precip_baseline():
    lats = np.linspace(*LAT_RANGE, GRID)
    lons = np.linspace(*LON_RANGE, GRID)
    lon_g, lat_g = np.meshgrid(lons, lats)
    d2 = (lat_g - P_LAT_C) ** 2 + (lon_g - P_LON_C) ** 2
    return P_MAX * np.exp(-d2 / (2 * P_SIGMA ** 2))


def habitat(P):
    def sig(x):
        z = np.clip(x / WIDTH, -50, 50)
        return 1.0 / (1.0 + np.exp(-z))
    return (sig(P - BARLEY_MIN) + sig(P - EMMER_MIN)
            + sig(P - EINKORN_MIN)) / 3.0


def non_shattering_fraction(t_bp):
    z = DOMESTICATION_RATE * (t_bp - DOMESTICATION_MIDPOINT_BP)
    return 1.0 / (1.0 + np.exp(z))


def farming_cost_ratio(t_bp):
    P = non_shattering_fraction(t_bp)
    return T_WILD - (T_WILD - T_DOMESTICATED) * P


class Model:
    def __init__(self, seed=42, climate_path="levant_climate.csv",
                 noise_sigma=0.0, noise_tau=None, **overrides):
        params = dict(DEFAULTS)
        params.update(overrides)
        for k, v in params.items():
            setattr(self, k, v)

        self.rng = np.random.default_rng(seed)
        self.P_base = build_precip_baseline()
        self.climate = load_climate(climate_path)

        self.noise_sigma = noise_sigma
        self.noise_tau = noise_tau if noise_tau is not None else NOISE_TAU
        self._log_noise = 0.0

        C0 = self.climate(T_START)
        H0 = habitat(self.P_base * C0)

        self.F = 0.02 * K_F * H0
        self.A = np.zeros((GRID, GRID))
        self.d = 0.3 * H0
        self.S = np.ones((GRID, GRID))

        self.first_conversion_bp = None

        self.history = {k: [] for k in
                        ["t", "F", "A", "H_mean", "d_mean",
                         "conv", "P_dom", "T_ratio",
                         "farmer_share", "conv_in_situ",
                         "conv_cultural",
                         "c_for_mean", "c_farm_mean"]}

    def _C(self, t_bp):
        C = self.climate(t_bp)
        if self.noise_sigma > 0:
            decay = 1.0 - DT / self.noise_tau
            innovation = (self.noise_sigma
                          * np.sqrt(DT)
                          * self.rng.normal(0.0, 1.0))
            self._log_noise = self._log_noise * decay + innovation
            self._log_noise = float(np.clip(
                self._log_noise, -NOISE_CLIP, NOISE_CLIP))
            C = C * float(np.exp(self._log_noise))
            C = max(C, 0.05)
        return C

    def step(self, t_bp):
        C = self._C(t_bp)
        P = self.P_base * C
        H = habitat(P)

        mask_F = self.F > 1e-6
        if mask_F.any():
            self.d = np.where(
                mask_F,
                self.d + (H - self.d) / self.TAU_D * DT,
                self.d,
            )
            self.d = np.clip(self.d, 0, 1)

        h = np.maximum(H, 0.01)
        c_for = (1.0 / h) * (1.0 + self.F / K_F)
        shortfall = np.maximum(0.0, self.d - H)
        c_for = c_for * (1.0 + self.ALPHA_SHORT * shortfall)

        c_farm = np.full_like(c_for, 1e6)
        viable = (t_bp < self.DOMESTICATION_START_BP) & (H > 0.15)
        T = farming_cost_ratio(t_bp)
        c_farm = np.where(viable,
                          T * (1.0 + self.A / K_A),
                          c_farm)

        can_conv = (self.d > D_MIN) & (c_farm < c_for) & (self.F > 0.1)
        conv_in_situ = np.zeros_like(self.F)
        if can_conv.any():
            adv = np.maximum(0.0, (c_for - c_farm) / np.maximum(c_for, 0.01))
            frac = np.minimum(0.05, self.K_CONV * adv * DT)
            conv_in_situ = self.F * frac * can_conv

        A_total = self.A.sum()
        F_total = self.F.sum()
        denom = A_total + GAMMA * F_total + 1e-9
        conv_cultural = self.F_CULT * A_total * self.F / denom * DT

        conv = np.minimum(conv_in_situ + conv_cultural, self.F)

        if self.first_conversion_bp is None and conv.sum() > 1e-6:
            self.first_conversion_bp = t_bp

        self.F -= conv
        self.A += conv

        in_situ_total = float(conv_in_situ.sum())
        cultural_total = float(conv_cultural.sum())

        K_A_local = K_A * np.minimum(1, H / 0.5) * self.S
        farmer_footprint = np.clip(
            self.A / np.maximum(K_A_local, 1.0), 0.0, 1.0)
        K_F_local = K_F * H * (1.0 - farmer_footprint)

        self.F += R_F * self.F * (1 - self.F / np.maximum(K_F_local, 0.1)) * DT
        self.F = np.maximum(self.F, 0)

        self.A += R_A * self.A * (1 - self.A / np.maximum(K_A_local, 0.1)) * DT
        self.A = np.maximum(self.A, 0)

        self.S -= self.SOIL_DEGR * (self.A / K_A) * DT
        self.S = np.maximum(self.S, 0.0)

        self.history["t"].append(t_bp)
        self.history["F"].append(self.F.sum())
        self.history["A"].append(self.A.sum())
        occ = (self.F + self.A) > 1e-6
        self.history["H_mean"].append(
            float(H[occ].mean()) if occ.any() else 0.0)
        self.history["d_mean"].append(
            float(self.d[mask_F].mean()) if mask_F.any() else 0.0)
        self.history["conv"].append(
            int(((self.A > self.F) & (self.A > 1.0)).sum()))
        self.history["P_dom"].append(non_shattering_fraction(t_bp))
        self.history["T_ratio"].append(farming_cost_ratio(t_bp))
        total = self.F.sum() + self.A.sum()
        self.history["farmer_share"].append(
            float(self.A.sum() / total) if total > 0 else 0.0)
        self.history["conv_in_situ"].append(in_situ_total)
        self.history["conv_cultural"].append(cultural_total)

        comparison_mask = mask_F & (H > 0.15)
        if comparison_mask.any():
            self.history["c_for_mean"].append(
                float(c_for[comparison_mask].mean()))
            self.history["c_farm_mean"].append(
                float(c_farm[comparison_mask].mean()))
        else:
            self.history["c_for_mean"].append(float("nan"))
            self.history["c_farm_mean"].append(float("nan"))

    def run(self, verbose=False):
        for s in range(N_STEPS):
            t = T_START - s * DT
            self.step(t)
            if verbose and s % 400 == 0:
                h = self.history
                print(f"t={t:7.0f}  F={h['F'][-1]:9.1f}  "
                      f"A={h['A'][-1]:9.1f}  "
                      f"share={h['farmer_share'][-1]:.3f}")
        return self.history