"""
Single-population ODE version of the cereal-availability model.

State variables are scalars, not fields:
    F   forager population
    A   farmer population
    d   forager dependence on wild cereals
    S   soil quality

There is no spatial grid. Habitat H(t) is derived directly from the
climate record via a fixed reference precipitation:

    H(t) = habitat_suitability(P_ref * C(t))

with P_ref = 550 mm, representative of the modern Levantine average
annual precipitation in the cereal-growing zone.

Conversion to farming has two routes:
    (1) In-situ (cost-mediated)
    (2) Cultural (contact transmission)

Farming is viable only for t < DOMESTICATION_START_BP.

Free parameters (4):
    TAU_D       dependence timescale
    K_CONV      in-situ conversion rate
    ALPHA_SHORT shortfall multiplier
    SOIL_DEGR   soil degradation rate

The migration and displacement parameters of the grid model are gone.
Foragers decline via competition for reduced carrying capacity, not
by leaving a cell.

Usage:
    from model_ode import Model
    m = Model(seed=42, climate_path="levant_climate.csv")
    h = m.run()
"""

import numpy as np
import pandas as pd

DT = 25.0
T_START = 23000.0
T_END = 7000.0
N_STEPS = int((T_START - T_END) / DT)

# Habitat
P_REF_MM = 550.0
BARLEY_MIN = 250.0
EMMER_MIN = 350.0
EINKORN_MIN = 400.0
WIDTH = 50.0

# Carrying capacities (per population unit)
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
    SOIL_DEGR=0.00015,
)


def load_climate(path="levant_climate.csv"):
    df = pd.read_csv(path)
    t = df["cal_bp"].values.astype(float)
    C = df["C"].values.astype(float)
    order = np.argsort(t)
    t, C = t[order], C[order]
    return lambda t_bp: float(np.interp(t_bp, t, C))


def _sig(x):
    z = np.clip(x / WIDTH, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def habitat_from_C(C):
    P = P_REF_MM * C
    return float((_sig(P - BARLEY_MIN)
                  + _sig(P - EMMER_MIN)
                  + _sig(P - EINKORN_MIN)) / 3.0)


def non_shattering_fraction(t_bp):
    z = DOMESTICATION_RATE * (t_bp - DOMESTICATION_MIDPOINT_BP)
    return 1.0 / (1.0 + np.exp(z))


def farming_cost_ratio(t_bp):
    P = non_shattering_fraction(t_bp)
    return T_WILD - (T_WILD - T_DOMESTICATED) * P


class Model:
    def __init__(self, seed=42, climate_path="levant_climate.csv",
                 noise_sigma=0.0, **overrides):
        params = dict(DEFAULTS)
        params.update(overrides)
        for k, v in params.items():
            setattr(self, k, v)

        self.rng = np.random.default_rng(seed)
        self.climate = load_climate(climate_path)
        self.noise_sigma = noise_sigma
        self._noise = 0.0

        C0 = self.climate(T_START)
        H0 = habitat_from_C(C0)

        self.F = 1.0
        self.A = 0.0
        self.d = 0.3 * H0
        self.S = 1.0
        self.first_conversion_bp = None

        self.history = {k: [] for k in
                        ["t", "F", "A", "H", "d", "S",
                         "share", "conv_in_situ", "conv_cultural",
                         "c_for", "c_farm", "T_ratio", "P_dom"]}

    def _C(self, t_bp):
        C = self.climate(t_bp)
        if self.noise_sigma > 0:
            eps = self.rng.normal(0, 1)
            self._noise = 0.99 * self._noise + 0.1 * eps
            C = C * (1.0 + self.noise_sigma * self._noise)
            C = max(C, 0.05)
        return C

    def step(self, t_bp):
        C = self._C(t_bp)
        H = habitat_from_C(C)

        if self.F > 1e-6:
            self.d = self.d + (H - self.d) / self.TAU_D * DT
            self.d = float(np.clip(self.d, 0, 1))

        # Forager cost
        c_for = (1.0 / max(H, 0.01)) * (1.0 + self.F / K_F)
        shortfall = max(0.0, self.d - H)
        c_for = c_for * (1.0 + self.ALPHA_SHORT * shortfall)

        # Farming cost
        c_farm = 1e6
        viable = (t_bp < DOMESTICATION_START_BP) and (H > 0.15)
        T = farming_cost_ratio(t_bp)
        if viable:
            c_farm = T * (1.0 + self.A / K_A)

        # In-situ conversion
        conv_in_situ = 0.0
        if viable and self.d > D_MIN and c_farm < c_for and self.F > 0.1:
            adv = max(0.0, (c_for - c_farm) / max(c_for, 0.01))
            frac = min(0.05, self.K_CONV * adv * DT)
            conv_in_situ = self.F * frac

        # Cultural conversion
        conv_cultural = (F_CULT * self.A * self.F
                         / (self.A + self.F + 1e-9) * DT)

        conv = min(conv_in_situ + conv_cultural, self.F)
        if self.first_conversion_bp is None and conv > 1e-6:
            self.first_conversion_bp = t_bp

        self.F -= conv
        self.A += conv

        # Carrying capacities
        K_A_local = K_A * min(1.0, H / 0.5) * self.S
        farmer_footprint = min(1.0, self.A / max(K_A_local, 1.0))
        K_F_local = K_F * H * (1.0 - farmer_footprint)

        # Growth
        self.F += R_F * self.F * (1.0 - self.F / max(K_F_local, 0.1)) * DT
        self.F = max(self.F, 0.0)

        self.A += R_A * self.A * (1.0 - self.A / max(K_A_local, 0.1)) * DT
        self.A = max(self.A, 0.0)

        # Soil
        self.S -= self.SOIL_DEGR * (self.A / K_A) * DT
        self.S = max(self.S, 0.0)

        # Record
        self.history["t"].append(t_bp)
        self.history["F"].append(self.F)
        self.history["A"].append(self.A)
        self.history["H"].append(H)
        self.history["d"].append(self.d)
        self.history["S"].append(self.S)
        total = self.F + self.A
        self.history["share"].append(
            self.A / total if total > 0 else 0.0)
        self.history["conv_in_situ"].append(conv_in_situ)
        self.history["conv_cultural"].append(conv_cultural)
        self.history["c_for"].append(c_for)
        self.history["c_farm"].append(
            c_farm if viable else float("nan"))
        self.history["T_ratio"].append(T)
        self.history["P_dom"].append(non_shattering_fraction(t_bp))

    def run(self, verbose=False):
        for s in range(N_STEPS):
            t = T_START - s * DT
            self.step(t)
            if verbose and s % 400 == 0:
                h = self.history
                print(f"t={t:7.0f}  F={h['F'][-1]:8.2f}  "
                      f"A={h['A'][-1]:8.2f}  "
                      f"H={h['H'][-1]:.3f}  "
                      f"share={h['share'][-1]:.3f}")
        return self.history