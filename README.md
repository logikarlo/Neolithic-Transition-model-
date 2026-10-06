# Cheapest-Alternative Model of the Levantine Neolithic Transition

An agent-based model of the transition from foraging to farming in the
Levantine corridor during the Late Pleistocene and Early Holocene
(23,000-7,000 cal BP).

The model formalizes one hypothesis:

> Cultivation emerges when foragers who have become dependent on wild
> cereals find the cereals disappearing. The cheapest alternative to
> migrating or starving is to cultivate.

The transition is driven by a single behavioural rule - choose the
cheapest available strategy - applied to two strategies (forage,
cultivate). No exogenous transition event is inserted.

---

## The mechanism

Each cell on a 40x40 grid carries a forager population `F`, a farmer
population `A`, a forager dependence on wild cereals `d`, and a soil
quality `S`.

Each time step, every agent compares the cost of foraging with the cost
of cultivating, per unit of food:

    c_for  = (1/H) * (1 + F/K_F) * (1 + alpha * max(0, d - H))
    c_farm = T(t) * (1 + A/K_A)

where `H` is the local habitat suitability for wild cereals and `T(t)`
is the ratio of farming cost to foraging cost, from Bowles (2011).

Dependence `d` relaxes toward `H` with timescale `tau`. When climate
deteriorates faster than dependence can fall, `d` exceeds `H`, and
`c_for` rises. If `c_for` exceeds `c_farm` in a cell where `d > 0.5`,
foragers there convert to farming.

Two conversion routes are modelled:

1. **In-situ** (cost-mediated): a dependent forager in a cell where
   farming has become cheaper switches to farming.
2. **Cultural** (contact-mediated): farmers convert foragers at rate
   `f = 0.001 /yr`; the cultural transmission rate is from 
   LaPolice et al. (2025), fitted to the European Neolithic expansion. 
   The paper's own finding is that cultural transmission contributed 
   minimally to that expansion. However, ancient DNA and isotope evidence 
   indicate that the Levant and the Near East more broadly show high 
   genetic continuity across the Neolithic transition, in contrast to 
   Europe where demic diffusion dominated. The Levantine transition 
   therefore involved a larger role for cultural transmission than the 
   European case. Using the European rate as the parameter for a 
   Levantine model is conservative: if the Levantine rate is higher, 
   the model underestimates the cultural transmission contribution, 
   and the results reported here are a lower bound on its effect.

Farmers spread by logistic growth. Foragers decline via reduced
carrying capacity as farmers fill cells; they do not migrate.

Farming is viable only after the archaeologically attested onset
(13,000 cal BP).

A single-population ODE version of the same mechanism is also
provided (`model_ode.py`) as a minimal baseline.

---

## Empirical anchors

Every parameter that could be anchored is anchored.

| Component | Value | Source |
|---|---|---|
| Climate multiplier C(t) | Soreq δ¹⁸O | Bar-Matthews et al. 2003; Orland et al. 2012 |
| Cereal precipitation thresholds | 250 / 350 / 400 mm | Willcox 2005; Hillman 1996 |
| Domestication curve P(t) | logistic, mid 10,050 BP, k = 0.0014 | Tanno & Willcox 2006; Fuller et al. 2014, 2016; Weide et al. 2015 |
| Cost ratio endpoints | T = 1.5, 1.0 | Bowles 2011, PNAS |
| Domestication gate | 13,000 cal BP | Arranz-Otaegui et al. 2016 |
| Cultural transmission rate | 0.001 /yr | LaPolice et al. 2025 (European Neolithic; conservative for the Levant) |

### Free parameters

Four parameters are free. Their status splits into two categories.

**Uncalibrated (flat MSE surface):**

| Parameter | Default | Controls |
|---|---|---|
| `TAU_D` | 600 yr | Dependence lag |
| `K_CONV` | 0.03 | In-situ conversion rate |
| `ALPHA_SHORT` | 5.0 | Shortfall multiplier |

These three parameters have flat MSE surfaces across their full
plausible ranges. No value produces a substantially better fit than
another, and the transition timing varies by at most 250 years. Their
defaults are not tuned to the data.

**Calibrated (identified by the fit):**

| Parameter | Value | Controls |
|---|---|---|
| `SOIL_DEGR` | 0.0004 | Soil degradation rate |

`SOIL_DEGR` is the only free parameter identified by the fit. The Dead
Sea core record gives a relative erosion enhancement (3–4× pre-Holocene 
(MIS 7c-2), Lu et al. 2017) but no absolute per-cell rate; converting it 
to a soil quality degradation rate for a 5-km cell requires a mapping 
assumption. The value of 0.0004 is set to the minimum of the MSE surface 
under the 2003 record.

Under the 2012 record the MSE minimum lies at 0.0006 (total MSE 0.0285
versus 0.0324 at 0.0004). The higher value is not used because it
removes the interior farmer peak under the 2003 record: at 0.0006 the
2003 farmer curve peaks before 9,800 BP and has no interior maximum in
the peak-and-decline window. The single value of 0.0004 is the only
value at which both records produce a farmer peak inside the empirical
peak-and-decline window.

Moving `SOIL_DEGR` from an uncalibrated default of 0.00015 to the
calibrated value of 0.0004 reduces total MSE from 0.0402 to 0.0331
under the 2012 record and from 0.0404 to 0.0360 under the 2003 record.
The improvement is entirely in the farmer curve. Forager MSE rises 17%,
farmer-share MSE rises 32%, while the first-conversion age and crossing
year are unchanged.

SOIL_DEGR controls the shape of the farmer curve through soil depletion. 
It does not affect the timing of the transition: the farmer-share crossing 
year is unchanged across the full tested range of the parameter. The timing 
responds only to TAU_D, K_CONV, and ALPHA_SHORT.

The calibration trades a substantially better farmer MSE for a smaller increase 
in the forager MSE and a moderate increase in the share MSE. The calibration 
improves the farmer fit at the cost of a smaller degradation of the forager and 
share fits. Whether that trade-off is worth it is a modelling choice: 
the mechanism's central claim is about the forager decline, and the calibrated 
parameter does not affect that claim's timing or its direction.

Definitional parameters (fixed, not free):

- `D_MIN = 0.5` - foragers are dependent if cereals supply half
  their diet.
- `GAMMA = 1.0` - no bias in cultural transmission.

---

## Results

### Under the Bar-Matthews (2003) climate record

    python run_model_grid.py levant_climate.csv

- Forager MSE (normalised): 0.0261
- Farmer MSE (normalised): 0.0088
- Total MSE: **0.0349**
- Farmer share MSE (unnormalised): 0.0265
- First conversion: 12,975 BP
- Farmer share crosses 0.5 at ~12,500 BP

Under this record the Younger Dryas appears as a drying event. The
shortfall term drives the transition. The conversion-flux panel of the
nine-panel figure shows a sharp in-situ spike of ~2,400 conversions
per step at 12,975 BP.

### Under the Orland (2012) climate record

    python run_model_grid.py levant_climate2.csv

- Forager MSE: 0.0234
- Farmer MSE: 0.0090
- Total MSE: **0.0324**
- Farmer share MSE: 0.0201
- First conversion: 12,975 BP
- Farmer share crosses 0.5 at ~12,475 BP

Under this record the Younger Dryas appears as a cooling plateau
without a strong precipitation anomaly. The shortfall is weaker; the
transition is carried by the declining farming cost ratio T(t) and by
competition for reduced carrying capacity.

### Ensemble

The model was run 200 times with Ornstein-Uhlenbeck noise on log C(t)
at `sigma = 0.005, tau = 2000 yr` (stationary std ~15%):

    python run_grid_ensemble.py 200 levant_climate.csv 0.005

Result:

- First conversion: **12,975 BP in every realization** (onset is
  deterministic).
- Crossing year: mode ~12,650 BP, mean 12,528, std 221 yr.
- Distribution: **unimodal with a heavy left tail** extending to
  11,500 BP. Every 25-year bin between 11,500 and 12,700 BP is
  populated.
- ~10% of realizations complete 400-1,100 years before the mode.

The mechanism has a single preferred completion window around
12,650 BP. Noise can accelerate completion but cannot delay it
substantially, because the underlying driver T(t) is monotonically
decreasing.

### Minimal model comparison

A single-population ODE version of the same mechanism is provided in
`model_ode.py` (four free parameters, no grid). Run with:

    python run_model_ode.py levant_climate.csv

The ODE fires the transition at the same time as the grid model, but
fits worse on every metric. The forager curve in the ODE is a sharp
crash rather than a decline with a tail. The grid model's spatial
distribution of habitat produces a gradual transition, a persistent
forager residual, and a multi-stage farmer rise. The grid fits better.

---

## Comparison to a null model

    python run_levant_null.py levant_climate.csv
    python run_levant_null.py levant_climate2.csv

A climate-driven logistic null model was fitted to the same SPD:

    dF/dt = r_F * F * (1 - F / (K_F * C(t)))
    dA/dt = r_A * A * (1 - A / (K_A * C(t)))

with farmer onset `A_START` and seed `A_SEED` as free parameters
(six total). Fitted by grid search. Both models were run at
`SOIL_DEGR = 0.0004`.

### Forager

| | Full MSE | YD window MSE | Direction (YD) |
|---|---|---|---|
| Null | 0.1215 | 0.1700 | **-0.99** |
| Cereal | 0.0273 | 0.0599 | **+0.89** |
| Ratio (c/n) | 0.225 | 0.352 | — |

Under 2012: null 0.1219 / 0.1682, cereal 0.0242 / 0.0242, ratio
0.199 / 0.144, direction null -0.99 / cereal +0.90.

**The cereal model reduces the forager MSE by a factor of four to
five.** The null is 2.25x worse than a flat-line baseline on the forager MSE.
Its direction correlation of -0.99 means the its forager curve moves
opposite to the empirical curve through the entire Younger Dryas
decline. The cereal model tracks at +0.89 to +0.90.

### Farmer

| Window | Null MSE | Cereal MSE | Ratio (c/n) |
|---|---|---|---|
| Full | 0.0149 | 0.0087 | 0.583 |
| Rise (12,500-9,800) | 0.0061 | 0.0349 | 5.726 |
| Peak+decline (9,800-7,000) | 0.0868 | 0.0165 | **0.190** |

Under 2012: full 0.0149 / 0.0089 (0.598), rise 0.0061 / 0.0083
(1.358), peak+decline 0.0868 / 0.0468 (0.539).

**The null wins the rise window because it has a fitted onset; the
cereal model's onset is emergent.** The null's `A_START` is placed at
11,500 BP by grid search, matching the empirical onset. The cereal
model's first conversion fires at 12,975 BP.

**The null wins the rise window because its onset is fitted to the empirical curve, 
while the cereal model's onset is gated at the earliest viable archaeologically attested date 
(13,000 BP).** The null's `A_START` is placed at 11,500 BP by grid search, 
matching the onset of the empirical curve. Because of the 25-year time step, 
the first conversion fires at 12,975 BP — exactly one step past the gate, 
with no dynamics involved in the timing. The fitted parameter gives the null 
a timing advantage that the fixed gate does not have. The comparison is therefore 
not neutral in the rise window.

**The cereal model wins the peak-and-decline window by a factor of
two to five.** The null's farmer curve rises monotonically through a
window where the empirical curve declines. Direction correlation is
-0.89 for the null (wrong direction) versus +0.96 (2003) or +0.79
(2012) for the cereal model.

The rise-window comparison is reported for completeness but is not a
discriminator: the empirical rise is approximately logistic, and a
climate-driven logistic with a fitted onset will always fit it well.

### Peak timing

| | Forager | Farmer |
|---|---|---|
| Empirical | 12,730 BP | 9,440 BP |
| Null | none | none |
| Cereal (2003) | 13,012 BP | 9,375 BP |
| Cereal (2012) | 13,012 BP | 8,725 BP |

Neither null curve has an interior peak in either window. The cereal
forager peak is 282 years late. The cereal farmer peak is 65 years
early under 2003 and 715 years early under 2012.

---

## Sensitivity

    python run_sensitivity.py levant_climate.csv
    python run_sensitivity.py levant_climate2.csv

The model was swept over plausible ranges of all four free parameters
under both climate records.

| Parameter | 2003 MSE | 2003 crossing | 2012 MSE | 2012 crossing |
|---|---|---|---|---|
| TAU_D | 0.0363-0.0404 | 12,450-12,600 | 0.0323-0.0349 | 12,375-12,600 |
| K_CONV | 0.0381-0.0410 | 12,525-12,600 | 0.0322-0.0329 | 12,325-12,500 |
| ALPHA_SHORT | 0.0358-0.0510 | 12,425-12,675 | 0.0324-0.0325 | 12,475 (constant) |
| SOIL_DEGR | 0.0349-0.0543 | 12,600 (constant) | 0.0285-0.0373 | 12,475 (constant) |

Three parameters (`TAU_D`, `K_CONV`, `ALPHA_SHORT`) have flat MSE
surfaces within their plausible ranges. The model is robust to their
specific values.

`SOIL_DEGR` is identified by the fit under the 2003 record with a
well-located minimum at 0.0004: MSE falls from 0.0381 at 0.00025 to
0.0349 at 0.0004, then rises to 0.0392 at 0.0006. Under the 2012
record the minimum lies at 0.0006. The farmer-share crossing sear is 
invariant to `SOIL_DEGR` in both records.

`ALPHA_SHORT` has zero effect under the 2012 record. Its MSE range
is 0.3% and its crossing year is constant across the full 1-20 range.
This confirms that the shortfall mechanism does not fire under 2012.

---

## Limitations

**1. The onset is a prior, not a prediction.**

The domestication gate (13,000 cal BP) is taken from the
archaeobotanical record. A test with the gate removed produced a
first conversion at 16,000 BP - 3,000 years before the
archaeologically attested onset. The mechanism's predictive content
is the shape and rate of the transition after onset, not the onset
itself.

**2. The erosion estimate precedes the proxy.**

The Dead Sea erosion record (Lu et al. 2017) shows the Holocene 
enhancement beginning at ~11,500 BP, roughly 1,500 years after the 
model's transition onset. The model applies soil degradation continuously 
from the first conversion, so its early soil signal precedes the proxy. 
The offset is consistent with the proxy recording erosion once farming 
dominates the landscape rather than at first appearance. The calibration 
of SOIL_DEGR to 0.0004 is driven by the fit to the empirical farmer curve; 
the erosion record enters only as a qualitative justification for the mechanism, 
not as a quantitative anchor for the parameter value.

**3. SOIL_DEGR is calibrated, not anchored.**

The value of 0.0004 is set to the MSE minimum under the 2003 record.
Under 2012 the MSE minimum lies at 0.0006, but that value removes
the interior farmer peak under 2003. The single value of 0.0004 is
the only one that preserves the mechanism's qualitative signature
under both records. The parameter does not affect the transition
timing, the forager result, or the crossing year.

**4. The two Soreq records are not independent.**

They come from the same cave and describe the same regional signal.
Running both is a conditional test of the YD-drought hypothesis, not
an independent replication.

**5. The rise window favors the null by construction.**

The null model's onset is fitted to the empirical onset while the
cereal model's onset is gated. The rise window is reported for 
completeness but is not used as a discriminator.

**6. The farmer peak is early, especially under 2012.**

Under 2003 the cereal farmer peak is 65 years early. Under 2012 it
is 715 years early. The soil degradation mechanism produces a peak
that arrives earlier than the empirical under the milder 2012
climate.

**7. The model produces a spatially complete transition.**

All 1,600 cells convert by 11,500 BP under both records. The
empirical record shows forager persistence in the arid margins into
the Pottery Neolithic. The model has no mechanism for spatial refugia.

**8. The crossing-time distribution is heavy-tailed.**

A small fraction of realizations (~10% at sigma = 0.005) complete
the transition 400-1,100 years before the mode.

**9. Spatial precipitation is a modern climatology.**

The spatial precipitation field is a Gaussian centered on the
Galilee. It captures the first-order gradient but does not resolve
paleo-spatial detail.

**10. The Southern Levantine data is 25% potentially unreliable.**

The radiocarbon corpus contains ~25% flagged dates. Bunbury (2025) 
curated 4,657 ¹⁴C dates from 582 Southern Levantine sites and flagged 
1,098 (≈25%) as unreliable or with potential reliability issues, including 
intrusive materials (96), mixed reservoirs (121), error values exceeding 
10% CRA (91), and unspecified material (327). The current pipeline applies 
no such screening; all XRONOS rows with valid coordinates, ages, and errors 
are included. The direction of the resulting bias is predominantly toward 
older ages, which affects the forager portion of the SPD more than the 
farmer portion. The magnitude of the bias in the 23,000–7,300 BP window is 
not quantified here; applying the Bunbury flags is the recommended next step.

---

## Repository structure


    forager-farmer-transition/
    ├── README.md
    ├── LICENSE
    ├── requirements.txt
    ├── .gitignore
    ├── models/
    │   ├── latest_levant_model.py
    │   ├── levant_null.py
    │   └── model_ode.py
    ├── xronos SPDs
    │   ├── levant_farmer_spd.npz
    │   ├── levant_farmers.csv
    │   ├── levant_forager_spd.npz
    │   ├── levant_foragers.csv
    │   ├── levant_raw_density.csv
    │   ├── levant_relative_density.csv
    │   └── xronos_SPD_levant.py
    ├── scripts/
    │   ├── latest_levant_ablation.py
    │   ├── run_model_grid.py
    │   ├── prepare_soreq2012.py
    │   ├── run_ODE_model.py
    │   ├── run_grid_ensemble.py
    │   ├── run_ensemble.py
    │   ├── run_levant_null.py
    │   ├── run_no_gate.py
    │   ├── run_sensitivity.py
    │   ├── levant_climate_processing.py
    │   └── levant_climate2_processing.py
    ├── data/
    │   ├── latest_levant_output_2003.txt
    │   ├── latest_levant_output_2012.txt
    │   ├── levant_climate.csv
    │   ├── levant_climate2.csv
    │   ├── levant_grid_corrected.npz
    │   ├── levant_suspicious_farmers.csv
    │   ├── levant_transition.csv
    │   ├── sensitivity_sweep_results.csv
    │   ├── soreq_2003_noaa
    │   └── soreq2012d18o-noaa
    ├── figures/
    │   ├── ablation_levant_climate.png
    │   ├── better_null_levant_climate.png
    │   ├── better_null_levant_climate2.png
    │   ├── grid_ensemble_levant_climate.png
    │   ├── grid_ensemble_levant_climate2.png
    │   ├── grid_levant_climate.png
    │   ├── grid_levant_climate2.png
    │   ├── levant_forager_farmer_spd.png
    │   ├── no_gate_levant_climate.png
    │   ├── no_gate_levant_climate2.png
    │   ├── null_both_levant_climate.png
    │   ├── null_both_levant_climate_0.0006.png
    │   ├── null_both_levant_climate2.png
    │   ├── ode_ensemble_levant_climate.png
    │   ├── ode_levant_climate.png
    │   ├── sensitivity_levant_climate.png
    │   └── sensitivity_levant_climate2.png
    └── notebooks/
        ├── levant_spd_notebook.ipynb
        └── levant_notebook.ipynb

---

## Reproducing the results

    # 1. Prepare climate data
    python prepare_soreq2012.py

    # 2. Main grid model - nine-panel comparison
    python run_model_grid.py levant_climate.csv
    python run_model_grid.py levant_climate2.csv

    # 3. Grid ensemble
    python run_grid_ensemble.py 200 levant_climate.csv 0.005

    # 4. Minimal ODE baseline
    python run_model.py levant_climate.csv

    # 5. Null comparison
    python run_levant_null.py levant_climate.csv
    python run_levant_null.py levant_climate2.csv

    # 6. Gate-removal test
    python run_no_gate.py levant_climate.csv

    # 7. Sensitivity sweep
    python run_sensitivity.py levant_climate.csv
    python run_sensitivity.py levant_climate2.csv

    # 8. Generate the walkthrough notebook
    python make_notebook.py
    jupyter notebook levant_walkthrough.ipynb

    # 9. Regenerate this README
    python write_readme.py

---

## References

Arranz-Otaegui, A., Colledge, S., Zapata, L., Teira-Mayolini, L.C.,
Ibanez, J.J. (2016). Regional diversity on the timing for the initial
appearance of cereal cultivation and domestication in southwest Asia.
*PNAS* 113, 14001-14006.

Bar-Matthews, M., Ayalon, A., Gilmour, M., Matthews, A., Hawkesworth,
C.J. (2003). Sea-land oxygen isotopic relationships from planktonic
foraminifera and speleothems in the Eastern Mediterranean region and
their application for paleorainfall during interglacial intervals.
*Geochimica et Cosmochimica Acta* 67, 3181-3199.

Bunbury, M. M. E. (2025). Towards Robust Demographic Models: A Systematic 
Approach to 14C Data Aggregation and Analysis: Lessons from the Southern 
Levant. *_Journal of Open Archaeology Data_*, 13(3).

Bocquet-Appel, J.-P. (2002). Paleoanthropological traces of a Neolithic
demographic transition. *Current Anthropology* 43, 637-650.

Bowles, S. (2011). Cultivation of cereals by the first farmers was not
more productive than foraging. *PNAS* 108, 4760-4765.

Fuller, D.Q., Denham, T., Arroyo-Kalin, M., et al. (2014). Convergent
evolution and parallelism in plant domestication revealed by an
expanding archaeological record. *PNAS* 111, 6147-6152.

Fuller, D.Q., et al. (2016). Convergent evolution and parallelism in
plant domestication. *Quaternary Science Reviews* 146, 1-14.

Hartman, G., Bar-Yosef, O., Brittingham, A., Grosman, L., Munro, N.D.
(2016). Hunted gazelles evidence cooling, but not drying, during the
Younger Dryas in the southern Levant. *PNAS* 113, 3997-4002.

Hillman, G.C. (1996). Late Pleistocene changes in wild plant-foods
available to hunter-gatherers of the northern Fertile Crescent. In
*The Origins and Spread of Agriculture and Pastoralism in Eurasia*,
ed. D.R. Harris, UCL Press.

LaPolice, T.M., Williams, M.P., Huber, C.D. (2025). Modeling the 
European Neolithic expansion suggests predominant within-group mating 
and limited cultural transmission. Nature Communications 16, 7905. 

Orland, I.J., Bar-Matthews, M., Ayalon, A., et al. (2012). Seasonal
resolution of Eastern Mediterranean climate change since 34 ka from a
Soreq Cave speleothem. *Geochimica et Cosmochimica Acta* 89, 240-255.

Pinhasi, R., Fort, J., Ammerman, A.J. (2005). Tracing the origin and
spread of agriculture in Europe. *PLoS Biology* 3, e410.

Tanno, K.-I., Willcox, G. (2006). How fast was wild wheat domesticated?
*Science* 311, 1886.

Lu, Y., Waldmann, N., Nadel, D., Marco, S. (2017). Increased sedimentation 
following the Neolithic Revolution in the Southern Levant. Global and 
Planetary Change 152, 1–9.

Weide, A., Riehl, S., Zeidi, M., Conard, N.J. (2015). Using new
morphological criteria to identify domesticated emmer wheat at the
aceramic Neolithic site of Chogha Golan (Iran). *Journal of
Archaeological Science* 57, 109-118.

Willcox, G. (2005). The distribution, natural habitats and
availability of wild cereals in relation to their domestication in
the Near East. *Vegetation History and Archaeobotany* 14, 534-541.
