# Macro-state conditioning of the tree models: yield-curve slope, credit spread, VIX (MACRO_CONDITION)

Implementation: `macro_condition.py` (run: `.venv/bin/python experiments/macro_condition/macro_condition.py`, about 15 minutes; needs network on the first run to download three FRED series into `cache/`).

## Objective
Test whether adding month-level macro variables (10y-2y yield-curve slope, Baa-10y credit spread, VIX) to the inputs of ExtraTrees / LightGBM lets the trees condition factor relations on the macro state and raises the universe rank IC over the same trees on the 18 factors alone - and whether any gain is real or the trees simply memorise the regime of each training month.

## Hypothesis
Macro-conditioned firm-factor relations (Gu-Kelly-Xiu style) improve prediction; the yield-curve slope is the variable one practitioner found useful. **Failure mode stated in advance**: a macro variable has one value per month, so a training set of 47-132 months holds only that many distinct values and trees can only use it to memorise each training month's regime. To measure that noise floor, three **placebo** arms feed the trees the yield-curve slope with its values permuted across months (same distribution, no macro information). Pre-registered kill rule (docs/RESEARCH.md Part I sec 4): adopt only if paired monthly universe-IC t >= 2 vs the control on two families, or IC >= 0.037 with t >= 2.

## Research origin
docs/RESEARCH.md Part I sec 3.3 item 7 (firm features x macro state; a 2026 GNN paper reports VIX / credit-spread conditioning helps; "only 68 test months; treat as low priority, high overfit risk") and docs/RESEARCH.md Part II sec 2.13 (r/algotrading 1rwp4z8: after 45 rounds of price/volume feature engineering the only feature that moved a 1,400-stock GBT+MLP ensemble was yield-curve slope; a commenter flags the 45 rounds themselves as an overfitting mechanism).

## Implementation
`add_macro_features` appends four columns to the model matrix for every row, taken at the row's characteristic month (last FRED observation on or before month-end, known at t): `ycurve_10y2y` (FRED T10Y2Y), `credit_baa10y` (BAA10Y), `vix` (VIXCLS) and the placebos. Everything else is the frozen walk-forward (`run_wf`, validation-IC selection against the raw return). Arms: `control` (18 factors), `plus_ycurve`, `plus_credit`, `plus_vix`, `plus_all_macro`, `placebo1..3` (shuffled slope; permutation seeds 1001-1003).

## Experimental setup
All four target experiments use the same frozen harness (a verbatim copy of the data / rank-transform / walk-forward / LP / cost / performance code of `experiments/largecap/largecap.py`, embedded in the script so it runs on its own).

- **Data**: `fiam/chars_final_with_names.parquet`, 523,125 stock-months after dropping rows without a next-month return; 18 factors chosen by economic group in `experiments/largecap` (`FACTOR_GROUPS`), cross-sectionally median-filled and rank-transformed to [-1, 1] over all stocks each month.
- **Universe** (training, validation and scoring rows): price >= $5, market cap >= $2,000M, 126-day dollar volume >= $10M, both betas observed (about 1,206 stocks per month; 153,392 rows).
- **Walk-forward**: for test year Y = 2021..2026, train on target months before Jan Y-2, validate on Y-2..Y-1, refit annually; candidate hyper-parameters are chosen on **validation rank IC against the raw next-month excess return** (for every arm, whatever the fit target), never on test data.
- **Models**: Extra-Trees (largecap grid, 300 trees) and forest-style LightGBM (num_leaves 7, min_child_samples {500, 2000}, extra_trees, lambda 100, checkpoints at 100/200/300 trees). Seed 42 unless stated.
- **Scoring**: universe rank IC of the OOS prediction vs the raw next-month return (68 months, 2021-01..2026-08), decile spread, and the frozen LP portfolios `lc_t10` (10% one-way turnover cap, headline) and `lc_free`, gross and net of the assumed tiered costs.
- **Pre-registered rule** (docs/RESEARCH.md Part I sec 4): adopt an alternative target only if paired monthly universe-IC t >= 2 vs the control on **two** model families (rule A), or IC >= the composite's 0.037 with t >= 2 (rule B).
- **Sanity check that the harness is faithful**: the control arm `et__control` reproduces `experiments/largecap` `et` exactly: IC 0.0164 (published 0.016), lc_t10 gross IR -0.22 (published -0.22).

## Baseline
`*__control`: same trees on the 18 factors (reproduces `largecap` `et`: IC 0.0164, gross IR -0.22). The **placebo arms are the noise-floor baseline** for any month-level feature.

## Commands
```
.venv/bin/python experiments/macro_condition/macro_condition.py
```

## Results (68 test months)
| model__arm | ic | ic_t | IC diff vs control | paired t | IR gross (lc_t10) | IR net (lc_t10) | importance on macro cols |
|---|---:|---:|---:|---:|---:|---:|---:|
| et__control | 0.0164 | 1.10 | +0.0000 |  | -0.216 | -0.302 | 0.00 |
| et__plus_ycurve | 0.0345 | 2.50 | +0.0181 | +1.19 | -0.176 | -0.269 | 0.57 |
| et__plus_credit | 0.0076 | 0.38 | -0.0088 | -0.84 | -0.278 | -0.358 | 0.56 |
| et__plus_vix | 0.0248 | 1.42 | +0.0085 | +1.02 | -0.100 | -0.179 | 0.63 |
| et__plus_all_macro | 0.0264 | 1.43 | +0.0100 | +0.77 | 0.159 | 0.076 | 0.88 |
| et__placebo1 | 0.0268 | 1.79 | +0.0104 | +1.17 | -0.407 | -0.491 | 0.61 |
| et__placebo2 | 0.0225 | 1.48 | +0.0061 | +0.81 | -0.138 | -0.217 | 0.46 |
| et__placebo3 | 0.0146 | 0.88 | -0.0018 | -0.23 | -0.346 | -0.430 | 0.49 |
| lgbm__control | 0.0160 | 1.04 | +0.0000 |  | -0.167 | -0.251 | 0.00 |
| lgbm__plus_ycurve | 0.0246 | 1.63 | +0.0086 | +1.15 | -0.360 | -0.439 | 0.71 |
| lgbm__plus_credit | 0.0092 | 0.45 | -0.0069 | -0.68 | -0.448 | -0.529 | 0.65 |
| lgbm__plus_vix | 0.0225 | 1.29 | +0.0065 | +0.57 | -0.165 | -0.241 | 0.76 |
| lgbm__plus_all_macro | 0.0220 | 1.09 | +0.0059 | +0.57 | -0.130 | -0.206 | 0.90 |
| lgbm__placebo1 | 0.0118 | 0.73 | -0.0042 | -0.41 | -0.359 | -0.444 | 0.75 |
| lgbm__placebo2 | 0.0185 | 1.36 | +0.0025 | +0.27 | -0.078 | -0.165 | 0.66 |
| lgbm__placebo3 | 0.0090 | 0.56 | -0.0071 | -1.14 | -0.203 | -0.285 | 0.68 |

Universe rank IC by test year (the control and the two families' slope and placebo arms):

| arm | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|---:|
| et__control | +0.050 | +0.009 | +0.032 | +0.032 | -0.021 | -0.013 |
| et__plus_ycurve | +0.069 | +0.068 | +0.052 | +0.024 | -0.016 | -0.003 |
| et__placebo1 | +0.044 | +0.017 | +0.031 | +0.070 | +0.012 | -0.034 |
| et__placebo2 | +0.049 | -0.002 | +0.060 | +0.013 | +0.014 | -0.011 |
| et__placebo3 | +0.044 | +0.019 | +0.045 | +0.018 | -0.040 | -0.003 |
| lgbm__control | +0.045 | +0.004 | +0.038 | +0.035 | -0.020 | -0.016 |
| lgbm__plus_ycurve | +0.068 | +0.013 | +0.023 | +0.022 | +0.005 | +0.012 |
| lgbm__placebo1 | +0.001 | +0.005 | +0.016 | +0.071 | -0.008 | -0.027 |
| lgbm__placebo2 | +0.068 | -0.020 | +0.049 | +0.025 | -0.011 | -0.008 |
| lgbm__placebo3 | +0.042 | -0.023 | +0.023 | +0.042 | -0.031 | -0.004 |

The 10y-2y slope was 0.0-0.6 in 2019-2020, 0.8-1.6 in 2021, then **inverted in 2022-2024** (2023: -1.06 to -0.19) and 0.3-0.7 in 2025-26.

**Process note.** The first run had no placebo arms; it showed the slope lifting ET's IC from 0.016 to 0.0345 (t 2.50), which prompted the placebo test (the first-run summary is kept in `output/first_run_without_placebo/`; its numbers for the shared arms are identical to the final ones, so the model was deterministic across the two runs).

## Interpretation
The yield-curve slope raises IC in both families (ET 0.0345 vs 0.0164, LightGBM 0.0246 vs 0.0160; +0.018 / +0.009), the credit spread hurts (-0.009 / -0.007) and VIX adds +0.007 / +0.009; but **none of the differences is significant** (paired t 1.2, 1.2, 1.0, 0.6), and the ET slope arm (IC 0.0345, t 2.5) is just below the composite's 0.037, so neither pre-registered adoption rule is met. The placebo arms show why to be cautious: feeding the trees a *month-shuffled* slope moves ET's IC by -0.002 to +0.010 and LightGBM's by -0.007 to +0.003, so the noise floor of "add any month-level feature" is about +-0.006 IC. The real slope beats all three placebos in both families (ET +0.018 vs +0.010 / +0.006 / -0.002; LightGBM +0.009 vs -0.004 / +0.003 / -0.007), which is mildly suggestive (with three placebos the probability that the real arm ranks first by chance is 1/4 per family), but the gain is **concentrated in a few months** (ET: the top 5 months account for 113% of the total paired gain, and 2022 alone goes from 0.009 to 0.068 - the year the curve first inverted in the sample) and is not consistent across families or years (LightGBM's gain is in 2021 and 2025-26). Trees put 45-90% of their importance on the macro columns in every arm, placebos included, which is regime memorisation, not information. The portfolios do not follow the IC: the `lc_t10` gross IR of the slope arms is -0.18 (ET) and -0.36 (LGBM); only `plus_all_macro` in ET is positive (+0.16 gross, +0.08 net) and is within IR noise. **Status: IMPLEMENTED_BUT_INCONCLUSIVE (rule not met; suggestive for the yield-curve slope, not credible enough to adopt).**

## Limitations
- Three placebos per family give only a coarse noise floor (the minimum achievable one-sided p is 0.25 per family); more placebos would sharpen it.
- 47-132 distinct macro values in training; a longer macro history than the 2015 start of the panel is available (FRED goes back decades) but the stock panel is what limits the sample.
- One seed, one path; the macro effect is confounded with time (the slope is a proxy for the calendar over 2021-24).
- Only three macro variables were tried; no macro x factor interaction was hand-built.

## Follow-up
If pursued, add many placebos (20+) and multiple seeds, and hand-build a small number of macro x factor interactions on the *composite* (no fitting of tree splits), which is where a macro signal could be tested without regime memorisation. Not run: 68 months cannot identify it.
