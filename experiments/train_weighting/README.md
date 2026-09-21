# Value-/size-weighted training loss, and "train on all stocks" (TRAIN_WEIGHTING)

Implementation: `train_weighting.py` (run: `.venv/bin/python experiments/train_weighting/train_weighting.py`, about 60 minutes: the all-stock arms train on up to 400,000 rows).

## Objective
Test whether weighting the training loss toward the tradeable end (sqrt of market cap, market cap, sqrt of dollar volume), or training on ALL stocks with such weights, improves the tradeable-universe IC of ExtraTrees / LightGBM on the 18 factors.

## Hypothesis
Restricting training to >= $2B stocks throws away ~70% of the rows and did not help (`experiments/largecap`: `et` vs `et_allrows` both about zero). Training on all stocks with weights that emphasise large names keeps the small-cap information and points the loss at the names the LP can hold, so universe IC rises over the universe-only unweighted control in at least two families.

## Research origin
docs/NEW.md sec 3.2 (Gu-Kelly-Xiu weight the loss by market value; Numerai ships a liquidity/residual-vol sample-weight vector; "weighting is a softer variant not yet tested. Own: try w = min(mcap, cap)^0.5").

## Implementation
`main_train.py`-derived driver (`make_weights`, `run_wf`): `weight` in {`sqrt_mcap` = sqrt(min(mcap, $50B)), `mcap` = min(mcap, $50B), `sqrt_dolvol`}, normalised to mean 1 in each training set; `train: all` uses every stock-month with a return as training rows while candidate selection and scoring stay on universe rows against the raw return. Weights enter `ExtraTreesRegressor.fit(sample_weight=)` and `LGBMRegressor.fit(sample_weight=)`.

## Experimental setup
All four target experiments use the same frozen harness (a verbatim copy of the data / rank-transform / walk-forward / LP / cost / performance code of `experiments/largecap/largecap.py`, embedded in the script so it runs on its own).

- **Data**: `fiam/chars_final_with_names.parquet`, 523,125 stock-months after dropping rows without a next-month return; 18 factors chosen by economic group in `experiments/largecap` (`FACTOR_GROUPS`), cross-sectionally median-filled and rank-transformed to [-1, 1] over all stocks each month.
- **Universe** (training, validation and scoring rows): price >= $5, market cap >= $2,000M, 126-day dollar volume >= $10M, both betas observed (about 1,206 stocks per month; 153,392 rows).
- **Walk-forward**: for test year Y = 2021..2026, train on target months before Jan Y-2, validate on Y-2..Y-1, refit annually; candidate hyper-parameters are chosen on **validation rank IC against the raw next-month excess return** (for every arm, whatever the fit target), never on test data.
- **Models**: Extra-Trees (largecap grid, 300 trees) and forest-style LightGBM (num_leaves 7, min_child_samples {500, 2000}, extra_trees, lambda 100, checkpoints at 100/200/300 trees). Seed 42 unless stated.
- **Scoring**: universe rank IC of the OOS prediction vs the raw next-month return (68 months, 2021-01..2026-08), decile spread, and the frozen LP portfolios `lc_t10` (10% one-way turnover cap, headline) and `lc_free`, gross and net of the assumed tiered costs.
- **Pre-registered rule** (docs/NEW.md sec 4): adopt an alternative target only if paired monthly universe-IC t >= 2 vs the control on **two** model families (rule A), or IC >= the composite's 0.037 with t >= 2 (rule B).
- **Sanity check that the harness is faithful**: the control arm `et__control` reproduces `experiments/largecap` `et` exactly: IC 0.0164 (published 0.016), lc_t10 gross IR -0.22 (published -0.22).

Arms: `control` (universe rows, unweighted), `w_*` (universe rows, weighted), `all_unweighted` (= the largecap `et_allrows` diagnostic; IC 0.0197 here vs 0.020 published), `all_w_sqrt_mcap`, `all_w_mcap`. The hyper-parameter grids are unchanged, which matters for the all-stock arms (`min_samples_leaf` 100-1000 is proportionally smaller on 400k rows).

## Baseline
`*__control`: same family, universe rows only, unweighted.

## Commands
```
.venv/bin/python experiments/train_weighting/train_weighting.py
```

## Results (seed 42, 68 test months)
| model__arm | universe rank IC | IC t | % months IC>0 | D10-D1 (%/m) | IR gross (lc_t10) | IR net (lc_t10) | IR gross (lc_free) | beta | max DD net % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| et__control | 0.0164 | 1.102 | 49 | -0.170 | -0.216 | -0.302 | -0.182 | -0.086 | -35 |
| et__w_sqrt_mcap | 0.0127 | 0.836 | 46 | -0.398 | -0.284 | -0.372 | -0.271 | -0.108 | -35 |
| et__w_mcap | 0.0106 | 0.687 | 47 | -0.419 | -0.306 | -0.400 | -0.300 | -0.067 | -37 |
| et__w_sqrt_dolvol | 0.0093 | 0.608 | 46 | -0.446 | -0.482 | -0.569 | -0.409 | -0.073 | -41 |
| et__all_unweighted | 0.0204 | 1.507 | 50 | -0.029 | -0.026 | -0.109 | -0.032 | -0.144 | -35 |
| et__all_w_sqrt_mcap | 0.0144 | 1.015 | 49 | -0.001 | -0.289 | -0.381 | -0.096 | -0.133 | -35 |
| et__all_w_mcap | 0.0098 | 0.697 | 49 | -0.208 | -0.035 | -0.130 | -0.262 | -0.099 | -34 |
| lgbm__control | 0.0160 | 1.044 | 49 | -0.175 | -0.167 | -0.251 | -0.264 | -0.122 | -33 |
| lgbm__w_sqrt_mcap | 0.0111 | 0.690 | 47 | -0.278 | -0.428 | -0.515 | -0.450 | -0.096 | -37 |
| lgbm__w_mcap | 0.0074 | 0.457 | 49 | -0.444 | -0.371 | -0.472 | -0.518 | -0.069 | -27 |
| lgbm__w_sqrt_dolvol | 0.0065 | 0.394 | 47 | -0.258 | -0.455 | -0.538 | -0.423 | -0.094 | -40 |
| lgbm__all_unweighted | 0.0230 | 1.830 | 54 | 0.259 | 0.079 | -0.000 | 0.152 | -0.064 | -34 |
| lgbm__all_w_sqrt_mcap | 0.0116 | 0.767 | 50 | -0.242 | -0.303 | -0.384 | -0.148 | -0.131 | -40 |
| lgbm__all_w_mcap | 0.0078 | 0.514 | 50 | -0.020 | -0.273 | -0.360 | -0.137 | -0.065 | -33 |

Paired monthly IC vs the same-family control:

| model | arm | ic | ic_control | IC diff | paired t | share of months better | ir_gross | ir_gross_control |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| et | w_sqrt_mcap | 0.0127 | 0.0164 | -0.0037 | -1.582 | 0.456 | -0.284 | -0.216 |
| et | w_mcap | 0.0106 | 0.0164 | -0.0058 | -1.430 | 0.412 | -0.306 | -0.216 |
| et | w_sqrt_dolvol | 0.0093 | 0.0164 | -0.0071 | -2.746 | 0.368 | -0.482 | -0.216 |
| et | all_unweighted | 0.0204 | 0.0164 | +0.0040 | 1.106 | 0.603 | -0.026 | -0.216 |
| et | all_w_sqrt_mcap | 0.0144 | 0.0164 | -0.0020 | -0.594 | 0.471 | -0.289 | -0.216 |
| et | all_w_mcap | 0.0098 | 0.0164 | -0.0066 | -1.438 | 0.426 | -0.035 | -0.216 |
| lgbm | w_sqrt_mcap | 0.0111 | 0.0160 | -0.0050 | -1.968 | 0.382 | -0.428 | -0.167 |
| lgbm | w_mcap | 0.0074 | 0.0160 | -0.0087 | -1.999 | 0.456 | -0.371 | -0.167 |
| lgbm | w_sqrt_dolvol | 0.0065 | 0.0160 | -0.0096 | -2.934 | 0.426 | -0.455 | -0.167 |
| lgbm | all_unweighted | 0.0230 | 0.0160 | +0.0070 | 0.942 | 0.559 | 0.079 | -0.167 |
| lgbm | all_w_sqrt_mcap | 0.0116 | 0.0160 | -0.0045 | -1.314 | 0.441 | -0.303 | -0.167 |
| lgbm | all_w_mcap | 0.0078 | 0.0160 | -0.0082 | -1.857 | 0.485 | -0.273 | -0.167 |

Pre-registered rule:
- `w_sqrt_mcap`: families with paired t >= 2: 0; rule A (two families t >= 2): **False**; rule B (IC >= 0.037 and t >= 2): **False**
- `w_mcap`: families with paired t >= 2: 0; rule A (two families t >= 2): **False**; rule B (IC >= 0.037 and t >= 2): **False**
- `w_sqrt_dolvol`: families with paired t >= 2: 0; rule A (two families t >= 2): **False**; rule B (IC >= 0.037 and t >= 2): **False**
- `all_unweighted`: families with paired t >= 2: 0; rule A (two families t >= 2): **False**; rule B (IC >= 0.037 and t >= 2): **False**
- `all_w_sqrt_mcap`: families with paired t >= 2: 0; rule A (two families t >= 2): **False**; rule B (IC >= 0.037 and t >= 2): **False**
- `all_w_mcap`: families with paired t >= 2: 0; rule A (two families t >= 2): **False**; rule B (IC >= 0.037 and t >= 2): **False**

## Interpretation
**Weighting toward large, liquid names hurts.** Every universe-row weighted arm has a lower IC than the control (0.006-0.013 vs 0.016), and the dollar-volume weights are significantly worse (paired t -2.7 for ET, -2.9 for LightGBM). The all-stock weighted arms are also below the control except `all_unweighted`. The best fitted arm in this whole exercise is **LightGBM trained on all stocks, unweighted** (IC 0.023, t 1.83, D10-D1 +0.26%/month, `lc_t10` gross IR +0.08), and ET all-stock (IC 0.020, t 1.51), but their paired gain over the control is +0.004 / +0.007 IC (t 1.1 / 0.9), not significant, and IR is at most zero net. So: down-weighting small caps removes the information the model uses to rank large caps (an effective-sample-size loss, and the small-cap cross-section teaches the same factor relations with much larger dispersion), while the mild positive of unweighted all-stock training is unconfirmed. No arm meets the pre-registered rule (which requires positive paired t). Status: IMPLEMENTED_BUT_FAILED (weighting hurts; all-stock training is a possible, unconfirmed +0.005).

## Limitations
- Weights sum to a fixed total, so heavy weights on a few mega caps reduce the effective number of training months per fold; a capped or smoothed weight was not tried beyond sqrt.
- Hyper-parameter grids were not rescaled for 400k-row training sets.
- One seed; IC s.e. about 0.015-0.02.

## Follow-up
A follow-up on `all_unweighted` with a tuned grid and more seeds would be the only continuation worth running; given the composite's IC of 0.037 vs 0.023 at best, it is a low-priority one.
