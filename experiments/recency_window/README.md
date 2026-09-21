# Recency weighting, rolling windows and era-stacked models (RECENCY_WINDOW)

Implementation: `recency_window.py` (run: `.venv/bin/python experiments/recency_window/recency_window.py`, about 25 minutes).

## Objective
Test whether down-weighting or dropping old training months (exponential half-lives 24 / 48 / 96 months; rolling windows of 24 / 48 months) or averaging models trained on different eras (Numerai "incremental" style) improves the tradeable-universe IC of ExtraTrees / LightGBM over the expanding-window control.

## Hypothesis
The characteristic-return relation is non-stationary (the composite's IC fell from 0.054 in 2021-23 to 0.018 in 2024-26), so recent data should be worth more than old data, and models trained on different eras and averaged should be more robust to shift. Pre-registered rule (docs/NEW.md sec 4): adopt only if paired monthly universe-IC t >= 2 vs the control on two model families, or IC >= 0.037 with t >= 2.

## Research origin
docs/NEW.md sec 3.2 (Delphic Alpha: a 6-month window beat 12 and 18; arXiv 2512.23596: window length and model complexity must be chosen jointly, +14% OOS R^2, strongest in recessions; "our expanding window never varies the window - add an exponential half-life {24, 48, 96 months} to the validation grid") and Numerai deep incremental learning (arXiv 2303.07925: stack models trained on different eras; a two-layer stack beat single models under distribution shift).

## Implementation
`make_weights` (w = 0.5^(age/half_life), age = months from the row's target month to the start of the validation window, normalised to mean 1), `make_select` (keep only the last N months before the validation window) and `make_blockwise_et` (the same ExtraTrees fitted on 3 contiguous month blocks of the training window - optionally plus the full-window model - and averaged: a two-layer stack whose second layer is the mean; ET only) plug into the generalised walk-forward `run_wf`. Recency and window choices are fixed arms (not tuned); candidate hyper-parameters are still selected on validation IC.

## Experimental setup
All four target experiments use the same frozen harness (a verbatim copy of the data / rank-transform / walk-forward / LP / cost / performance code of `experiments/largecap/largecap.py`, embedded in the script so it runs on its own).

- **Data**: `fiam/chars_final_with_names.parquet`, 523,125 stock-months after dropping rows without a next-month return; 18 factors chosen by economic group in `experiments/largecap` (`FACTOR_GROUPS`), cross-sectionally median-filled and rank-transformed to [-1, 1] over all stocks each month.
- **Universe** (training, validation and scoring rows): price >= $5, market cap >= $2,000M, 126-day dollar volume >= $10M, both betas observed (about 1,206 stocks per month; 153,392 rows).
- **Walk-forward**: for test year Y = 2021..2026, train on target months before Jan Y-2, validate on Y-2..Y-1, refit annually; candidate hyper-parameters are chosen on **validation rank IC against the raw next-month excess return** (for every arm, whatever the fit target), never on test data.
- **Models**: Extra-Trees (largecap grid, 300 trees) and forest-style LightGBM (num_leaves 7, min_child_samples {500, 2000}, extra_trees, lambda 100, checkpoints at 100/200/300 trees). Seed 42 unless stated.
- **Scoring**: universe rank IC of the OOS prediction vs the raw next-month return (68 months, 2021-01..2026-08), decile spread, and the frozen LP portfolios `lc_t10` (10% one-way turnover cap, headline) and `lc_free`, gross and net of the assumed tiered costs.
- **Pre-registered rule** (docs/NEW.md sec 4): adopt an alternative target only if paired monthly universe-IC t >= 2 vs the control on **two** model families (rule A), or IC >= the composite's 0.037 with t >= 2 (rule B).
- **Sanity check that the harness is faithful**: the control arm `et__control` reproduces `experiments/largecap` `et` exactly: IC 0.0164 (published 0.016), lc_t10 gross IR -0.22 (published -0.22).

Arms: `control` (expanding window), `hl24`, `hl48`, `hl96`, `win24`, `win48`, and for ET `era_stack3`, `era_stack3_full`. Note: the first fold has only 47 training months, so a 48-month window equals the control there.

## Baseline
`*__control` (expanding window, unweighted, universe rows).

## Commands
```
.venv/bin/python experiments/recency_window/recency_window.py
```

## Results (seed 42, 68 test months)
| model__arm | universe rank IC | IC t | % months IC>0 | D10-D1 (%/m) | IR gross (lc_t10) | IR net (lc_t10) | IR gross (lc_free) | beta | max DD net % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| et__control | 0.0164 | 1.102 | 49 | -0.170 | -0.216 | -0.302 | -0.182 | -0.086 | -35 |
| et__hl24 | 0.0141 | 0.867 | 51 | -0.134 | -0.419 | -0.509 | -0.425 | -0.148 | -37 |
| et__hl48 | 0.0140 | 0.914 | 49 | -0.284 | -0.370 | -0.461 | -0.330 | -0.136 | -38 |
| et__hl96 | 0.0153 | 1.021 | 49 | -0.270 | -0.369 | -0.456 | -0.346 | -0.117 | -39 |
| et__win24 | 0.0075 | 0.358 | 50 | -0.367 | -0.313 | -0.400 | -0.674 | -0.167 | -28 |
| et__win48 | 0.0130 | 0.896 | 51 | -0.135 | -0.220 | -0.314 | -0.181 | -0.095 | -31 |
| et__era_stack3 | 0.0196 | 1.341 | 47 | -0.202 | -0.241 | -0.327 | -0.226 | -0.101 | -38 |
| et__era_stack3_full | 0.0190 | 1.284 | 47 | -0.287 | -0.199 | -0.284 | -0.239 | -0.097 | -34 |
| lgbm__control | 0.0160 | 1.044 | 49 | -0.175 | -0.167 | -0.251 | -0.264 | -0.122 | -33 |
| lgbm__hl24 | 0.0079 | 0.468 | 49 | -0.438 | -0.487 | -0.576 | -0.452 | -0.135 | -40 |
| lgbm__hl48 | 0.0121 | 0.775 | 49 | -0.247 | -0.342 | -0.434 | -0.481 | -0.164 | -34 |
| lgbm__hl96 | 0.0135 | 0.856 | 49 | -0.277 | -0.543 | -0.630 | -0.371 | -0.094 | -41 |
| lgbm__win24 | 0.0065 | 0.313 | 50 | -0.605 | -0.556 | -0.638 | -0.733 | -0.107 | -39 |
| lgbm__win48 | 0.0111 | 0.741 | 51 | 0.094 | -0.383 | -0.475 | -0.294 | -0.100 | -33 |

Paired monthly IC vs the same-family control:

| model | arm | ic | ic_control | IC diff | paired t | share of months better | ir_gross | ir_gross_control |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| et | hl24 | 0.0141 | 0.0164 | -0.0023 | -0.362 | 0.471 | -0.419 | -0.216 |
| et | hl48 | 0.0140 | 0.0164 | -0.0024 | -0.671 | 0.441 | -0.370 | -0.216 |
| et | hl96 | 0.0153 | 0.0164 | -0.0011 | -0.493 | 0.456 | -0.369 | -0.216 |
| et | win24 | 0.0075 | 0.0164 | -0.0089 | -0.668 | 0.456 | -0.313 | -0.216 |
| et | win48 | 0.0130 | 0.0164 | -0.0034 | -0.663 | 0.309 | -0.220 | -0.216 |
| et | era_stack3 | 0.0196 | 0.0164 | +0.0032 | 1.435 | 0.588 | -0.241 | -0.216 |
| et | era_stack3_full | 0.0190 | 0.0164 | +0.0026 | 1.547 | 0.603 | -0.199 | -0.216 |
| lgbm | hl24 | 0.0079 | 0.0160 | -0.0081 | -1.076 | 0.412 | -0.487 | -0.167 |
| lgbm | hl48 | 0.0121 | 0.0160 | -0.0040 | -0.866 | 0.456 | -0.342 | -0.167 |
| lgbm | hl96 | 0.0135 | 0.0160 | -0.0026 | -0.898 | 0.471 | -0.543 | -0.167 |
| lgbm | win24 | 0.0065 | 0.0160 | -0.0095 | -0.716 | 0.441 | -0.556 | -0.167 |
| lgbm | win48 | 0.0111 | 0.0160 | -0.0049 | -0.955 | 0.338 | -0.383 | -0.167 |

Pre-registered rule:
- `hl24`: families with paired t >= 2: 0; rule A (two families t >= 2): **False**; rule B (IC >= 0.037 and t >= 2): **False**
- `hl48`: families with paired t >= 2: 0; rule A (two families t >= 2): **False**; rule B (IC >= 0.037 and t >= 2): **False**
- `hl96`: families with paired t >= 2: 0; rule A (two families t >= 2): **False**; rule B (IC >= 0.037 and t >= 2): **False**
- `win24`: families with paired t >= 2: 0; rule A (two families t >= 2): **False**; rule B (IC >= 0.037 and t >= 2): **False**
- `win48`: families with paired t >= 2: 0; rule A (two families t >= 2): **False**; rule B (IC >= 0.037 and t >= 2): **False**
- `era_stack3`: families with paired t >= 2: 0; rule A (two families t >= 2): **False**; rule B (IC >= 0.037 and t >= 2): **False**
- `era_stack3_full`: families with paired t >= 2: 0; rule A (two families t >= 2): **False**; rule B (IC >= 0.037 and t >= 2): **False**

## Interpretation
**Recency does not help; if anything it hurts.** Every half-life and window arm has a *lower* IC than the expanding-window control in both families (ET 0.014-0.015 vs 0.016; LightGBM 0.008-0.013 vs 0.016; 24-month windows are worst, IC 0.007 in both), and none is significant (paired t -0.4 to -1.1). Shorter memory removes training rows without removing the non-stationarity, so the fit gets noisier rather than more current. The only positive arms are the ET era-stacks (`era_stack3` IC 0.020, `era_stack3_full` 0.019; +0.003 IC, paired t 1.4 / 1.5), the same order as the rank-target and all-stock effects, and their portfolios are still negative (gross IR -0.24 / -0.20). **No arm meets the pre-registered rule.** The Delphic / arXiv 2512.23596 window results (a short window helps in recessions) are not visible in this 2021-26 large-cap universe. Status: IMPLEMENTED_BUT_FAILED.

## Limitations
- Half-lives and windows are fixed arms, not selected jointly with model complexity as in the paper; the validation window (2 years) is too short to choose them robustly.
- 2015-onwards training data; a 96-month half-life is close to the whole sample at the last folds.
- Single seed; IC s.e. about 0.015-0.02.

## Follow-up
None. Era-stacking (+0.003) is the only non-negative arm and is not significant.
