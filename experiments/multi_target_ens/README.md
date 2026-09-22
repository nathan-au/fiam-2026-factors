# Multi-target, multi-horizon ensemble with a ridge meta-model (MULTI_TARGET_ENS)

Implementation: `multi_target_ens.py` (run: `.venv/bin/python experiments/multi_target_ens/multi_target_ens.py`, about 11 minutes).

## Objective
Test the Numerai-style recipe: models trained on different targets (1-month raw, Gaussian-rank, sector-demeaned, 3-month, 6-month excess return), each averaged over 3 seeds, combined by an equal-weight rank average or a ridge meta-model fitted on the validation rows, optionally with 50% feature neutralisation.

## Hypothesis
Averaging models trained on different targets diversifies target-specific noise, so the ensemble (especially the ridge-weighted and the partially feature-neutralised one) beats the single winsorised 1-month target on universe rank IC; the longer-horizon members should give a slower, lower-turnover signal. Pre-registered kill rule (docs/RESEARCH.md Part I sec 2 #9): no paired-IC gain vs the single-target control (paired monthly t < 1).

## Research origin
docs/RESEARCH.md Part I sec 2 #9 and sec 3.1e (Numerai LightGBM ensemble: one model per target - 20d and 60d horizons, different neutralisations - 3 seeds each, ridge meta-model on validation eras, then partial feature neutralisation; feature exposure 0.25 -> 0.17 and max drawdown halved; 3-6 month targets as the standard route to lower turnover).

## Implementation
`horizon_returns` compounds the panel's monthly `ret_exc` over t+1..t+h (h = 1, 3, 6). **Leakage guard**: a training row is used only if its whole h-month forward window ends before the validation window starts (`month index <= validation_start - h`); validation candidate selection is always against the raw 1-month return. `fit_et_seeds`: the harness's ExtraTrees grid, each candidate the average of 3 seeds (42, 43, 44). Ensembles: `ens_equal` (mean of within-month rank-z predictions), `ens_ridge` (`RidgeCV`, alphas 1..1e4, no intercept, fitted on the validation rows' rank-z predictions -> rank-z raw return), `ens_ridge_neutral` (`neutralise`: subtract 50% of the within-month OLS projection of the ridge score on the 18 factors). Control = `t_raw1` (the harness's winsorised raw 1-month target, 3-seed ET).

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
`t_raw1`: single winsorised 1-month raw target, same 3-seed ET grid (IC 0.0165; the single-seed `et` control of the target experiments is 0.0164).

## Commands
```
.venv/bin/python experiments/multi_target_ens/multi_target_ens.py
```

## Results (68 test months)
| arm | IC | IC t | IC diff vs t_raw1 | paired t | pred rank autocorr | IR gross (lc_t10) | IR net (lc_t10) | one-way turnover lc_free % | max DD % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| t_raw1 | 0.0165 | 1.110 | +0.0000 |  | 0.80 | -0.283 | -0.369 | 46 | -37 |
| t_gauss1 | 0.0212 | 1.383 | +0.0047 | +1.17 | 0.84 | -0.330 | -0.411 | 40 | -37 |
| t_sector1 | 0.0198 | 1.248 | +0.0032 | +0.68 | 0.80 | -0.190 | -0.277 | 46 | -37 |
| t_raw3 | 0.0132 | 0.869 | -0.0033 | -0.80 | 0.87 | -0.347 | -0.432 | 36 | -35 |
| t_raw6 | 0.0134 | 0.872 | -0.0031 | -0.43 | 0.89 | -0.231 | -0.313 | 36 | -39 |
| ens_equal | 0.0168 | 1.080 | +0.0003 | +0.11 | 0.85 | -0.357 | -0.447 | 40 | -33 |
| ens_ridge | -0.0025 | -0.191 | -0.0190 | -1.06 | 0.74 | -1.180 | -1.294 | 53 | -35 |
| ens_ridge_neutral | -0.0031 | -0.316 | -0.0197 | -1.24 | 0.68 | -1.157 | -1.282 | 56 | -30 |

Validation IC per target and for the ensembles in each fold (validation = the two years before the test year; `ridge val IC` is in-sample for the ridge fit):

| test year | val IC t_raw1 | val IC t_gauss1 | val IC t_sector1 | val IC t_raw3 | val IC t_raw6 | ens_equal val IC | ridge val IC (in-sample) | ridge alpha |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2021 | -0.006 | 0.005 | -0.015 | 0.001 | 0.014 | -0.000 | 0.062 | 100.000 |
| 2022 | 0.042 | 0.041 | 0.026 | 0.040 | 0.032 | 0.038 | 0.052 | 100.000 |
| 2023 | -0.033 | -0.023 | -0.032 | -0.040 | -0.032 | -0.034 | 0.051 | 10.000 |
| 2024 | 0.023 | 0.027 | 0.024 | 0.001 | -0.005 | 0.016 | 0.042 | 100.000 |
| 2025 | 0.030 | 0.037 | 0.026 | 0.031 | 0.031 | 0.032 | 0.045 | 100.000 |
| 2026 | 0.012 | 0.016 | 0.004 | 0.011 | 0.007 | 0.010 | 0.029 | 1000.000 |

Ridge meta-model weights by fold (rank-z predictions of each target):

| test year | t_raw1 | t_gauss1 | t_sector1 | t_raw3 | t_raw6 |
|---|---:|---:|---:|---:|---:|
| 2021 | -0.021 | +0.093 | -0.108 | -0.048 | +0.085 |
| 2022 | +0.035 | +0.028 | -0.047 | +0.081 | -0.058 |
| 2023 | +0.060 | +0.025 | -0.010 | -0.182 | +0.079 |
| 2024 | +0.093 | +0.010 | -0.007 | -0.083 | -0.001 |
| 2025 | -0.010 | +0.093 | -0.069 | -0.003 | +0.027 |
| 2026 | +0.008 | +0.044 | -0.047 | +0.014 | -0.008 |

## Interpretation
**The ensembles do not beat the single target.** The equal-weight ensemble is identical to the control (IC 0.0168 vs 0.0165; paired t 0.11). The ridge meta-model is *worse*: IC -0.0025 (paired t -1.06) and its partially feature-neutralised version -0.0031 (-1.24), with strongly negative gross IR (-1.18 / -1.16). The reason is visible in the fold table: the ridge fit reaches a validation IC of 0.03-0.06 (in-sample) but its weights are unstable and often large and negative (e.g. -0.18 on the 3-month target in 2023, -0.11 on the sector-demeaned target in 2021), i.e. it fits the validation eras' noise - the 2-year validation window contains only 24 cross-sections. The 3- and 6-month targets are individually worse than the 1-month target (IC 0.013 vs 0.0165; paired t -0.8 / -0.4), and the Gaussian-rank 1-month target is the best single member (+0.0047 IC, t 1.17, the same small rank-target effect as in `experiments/target_rank`). Longer horizons *did* make the signal slower, as the source says (prediction rank autocorrelation 0.875 / 0.888 for the 3- / 6-month targets vs 0.798, and one-way turnover of the uncapped LP 36% vs 46% of gross), but at a cost in IC (0.013 vs 0.0165), so the trade-off is not favourable; and the partial feature neutralisation did not rescue the ridge ensemble. **Kill rule fires for the ensembles (paired t < 1 or negative).** Status: IMPLEMENTED_BUT_FAILED.

## Limitations
- The validation window (24 monthly cross-sections) is the meta-model's only training data; Numerai fits on hundreds of eras.
- ET with a fixed grid (not LightGBM with the Numerai regularisation of 15 leaves / 2000 rows per leaf / L2 = 10); only 5 targets rather than 40.
- The ridge meta-model was not constrained (non-negative weights would be safer); an unconstrained fit was the specified recipe.
- One 68-month path.

## Follow-up
A non-negative or equal-weight ensemble is the only sensible variant and it equals the single target, so no follow-up.
