# Decile classification instead of return regression (TARGET_DECILE_CLASS)

Implementation: `target_decile_class.py` (run: `.venv/bin/python experiments/target_decile_class/target_decile_class.py`, about 12 minutes).

## Objective
Test whether predicting the within-month decile of next-month excess return (10 classes) and ranking by the probability-weighted expected decile beats regressing the raw return, in ExtraTrees and LightGBM on the 18 factors.

## Hypothesis
Classification is robust to return outliers and optimises what a long/short book needs (ordering), so its universe rank IC exceeds the regression control in at least two families. Ablations: P(top decile) - P(bottom decile) as the score, and the hard argmax class (which the source says should be worse).

## Research origin
docs/NEW.md sec 3.1b (Bai & Pukthuanthong, arXiv 2108.02283: matched models, classification value-weighted Sharpe 2.08 vs 1.39; Quantitativo "probabilistic momentum": expected value over class probabilities, not argmax).

## Implementation
Labels: within-month decile 0..9 on universe rows. `make_fit_et_clf` (ExtraTreesClassifier, same grid) and `make_fit_lgbm_clf` (multiclass LGBMClassifier, same settings, 300 rounds) return `predict_proba`, converted by `_score_from_proba` to `expdec` (P @ class index), `top_minus_bottom` or `argmax` (+ tiny tie-break). Candidate selection: validation rank IC of the resulting score against the raw return.

## Experimental setup
All four target experiments use the same frozen harness (a verbatim copy of the data / rank-transform / walk-forward / LP / cost / performance code of `experiments/largecap/largecap.py`, embedded in the script so it runs on its own).

- **Data**: `fiam/chars_final_with_names.parquet`, 523,125 stock-months after dropping rows without a next-month return; 18 factors chosen by economic group in `experiments/largecap` (`FACTOR_GROUPS`), cross-sectionally median-filled and rank-transformed to [-1, 1] over all stocks each month.
- **Universe** (training, validation and scoring rows): price >= $5, market cap >= $2,000M, 126-day dollar volume >= $10M, both betas observed (about 1,206 stocks per month; 153,392 rows).
- **Walk-forward**: for test year Y = 2021..2026, train on target months before Jan Y-2, validate on Y-2..Y-1, refit annually; candidate hyper-parameters are chosen on **validation rank IC against the raw next-month excess return** (for every arm, whatever the fit target), never on test data.
- **Models**: Extra-Trees (largecap grid, 300 trees) and forest-style LightGBM (num_leaves 7, min_child_samples {500, 2000}, extra_trees, lambda 100, checkpoints at 100/200/300 trees). Seed 42 unless stated.
- **Scoring**: universe rank IC of the OOS prediction vs the raw next-month return (68 months, 2021-01..2026-08), decile spread, and the frozen LP portfolios `lc_t10` (10% one-way turnover cap, headline) and `lc_free`, gross and net of the assumed tiered costs.
- **Pre-registered rule** (docs/NEW.md sec 4): adopt an alternative target only if paired monthly universe-IC t >= 2 vs the control on **two** model families (rule A), or IC >= the composite's 0.037 with t >= 2 (rule B).
- **Sanity check that the harness is faithful**: the control arm `et__control` reproduces `experiments/largecap` `et` exactly: IC 0.0164 (published 0.016), lc_t10 gross IR -0.22 (published -0.22).

## Baseline
Same family regression control (`*__control`, winsorised raw target); the composite (IC 0.037) as the practical bar.

## Commands
```
.venv/bin/python experiments/target_decile_class/target_decile_class.py
```

## Results (seed 42, 68 test months)
| model__arm | universe rank IC | IC t | % months IC>0 | D10-D1 (%/m) | IR gross (lc_t10) | IR net (lc_t10) | IR gross (lc_free) | beta | max DD net % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| et__control | 0.0164 | 1.102 | 49 | -0.170 | -0.216 | -0.302 | -0.182 | -0.086 | -35 |
| et__clf_expdec | 0.0244 | 1.458 | 62 | -0.125 | -0.351 | -0.430 | -0.168 | -0.063 | -40 |
| et__clf_top_minus_bottom | 0.0111 | 0.752 | 51 | -0.440 | -0.495 | -0.580 | -0.346 | -0.217 | -39 |
| et__clf_argmax | 0.0147 | 1.261 | 50 | -0.264 | -0.483 | -0.580 | -0.318 | -0.094 | -35 |
| lgbm__control | 0.0160 | 1.044 | 49 | -0.175 | -0.167 | -0.251 | -0.264 | -0.122 | -33 |
| lgbm__clf_expdec | 0.0212 | 1.369 | 59 | -0.253 | -0.325 | -0.404 | -0.185 | -0.069 | -43 |
| lgbm__clf_top_minus_bottom | 0.0059 | 0.456 | 51 | -0.622 | -0.451 | -0.542 | -0.386 | -0.142 | -37 |
| lgbm__clf_argmax | 0.0143 | 1.393 | 56 | -0.335 | -0.200 | -0.293 | -0.390 | -0.075 | -34 |

Paired monthly IC vs the same-family control:

| model | arm | ic | ic_control | IC diff | paired t | share of months better | ir_gross | ir_gross_control |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| et | clf_expdec | 0.0244 | 0.0164 | +0.0080 | 1.039 | 0.529 | -0.351 | -0.216 |
| et | clf_top_minus_bottom | 0.0111 | 0.0164 | -0.0052 | -0.801 | 0.456 | -0.495 | -0.216 |
| et | clf_argmax | 0.0147 | 0.0164 | -0.0017 | -0.244 | 0.529 | -0.483 | -0.216 |
| lgbm | clf_expdec | 0.0212 | 0.0160 | +0.0051 | 0.810 | 0.544 | -0.325 | -0.167 |
| lgbm | clf_top_minus_bottom | 0.0059 | 0.0160 | -0.0102 | -1.244 | 0.412 | -0.451 | -0.167 |
| lgbm | clf_argmax | 0.0143 | 0.0160 | -0.0017 | -0.190 | 0.544 | -0.200 | -0.167 |

Pre-registered rule:
- `clf_expdec`: families with paired t >= 2: 0; rule A (two families t >= 2): **False**; rule B (IC >= 0.037 and t >= 2): **False**
- `clf_top_minus_bottom`: families with paired t >= 2: 0; rule A (two families t >= 2): **False**; rule B (IC >= 0.037 and t >= 2): **False**
- `clf_argmax`: families with paired t >= 2: 0; rule A (two families t >= 2): **False**; rule B (IC >= 0.037 and t >= 2): **False**

## Interpretation
The expected-decile classifier has the highest IC of the arms (ET 0.024, LGBM 0.021 vs 0.016; % of months with positive IC rises from 49% to 59-62%) but its paired t is only 1.0 and 0.8, and its portfolios are worse than the control (gross IR -0.32 to -0.35 vs -0.17 to -0.22). `top_minus_bottom` and `argmax` are worse than or equal to the control, which agrees with the source's point that hard/extreme-class scores throw information away (argmax prediction autocorrelation 0.55-0.58: unstable rankings). **Not adopted: neither pre-registered rule is met.** The Bai-Pukthuanthong Sharpe gain (2.08 vs 1.39) is not reproduced on the tradeable large-cap universe. Status: IMPLEMENTED_BUT_FAILED.

## Limitations
- Their result is on value-weighted decile portfolios on all US stocks; ours is a beta/sector-constrained LP on a $2B universe where the signal is much weaker.
- Ten classes with ~100 training rows per class-month cell make per-class probabilities noisy; fewer classes (3-5) were not tried.
- Single seed; IC s.e. about 0.015-0.02.

## Follow-up
None planned; expected-decile is the only variant with any positive sign and it adds +0.005 to +0.008 IC (the same order as rank targets), so it is not independent evidence.
