# Composite as prior: ML may only add what survives validation (COMPOSITE_PRIOR)

Implementation: `composite_prior.py` (run: `.venv/bin/python experiments/composite_prior/composite_prior.py`, about 3 minutes).

## Objective
Test whether a model that starts from the frozen composite and may move away from it only when validation IC says so (LightGBM with `init_score`, ExtraTrees on the composite's residual, ridge shrunk toward the composite's weights, and an orthogonal-residual variant) can beat the composite on the >= $2B universe. "No ML" is always a candidate, so the arm can always fall back to the composite.

## Hypothesis
Every fitted model on these 18 factors lost to the no-fit composite (ET IC 0.016 vs 0.037). If the factors hold any structure beyond the composite's linear-in-rank form, a model anchored to the composite should beat it; if not, validation should pick "no ML". Reddit H2: ML trained on `y - a*composite` and orthogonalised to the composite can only produce what the shared core does not.
Pre-registered kill rules: ML-on-top adds < +0.005 paired IC (docs/NEW.md sec 2 #2); orthogonal-residual IC < 0.01 (docs/REDDIT_RESEARCH.md H2).

## Research origin
docs/NEW.md sec 2 #2 and sec 3.5 (composite-as-prior: LightGBM `init_score`, <= 50 shallow trees, or ridge toward composite group weights); docs/REDDIT_RESEARCH.md sec 2.12 (r/quant top advice "encode your heuristic as a feature", 60 upvotes) and H2.

## Implementation
Per fold (train < Jan Y-2, validate Y-2..Y-1, test Y), on universe rows: `a` = slope of the month-demeaned winsorised return on the composite score, **floored at +0.005** (see the bug note below). Arms: `lgbm_init` (LGBM regression on the demeaned return with `init_score = a*composite`, num_leaves 4 or 7, up to 50 rounds at lr 0.05; candidates are 0, 10, 25 or 50 trees, chosen on validation rank IC; 0 trees = the composite); `et_resid` (prediction = a*composite + s*ExtraTrees(residual y - a*composite), s in {0, 0.25, 0.5, 1}, grid chosen on validation IC); `ridge_prior` (ridge on the composite's own 18 universe-ranked inputs shrunk toward the composite's weights, lambda in {1e2..1e5, inf}); `orth_alone` and `comp_plus_orth` (H2: the ET-on-residual output with the composite direction removed each month, alone and added 1:1 in z-units to the composite). `*_forced*` arms take the ML candidate regardless of validation.

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
The frozen composite itself (`comp`): IC 0.0366, `lc_t10` gross IR 0.615 / net 0.541.

## Commands
```
.venv/bin/python experiments/composite_prior/composite_prior.py
```

## Results (68 test months)
| arm | ic | ic_t | IC diff vs comp | paired t | resid IC | resid t | corr w/ comp | IC small | IC mid | IC large | IR gross | IR net | max DD % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| comp | 0.0366 | 1.91 | +0.0000 |  |  |  |  | 0.0512 | 0.0359 | 0.0145 | 0.615 | 0.541 | -19 |
| lgbm_init | 0.0230 | 1.34 | -0.0137 | -1.547 | -0.0162 | -0.880 | 0.732 | 0.0414 | 0.0240 | -0.0064 | 0.111 | 0.015 | -25 |
| lgbm_init_forced50 | 0.0255 | 1.78 | -0.0111 | -0.666 | +0.0022 | 0.167 | 0.520 | 0.0359 | 0.0223 | 0.0123 | -0.040 | -0.120 | -29 |
| et_resid | 0.0232 | 1.36 | -0.0134 | -1.385 | -0.0085 | -0.538 | 0.747 | 0.0430 | 0.0194 | -0.0048 | 0.080 | -0.018 | -25 |
| et_resid_forced_s1 | 0.0254 | 1.70 | -0.0113 | -0.704 | +0.0017 | 0.119 | 0.552 | 0.0382 | 0.0194 | 0.0102 | 0.001 | -0.084 | -30 |
| ridge_prior | 0.0185 | 1.08 | -0.0181 | -1.746 | -0.0292 | -1.411 | 0.631 | 0.0365 | 0.0178 | -0.0112 | -0.293 | -0.386 | -36 |
| orth_alone | 0.0005 | 0.03 | -0.0361 | -1.337 | +0.0005 | 0.033 | -0.071 | 0.0089 | -0.0108 | -0.0053 | -0.864 | -0.930 | -58 |
| comp_plus_orth | 0.0261 | 1.77 | -0.0106 | -0.731 | +0.0005 | 0.033 | 0.638 | 0.0406 | 0.0181 | 0.0121 | -0.310 | -0.390 | -41 |

Validation choices per fold (`slope_a_used` = floored slope; `slope_a_fitted` = raw fitted slope):

| test_year | et_resid_shrink | lgbm_n_trees | ridge_lambda | slope_a_used | slope_a_fitted |
|---|---:|---:|---:|---:|---:|
| 2021 | 1.00 | 50 | 100 | 0.0158 | +0.0158 |
| 2022 | 1.00 | 50 | 1000 | 0.0104 | +0.0104 |
| 2023 | 0.00 | 0 | inf | 0.0050 | -0.0004 |
| 2024 | 0.25 | 0 | inf | 0.0064 | +0.0064 |
| 2025 | 1.00 | 50 | 10000 | 0.0113 | +0.0113 |
| 2026 | 0.00 | 0 | inf | 0.0088 | +0.0088 |

**Implementation bug found and fixed (kept for the record).** The first run fitted `a` without a floor. In the 2023 fold the fitted slope on the 2015-2020 training data was -0.0004, so the "0 trees" candidate `a*composite` **flipped the composite's sign** (validation IC -0.077 instead of +0.077) and the arm chose a deviating model instead. The fix floors `a` at +0.005 (the composite is a no-fit rule with a pre-specified sign, so its scale must stay positive); the fixed run correctly falls back to the composite in 2023. The invalid first-run summary is kept in `output/first_run_invalid_negative_scale/`:

| arm | ic | d_ic_vs_comp | paired_t_vs_comp | ir_gross | ir_net |
|---|---:|---:|---:|---:|---:|
| comp | 0.0366 | +0.0000 |  | 0.615 | 0.541 |
| lgbm_init | 0.0296 | -0.0071 | -0.457 | -0.115 | -0.208 |
| lgbm_init_forced50 | 0.0254 | -0.0113 | -0.633 | -0.033 | -0.117 |
| et_resid | 0.0290 | -0.0076 | -0.454 | -0.197 | -0.290 |
| et_resid_forced_s1 | 0.0266 | -0.0100 | -0.554 | -0.149 | -0.234 |
| ridge_prior | 0.0217 | -0.0149 | -0.934 | -0.544 | -0.636 |
| orth_alone | -0.0004 | -0.0371 | -1.371 | -0.858 | -0.924 |
| comp_plus_orth | 0.0256 | -0.0111 | -0.765 | -0.343 | -0.424 |

The conclusion did not change: after the fix every ML arm is still below the composite (the fix moved the arms' IC by between -0.007 and +0.001).

## Interpretation
Even with "no ML" as a candidate and shrinkage grids, every ML-on-top arm is **worse** than the composite it started from: paired IC -0.011 to -0.018 (t -0.7 to -1.7), gross IR -0.29 to +0.11 vs +0.61. Validation-gated deviation does not work because validation IC is a poor guide to test IC here: in 3 of 6 folds (2021, 2022, 2025) the ET-on-residual variant was chosen because its validation IC beat the composite's (by +0.045, +0.041 and +0.006 IC respectively, from the run log), and the resulting arm is worse than the composite out of sample overall; in the folds where validation IC preferred the composite (2023, 2026, and 2024 for LightGBM/ridge) it stayed put. The H2 orthogonal residual has essentially zero IC (0.0005 alone; the residual IC of every ML arm after removing the composite is between -0.029 and +0.002, below the 0.01 bar), i.e. **the 18 factors contain no learnable structure beyond the composite that survives out of sample**, which fits `experiments/composite_overlap` (the composite's information is mostly value/investment) and `experiments/factor_filter`. Both kill rules fire. Status: IMPLEMENTED_BUT_FAILED.

## Limitations
- Training window 2015 onwards (72 months at the first fold, 132 at the last) is short for fitting any interaction structure; the composite is the same information without a fit.
- The scale `a` is estimated on a window in which the composite's slope was tiny or unstable, which is the reason for the floor; a different scale rule could change the ML arms' weight against the prior.
- A single seed.

## Follow-up
None on these 18 factors. A prior-based approach could only add value with genuinely new information (external data blocks) fed to the ML residual, not more model capacity on the same inputs.
