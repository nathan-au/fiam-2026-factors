# Size-/industry-demeaned and factor-residual training targets (TARGET_NEUTRAL)

Implementation: `target_neutral.py` (run: `.venv/bin/python experiments/target_neutral/target_neutral.py`, about 6 minutes).

## Objective
Test whether training on next-month return with the size-tercile median, the GICS-sector median, both, or a Numerai-style factor residual removed improves the tradeable-universe IC (against the raw return) of Extra-Trees / LightGBM on the 18 factors.

## Hypothesis
Fitted models learned size / ivol / sector / momentum structure that does not persist in the tradeable universe. Removing that structure from the target forces the model to look for other cross-sectional information and raises the raw-return universe IC (Howard's "target regularisation"; Numerai's factor-neutral target).

## Research origin
docs/RESEARCH.md Part I sec 3.1c (Howard, *Less is More?*: subtracting the size-group median from the target gives most of the size-bucketed ensemble's gain) and sec 3.1d (Numerai Signals target neutral to country / sector / beta / momentum / size); consistent with docs/RESEARCH.md Part II sec 2.5.

## Implementation
`build_targets`, all within month on universe rows: `demean_size` = y minus the median of y within the market-cap tercile; `demean_sector` = y minus the 2-digit GICS sector median; `demean_size_sector` = y minus the (size x sector) median (falls back to the sector median for cells with < 5 stocks); `resid_factor` = OLS residual of y on log market cap, betabab, ivol, 12-1 momentum rank and sector dummies, fitted cross-sectionally each month with exposures at the characteristic month. Fit targets are winsorised 1/99 at the training fold; candidate selection and scoring use the raw return. IC against the residual return and the sector-demeaned return is also reported (`ic_vs_*`, scoring only).

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
Same model, features and universe with the winsorised raw target (`*__control`); the composite (IC 0.037, gross IR 0.61) as the practical bar.

## Commands
```
.venv/bin/python experiments/target_neutral/target_neutral.py
```

## Results (seed 42, 68 test months)
| model__arm | universe rank IC | IC t | % months IC>0 | D10-D1 (%/m) | IR gross (lc_t10) | IR net (lc_t10) | IR gross (lc_free) | beta | max DD net % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| et__control | 0.0164 | 1.102 | 49 | -0.170 | -0.216 | -0.302 | -0.182 | -0.086 | -35 |
| et__demean_size | 0.0119 | 0.895 | 50 | -0.356 | -0.514 | -0.595 | -0.303 | -0.102 | -44 |
| et__demean_sector | 0.0206 | 1.313 | 49 | -0.065 | -0.274 | -0.359 | -0.089 | -0.079 | -40 |
| et__demean_size_sector | 0.0159 | 1.145 | 54 | 0.012 | -0.281 | -0.365 | -0.071 | -0.123 | -39 |
| et__resid_factor | 0.0211 | 1.597 | 57 | 0.284 | -0.113 | -0.201 | -0.080 | -0.089 | -39 |
| lgbm__control | 0.0160 | 1.044 | 49 | -0.175 | -0.167 | -0.251 | -0.264 | -0.122 | -33 |
| lgbm__demean_size | 0.0121 | 0.876 | 53 | -0.251 | -0.447 | -0.528 | -0.270 | -0.119 | -42 |
| lgbm__demean_sector | 0.0210 | 1.326 | 50 | 0.139 | -0.123 | -0.208 | -0.073 | -0.089 | -38 |
| lgbm__demean_size_sector | 0.0120 | 0.903 | 53 | -0.108 | -0.161 | -0.253 | -0.145 | -0.104 | -32 |
| lgbm__resid_factor | 0.0183 | 1.352 | 56 | 0.157 | -0.043 | -0.138 | 0.044 | -0.070 | -34 |

IC against the scoring-only alternative targets (control): 0.012 vs the factor-residual return and 0.023-0.024 vs the sector-demeaned return; per-arm values are in `results.json` (`ic_vs_resid_factor`, `ic_vs_demean_sector`) and `summary.csv`. Every arm has a higher IC against those alternative targets than against the raw return, as expected (they remove variance the model cannot predict), but the ordering of arms is the same.

Paired monthly IC vs the same-family control:

| model | arm | ic | ic_control | IC diff | paired t | share of months better | ir_gross | ir_gross_control |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| et | demean_size | 0.0119 | 0.0164 | -0.0045 | -0.655 | 0.485 | -0.514 | -0.216 |
| et | demean_sector | 0.0206 | 0.0164 | +0.0042 | 0.923 | 0.544 | -0.274 | -0.216 |
| et | demean_size_sector | 0.0159 | 0.0164 | -0.0005 | -0.067 | 0.485 | -0.281 | -0.216 |
| et | resid_factor | 0.0211 | 0.0164 | +0.0047 | 0.607 | 0.529 | -0.113 | -0.216 |
| lgbm | demean_size | 0.0121 | 0.0160 | -0.0040 | -0.540 | 0.456 | -0.447 | -0.167 |
| lgbm | demean_sector | 0.0210 | 0.0160 | +0.0050 | 0.918 | 0.529 | -0.123 | -0.167 |
| lgbm | demean_size_sector | 0.0120 | 0.0160 | -0.0040 | -0.559 | 0.456 | -0.161 | -0.167 |
| lgbm | resid_factor | 0.0183 | 0.0160 | +0.0023 | 0.287 | 0.500 | -0.043 | -0.167 |

Pre-registered rule:
- `demean_size`: families with paired t >= 2: 0; rule A (two families t >= 2): **False**; rule B (IC >= 0.037 and t >= 2): **False**
- `demean_sector`: families with paired t >= 2: 0; rule A (two families t >= 2): **False**; rule B (IC >= 0.037 and t >= 2): **False**
- `demean_size_sector`: families with paired t >= 2: 0; rule A (two families t >= 2): **False**; rule B (IC >= 0.037 and t >= 2): **False**
- `resid_factor`: families with paired t >= 2: 0; rule A (two families t >= 2): **False**; rule B (IC >= 0.037 and t >= 2): **False**

## Interpretation
`demean_sector` and `resid_factor` give a small IC lift in both families (+0.002 to +0.005, paired t 0.3-0.9), and `resid_factor` has the least-bad portfolios (LightGBM `lc_t10` gross IR -0.04, ET -0.11), but none of it is significant and every book is still at or below zero. **Removing the size-tercile median hurts** (IC 0.012 vs 0.016, gross IR -0.45 to -0.51): inside a >= $2B universe, size carries information the model was using; Howard's size-median result does not transfer to a large-cap-only universe. No arm meets the pre-registered rule. Status: IMPLEMENTED_BUT_FAILED.

## Limitations
- The residual target regresses on exposures that the LP later neutralises anyway, which may be why `resid_factor` books are slightly better; it is a cheap, six-exposure version of Numerai's 200-column neutraliser.
- Single path, 68 months, IR s.e. about 0.46: IR differences of 0.1-0.2 are noise.
- Size terciles inside a $2B floor are narrow in market-cap terms, so "size neutral" here is weaker than in the paper's all-cap setting.

## Follow-up
If a factor-residual target is used at all, use it with the composite as prior (`experiments/composite_prior`); do not spend more time on size-median removal in a large-cap universe.
