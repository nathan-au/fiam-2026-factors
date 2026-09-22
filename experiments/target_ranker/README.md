# Learning-to-rank objectives (TARGET_RANKER)

Implementation: `target_ranker.py` (run: `.venv/bin/python experiments/target_ranker/target_ranker.py`, about 30 minutes; the custom-objective arm dominates).

## Objective
Test whether ranking losses over monthly query groups (XGBRanker `rank:ndcg` / `rank:pairwise`, LightGBM `lambdarank`, and a rank-IC-weighted pairwise objective) beat squared-error regression on universe rank IC.

## Hypothesis
At IC ~0.03 (low signal-to-noise), fitting the cross-sectional ordering directly and ignoring return magnitudes generalises better than regression of the raw return.

## Research origin
docs/RESEARCH.md Part I sec 3.1f (Quantitativo LambdaMART on the Russell 3000, Sharpe 1.62; LambdaRankIC, arXiv 2605.00501: a custom XGBoost objective optimising Rank IC, best under low SNR and heavy tails).

## Implementation
`make_fit_xgb_rank(objective)` (xgboost.XGBRanker, monthly query groups, decile labels 0..9, linear NDCG gain), `make_fit_lgbm_lambdarank` (LGBMRanker), and `make_fit_xgb_rankic`, **my re-implementation** of a rank-IC-weighted pairwise objective as a custom XGBoost objective (only the LambdaRankIC abstract was read, so this tests the idea, not the paper's exact gradients): for a pair (i, j) with target rank r_i > r_j the change in Spearman IC from putting i above j is proportional to |r_i - r_j| * |rank(p_i) - rank(p_j)|, and the lambda is that weight times the logistic mis-order probability; 12 random partners per row per round. The xgb family's regression control (`fit_xgb_reg`) uses the same booster settings with squared error.

## Experimental setup
All four target experiments use the same frozen harness (a verbatim copy of the data / rank-transform / walk-forward / LP / cost / performance code of `experiments/largecap/largecap.py`, embedded in the script so it runs on its own).

- **Data**: `fiam/chars_final_with_names.parquet`, 523,125 stock-months after dropping rows without a next-month return; 18 factors chosen by economic group in `experiments/largecap` (`FACTOR_GROUPS`), cross-sectionally median-filled and rank-transformed to [-1, 1] over all stocks each month.
- **Universe** (training, validation and scoring rows): price >= $5, market cap >= $2,000M, 126-day dollar volume >= $10M, both betas observed (about 1,206 stocks per month; 153,392 rows).
- **Walk-forward**: for test year Y = 2021..2026, train on target months before Jan Y-2, validate on Y-2..Y-1, refit annually; candidate hyper-parameters are chosen on **validation rank IC against the raw next-month excess return** (for every arm, whatever the fit target), never on test data.
- **Models**: Extra-Trees (largecap grid, 300 trees) and forest-style LightGBM (num_leaves 7, min_child_samples {500, 2000}, extra_trees, lambda 100, checkpoints at 100/200/300 trees). Seed 42 unless stated.
- **Scoring**: universe rank IC of the OOS prediction vs the raw next-month return (68 months, 2021-01..2026-08), decile spread, and the frozen LP portfolios `lc_t10` (10% one-way turnover cap, headline) and `lc_free`, gross and net of the assumed tiered costs.
- **Pre-registered rule** (docs/RESEARCH.md Part I sec 4): adopt an alternative target only if paired monthly universe-IC t >= 2 vs the control on **two** model families (rule A), or IC >= the composite's 0.037 with t >= 2 (rule B).
- **Sanity check that the harness is faithful**: the control arm `et__control` reproduces `experiments/largecap` `et` exactly: IC 0.0164 (published 0.016), lc_t10 gross IR -0.22 (published -0.22).

Families here are `xgb` (control: XGBRegressor, depth {2, 4}, 400 rounds at lr 0.05 with checkpoints 100/200/400, subsample 0.5, colsample 0.5, lambda 100) and `lgbm` (control: forest-style LGBMRegressor).

## Baseline
`xgb__control` (XGBRegressor, winsorised raw target) and `lgbm__control`; the composite (IC 0.037) as the practical bar.

## Commands
```
.venv/bin/python experiments/target_ranker/target_ranker.py
```

## Results (seed 42, 68 test months)
| model__arm | universe rank IC | IC t | % months IC>0 | D10-D1 (%/m) | IR gross (lc_t10) | IR net (lc_t10) | IR gross (lc_free) | beta | max DD net % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| xgb__control | 0.0055 | 0.457 | 51 | 0.066 | -0.065 | -0.167 | 0.046 | -0.109 | -26 |
| xgb__rank_ndcg | 0.0109 | 0.917 | 56 | 0.321 | -0.389 | -0.513 | 0.052 | 0.049 | -17 |
| xgb__rank_pairwise | 0.0152 | 1.364 | 59 | 0.055 | -0.315 | -0.421 | -0.436 | -0.040 | -27 |
| xgb__rankic_pairwise | 0.0129 | 0.948 | 49 | -0.109 | -0.681 | -0.777 | -0.387 | -0.119 | -40 |
| lgbm__control | 0.0160 | 1.044 | 49 | -0.175 | -0.167 | -0.251 | -0.264 | -0.122 | -33 |
| lgbm__lambdarank | 0.0163 | 1.027 | 51 | 0.084 | -0.365 | -0.452 | -0.159 | 0.008 | -33 |

Paired monthly IC vs the same-family control:

| model | arm | ic | ic_control | IC diff | paired t | share of months better | ir_gross | ir_gross_control |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| xgb | rank_ndcg | 0.0109 | 0.0055 | +0.0054 | 0.446 | 0.471 | -0.389 | -0.065 |
| xgb | rank_pairwise | 0.0152 | 0.0055 | +0.0097 | 1.571 | 0.662 | -0.315 | -0.065 |
| xgb | rankic_pairwise | 0.0129 | 0.0055 | +0.0073 | 1.117 | 0.559 | -0.681 | -0.065 |
| lgbm | lambdarank | 0.0163 | 0.0160 | +0.0002 | 0.028 | 0.544 | -0.365 | -0.167 |

Pre-registered rule:
- `rank_ndcg`: families with paired t >= 2: 0; rule A (two families t >= 2): **False**; rule B (IC >= 0.037 and t >= 2): **False**
- `rank_pairwise`: families with paired t >= 2: 0; rule A (two families t >= 2): **False**; rule B (IC >= 0.037 and t >= 2): **False**
- `lambdarank`: families with paired t >= 2: 0; rule A (two families t >= 2): **False**; rule B (IC >= 0.037 and t >= 2): **False**
- `rankic_pairwise`: families with paired t >= 2: 0; rule A (two families t >= 2): **False**; rule B (IC >= 0.037 and t >= 2): **False**

## Interpretation
The XGBoost regression control is weak (IC 0.006), so the ranking arms look better relative to it (+0.005 to +0.010 IC, best `rank_pairwise` paired t 1.6), but in absolute terms they are still below the LightGBM regression control (0.016) and far below the composite. LightGBM `lambdarank` is identical to its regression control (IC 0.016, paired t 0.03) with worse portfolios. Every ranking-arm portfolio is negative (gross IR -0.32 to -0.68). The custom rank-IC objective is not better than off-the-shelf pairwise. **No arm meets the pre-registered rule.** Status: IMPLEMENTED_BUT_FAILED.

## Limitations
- The rank-IC objective is my reconstruction, not the paper's; a faithful LambdaRankIC might behave differently.
- The xgb control is a weak baseline (IC 0.006 despite the same booster settings), which inflates the apparent gain of the ranker arms; the fairer comparison is against the best control (lgbm, 0.016).
- Single seed, 68 months, IC s.e. about 0.015-0.02.

## Follow-up
Not worth pursuing on these 18 factors: the ranking objective changes the fit, not the information available.
