# Rank / Gaussian-rank training targets (TARGET_RANK)

Implementation: `target_rank.py` (run: `.venv/bin/python experiments/target_rank/target_rank.py`, about 6 minutes; seed follow-ups below). Everything the script writes goes to `output/`.

## Objective
Test whether training on a within-month rank of next-month excess return (instead of the 1/99-winsorised raw return the whole project uses) improves the tradeable-universe rank IC of the same models on the same 18 factors.

## Hypothesis
Cross-sectional rank targets remove the influence of extreme returns, so the fitted model generalises better and its universe rank IC rises above the raw-target control in at least two model families. Mechanism: the raw target's variance is dominated by a few extreme names whose behaviour does not persist (docs/NEGATIVE_RESULT.md).

## Research origin
docs/NEW.md sec 3.1a and shortlist #1 (Cakici & Zaremba, *Getting the Target Right in Return Prediction*: rank targets "roughly doubled returns and Sharpe in large-cap universes"; the repo rank-transforms features but not the target). Reddit support is thin (docs/REDDIT_RESEARCH.md sec 5 item 10: r/quant 1c6t9b0 is the only on-topic thread).

## Implementation
`build_targets` computes, on universe rows and within each month, (a) the uniform rank mapped to [-1, 1] and (b) the Gaussianised rank `norm.ppf((rank - 0.5)/n)`. `run_wf` is the frozen walk-forward with two changes: the fit target is the transformed target (not winsorised, ranks have no outliers) and validation-IC candidate selection is always against the raw return. The control fits the harness's winsorised raw return.

## Experimental setup
All four target experiments use the same frozen harness (a verbatim copy of the data / rank-transform / walk-forward / LP / cost / performance code of `experiments/largecap/largecap.py`, embedded in the script so it runs on its own).

- **Data**: `fiam/chars_final_with_names.parquet`, 523,125 stock-months after dropping rows without a next-month return; 18 factors chosen by economic group in `experiments/largecap` (`FACTOR_GROUPS`), cross-sectionally median-filled and rank-transformed to [-1, 1] over all stocks each month.
- **Universe** (training, validation and scoring rows): price >= $5, market cap >= $2,000M, 126-day dollar volume >= $10M, both betas observed (about 1,206 stocks per month; 153,392 rows).
- **Walk-forward**: for test year Y = 2021..2026, train on target months before Jan Y-2, validate on Y-2..Y-1, refit annually; candidate hyper-parameters are chosen on **validation rank IC against the raw next-month excess return** (for every arm, whatever the fit target), never on test data.
- **Models**: Extra-Trees (largecap grid, 300 trees) and forest-style LightGBM (num_leaves 7, min_child_samples {500, 2000}, extra_trees, lambda 100, checkpoints at 100/200/300 trees). Seed 42 unless stated.
- **Scoring**: universe rank IC of the OOS prediction vs the raw next-month return (68 months, 2021-01..2026-08), decile spread, and the frozen LP portfolios `lc_t10` (10% one-way turnover cap, headline) and `lc_free`, gross and net of the assumed tiered costs.
- **Pre-registered rule** (docs/NEW.md sec 4): adopt an alternative target only if paired monthly universe-IC t >= 2 vs the control on **two** model families (rule A), or IC >= the composite's 0.037 with t >= 2 (rule B).
- **Sanity check that the harness is faithful**: the control arm `et__control` reproduces `experiments/largecap` `et` exactly: IC 0.0164 (published 0.016), lc_t10 gross IR -0.22 (published -0.22).

Arms: `control` (winsorised raw target), `rank_uniform`, `rank_gauss`, for Extra-Trees and LightGBM. Seeds 43 and 44 were added for `control` and `rank_uniform` because the seed-42 result was ambiguous.

## Baseline
Same model, features, universe and selection procedure with the winsorised raw target (`*__control`). Practical bar: the no-fit composite (IC 0.037, `lc_t10` gross IR 0.61 / net 0.54 in `experiments/largecap`).

## Commands
```
.venv/bin/python experiments/target_rank/target_rank.py
.venv/bin/python experiments/target_rank/target_rank.py --models et,lgbm --arms control,rank_uniform --seed 43 --suffix _seed43
.venv/bin/python experiments/target_rank/target_rank.py --models et,lgbm --arms control,rank_uniform --seed 44 --suffix _seed44
```

## Results (seed 42, 68 test months)
| model__arm | universe rank IC | IC t | % months IC>0 | D10-D1 (%/m) | IR gross (lc_t10) | IR net (lc_t10) | IR gross (lc_free) | beta | max DD net % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| et__control | 0.0164 | 1.102 | 49 | -0.170 | -0.216 | -0.302 | -0.182 | -0.086 | -35 |
| et__rank_uniform | 0.0222 | 1.442 | 50 | -0.084 | -0.141 | -0.223 | -0.099 | 0.007 | -32 |
| et__rank_gauss | 0.0198 | 1.320 | 50 | -0.098 | -0.361 | -0.441 | -0.171 | -0.065 | -38 |
| lgbm__control | 0.0160 | 1.044 | 49 | -0.175 | -0.167 | -0.251 | -0.264 | -0.122 | -33 |
| lgbm__rank_uniform | 0.0210 | 1.376 | 51 | -0.104 | -0.225 | -0.306 | -0.322 | -0.054 | -32 |
| lgbm__rank_gauss | 0.0191 | 1.277 | 50 | -0.067 | -0.253 | -0.336 | -0.179 | -0.072 | -37 |

Paired monthly IC vs the same-family control:

| model | arm | ic | ic_control | IC diff | paired t | share of months better | ir_gross | ir_gross_control |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| et | rank_uniform | 0.0222 | 0.0164 | +0.0058 | 1.379 | 0.529 | -0.141 | -0.216 |
| et | rank_gauss | 0.0198 | 0.0164 | +0.0034 | 0.844 | 0.471 | -0.361 | -0.216 |
| lgbm | rank_uniform | 0.0210 | 0.0160 | +0.0050 | 1.283 | 0.574 | -0.225 | -0.167 |
| lgbm | rank_gauss | 0.0191 | 0.0160 | +0.0031 | 0.730 | 0.574 | -0.253 | -0.167 |

Pre-registered rule:
- `rank_uniform`: families with paired t >= 2: 0; rule A (two families t >= 2): **False**; rule B (IC >= 0.037 and t >= 2): **False**
- `rank_gauss`: families with paired t >= 2: 0; rule A (two families t >= 2): **False**; rule B (IC >= 0.037 and t >= 2): **False**

Seed check (universe rank IC per seed; the seed changes only the model randomness):

| model__arm | seed 42 | seed 43 | seed 44 | mean |
|---|---:|---:|---:|---:|
| et__control | 0.0164 | 0.0169 | 0.0170 | 0.0168 |
| et__rank_uniform | 0.0222 | 0.0226 | 0.0209 | 0.0219 |
| et__rank_gauss | 0.0198 |  |  | 0.0198 |
| lgbm__control | 0.0160 | 0.0147 | 0.0160 | 0.0156 |
| lgbm__rank_uniform | 0.0210 | 0.0205 | 0.0221 | 0.0212 |
| lgbm__rank_gauss | 0.0191 |  |  | 0.0191 |

Seed-averaged paired IC (uniform rank minus control):
- et: seed-averaged IC difference +0.0051, paired t 1.23, better in 51% of months
- lgbm: seed-averaged IC difference +0.0056, paired t 1.44, better in 65% of months

Sanity checks: no NaN predictions (`n_nan_pred` = 0 in `results.json`); the control equals the published `et` arm; the rank arms really differ from the control (prediction rank autocorrelation 0.83-0.86 vs 0.79-0.80).

## Interpretation
Every rank-target arm has a higher universe IC than its control (+0.003 to +0.006), in both families and in all three seeds, so the direction agrees with the paper. But the paired-IC t is only 0.7-1.4 (seed-averaged 1.2-1.4), far from the pre-registered t >= 2, and the fitted models stay far below the composite's 0.037 IC. The portfolios do not improve: every `lc_t10` book is still negative (gross IR -0.14 to -0.36), and Gaussian rank is no better than uniform rank. **The idea fails the pre-registered rule: a consistent but small (about +0.005 IC) and statistically unconfirmed gain, not enough to turn a fitted model into a tradeable one.** Status: IMPLEMENTED_BUT_FAILED (kill rule), direction of effect noted.

## Limitations
- One 68-month path; the IC s.e. per model is about 0.015-0.02, so a +0.005 difference is undetectable even paired (paired t about 1.3).
- Only 18 economic-group factors and only Extra-Trees / LightGBM; the paper's gain is on 147-characteristic global data with other models and value-weighted portfolios.
- The LP is a constrained, beta-neutral optimiser with very noisy single-path IR (s.e. about 0.46).

## Follow-up
Keep rank targets as a cheap default whenever a model is fitted on top of another signal, but they do not rescue a fitted model on their own. The combined test (rank target on top of the composite prior) is in `experiments/composite_prior` and also failed.
