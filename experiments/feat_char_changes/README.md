# Characteristic-change (momentum in attributes) blocks (FEAT_CHAR_CHANGES)

Implementation: `feat_char_changes.py` (run: `.venv/bin/python experiments/feat_char_changes/feat_char_changes.py`). Everything it writes goes to `output/`.

## Objective
Test whether the 3- and 12-month change in the cross-sectional percentile rank of profitability, valuation, quality, earnings surprise and investment characteristics carries information the levels in the composite do not.

## Hypothesis
Returns are driven by *changes* in characteristics ("momentum in firm attributes"), so signed rank changes of key characteristics, added as a group, raise universe IC.

## Research origin
docs/NEW.md sec 3.3 item 5 (equilibrium characteristics-driven returns, arXiv 2203.07865; grade C).

## Implementation
`chg_builder` computes, from the raw panel columns, the cross-sectional percentile rank (all stocks) of `qmj`, `gp_at`, `be_me`, `niq_su`, `at_gr1` each month, calendar-lagged by 3 and 12 months, and differences them. Signs = the level's sign in the composite (`at_gr1` -1, others +1). Blocks: `chg12m_composite` and `chg3m_composite` (equal-weight of the five signed changes) and each 12-month change separately.

## Experimental setup
Common harness (a verbatim copy of the data / rank-transform / walk-forward / LP / cost / performance code of `experiments/largecap/largecap.py`, embedded so the script runs on its own):

- **Data**: `fiam/chars_final_with_names.parquet` (523,125 stock-months with a next-month return; 147 characteristics; `ret`, `me`, `ff49`, `gvkey`, `ticker`, ... for feature building).
- **Universe** (IC scoring and LP): price >= $5, market cap >= $2,000M, 126-day dollar volume >= $10M, both betas observed; about 1,206 stocks per month; 68 test months 2021-01..2026-08 (82,026 universe test rows).
- **No model is fitted.** The baseline is the frozen composite: equal-weight of 7 factor groups over 18 pre-specified factors, re-ranked within the universe each month. Each candidate block is added as an **8th equal-weight group** with a sign fixed before any result was seen (`sign` column below; the sign is applied to the within-month universe percentile rank of the block, NaN -> neutral 0).
- **Tests per block**: (1) IC of the signed block alone; (2) paired monthly IC gain of composite+block vs the composite (mean difference, t); (3) residual IC after orthogonalising the block to the composite each month (the "shared-core" check, docs/REDDIT_RESEARCH.md sec 2.5); (4) the gain within each size tercile; (5) a within-(month, sector) permutation null, 200 shuffles, for the additive gain (docs/REDDIT_RESEARCH.md H4). **Note**: a random block *dilutes* the composite, so the null gain is negative on average; the permutation p therefore tests "does this block carry information", while the paired t tests "does adding it improve the composite". (6) LP `lc_t10` gross / net IR (**single-path IR s.e. is about 0.46**, and LP corner solutions make IR jump by +-0.3 for changes that leave IC unchanged; IR is reported but never used for decisions). (7) A **truncation-invariance test**: the feature builder is re-run on rows with eom <= t only for random test dates t and must give identical values at t for every stock.
- **Pre-registered rule** (docs/NEW.md sec 2 / 4): KILL if the paired IC gain has t < 1 or the gain is confined to the smallest size tercile. PASS requires paired t >= 2, Bonferroni-adjusted permutation p <= 0.05 (factor = number of blocks in the file) and not small-tercile-only.
- **Baseline sanity**: the composite reproduces `experiments/largecap` (IC 0.0366, t 1.91; `lc_t10` gross IR 0.615 / net 0.541).


## Baseline
The frozen 7-group composite of `experiments/largecap` on the same universe and months (composite IC 0.0366 (t 1.91); lc_t10 gross IR 0.615, net IR 0.541, 2025 net return -12.4%); every block is compared with it (paired), and the LP book with it (gross/net IR, single path).

## Commands
```
.venv/bin/python experiments/feat_char_changes/feat_char_changes.py
```

## Results
Additive test of each block on the composite (sign fixed in advance; `gain` = IC(composite + block) - IC(composite), paired over 68 months; `Bonferroni p` = permutation p x number of blocks in this file):

| name | sign | coverage | block IC | IC t | comp+block IC | IC gain | paired t | resid IC | resid t | corr w/ comp | perm p | Bonferroni p | gain small | gain mid | gain large | IR gross | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chg12m_composite | +1 | 1.00 | +0.0010 | +0.11 | 0.0317 | -0.0050 | -1.47 | -0.0110 | -1.36 | +0.25 | 0.970 | 1.000 | -0.0094 | -0.0027 | -0.0008 | 0.56 | kill |
| chg3m_composite | +1 | 1.00 | +0.0082 | +1.05 | 0.0355 | -0.0012 | -0.36 | +0.0010 | +0.13 | +0.16 | 0.861 | 1.000 | -0.0063 | -0.0006 | +0.0038 | 0.61 | kill |
| d12_qmj | +1 | 0.93 | +0.0075 | +0.82 | 0.0346 | -0.0020 | -0.60 | -0.0063 | -0.67 | +0.31 | 0.269 | 1.000 | -0.0043 | -0.0016 | +0.0026 | 0.70 | kill |
| d12_gp_at | +1 | 1.00 | +0.0090 | +1.03 | 0.0350 | -0.0016 | -0.47 | -0.0015 | -0.18 | +0.20 | 0.478 | 1.000 | -0.0051 | -0.0011 | +0.0039 | 0.66 | kill |
| d12_be_me | +1 | 0.95 | -0.0082 | -0.53 | 0.0314 | -0.0052 | -0.93 | -0.0084 | -0.58 | -0.03 | 1.000 | 1.000 | -0.0094 | -0.0010 | -0.0065 | 0.07 | kill |
| d12_niq_su | +1 | 0.99 | +0.0011 | +0.13 | 0.0330 | -0.0036 | -1.03 | -0.0053 | -0.64 | +0.18 | 0.980 | 1.000 | -0.0056 | -0.0044 | +0.0004 | 0.70 | kill |
| d12_at_gr1 | -1 | 1.00 | -0.0053 | -0.83 | 0.0326 | -0.0040 | -1.46 | -0.0031 | -0.51 | -0.04 | 0.716 | 1.000 | -0.0040 | -0.0030 | -0.0041 | 0.60 | kill |

Standalone block IC by calendar year and by size tercile (all blocks, signed as tested):

| block | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | small | mid | large |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chg12m_composite | +0.018 | -0.028 | +0.003 | +0.007 | +0.003 | +0.003 | -0.009 | +0.006 | +0.005 |
| chg3m_composite | +0.019 | +0.000 | +0.006 | +0.028 | -0.003 | -0.007 | -0.003 | +0.010 | +0.015 |
| d12_qmj | +0.004 | +0.006 | -0.017 | +0.029 | +0.015 | +0.007 | +0.005 | +0.007 | +0.010 |
| d12_gp_at | +0.017 | -0.021 | -0.001 | +0.034 | +0.021 | +0.001 | +0.003 | +0.010 | +0.016 |
| d12_be_me | +0.020 | -0.050 | +0.047 | -0.033 | -0.018 | -0.019 | -0.015 | +0.002 | -0.017 |
| d12_niq_su | +0.004 | +0.021 | -0.017 | +0.000 | -0.012 | +0.017 | -0.002 | -0.005 | +0.008 |
| d12_at_gr1 | +0.011 | -0.024 | -0.004 | -0.009 | -0.001 | -0.004 | -0.005 | -0.003 | -0.008 |

Truncation-invariance test of the feature builder: passed = **True**; 100461 values compared at dates ['2024-07-31', '2023-11-30', '2025-08-31']; max |difference| over features = 0.0.


## Interpretation
No block passes. The 12-month change composite has *negative* paired gain (-0.0050, t -1.47) and the individual changes are between -0.0052 and -0.0016 (t between -0.47 and -1.46). Changes in valuation (`d12_be_me`) and investment (`d12_at_gr1`) are slightly negative standalone. One early-look number worth recording: in the 1/3-stock smoke run `d12_at_gr1` looked good (gain +0.008); on the full sample it is -0.0040, which shows how easily a subsample flatters a block. **All killed.**

**Status: IMPLEMENTED_BUT_FAILED (all blocks killed by the pre-registered rule)**

## Limitations
- Rank changes of accounting characteristics that update quarterly are step functions with many zeros at a monthly cadence; 3-month changes are noisy.
- Seven blocks tested; Bonferroni factor 7.
- Only five characteristics were changed; a broader set was not tried.

## Follow-up
None.
