# Price-path-shape features from monthly returns (FEAT_PATH)

Implementation: `feat_path.py` (run: `.venv/bin/python experiments/feat_path/feat_path.py`). Everything it writes goes to `output/`.

## Objective
Test whether features describing the *shape* of the last 11 monthly returns (information discreteness, smoothness/trend strength, up-month fraction, drawdown inside the path) add IC to the large-cap composite. The composite has no momentum content, so the plain 12-1 momentum control answers whether shape adds anything beyond momentum itself.

## Hypothesis
The 147 characteristics are levels; the path shape carries retail-behaviour information that survives in large caps ("frog in the pan": momentum is 5.94% for continuous- vs -2.07% for discrete-information stocks). Each shape block added as an 8th group should raise universe IC.

## Research origin
docs/RESEARCH.md Part I sec 3.3 items 1-2 and Round B: Da-Gurun-Warachka information discreteness (*Review of Financial Studies* 2014, evidence grade A) and the Quantitativo slope/smoothness replication (7.9% annual alpha, t 3.44; grade B).

## Implementation
`path_builder` builds, from the monthly `ret` column only, for characteristic month t the compounded return of months t-11..t-1 (the same window as the panel's `ret_12_1`, verified equal to 1e-16), the share of up and down months, `ID = sign(cum) * (%neg - %pos)`, the 11-point cumulative log-return path, its slope and R^2 and its worst internal drawdown. Blocks (sign fixed in advance, all +): `mom_x_continuity` (12-1 momentum rank scaled by (1 + continuity rank)/2), `trend_strength` (slope x R^2), `up_minus_down_frac`, `path_maxdd`, and the control `mom_12_1_plain`.

## Experimental setup
Common harness (a verbatim copy of the data / rank-transform / walk-forward / LP / cost / performance code of `experiments/largecap/largecap.py`, embedded so the script runs on its own):

- **Data**: `fiam/chars_final_with_names.parquet` (523,125 stock-months with a next-month return; 147 characteristics; `ret`, `me`, `ff49`, `gvkey`, `ticker`, ... for feature building).
- **Universe** (IC scoring and LP): price >= $5, market cap >= $2,000M, 126-day dollar volume >= $10M, both betas observed; about 1,206 stocks per month; 68 test months 2021-01..2026-08 (82,026 universe test rows).
- **No model is fitted.** The baseline is the frozen composite: equal-weight of 7 factor groups over 18 pre-specified factors, re-ranked within the universe each month. Each candidate block is added as an **8th equal-weight group** with a sign fixed before any result was seen (`sign` column below; the sign is applied to the within-month universe percentile rank of the block, NaN -> neutral 0).
- **Tests per block**: (1) IC of the signed block alone; (2) paired monthly IC gain of composite+block vs the composite (mean difference, t); (3) residual IC after orthogonalising the block to the composite each month (the "shared-core" check, docs/RESEARCH.md Part II sec 2.5); (4) the gain within each size tercile; (5) a within-(month, sector) permutation null, 200 shuffles, for the additive gain (docs/RESEARCH.md Part II H4). **Note**: a random block *dilutes* the composite, so the null gain is negative on average; the permutation p therefore tests "does this block carry information", while the paired t tests "does adding it improve the composite". (6) LP `lc_t10` gross / net IR (**single-path IR s.e. is about 0.46**, and LP corner solutions make IR jump by +-0.3 for changes that leave IC unchanged; IR is reported but never used for decisions). (7) A **truncation-invariance test**: the feature builder is re-run on rows with eom <= t only for random test dates t and must give identical values at t for every stock.
- **Pre-registered rule** (docs/RESEARCH.md Part I sec 2 / 4): KILL if the paired IC gain has t < 1 or the gain is confined to the smallest size tercile. PASS requires paired t >= 2, Bonferroni-adjusted permutation p <= 0.05 (factor = number of blocks in the file) and not small-tercile-only.
- **Baseline sanity**: the composite reproduces `experiments/largecap` (IC 0.0366, t 1.91; `lc_t10` gross IR 0.615 / net 0.541).


## Baseline
The frozen 7-group composite of `experiments/largecap` on the same universe and months (composite IC 0.0366 (t 1.91); lc_t10 gross IR 0.615, net IR 0.541, 2025 net return -12.4%); every block is compared with it (paired), and the LP book with it (gross/net IR, single path).

## Commands
```
.venv/bin/python experiments/feat_path/feat_path.py
```

## Results
Additive test of each block on the composite (sign fixed in advance; `gain` = IC(composite + block) - IC(composite), paired over 68 months; `Bonferroni p` = permutation p x number of blocks in this file):

| name | sign | coverage | block IC | IC t | comp+block IC | IC gain | paired t | resid IC | resid t | corr w/ comp | perm p | Bonferroni p | gain small | gain mid | gain large | IR gross | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mom_x_continuity | +1 | 1.00 | +0.0036 | +0.19 | 0.0357 | -0.0010 | -0.15 | +0.0000 | +0.00 | +0.12 | 0.149 | 0.746 | +0.0021 | -0.0038 | +0.0008 | 0.95 | kill |
| trend_strength | +1 | 1.00 | +0.0117 | +0.63 | 0.0378 | +0.0011 | +0.17 | +0.0074 | +0.41 | +0.13 | 0.065 | 0.323 | +0.0031 | -0.0013 | +0.0037 | 1.03 | kill |
| up_minus_down_frac | +1 | 1.00 | +0.0091 | +0.62 | 0.0350 | -0.0016 | -0.32 | -0.0001 | -0.00 | +0.17 | 0.159 | 0.796 | +0.0026 | -0.0054 | -0.0026 | 0.88 | kill |
| path_maxdd | +1 | 1.00 | +0.0203 | +0.96 | 0.0359 | -0.0008 | -0.14 | +0.0040 | +0.22 | +0.43 | 0.114 | 0.572 | +0.0017 | -0.0013 | -0.0034 | 0.61 | kill |
| mom_12_1_plain | +1 | 1.00 | +0.0045 | +0.24 | 0.0363 | -0.0003 | -0.05 | +0.0017 | +0.09 | +0.11 | 0.075 | 0.373 | +0.0021 | -0.0041 | +0.0036 | 0.91 | kill |

Standalone block IC by calendar year and by size tercile (all blocks, signed as tested):

| block | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | small | mid | large |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mom_x_continuity | -0.048 | +0.045 | -0.042 | +0.064 | -0.003 | +0.005 | +0.023 | -0.008 | -0.000 |
| trend_strength | -0.004 | +0.031 | -0.044 | +0.077 | +0.005 | +0.002 | +0.030 | -0.000 | +0.007 |
| up_minus_down_frac | +0.006 | +0.025 | -0.041 | +0.050 | +0.003 | +0.011 | +0.028 | -0.003 | -0.002 |
| path_maxdd | +0.022 | +0.069 | -0.060 | +0.094 | -0.019 | +0.014 | +0.040 | +0.014 | -0.002 |
| mom_12_1_plain | -0.048 | +0.052 | -0.038 | +0.058 | +0.002 | -0.001 | +0.021 | -0.010 | +0.007 |

Truncation-invariance test of the feature builder: passed = **True**; 56345 values compared at dates ['2024-07-31', '2023-11-30', '2025-08-31']; max |difference| over features = 0.0.


## Interpretation
No path block helps. All five have paired gain t between -0.32 and +0.17 (gains of -0.0016 to +0.0011 IC) and none is significant on the permutation null after Bonferroni. Continuity does not add to momentum: `mom_x_continuity` has IC 0.004 vs 0.005 for plain momentum, so the frog-in-the-pan interaction is not visible in this universe and window (which includes the 2025-26 momentum unwinds). Standalone IC is largest for `path_maxdd` (+0.020, t 0.96) but adds nothing (paired t -0.14). The LP gross IR moves from 0.615 to 0.61-1.03, a spread that is pure single-path noise given IC is unchanged (see the note on IR in the setup). **The kill rule fires for all five blocks.**

**Status: IMPLEMENTED_BUT_FAILED (all blocks killed by the pre-registered rule)**

## Limitations
- Only monthly returns are available; the literature's information-discreteness measure uses daily returns (a 12-month window of ~250 daily signs), of which the monthly version is a coarse proxy (11 signs).
- One path, 68 months, the window contains a momentum unwind; the effect might exist in a longer sample.
- Five blocks tested together; the Bonferroni factor is 5.

## Follow-up
Daily returns are the natural next step for this idea (see the blocked daily-return items in the final report); with monthly data there is nothing more to try cheaply.
