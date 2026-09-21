# Industry-relative momentum and industry-relative composites (FEAT_INDUSTRY_REL)

Implementation: `feat_industry_rel.py` (run: `.venv/bin/python experiments/feat_industry_rel/feat_industry_rel.py`). Everything it writes goes to `output/`.

## Objective
Test defensible momentum variants (industry-relative, 52-week-high-neutral, industry momentum, the panel's existing FF3-residual momentum) as extra composite groups, and test whether ranking every composite factor *within industry* (GICS sector, FF49, or a 50/50 blend) improves the composite itself.

## Hypothesis
Raw momentum crashes in sector rotations; industry-adjusted variants have lower crash risk and add IC. Sector-adjusting characteristics is reported to help most in value-weighted large-cap strategies, so within-industry ranking should raise the composite's universe IC.

## Research origin
docs/NEW.md sec 3.3 items 3-4 (residual / industry-neutral / 52-week-high-neutral momentum; "adjusting for sector boosts [the better result] from 20% to 78% of the time", largest for value-weighted large-cap strategies; grade C).

## Implementation
`make_blocks`: `ind_rel_mom` = 12-1 momentum minus its FF49 universe mean; `ind_mom_12_1` = FF49 industry 12-1 momentum; `mom_52wk_high_neutral` = month-by-month residual of the 12-1 momentum rank on the nearness-to-52-week-high rank (`prc_highprc_252d`); `resff3_12_1_existing` = reference (already in the panel). Replacement composites via `grp_scores(key=...)`: every factor ranked within GICS sector (`comp_indrel_gics2`), within FF49 (`comp_indrel_ff49`), and 50/50 blends of the universe-rank and within-industry composites. All signs +1.

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
.venv/bin/python experiments/feat_industry_rel/feat_industry_rel.py
```

## Results
Additive test of each block on the composite (sign fixed in advance; `gain` = IC(composite + block) - IC(composite), paired over 68 months; `Bonferroni p` = permutation p x number of blocks in this file):

| name | sign | coverage | block IC | IC t | comp+block IC | IC gain | paired t | resid IC | resid t | corr w/ comp | perm p | Bonferroni p | gain small | gain mid | gain large | IR gross | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ind_rel_mom | +1 | 0.99 | +0.0086 | +0.60 | 0.0367 | +0.0000 | +0.01 | +0.0052 | +0.39 | +0.13 | 0.294 | 1.000 | +0.0024 | -0.0023 | +0.0015 | 0.92 | kill |
| ind_mom_12_1 | +1 | 0.99 | -0.0007 | -0.05 | 0.0359 | -0.0007 | -0.13 | -0.0002 | -0.02 | -0.02 | 0.010 | 0.040 | -0.0001 | -0.0032 | +0.0037 | 0.83 | kill |
| mom_52wk_high_neutral | +1 | 1.00 | +0.0008 | +0.05 | 0.0368 | +0.0002 | +0.03 | +0.0009 | +0.06 | -0.00 | 0.303 | 1.000 | +0.0020 | -0.0050 | +0.0065 | 0.99 | kill |
| resff3_12_1_existing | +1 | 1.00 | +0.0134 | +1.00 | 0.0397 | +0.0030 | +0.60 | +0.0115 | +0.89 | +0.08 | 0.025 | 0.100 | +0.0023 | +0.0026 | +0.0065 | 0.95 | kill |

Replacement composites (compared directly with the frozen composite):

| name | IC | IC t | composite IC | IC diff vs composite | paired t | rank corr w/ composite | diff small | diff mid | diff large | IR gross | IR net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| comp_indrel_gics2 | 0.0324 | 1.92 | 0.0366 | -0.0043 | -0.73 | 0.92 | -0.0088 | -0.0027 | -0.0006 | 0.606 | 0.529 |
| comp_indrel_ff49 | 0.0324 | 2.14 | 0.0366 | -0.0042 | -0.62 | 0.87 | -0.0086 | -0.0026 | -0.0005 | 0.974 | 0.877 |
| comp_half_indrel_ff49 | 0.0356 | 2.04 | 0.0366 | -0.0010 | -0.33 | 0.97 | -0.0027 | -0.0005 | +0.0006 | 0.885 | 0.803 |
| comp_half_indrel_gics2 | 0.0347 | 1.92 | 0.0366 | -0.0020 | -0.70 | 0.98 | -0.0041 | -0.0016 | -0.0002 | 0.641 | 0.567 |

Standalone block IC by calendar year and by size tercile (all blocks, signed as tested):

| block | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | small | mid | large |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ind_rel_mom | -0.013 | +0.024 | -0.024 | +0.063 | -0.003 | +0.002 | +0.026 | -0.002 | +0.003 |
| ind_mom_12_1 | -0.047 | +0.070 | -0.035 | +0.005 | -0.005 | +0.014 | +0.003 | -0.010 | +0.008 |
| mom_52wk_high_neutral | -0.049 | +0.023 | +0.007 | +0.032 | -0.005 | -0.005 | +0.016 | -0.016 | +0.007 |
| resff3_12_1_existing | +0.004 | +0.024 | +0.006 | +0.033 | -0.008 | +0.025 | +0.018 | +0.011 | +0.017 |

Truncation-invariance test of the feature builder: passed = **True**; 22695 values compared at dates ['2024-07-31', '2023-11-30', '2025-08-31']; max |difference| over features = 0.0.


## Interpretation
No block or replacement composite improves the composite. Momentum variants have gains of -0.0007 to +0.0030 IC (paired t between -0.13 and +0.60); the existing FF3-residual momentum is the largest at +0.0030. Within-industry composites are slightly *worse* (IC 0.0324 vs 0.0366, paired t -0.6 to -0.7) and highly correlated with the composite (rank corr 0.87-0.98); the FF49 version has a higher gross IR (0.97) with lower IC, an example of the LP-noise caveat. The industry-neutral claim does not show up for the composite: value/investment tilts here are partly *between*-industry effects that within-industry ranking removes. `ind_mom_12_1` has a low permutation p (0.010) because the shuffled null is negative by construction (dilution) - its gain is -0.0007, i.e. no better than nothing. **All killed / not adopted.**

**Status: IMPLEMENTED_BUT_FAILED (all blocks killed by the pre-registered rule)**

## Limitations
- FF49 groups inside a $2B universe contain about 24 stocks on average, so within-FF49 ranks are coarse; GICS-2 (about 110 per sector) is the better-powered version and is also not better.
- The 68-month window contains a sector-rotation heavy 2025-26; industry momentum has known regime dependence.
- Four blocks tested; Bonferroni factor 4.

## Follow-up
None: the industry-relative ideas are exhausted with the panel's data.
