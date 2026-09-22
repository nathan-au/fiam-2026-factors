# FINRA short-interest blocks on the large-cap composite (FEAT_SHORT_INTEREST)

Implementation: `feat_short_interest.py` (run: `.venv/bin/python experiments/feat_short_interest/feat_short_interest.py`). Everything it writes goes to `output/`.

## Objective
Test the short-interest ratio, its 3-month change, and days-to-cover (FINRA bi-monthly short interest) as extra composite groups, and see whether short interest identifies names to avoid in the short book.

## Hypothesis
Heavily shorted stocks and stocks with rising short interest underperform (informed short sellers). As extra composite groups they add IC in the >= $2B universe. Signs fixed in advance: all -1 (more short interest / days-to-cover -> lower next-month return).

## Research origin
docs/RESEARCH.md Part I sec 3.7 (FINRA short interest: surprise in short interest negatively predicts the cross-section; "more valuable as a short-selection / squeeze-risk filter given our Jan-2021 loss"; free data; published ~1 week after settlement).

## Implementation
`fetch_finra` downloads `shrt<YYYYMMDD>.csv` from `cdn.finra.org` into `cache/finra_short_interest/` (74 mid-month settlement files, 2020-06..2026-07; FINRA rejects urllib's default user agent). **Timing**: for characteristic month t only the mid-month settlement file (15th or the prior business day, published about 7 business days later, before month-end) is used - never the month-end file. Symbols are matched to `permno` through the panel's `ticker` at month t (95.9% of universe test rows match); `si_ratio` = `currentShortPositionQuantity` / (`shares` x 1e6), `si_change_3m` = change in the ratio vs 3 months earlier, `days_to_cover` = FINRA `daysToCoverQuantity`.

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
.venv/bin/python experiments/feat_short_interest/feat_short_interest.py
```

## Results
Additive test of each block on the composite (sign fixed in advance; `gain` = IC(composite + block) - IC(composite), paired over 68 months; `Bonferroni p` = permutation p x number of blocks in this file):

| name | sign | coverage | block IC | IC t | comp+block IC | IC gain | paired t | resid IC | resid t | corr w/ comp | perm p | Bonferroni p | gain small | gain mid | gain large | IR gross | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| si_ratio | -1 | 0.96 | +0.0446 | +2.82 | 0.0432 | +0.0066 | +1.80 | +0.0315 | +2.60 | +0.38 | 0.005 | 0.015 | +0.0043 | +0.0064 | +0.0054 | 0.63 | no pass |
| si_change_3m | -1 | 0.95 | -0.0058 | -0.92 | 0.0336 | -0.0030 | -1.24 | -0.0037 | -0.63 | -0.00 | 0.443 | 1.000 | -0.0058 | -0.0020 | -0.0020 | 0.69 | kill |
| days_to_cover | -1 | 0.96 | +0.0351 | +3.51 | 0.0455 | +0.0089 | +2.67 | +0.0334 | +3.60 | +0.02 | 0.005 | 0.015 | +0.0055 | +0.0111 | +0.0086 | 0.66 | PASS |

Standalone block IC by calendar year and by size tercile (all blocks, signed as tested):

| block | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | small | mid | large |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| si_ratio | +0.070 | +0.069 | +0.001 | +0.069 | +0.037 | +0.010 | +0.048 | +0.042 | +0.031 |
| si_change_3m | -0.024 | -0.000 | -0.003 | +0.005 | -0.002 | -0.010 | -0.008 | -0.007 | -0.005 |
| days_to_cover | +0.050 | +0.035 | +0.020 | +0.044 | +0.062 | -0.018 | +0.033 | +0.034 | +0.027 |

Truncation-invariance test of the feature builder: passed = **True**; 22379 values compared at dates ['2025-09-30', '2024-07-31']; max |difference| over features = 0.0.


## Interpretation
**Days-to-cover is the first block in this round to pass the pre-registered rule**: standalone IC +0.0351 (t 3.51), paired gain on the composite **+0.0089 (t 2.67)**, residual IC after orthogonalising to the composite +0.0334 (t 3.60), within-(month, sector) permutation p 0.005 (the smallest resolvable with 200 shuffles; Bonferroni over this file's 3 blocks: 0.015), positive in all three size terciles (small +0.0055, mid +0.0111, large +0.0086 gain), and positive in 5 of 6 calendar years (2021 +0.050, 2022 +0.035, 2023 +0.020, 2024 +0.044, 2025 +0.062, 2026 -0.018). The short-interest ratio has standalone IC +0.0446 (t 2.82, permutation p 0.005), residual IC +0.0315 (t 2.60) and a gain of +0.0066 (paired t 1.80: just misses t >= 2). The 3-month change carries nothing (IC -0.0058, t -0.92, the opposite of "surprise"). The LP gross IR moves only from 0.615 to 0.63-0.69, which is inside single-path noise (the IC gain of +0.009 would imply about +0.15 IR at best). Because the result is unusually good, it was scrutinised in `experiments/feat_si_followup` (leg attribution, staleness, borrow stress, short-side filter). **Two cautions on how much to believe it**: (a) 3 blocks were tested in this file, but the project-wide family is about 46 blocks, and a Bonferroni factor of 46 turns p 0.005 into 0.23 (this is a prior-motivated hypothesis from the literature, not a data-mined one, which supports it but does not remove the discount); (b) high short-interest names are the hard-to-borrow ones, so the short leg's costs are the weakest part of the assumptions.

**Status: IMPLEMENTED_AND_TESTED (days-to-cover passes the pre-registered rule; see follow-up for how far to trust it)**

## Limitations
- 2021-26 contains the meme / short-squeeze episodes and 2026 has one weak year; the effect is a persistent state variable (see the follow-up), not a timing signal.
- Ticker matching uses the panel's `ticker` column; 4% of universe rows do not match (neutral).
- No borrow data: whether the high-SI shorts can be shorted at the assumed cost is not verified.
- Prior-motivated but tested inside a large family of ideas; treat the p-values accordingly.

## Follow-up
`experiments/feat_si_followup` (done). Next would be pulling borrow-fee data (not freely available) and a longer FINRA history (files exist back to 2015).
