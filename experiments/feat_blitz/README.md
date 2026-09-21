# Blitz et al. short-term composite as a large-cap block (FEAT_BLITZ)

Implementation: `feat_blitz.py` (run: `.venv/bin/python experiments/feat_blitz/feat_blitz.py`). Everything it writes goes to `output/`.

## Objective
Test the large-cap-native short-term composite of Blitz-Hanauer-Honarvar-Huisman-van Vliet (industry-relative 1-month reversal, 1-month industry momentum, same-calendar-month seasonality, idiosyncratic volatility) as an extra composite group. Analyst revisions, the paper's fourth signal, are not in the panel.

## Hypothesis
A composite of short-horizon signals defined only on benchmark constituents is orthogonal to the frozen composite (which has no short-term content) and adds IC in the >= $2B universe. Risk noted in advance: the 2021-26 window contains the Jul-2025 / Jan-2026 short-horizon unwinds.

## Research origin
docs/REDDIT_RESEARCH.md sec 2.2 (Blitz et al., *FAJ* 2023: MSCI World constituents only, individual gross returns 5-8%/yr, composite > 12% six-factor alpha, break-even cost > 30bp, alpha persists post-publication; grade: demonstrated, not on our sample).

## Implementation
`make_blocks`: `blitz_indrev_1m` = -(ret_1_0 - FF49 universe mean) (sign -1 on the relative return), `blitz_indmom_1m` = FF49 industry mean return of month t (+), `blitz_seasonality` = mean rank of `seas_1_1na` and `seas_2_5na` (+), `blitz_composite3` = equal-weight of those three, `blitz_composite4_with_ivol` adds `-ivol_capm_21d` (already inside the composite's volatility group). Analyst revisions omitted (need I/B/E/S). The paper's buy-10/hold-50 buffer combination is not tested here (see `experiments/portfolio_buffer`, which failed on the composite).

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
.venv/bin/python experiments/feat_blitz/feat_blitz.py
```

## Results
Additive test of each block on the composite (sign fixed in advance; `gain` = IC(composite + block) - IC(composite), paired over 68 months; `Bonferroni p` = permutation p x number of blocks in this file):

| name | sign | coverage | block IC | IC t | comp+block IC | IC gain | paired t | resid IC | resid t | corr w/ comp | perm p | Bonferroni p | gain small | gain mid | gain large | IR gross | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| blitz_indrev_1m | -1 | 0.99 | -0.0105 | -1.03 | 0.0314 | -0.0052 | -1.50 | -0.0098 | -1.07 | +0.01 | 0.995 | 1.000 | -0.0034 | -0.0066 | -0.0057 | 0.42 | kill |
| blitz_indmom_1m | +1 | 0.99 | +0.0043 | +0.26 | 0.0366 | -0.0000 | -0.00 | +0.0043 | +0.27 | -0.04 | 0.050 | 0.249 | +0.0018 | -0.0020 | +0.0025 | 0.79 | kill |
| blitz_seasonality | +1 | 1.00 | -0.0011 | -0.06 | 0.0370 | +0.0004 | +0.06 | +0.0023 | +0.13 | -0.04 | 0.005 | 0.025 | +0.0038 | -0.0046 | +0.0037 | 0.90 | kill |
| blitz_composite3 | +1 | 1.00 | -0.0050 | -0.49 | 0.0346 | -0.0020 | -0.47 | -0.0025 | -0.25 | -0.03 | 0.308 | 1.000 | +0.0020 | -0.0065 | -0.0007 | 0.73 | kill |
| blitz_composite4_with_ivol | +1 | 1.00 | +0.0120 | +1.07 | 0.0366 | -0.0000 | -0.00 | +0.0020 | +0.19 | +0.25 | 0.045 | 0.224 | +0.0038 | -0.0034 | -0.0016 | 0.61 | kill |

Standalone block IC by calendar year and by size tercile (all blocks, signed as tested):

| block | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | small | mid | large |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| blitz_indrev_1m | +0.010 | -0.012 | -0.001 | -0.006 | -0.033 | -0.027 | -0.002 | -0.021 | -0.010 |
| blitz_indmom_1m | -0.022 | -0.035 | +0.023 | -0.018 | +0.050 | +0.040 | +0.010 | -0.004 | +0.010 |
| blitz_seasonality | -0.027 | -0.057 | -0.003 | +0.031 | +0.042 | +0.012 | +0.010 | -0.016 | +0.003 |
| blitz_composite3 | -0.019 | -0.055 | +0.015 | -0.002 | +0.025 | +0.012 | +0.008 | -0.022 | +0.002 |
| blitz_composite4_with_ivol | +0.008 | -0.011 | +0.006 | +0.033 | +0.026 | +0.010 | +0.027 | -0.003 | +0.006 |

Truncation-invariance test of the feature builder: passed = **True**; 23101 values compared at dates ['2024-07-31', '2023-11-30', '2025-08-31']; max |difference| over features = 0.0.


## Interpretation
Nothing adds IC: paired gains are -0.0052 to +0.0004 (t -1.50 to +0.06), and the industry-relative reversal is *negative* standalone (IC -0.0105, t -1.03; paired t -1.50 on the composite). The composite of the three signals has IC -0.005 standalone. Seasonality and industry momentum have low permutation p only because of the dilution effect described in the setup. The 2021-26 window is hostile to short-term signals (the paper's sample ends 2021), and large-cap short-term reversal is reported to have faded. **All killed.**

**Status: IMPLEMENTED_BUT_FAILED (all blocks killed; the analyst-revision leg is BLOCKED for lack of I/B/E/S data)**

## Limitations
- The paper's strongest signal (analyst revisions) is missing; the composite here is the weaker three-signal version.
- High-turnover signals in a book with a 10% one-way turnover cap are structurally handicapped; this experiment measures IC only.
- Five blocks; Bonferroni factor 5.

## Follow-up
If I/B/E/S revisions become available (see the blocked-external-data items), re-run with the full five-signal composite.
