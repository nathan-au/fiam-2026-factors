# Follow-up: is the 8-K item 5.02 (officer/director change) intensity signal robust? (FEAT_8K_502_FOLLOWUP)

Implementation: `feat_8k_502_followup.py` (run: `.venv/bin/python experiments/feat_8k_502_followup/feat_8k_502_followup.py`). Everything it writes goes to `output/`.

## Objective
Check whether the one 8-K metadata block with a notable standalone t-stat (`mgmt_5_02_decayed_hl6`: IC +0.0125, t 2.87, permutation p 0.010) is robust to the decay half-life, to a plain count, to orthogonalisation to size / liquidity / firm age, to the size tercile, to the calendar year, and to weighting in the composite - or is a proxy for large, frequently-filing firms, or multiple-testing noise.

## Hypothesis
If real, the signal survives half-lives 3/6/12, a plain 12-month count, residualisation to size, turnover, Amihud and age, is present in both halves of the sample and in mid/large caps, and improves the composite's IC when weighted more. If it fails the size/liquidity residualisation it is a proxy; if it fails the composite test it is too weak to matter.

## Research origin
My own follow-up to `experiments/feat_8k_meta` (docs/RESEARCH.md Part I sec 3.4 item 3: 5.02 among the event flags). Only the structured item code and filing date are used; no text, no LLM. Family-wise: the original test was 1 of 10, so the Bonferroni factor stays 10; the variants here share one signal and are not independent tests.

## Implementation
`f502_builder` (decayed count of 5.02 filings, half-lives 3/6/12 months, plain 3- and 12-month counts; NaN when the stock has no 8-K in the trailing 24 months); `orth_raw` residualises the hl6 intensity rank each month on log size, turnover (`turnover_126d`), Amihud (`ami_126d`) and `age` (or on log size only). Replacement composites `comp_plus_{1,2,4}x_m502_hl6` = (sum of the 7 group scores + k * block) / (7 + k). All blocks sign -1.

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
.venv/bin/python experiments/feat_8k_502_followup/feat_8k_502_followup.py
```

## Results
Additive test of each block on the composite (sign fixed in advance; `gain` = IC(composite + block) - IC(composite), paired over 68 months; `Bonferroni p` = permutation p x number of blocks in this file):

| name | sign | coverage | block IC | IC t | comp+block IC | IC gain | paired t | resid IC | resid t | corr w/ comp | perm p | Bonferroni p | gain small | gain mid | gain large | IR gross | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| m502_decayed_hl3 | -1 | 0.94 | +0.0101 | +2.37 | 0.0368 | +0.0001 | +0.06 | +0.0062 | +1.37 | +0.08 | 0.030 | 0.209 | +0.0007 | +0.0015 | +0.0009 | 0.91 | kill |
| m502_decayed_hl6 | -1 | 0.94 | +0.0125 | +2.87 | 0.0377 | +0.0010 | +0.47 | +0.0084 | +1.78 | +0.09 | 0.010 | 0.070 | +0.0023 | +0.0022 | +0.0015 | 0.92 | kill |
| m502_decayed_hl12 | -1 | 0.94 | +0.0125 | +2.74 | 0.0379 | +0.0012 | +0.55 | +0.0084 | +1.66 | +0.09 | 0.010 | 0.070 | +0.0028 | +0.0023 | +0.0016 | 0.96 | kill |
| m502_count_12m | -1 | 0.94 | +0.0117 | +2.82 | 0.0376 | +0.0010 | +0.49 | +0.0061 | +1.22 | +0.09 | 0.015 | 0.104 | +0.0026 | +0.0022 | +0.0010 | 0.66 | kill |
| m502_count_3m | -1 | 0.94 | +0.0061 | +1.43 | 0.0356 | -0.0011 | -0.60 | -0.0086 | -1.01 | +0.05 | 0.308 | 1.000 | -0.0016 | -0.0003 | +0.0002 | 0.76 | kill |
| m502_decayed_hl6_orth_size_liq_age | -1 | 1.00 | +0.0114 | +2.84 | 0.0379 | +0.0013 | +0.61 | +0.0097 | +2.30 | +0.04 | 0.005 | 0.035 | +0.0022 | +0.0027 | +0.0016 | 0.94 | kill |
| m502_decayed_hl6_orth_size_only | -1 | 1.00 | +0.0133 | +3.00 | 0.0379 | +0.0012 | +0.56 | +0.0092 | +1.98 | +0.10 | 0.005 | 0.035 | +0.0024 | +0.0022 | +0.0017 | 0.96 | kill |

Replacement composites (compared directly with the frozen composite):

| name | IC | IC t | composite IC | IC diff vs composite | paired t | rank corr w/ composite | diff small | diff mid | diff large | IR gross | IR net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| comp_plus_1x_m502_hl6 | 0.0377 | 2.14 | 0.0366 | +0.0010 | +0.47 | 0.92 | +0.0023 | +0.0022 | +0.0015 | 0.915 | 0.835 |
| comp_plus_2x_m502_hl6 | 0.0367 | 2.42 | 0.0366 | +0.0000 | +0.01 | 0.78 | +0.0013 | +0.0011 | +0.0031 | 0.837 | 0.750 |
| comp_plus_4x_m502_hl6 | 0.0315 | 2.84 | 0.0366 | -0.0051 | -0.55 | 0.56 | -0.0056 | -0.0034 | +0.0027 | 0.712 | 0.610 |

Standalone block IC by calendar year and by size tercile (all blocks, signed as tested):

| block | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | small | mid | large |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| m502_decayed_hl3 | +0.013 | -0.001 | +0.018 | +0.017 | +0.004 | +0.011 | +0.016 | +0.011 | +0.009 |
| m502_decayed_hl6 | +0.016 | +0.003 | +0.019 | +0.022 | +0.003 | +0.012 | +0.021 | +0.013 | +0.009 |
| m502_decayed_hl12 | +0.015 | +0.005 | +0.018 | +0.024 | +0.000 | +0.012 | +0.023 | +0.014 | +0.008 |
| m502_count_12m | +0.014 | -0.000 | +0.016 | +0.018 | +0.008 | +0.016 | +0.020 | +0.013 | +0.009 |
| m502_count_3m | +0.008 | -0.006 | +0.012 | +0.008 | +0.010 | +0.005 | +0.006 | +0.006 | +0.008 |
| m502_decayed_hl6_orth_size_liq_age | +0.014 | -0.000 | +0.017 | +0.024 | +0.007 | +0.005 | +0.018 | +0.012 | +0.008 |
| m502_decayed_hl6_orth_size_only | +0.017 | +0.003 | +0.018 | +0.026 | +0.005 | +0.011 | +0.021 | +0.013 | +0.010 |

Truncation-invariance test of the feature builder: passed = **True**; 35675 values compared at dates ['2025-09-30', '2024-07-31']; max |difference| over features = 0.0.


## Interpretation
The signal is a **weak but robust standalone predictor and is not a proxy for size or liquidity**: IC +0.0101 to +0.0133 (t 2.4-3.0) for every half-life, the 12-month count (+0.0117, t 2.82), the size-and-liquidity-and-age-orthogonalised version (+0.0114, t 2.84, residual IC vs the composite +0.0097, t 2.30, permutation p 0.005) and the size-only orthogonalised version (+0.0133, t 3.00); only the 3-month count is insignificant (+0.0061). But it **does not improve the composite at any weight**: paired gain -0.0011 (3-month count) to +0.0013 (t between -0.6 and +0.7) at 1/8 weight, +0.0000 at 2x weight and -0.0051 at 4x weight (IC 0.0315, whose t rises to 2.84 only because the block's IC is less volatile than the composite's while the mean falls). A 0.012 IC is a third of the composite's, so blending it in equal weight can only dilute the mean IC. Two honest cautions: the Bonferroni-adjusted p over the original family of 10 is 0.05-0.10 (borderline), and the project-wide family (see the final report) is much larger. **Status: IMPLEMENTED_BUT_INCONCLUSIVE - a real-looking, tiny effect (IC ~0.012) of no practical value on its own.**

**Status: IMPLEMENTED_BUT_INCONCLUSIVE**

## Limitations
- Item 5.02 mixes departures, appointments, director elections and compensation arrangements, which cannot be separated without the filing text (out of scope); the sign of the effect (more changes -> lower return) is an average.
- 68 months; the IC is positive in most years but small each year.
- Only permutation p and standalone t are informative for such a weak signal; a longer sample would be needed for a practical assessment.

## Follow-up
None practical: an IC of 0.012 is not worth a group in the composite. The item-level content (which needs text) is where any information would be, and that is out of scope.
