# SEC Form 4 insider-trading blocks on the large-cap composite (FEAT_INSIDER_FORM4)

Implementation: `feat_insider_form4.py` (run: `.venv/bin/python experiments/feat_insider_form4/feat_insider_form4.py`). Everything it writes goes to `output/`.

## Objective
Test insider open-market purchases / sales (value scaled by market cap) and cluster buying (number of buying insiders) from SEC Form 4 filings as extra composite groups.

## Hypothesis
Insider open-market purchases (and clusters of buyers) predict higher, and insider sales lower, next-month returns in mid/large caps and add IC to the composite. Signs fixed in advance: purchases +, net purchases +, cluster +, sales -.

## Research origin
docs/RESEARCH.md Part I sec 3.7 (insider composite of role, size, clustering gives >= 1%/mo alpha equal-weighted, "works best for mid/large caps", alpha compressed since pre-2010; SEC EDGAR, free, filed within 2 business days).

## Implementation
`fetch_quarter` downloads the SEC DERA *Insider Transactions Data Sets* (`<YYYY>q<Q>_form345.zip`, ~16 MB each, 2020q1-2026q1 available; 2026q2+ returned HTTP errors) into `cache/sec_form345/` using a generic user agent (no personal contact detail was sent). Form 4 ('4') non-derivative open-market transactions with code P (purchase) or S (sale); value = shares x price; dated by `FILING_DATE` (public availability) and aggregated by filing month <= t; issuer CIK -> permno via the identified 8-K file (94.5% of universe test rows covered; uncovered = NaN, not "no trades"). Blocks: `ins_buy_6m_over_mcap`, `ins_net_buy_6m_over_mcap`, `ins_sell_6m_over_mcap` (sign -), `ins_buyers_3m` (sum of monthly distinct buying insiders over 3 months).

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
.venv/bin/python experiments/feat_insider_form4/feat_insider_form4.py
```

## Results
Additive test of each block on the composite (sign fixed in advance; `gain` = IC(composite + block) - IC(composite), paired over 68 months; `Bonferroni p` = permutation p x number of blocks in this file):

| name | sign | coverage | block IC | IC t | comp+block IC | IC gain | paired t | resid IC | resid t | corr w/ comp | perm p | Bonferroni p | gain small | gain mid | gain large | IR gross | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ins_buy_6m_over_mcap | +1 | 0.95 | +0.0003 | +0.05 | 0.0366 | -0.0001 | -0.05 | +0.0182 | +1.63 | -0.09 | 0.542 | 1.000 | -0.0021 | +0.0021 | +0.0001 | 0.57 | kill |
| ins_net_buy_6m_over_mcap | +1 | 0.95 | +0.0161 | +1.36 | 0.0375 | +0.0009 | +0.24 | +0.0134 | +1.23 | +0.04 | 0.214 | 0.856 | -0.0030 | +0.0022 | +0.0021 | 0.43 | kill |
| ins_sell_6m_over_mcap | -1 | 0.95 | +0.0188 | +1.55 | 0.0384 | +0.0017 | +0.49 | +0.0152 | +1.38 | +0.07 | 0.055 | 0.219 | -0.0020 | +0.0026 | +0.0033 | 0.48 | kill |
| ins_buyers_3m | +1 | 0.95 | -0.0035 | -0.63 | 0.0368 | +0.0002 | +0.13 | +0.0030 | +0.21 | -0.07 | 0.294 | 1.000 | -0.0001 | +0.0002 | -0.0003 | 0.50 | kill |

Standalone block IC by calendar year and by size tercile (all blocks, signed as tested):

| block | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | small | mid | large |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ins_buy_6m_over_mcap | -0.003 | -0.005 | +0.006 | -0.002 | +0.003 | +0.003 | -0.001 | +0.002 | +0.001 |
| ins_net_buy_6m_over_mcap | +0.040 | +0.056 | -0.004 | -0.015 | +0.008 | +0.008 | +0.008 | +0.018 | +0.016 |
| ins_sell_6m_over_mcap | +0.042 | +0.066 | -0.005 | -0.014 | +0.009 | +0.013 | +0.011 | +0.020 | +0.018 |
| ins_buyers_3m | -0.002 | -0.003 | -0.001 | +0.001 | -0.001 | -0.020 | +0.001 | -0.009 | -0.006 |

Truncation-invariance test of the feature builder: passed = **True**; 30573 values compared at dates ['2024-07-31', '2023-11-30', '2025-08-31']; max |difference| over features = 0.0.


## Interpretation
No insider block passes. Standalone ICs are small and mostly insignificant (`ins_net_buy` +0.0161, t 1.36; `ins_sell` +0.0188, t 1.55 with the pre-registered negative sign, i.e. more selling -> lower returns, permutation p 0.055; `ins_buy` +0.0003; `ins_buyers` -0.0035) and the paired gains on the composite are +0.0017 or less (t <= 0.49). Insider *sales* have the only suggestive signal (residual IC +0.0152, t 1.38), the same direction as the literature, but the effect is well below what the composite needs. Insider purchases are rare in $2B+ stocks (most large-cap insider activity is scheduled selling), which reduces power. **All killed.**

**Status: IMPLEMENTED_BUT_FAILED (all blocks killed by the pre-registered rule)**

## Limitations
- 2021-2026Q1 only for the data (2026Q2-Q3 not published in the data sets), so the last ~5 test months have stale 6-month windows.
- Role information (CEO vs director) and the routine-vs-opportunistic split were not used; the literature's alpha comes from those refinements.
- Four blocks; Bonferroni factor 4.

## Follow-up
A role-weighted / opportunistic-only insider composite would be the natural refinement; not run because none of the simple blocks shows a signal.
