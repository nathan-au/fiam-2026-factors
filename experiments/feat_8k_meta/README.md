# 8-K item-code metadata: intensity, news/no-news reversal, event flags, earnings timing (FEAT_8K_META)

Implementation: `feat_8k_meta.py` (run: `.venv/bin/python experiments/feat_8k_meta/feat_8k_meta.py`). Everything it writes goes to `output/`.

## Objective
Test structured 8-K metadata (filing dates and item codes only) as composite blocks: filing frequency, decayed bad-event and management-change intensity, reversal restricted to no-news months, and expected-earnings-month flags.

## Hypothesis
8-K metadata carries mid/large-cap information the 147 characteristics lack: more filings -> lower returns (-), decayed bad-event intensity (-), short-term reversal only for stocks with no 8-K in the month (+), and a stock expected to announce earnings next month earns a premium (+, a falsification test since the premium is reported to have disappeared).

## Research origin
docs/RESEARCH.md Part I sec 3.4 items 1-4 (8-K filing frequency predicts lower returns and volatility, ~4.3%/yr spread, *Management Science*; reversal is liquidity provision and works for non-news moves; 4.02 / 4.01 / 2.05 event flags; Frazzini-Lamont earnings-announcement premium vs Heitz et al. 2020), docs/RESEARCH.md Part II sec 2.11 (r/algotrading 1u19hnl: `has_earnings_in_window` a top XGBoost feature) and H7 (graded decayed event intensity instead of sparse flags). **Scope**: only the structured `filing_date` and `items` fields are used; no filing text and no LLM (the text/LLM 8-K ideas are excluded from this round).

## Implementation
`k8_builder`: from `fiam/8k_20150101_20260831_identified.parquet` (373k filings, `permno`-linked), a filing dated in month m is known at the end of month m; monthly counts of filings (each filing once), of items 2.02, of bad items (4.02, 4.01, 2.05, 2.06) and of 5.02. Features are NaN if the stock has no 8-K in the trailing 24 months (unmapped / dormant; 94.4% of universe test rows are covered), so "no filings" is never confused with "not in the 8-K file". Blocks: `k8_count_12m` (-), `k8_abnormal_3m` (-; last-3-month count minus the prior-12-month quarterly average), `k8_decayed_all_hl6` (-; exponentially decayed count, half-life 6 months), `rev_nonews` (+; -rank(ret_1_0) only for stocks with no 8-K in month t) and its control `rev_news_control`, `bad_event_decayed_hl6` (-), `bad_event_flag_3m` (-), `mgmt_5_02_decayed_hl6` (-), `eap_same_month_last_year` (+; a 2.02 filing in the same calendar month a year earlier) and `eap_any_quarter` (+; a 2.02 at t-2/-5/-8/-11).

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
.venv/bin/python experiments/feat_8k_meta/feat_8k_meta.py
```

## Results
Additive test of each block on the composite (sign fixed in advance; `gain` = IC(composite + block) - IC(composite), paired over 68 months; `Bonferroni p` = permutation p x number of blocks in this file):

| name | sign | coverage | block IC | IC t | comp+block IC | IC gain | paired t | resid IC | resid t | corr w/ comp | perm p | Bonferroni p | gain small | gain mid | gain large | IR gross | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| k8_count_12m | -1 | 0.94 | -0.0013 | -0.18 | 0.0330 | -0.0036 | -1.05 | -0.0073 | -0.90 | +0.13 | 0.871 | 1.000 | -0.0019 | -0.0008 | -0.0066 | 0.50 | kill |
| k8_abnormal_3m | -1 | 0.94 | +0.0019 | +0.46 | 0.0366 | -0.0000 | -0.01 | +0.0028 | +0.69 | -0.02 | 0.239 | 1.000 | -0.0046 | -0.0002 | +0.0054 | 0.44 | kill |
| k8_decayed_all_hl6 | -1 | 0.94 | -0.0002 | -0.02 | 0.0333 | -0.0033 | -0.91 | -0.0067 | -0.77 | +0.13 | 0.831 | 1.000 | -0.0015 | +0.0003 | -0.0069 | 0.48 | kill |
| rev_nonews | +1 | 0.94 | -0.0050 | -0.52 | 0.0337 | -0.0029 | -0.97 | +0.0007 | +0.07 | +0.03 | 0.970 | 1.000 | -0.0044 | -0.0006 | -0.0034 | 0.48 | kill |
| rev_news_control | +1 | 0.94 | -0.0084 | -0.77 | 0.0328 | -0.0038 | -1.00 | -0.0081 | -0.88 | +0.02 | 0.796 | 1.000 | -0.0016 | -0.0052 | -0.0060 | 0.58 | kill |
| bad_event_decayed_hl6 | -1 | 0.94 | +0.0043 | +0.67 | 0.0360 | -0.0006 | -0.38 | -0.0105 | -1.65 | +0.10 | 0.577 | 1.000 | -0.0017 | -0.0012 | +0.0024 | 0.64 | kill |
| bad_event_flag_3m | -1 | 0.94 | -0.0013 | -0.23 | 0.0362 | -0.0005 | -1.48 | -0.0315 | -1.87 | +0.10 | 0.846 | 1.000 | -0.0011 | -0.0005 | +0.0004 | 0.70 | kill |
| mgmt_5_02_decayed_hl6 | -1 | 0.94 | +0.0125 | +2.87 | 0.0377 | +0.0010 | +0.47 | +0.0084 | +1.78 | +0.09 | 0.010 | 0.100 | +0.0023 | +0.0022 | +0.0015 | 0.92 | kill |
| eap_same_month_last_year | +1 | 0.94 | -0.0004 | -0.06 | 0.0348 | -0.0018 | -1.13 | -0.0038 | -0.37 | -0.01 | 0.861 | 1.000 | -0.0054 | +0.0010 | +0.0001 | 0.59 | kill |
| eap_any_quarter | +1 | 0.94 | -0.0071 | -0.99 | 0.0334 | -0.0033 | -1.68 | +0.0075 | +0.69 | +0.00 | 0.990 | 1.000 | -0.0052 | -0.0006 | -0.0043 | 0.73 | kill |

Standalone block IC by calendar year and by size tercile (all blocks, signed as tested):

| block | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | small | mid | large |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| k8_count_12m | +0.001 | -0.015 | +0.021 | +0.001 | -0.022 | +0.012 | +0.007 | +0.008 | -0.013 |
| k8_abnormal_3m | +0.004 | +0.001 | +0.008 | -0.013 | +0.011 | -0.000 | -0.011 | +0.004 | +0.012 |
| k8_decayed_all_hl6 | +0.007 | -0.017 | +0.026 | +0.006 | -0.032 | +0.013 | +0.008 | +0.012 | -0.014 |
| rev_nonews | -0.003 | -0.000 | -0.005 | +0.015 | -0.024 | -0.016 | -0.009 | +0.003 | -0.006 |
| rev_news_control | +0.016 | +0.001 | +0.005 | -0.013 | -0.036 | -0.031 | +0.001 | -0.016 | -0.014 |
| bad_event_decayed_hl6 | +0.008 | -0.006 | +0.012 | +0.024 | -0.012 | -0.004 | +0.001 | +0.002 | +0.009 |
| bad_event_flag_3m | +0.016 | -0.002 | +0.001 | -0.002 | -0.014 | -0.009 | -0.014 | +0.004 | +0.004 |
| mgmt_5_02_decayed_hl6 | +0.016 | +0.003 | +0.019 | +0.022 | +0.003 | +0.012 | +0.021 | +0.013 | +0.009 |
| eap_same_month_last_year | -0.011 | +0.015 | -0.006 | +0.021 | -0.031 | +0.016 | -0.008 | +0.009 | -0.001 |
| eap_any_quarter | -0.016 | +0.005 | -0.011 | +0.020 | -0.043 | +0.006 | -0.014 | +0.005 | -0.014 |

Truncation-invariance test of the feature builder: passed = **True**; 64215 values compared at dates ['2025-09-30', '2024-07-31']; max |difference| over features = 0.0.


## Interpretation
None of the ten blocks passes the pre-registered rule. The falsification tests behave as the sceptical literature says: **the earnings-announcement premium is absent** (`eap_same_month_last_year` IC -0.0004; `eap_any_quarter` -0.0071, paired gain -0.0033, t -1.68), **filing frequency does not predict lower returns** (`k8_count_12m` IC -0.0013, gain -0.0036, t -1.05; the Management Science spread is not there in $2B+ stocks over 2021-26), and **reversal is not stronger without news** (`rev_nonews` IC -0.0050 with the wrong sign, and the control `rev_news_control` -0.0084 is equally absent). Bad-event flags (4.02/4.01/2.05/2.06) have wrong-signed residual IC (`bad_event_flag_3m` residual IC -0.0315, t -1.87) - the sparse-event mismatch Reddit warned about (H7's decayed intensity is not better: gain -0.0006, t -0.39). The one block with a notable standalone t-stat is **`mgmt_5_02_decayed_hl6`** (5.02: officer/director changes; IC +0.0125, t 2.87, permutation p 0.010, Bonferroni 0.10 over 10 blocks; residual IC +0.0084, t 1.78) whose paired gain on the composite is only +0.0010 (t 0.48): a genuine-looking but very weak signal that was followed up separately in `experiments/feat_8k_502_followup`. Several LP gross IRs rise to 0.9 for unchanged IC; that is noise (see the setup note).

**Status: IMPLEMENTED_BUT_FAILED (all ten blocks fail the pre-registered rule; one weak standalone signal was followed up separately)**

## Limitations
- Coverage 94%: unmapped stocks are neutral.
- The 8-K main text excludes press-release exhibits (median 3.9k characters), so 2.02 filings are pointers; item codes carry only the *fact* of an event, not its content or sign.
- Ten blocks tested (Bonferroni 10); signs were fixed from the literature before results, which is why `rev_nonews` and the bad-event flags count as failures rather than as sign-flipped discoveries.
- Earnings dates come from filing months, so the expected-month flag is a monthly, not daily, proxy.

## Follow-up
The 5.02 follow-up (below) is the only continuation. Any further 8-K work is text based (excluded) or needs event content.
