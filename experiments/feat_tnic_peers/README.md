# Text-based-industry (TNIC-3) peer signals on the large-cap composite (FEAT_TNIC_PEERS)

Implementation: `feat_tnic_peers.py` (run: `.venv/bin/python experiments/feat_tnic_peers/feat_tnic_peers.py`). Everything it writes goes to `output/`.

## Objective
Test whether the returns of a stock's most text-similar rivals (Hoberg-Phillips TNIC-3 network, top-5 by similarity) - peer return, own-minus-peer gap, peer momentum - add IC to the composite, beyond a plain industry-return control.

## Hypothesis
A stock's most similar rivals' recent returns carry information neither its own characteristics nor plain industry returns do ("TNIC peer momentum is substantially more significant than SIC-based peers or own-firm momentum"). Signs fixed in advance: peer return +, own-minus-peer gap - (laggards catch up), peer momentum +.

## Research origin
docs/RESEARCH.md Part I sec 3.6 item 2 (Hoberg-Phillips TNIC data, free, keyed by gvkey).

## Implementation
`load_tnic_top` streams `tnic3_data.txt` (27M rows, 680 MB) from the 151 MB `cache/tnic/tnic3_data.zip` (downloaded on first run), keeps byear >= 2019 and each firm's top 15 rivals by similarity score. **Timing**: the network of business-description year `byear` is applied to months of year >= byear + 1 (all of that year's 10-K text is public by year-end); the database ends at 2023, so months of 2025-26 use the 2023 network (2-3 years stale). Peers = the 5 highest-scoring rivals of the stock's gvkey that have a month-t return and market cap >= $500M (>= 3 needed; 77.5% of universe rows have peers). gvkey -> permno through the panel (largest share class). Control: FF49 mean return excluding self.

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
.venv/bin/python experiments/feat_tnic_peers/feat_tnic_peers.py
```

## Results
Additive test of each block on the composite (sign fixed in advance; `gain` = IC(composite + block) - IC(composite), paired over 68 months; `Bonferroni p` = permutation p x number of blocks in this file):

| name | sign | coverage | block IC | IC t | comp+block IC | IC gain | paired t | resid IC | resid t | corr w/ comp | perm p | Bonferroni p | gain small | gain mid | gain large | IR gross | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tnic_peer_ret_1m | +1 | 0.77 | +0.0091 | +0.69 | 0.0368 | +0.0001 | +0.03 | +0.0072 | +0.59 | -0.02 | 0.095 | 0.378 | +0.0001 | +0.0004 | +0.0013 | 0.75 | kill |
| tnic_peer_gap_1m | -1 | 0.77 | -0.0113 | -1.80 | 0.0320 | -0.0046 | -2.07 | -0.0102 | -1.79 | +0.01 | 0.990 | 1.000 | -0.0047 | -0.0066 | -0.0031 | 0.51 | kill |
| tnic_peer_ret_12_1 | +1 | 0.77 | -0.0074 | -0.56 | 0.0332 | -0.0035 | -0.83 | -0.0064 | -0.53 | +0.01 | 0.502 | 1.000 | -0.0039 | -0.0043 | -0.0005 | 0.66 | kill |
| ind_ret_1m_control | +1 | 0.99 | +0.0012 | +0.08 | 0.0349 | -0.0017 | -0.30 | -0.0006 | -0.04 | -0.03 | 0.075 | 0.299 | +0.0003 | -0.0030 | -0.0011 | 0.60 | kill |

Standalone block IC by calendar year and by size tercile (all blocks, signed as tested):

| block | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | small | mid | large |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tnic_peer_ret_1m | -0.010 | -0.009 | -0.001 | -0.003 | +0.043 | +0.048 | +0.010 | +0.011 | +0.005 |
| tnic_peer_gap_1m | -0.004 | -0.015 | -0.011 | -0.010 | -0.021 | -0.006 | -0.010 | -0.020 | -0.006 |
| tnic_peer_ret_12_1 | -0.054 | +0.063 | -0.030 | -0.002 | -0.023 | +0.007 | -0.005 | -0.010 | -0.003 |
| ind_ret_1m_control | -0.017 | -0.015 | +0.000 | -0.018 | +0.037 | +0.031 | +0.008 | -0.006 | +0.004 |

Truncation-invariance test of the feature builder: passed = **True**; 18486 values compared at dates ['2025-09-30', '2024-07-31']; max |difference| over features = 0.0.


## Interpretation
No TNIC block passes, and the pre-registered *sign* of the gap is wrong: `tnic_peer_gap_1m` (own month-t return minus peers', predicted to reverse) has standalone IC -0.0113 (t -1.80) and a **negative** paired gain of -0.0046 (t -2.07) - stocks that outperformed their text-similar peers keep outperforming, the reverse of "laggards catch up" (I did not flip the pre-registered sign). `tnic_peer_ret_1m` has IC +0.0091 (t 0.69) and no gain (+0.0001), `tnic_peer_ret_12_1` -0.0074 (t -0.56), and the control industry return IC +0.0012. The by-year pattern of `tnic_peer_ret_1m` is negative or zero in 2021-24 (-0.010, -0.009, -0.001, -0.003) and positive only in 2025-26 (+0.043, +0.048), the years in which the network is 2-3 years stale, so it is not a coherent lead-lag effect. **All killed.**

**Status: IMPLEMENTED_BUT_FAILED (all blocks killed by the pre-registered rule)**

## Limitations
- The network is annual and the last 2-3 test years use a stale network; only 77.5% coverage of the universe.
- The one-year lag is conservative; the paper's strongest results use the contemporaneous network (which cannot be known at month t).
- Four blocks; Bonferroni factor 4.

## Follow-up
None. Combined with `experiments/feat_peer_gap` (correlation peers) the peer-return idea is exhausted at monthly frequency.
