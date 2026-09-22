# Correlation-peer return-gap signals (FEAT_PEER_GAP)

Implementation: `feat_peer_gap.py` (run: `.venv/bin/python experiments/feat_peer_gap/feat_peer_gap.py`). Everything it writes goes to `output/`.

## Objective
Test whether the returns of a stock's five most-correlated peers (within FF49, market cap >= $500M, 60-month correlation) - peer return, own-minus-peer gap, peer momentum - add IC to the composite, beyond a plain industry-return control.

## Hypothesis
Cross-stock information is absent from a stock's own characteristics: peers' month-t returns predict its next month (lead-lag / Peer Index), and laggards catch up (peer return gap).

## Research origin
docs/RESEARCH.md Part I sec 3.6 item 1 (Peer Return Gap 1.26%/mo, t 3.81 in China; Avramov-Ge, *Dual peer effects*, JFE 2026: US Peer Index predicts returns without reversal and ML on own characteristics does not subsume it; grade C).

## Implementation
`peer_builder`: for each characteristic month t from 2020-12, within FF49 among stocks with market cap >= $500M and a month-t return, correlation of monthly returns over months t-59..t (>= 36 overlapping months); peers = top-5 by correlation. Features: `peer_ret_1m` (+), `peer_gap_1m` = own - peer month-t return (-), `peer_ret_12_1` (+), control `ind_ret_1m_control` (+, FF49 mean excluding self). Only returns up to month t are used.

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
.venv/bin/python experiments/feat_peer_gap/feat_peer_gap.py
```

## Results
Additive test of each block on the composite (sign fixed in advance; `gain` = IC(composite + block) - IC(composite), paired over 68 months; `Bonferroni p` = permutation p x number of blocks in this file):

| name | sign | coverage | block IC | IC t | comp+block IC | IC gain | paired t | resid IC | resid t | corr w/ comp | perm p | Bonferroni p | gain small | gain mid | gain large | IR gross | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| peer_ret_1m | +1 | 0.98 | +0.0079 | +0.47 | 0.0371 | +0.0005 | +0.07 | +0.0070 | +0.45 | -0.03 | 0.114 | 0.458 | +0.0005 | -0.0008 | +0.0032 | 0.70 | kill |
| peer_gap_1m | -1 | 0.98 | -0.0071 | -1.02 | 0.0325 | -0.0042 | -1.54 | -0.0066 | -0.99 | +0.01 | 0.990 | 1.000 | -0.0028 | -0.0069 | -0.0031 | 0.43 | kill |
| peer_ret_12_1 | +1 | 0.98 | -0.0018 | -0.11 | 0.0361 | -0.0005 | -0.09 | -0.0031 | -0.20 | +0.04 | 0.075 | 0.299 | -0.0030 | -0.0002 | +0.0049 | 0.90 | kill |
| ind_ret_1m_control | +1 | 0.98 | +0.0003 | +0.02 | 0.0346 | -0.0020 | -0.35 | -0.0014 | -0.09 | -0.03 | 0.104 | 0.418 | -0.0003 | -0.0031 | -0.0013 | 0.67 | kill |

Standalone block IC by calendar year and by size tercile (all blocks, signed as tested):

| block | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | small | mid | large |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| peer_ret_1m | -0.018 | -0.008 | -0.004 | -0.018 | +0.045 | +0.072 | +0.012 | +0.002 | +0.008 |
| peer_gap_1m | +0.000 | -0.010 | -0.008 | -0.010 | -0.020 | +0.012 | +0.000 | -0.018 | -0.006 |
| peer_ret_12_1 | -0.058 | +0.062 | -0.036 | +0.026 | +0.001 | -0.007 | -0.002 | -0.003 | +0.007 |
| ind_ret_1m_control | -0.018 | -0.015 | -0.002 | -0.018 | +0.034 | +0.032 | +0.006 | -0.007 | +0.003 |

Truncation-invariance test of the feature builder: passed = **True**; 15615 values compared at dates ['2025-09-30', '2024-07-31']; max |difference| over features = 0.0.


## Interpretation
No peer feature helps: `peer_ret_1m` gain +0.0005 (t 0.07), `peer_gap_1m` -0.0042 (t -1.54), `peer_ret_12_1` -0.0005, control -0.0020. Correlation peers are no better than plain industry returns (control IC 0.0003 vs `peer_ret_1m` 0.0079, both insignificant). The US large-cap lead-lag effect is not visible at monthly frequency, which is consistent with the literature finding it mostly at weekly/daily frequency and in small-cap / low-attention stocks. **All killed.**

**Status: IMPLEMENTED_BUT_FAILED (all blocks killed by the pre-registered rule)**

## Limitations
- Correlation peers from 60 months of monthly returns are noisy (60 observations); the paper uses daily returns.
- Peers restricted to >= $500M so both sides are reasonably liquid; smaller peers were not tried.
- Four blocks; Bonferroni factor 4.

## Follow-up
TNIC text-based peers (`experiments/feat_tnic_peers`) replace the noisy correlation peers with a cleaner definition.
