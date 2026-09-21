# Follow-up: where does the FINRA short-interest signal come from, and does it survive borrow stress? (FEAT_SI_FOLLOWUP)

Implementation: `feat_si_followup.py` (run: `.venv/bin/python experiments/feat_si_followup/feat_si_followup.py`, about 1 minute after the FINRA cache is filled; needs network on the first run).

## Objective
Scrutinise the one result in this round that passed its pre-registered rule (`experiments/feat_short_interest`: FINRA days-to-cover adds +0.0089 IC to the composite, paired t 2.67, Bonferroni-in-file p 0.015; short-interest ratio standalone IC 0.045): which leg carries it, is it a persistent state or a timing effect, does it survive when high-short-interest shorts are expensive to borrow, and does it work as a short-side filter?

## Hypothesis
(a) The effect is concentrated in the high-SI decile's underperformance (short leg); (b) the information persists (an older file still predicts), i.e. it is a state, not an information-flow effect; (c) the book's net IR falls sharply when the highest-SI fifth of its shorts pays x3 / x5 / x10 the assumed borrow (Reddit sec 2.10: "the reason you want to short is correlated with the borrow becoming expensive"); (d) capping the short book at SI ratio <= 10% (squeeze / borrow filter, as docs/NEW.md sec 3.7 suggests) keeps most of the gain. No adopt rule beyond the parent's; this is a robustness study.

## Research origin
Own follow-up; docs/REDDIT_RESEARCH.md sec 2.10 (borrow adverse selection) and docs/NEW.md sec 3.7 (short interest "more valuable as a short-selection / squeeze-risk filter").

## Implementation
Reuses the FINRA feature builder of `feat_short_interest` (mid-month settlement file, published before month-end; symbol -> permno through the panel ticker) and the extended LP (`short_cap_col`, `short_cap_val`: shorts allowed only where the SI ratio <= 10%). (1) mean next-month excess return by SI-ratio and days-to-cover decile over universe rows; (2) IC of the signal built from the file k = 0, 1, 2, 3, 6 months older; (3) LP `lc_t10` books for the composite, +days-to-cover, +SI ratio, +both, and the composite (+dtc) with the short cap; for each, net IR with the highest-SI fifth of the short book paying x1 / x3 / x5 / x10 the tiered borrow (state-dependent stress).

## Experimental setup
Common harness (a verbatim copy of the data / rank-transform / LP / cost / performance code of `experiments/largecap/largecap.py`, embedded so the script runs on its own; the LP functions `_solve_lp` / `build_portfolio` are redefined in the script with optional extra behaviours and reproduce the frozen LP when the options are off).

- **Signal**: the frozen composite (18 factors in 7 economic groups, equal weight, re-ranked in the universe each month; no fitting). It reproduces `experiments/largecap`: universe rank IC 0.0366 (t 1.91), `lc_t10` gross IR 0.615 / net IR 0.541, 2025 net return -12.4%.
- **Universe**: price >= $5, market cap >= $2,000M, 126-day dollar volume >= $10M, both betas observed (about 1,206 stocks per month); 68 test months 2021-01..2026-08.
- **Portfolio**: the frozen LP (dollar-neutral, neutral to `beta_60m` and `betabab_1260d`, net sector <= 5% of NAV, gross sector share <= 35%, gross 200%, per-name cap 1%, 10% one-way turnover cap = `lc_t10`), with tiered *assumed* trading and borrow costs (5/10/20 bp trading and 30/75/200 bp borrow by market-cap tier; not measured).
- **Caveat that applies to every number below**: a single 68-month path; the standard error of a single-path IR is about 0.46, so IR differences under ~0.5 are not distinguishable from noise on their own. Paired monthly net-return t-stats are reported next to IR; with 20+ variants per file, a few |t| > 1 are expected by chance.

## Baseline
The frozen composite `lc_t10` book (net IR 0.541) and its own SI-stress row.

## Commands
```
.venv/bin/python experiments/feat_si_followup/feat_si_followup.py
```

## Results
Match rate 95.9% of universe test rows; universe SI ratio quantiles (10/50/90/99%): 1.1%, 3.0%, 9.5%, 19.8%; days-to-cover 1.7, 3.5, 8.2, 16.0.

Mean next-month excess return (%/month) by decile (1 = lowest value of the signal), universe rows, 68 months:

| decile (1 = lowest) | si_ratio | days_to_cover |
|---|---:|---:|
| 1 | 0.96 | 1.25 |
| 2 | 0.86 | 0.91 |
| 3 | 0.73 | 0.93 |
| 4 | 0.96 | 0.89 |
| 5 | 0.74 | 0.81 |
| 6 | 0.94 | 0.85 |
| 7 | 0.65 | 0.49 |
| 8 | 0.50 | 0.74 |
| 9 | 0.35 | 0.07 |
| 10 | 0.44 | 0.19 |

D1 - D10 is +0.52 %/month for the SI ratio and +1.06 %/month for days-to-cover. Relative to the universe mean, D10 is -0.28 / -0.52 and D1 is +0.25 / +0.54: the short-side (high-SI) underperformance accounts for about 53% (SI ratio) and 49% (days-to-cover) of the spread; the low-SI names outperform by about as much.

Staleness (IC of the signed rank built from an older FINRA file):

| signal built from the file k months older | IC (of -rank) | t |
|---|---:|---:|
| si_ratio_lag0m | 0.0446 | 2.82 |
| days_to_cover_lag0m | 0.0351 | 3.51 |
| si_ratio_lag1m | 0.0455 | 2.84 |
| days_to_cover_lag1m | 0.0305 | 3.10 |
| si_ratio_lag2m | 0.0452 | 2.83 |
| days_to_cover_lag2m | 0.0361 | 3.70 |
| si_ratio_lag3m | 0.0462 | 2.98 |
| days_to_cover_lag3m | 0.0364 | 3.71 |
| si_ratio_lag6m | 0.0446 | 2.87 |
| days_to_cover_lag6m | 0.0370 | 3.98 |

Portfolios (`lc_t10`, single path; "x3 / x5 / x10" = net IR when the highest-SI fifth of the short book pays that multiple of the assumed borrow):

| variant | IR gross | IR net | paired t vs comp | avg SI ratio, shorts | avg SI ratio, longs | net IR, top-SI-fifth shorts pay x3 | x5 | x10 | max DD % | 2025 net | beta | short_median_mcap_musd |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| comp | 0.615 | 0.541 |  | 0.074 | 0.037 | 0.521 | 0.501 | 0.451 | -18.9 | -0.124 | -0.061 | 7,340 |
| comp_plus_dtc | 0.659 | 0.584 | 0.378 | 0.084 | 0.026 | 0.565 | 0.545 | 0.496 | -20.3 | -0.128 | -0.022 | 5,610 |
| comp_plus_sir | 0.629 | 0.554 | 0.196 | 0.086 | 0.023 | 0.534 | 0.514 | 0.465 | -22.5 | -0.136 | -0.077 | 5,791 |
| comp_plus_dtc_sir | 0.544 | 0.468 | -0.309 | 0.092 | 0.018 | 0.448 | 0.429 | 0.379 | -23.2 | -0.127 | -0.030 | 5,040 |
| comp_shortcap10 | 0.590 | 0.508 | -0.565 | 0.047 | 0.037 | 0.485 | 0.462 | 0.405 | -9.8 | -0.066 | 0.002 | 9,682 |
| comp_plus_dtc_shortcap10 | 0.721 | 0.637 | 0.095 | 0.054 | 0.023 | 0.614 | 0.591 | 0.534 | -9.8 | -0.010 | 0.023 | 6,980 |

## Interpretation
(a) **Not just a short-leg effect**: about half of the decile spread comes from low-SI *outperformance* (D1 above the universe mean by +0.25 / +0.54 %/month). (b) **It is a persistent state, not a timing signal**: the IC of a 6-month-old file (0.0446 SI ratio, 0.037 days-to-cover) equals the fresh one (0.0446, 0.0351); short-interest ranks are very persistent, so the signal behaves like a slow characteristic. That makes it a legitimate cross-sectional characteristic, but it also means the +0.009 IC gain is partly a re-labelling of a known style exposure (its residual IC after the composite is still +0.033, t 3.6). (c) **Borrow stress barely matters here**: making the highest-SI fifth of shorts pay 10x the assumed borrow lowers net IR by only 0.09-0.10 for every book (composite 0.541 -> 0.451; composite+dtc 0.584 -> 0.496; with the short cap 0.508 -> 0.405), because the assumed borrow (30-200 bp) is small relative to the alpha and the book's shorts are $5-7B median names; the true cost of borrowing genuinely hard-to-borrow names is not in the panel, so this is a stress of my assumptions, not a measurement. (d) **The short-side SI cap is a strong risk filter**: barring shorts with SI ratio > 10% cuts the max drawdown from -18.9% to -9.8% and the 2025 loss from -12.4% to -6.6% at a small IR cost for the composite alone (net IR 0.508 vs 0.541; paired t -0.57), and the composite + days-to-cover with the cap has net IR 0.637 and the same -9.8% drawdown (paired t vs composite +0.10). The alpha *gain* from days-to-cover does not translate into a significant LP gain (composite+dtc net IR 0.584, paired t 0.38; adding both dtc and SI ratio is worse, 0.468), which is expected given IR s.e. ~0.46.
**Status: IMPLEMENTED_AND_TESTED. Short interest as a risk filter on the short book (squeeze / borrow control) is the most defensible use; as an alpha block it is a real cross-sectional characteristic but its portfolio benefit is unconfirmed.**

## Limitations
- No borrow-fee or availability data: the stress multiplies an assumption. High-SI names can be *unavailable* (a discontinuity), which this does not model; the SI cap in (d) is the practical protection.
- The drawdown improvement rests on a handful of months (Jan-2021 squeeze, 2025) - single path, 68 months.
- Ticker-based matching (4% of universe rows unmatched) and the panel's `shares` for the denominator.
- The 10% cap level was set a priori (roughly the 90th percentile of universe SI ratios), not tuned; no other level was tried, so no selection was made on these results.

## Follow-up
Test the short-side SI cap on a longer history (FINRA files exist back to 2015; the panel starts 2015-01) and on the ET / LightGBM books. If accepted, it is a one-line constraint in the LP.
