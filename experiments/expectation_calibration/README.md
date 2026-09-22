# Expectation calibration: live-IC haircut, fundamental-law realisation, power analysis, bootstrap uncertainty (EXPECTATION_CALIBRATION)

Implementation: `expectation_calibration.py` (run: `.venv/bin/python experiments/expectation_calibration/expectation_calibration.py`, about 20 seconds).

## Objective
Calibrate how the composite's headline numbers should be quoted: how much of the IC survives production (haircut), how much of the fundamental law an LP-constrained book realises, what 68 months can and cannot detect, and how uncertain the single-path IC and IR are.

## Hypothesis
(a) docs/RESEARCH.md's Part I fundamental-law expectation ("IC 0.03 -> IR 0.5-0.7") is optimistic because it omits the sqrt(12) for monthly bets and a transfer coefficient, and the constrained LP realises a fraction of the law. (b) After a 40-50% live haircut the composite's IR is about 0.3 gross and lower net. (c) 68 months cannot distinguish the composite's IC from zero at 80% power, and the bootstrap standard error of the IR is close to the theoretical ~0.45. No adopt/kill rule: the outputs set how headline numbers should be quoted.

## Research origin
docs/RESEARCH.md Part II sec 2.7 (r/quant 1q02d6g "Decline in IC going into prod": ~40-50% survives at multi-strats, 20-40% haircut as a general range; Azevedo-Hoegner-Velikov: 57% cumulative reduction for ML strategies), sec 2.13 (power analysis before backtesting, r/quant 1r6rc2d) and docs/RESEARCH.md Part I sec 5.

## Implementation
`main()`: book effective breadth = mean of 1/sum((w/gross)^2) over months; unconstrained law IR = IC x sqrt(N_eff x 12); implied transfer coefficient = realised gross IR / law IR; haircut table (0/20/40/50/57% applied to the monthly gross return, costs held at the assumed tiers, hurdle 4%/yr); minimum detectable IC and months needed = (2.80 x sd(IC)/IC)^2; stationary bootstrap (6-month blocks, 2,000 draws) of the mean IC and of gross / net IR; first-vs-second-half IC (Welch t) and a trend regression of monthly IC on time.

## Experimental setup
Common harness (a verbatim copy of the data / rank-transform / LP / cost / performance code of `experiments/largecap/largecap.py`, embedded so the script runs on its own; the LP functions `_solve_lp` / `build_portfolio` are redefined in the script with optional extra behaviours and reproduce the frozen LP when the options are off).

- **Signal**: the frozen composite (18 factors in 7 economic groups, equal weight, re-ranked in the universe each month; no fitting). It reproduces `experiments/largecap`: universe rank IC 0.0366 (t 1.91), `lc_t10` gross IR 0.615 / net IR 0.541, 2025 net return -12.4%.
- **Universe**: price >= $5, market cap >= $2,000M, 126-day dollar volume >= $10M, both betas observed (about 1,206 stocks per month); 68 test months 2021-01..2026-08.
- **Portfolio**: the frozen LP (dollar-neutral, neutral to `beta_60m` and `betabab_1260d`, net sector <= 5% of NAV, gross sector share <= 35%, gross 200%, per-name cap 1%, 10% one-way turnover cap = `lc_t10`), with tiered *assumed* trading and borrow costs (5/10/20 bp trading and 30/75/200 bp borrow by market-cap tier; not measured).
- **Caveat that applies to every number below**: a single 68-month path; the standard error of a single-path IR is about 0.46, so IR differences under ~0.5 are not distinguishable from noise on their own. Paired monthly net-return t-stats are reported next to IR; with 20+ variants per file, a few |t| > 1 are expected by chance.

## Baseline
The composite book at the assumed costs (IC 0.0366, gross IR 0.615, net 0.541).

## Commands
```
.venv/bin/python experiments/expectation_calibration/expectation_calibration.py
```

## Results
Fundamental law: composite IC 0.0366, effective breadth N_eff = 215 names, unconstrained IR = 0.0366 x sqrt(215 x 12) = 1.86, realised gross IR 0.615, **implied transfer coefficient 0.33**.

Haircut table:

| haircut on gross alpha | expected IC | expected IR gross | expected IR net (assumed costs) |
|---|---:|---:|---:|
| 0% | 0.037 | 0.615 | 0.541 |
| 20% | 0.029 | 0.433 | 0.359 |
| 40% | 0.022 | 0.251 | 0.178 |
| 50% | 0.018 | 0.160 | 0.087 |
| 57% | 0.016 | 0.097 | 0.023 |

Power: the monthly IC has s.d. 0.158, so the s.e. of the 68-month mean IC is 0.0191 and the minimum detectable IC at 80% power is **0.054**. Months needed to detect an IC of:

| IC to detect | months needed (two-sided 5%, 80% power) |
|---|---:|
| 0.050 | 78 |
| 0.037 | 143 |
| 0.030 | 217 |
| 0.020 | 488 |
| 0.015 | 868 |
| 0.010 | 1953 |

Bootstrap (stationary, 6-month blocks, 2,000 draws), 5th / 50th / 95th percentiles: mean IC **[0.006, 0.036, 0.065]**; gross IR **[-0.09, 0.61, 1.30]**; net IR **[-0.17, 0.54, 1.22]**; bootstrap s.e. of net IR 0.43; share of draws with net IR > 0: 0.89; with IC > 0: 0.98.

Decay within the sample: IC 2021-23 = 0.0536, 2024-26 = 0.0175 (Welch t of the difference 0.95); trend in monthly IC -0.00148 per month (t -1.53); by year: 2021: +0.077, 2022: +0.077, 2023: +0.006, 2024: +0.062, 2025: -0.026, 2026: +0.016.

## Interpretation
(a) **The fundamental law overstates the book by 3x**: the unconstrained law gives IR 1.86 for IC 0.0366 with ~215 effective names and 12 monthly bets a year; the beta-neutral, sector-capped, turnover-capped, 1%-per-name LP realises 0.615 gross (transfer coefficient 0.33). docs/RESEARCH.md's Part I "IC 0.03 -> IR 0.5-0.7" therefore happens to be about right for this constrained book but by an accident of two errors (missing sqrt(12) and missing transfer coefficient). (b) **A 40-50% haircut leaves gross IR 0.16-0.25 and net IR 0.09-0.18**; with the 57% cumulative reduction net IR is ~0.02, i.e. nothing after costs: the honest expected live IR of the composite is close to zero to modest, not 0.54. (c) **68 months can only detect IC >= 0.054** (the composite's 0.0366 needs 143 months; a haircut IC of 0.02 needs ~490 months). The bootstrap 5-95% band for the mean IC (0.006, 0.065) and net IR (-0.17, 1.22) contains zero at the low end: 89% of bootstrap draws have positive net IR and 98% positive IC, so the sign is likely but the size is unknowable; the bootstrap s.e. of net IR (0.43) matches the theoretical 0.46 used throughout the project. (d) The composite's IC **fell within the sample**: 0.054 in 2021-23 vs 0.018 in 2024-26 (Welch t 0.95; trend t -1.5, not significant) with a negative 2025 - consistent with the haircut logic (and with the composite being the crowded core, see `experiments/composite_overlap`). Status: IMPLEMENTED_AND_TESTED (calibration).

## Limitations
- The haircuts (20-57%) come from anecdotes and one paper about ML strategies; whether they apply to a factor-composite book is unknown (docs/RESEARCH.md Part II sec 5 item 5).
- The transfer coefficient is specific to this LP; the effective-breadth measure uses position concentration, not independent bets.
- Bootstrap blocks of 6 months are a choice; longer blocks would widen the bands.

## Follow-up
Quote the composite result as "IC 0.037 (t 1.9), net IR 0.54 in-sample, expect roughly half of that live" in the deck.
