# Crowding-state / confidence gates and volatility scaling on the composite book (CROWDING_GATE)

Implementation: `crowding_gate.py` (run: `.venv/bin/python experiments/crowding_gate/crowding_gate.py`, about 30 seconds).

## Objective
Test whether gates on crowding-state variables known before each month (trailing IC, recent factor-sleeve moves, the composite's own last-month D10-D1, S&P realised volatility) or continuous volatility scaling reduce the composite book's drawdown and its losses in the Jun-Jul 2025 / Jan-2026 / Jul-2026 unwinds at little cost in IR - and show that a gate cutoff tuned on a validation slice does not transfer.

## Hypothesis
(a) A gate on trailing-12-month IC (docs/RESEARCH.md Part I) will NOT fire before the unwinds because crowding leaves IC intact while impairing P&L (docs/RESEARCH.md Part II sec 2.6). (b) Gates on variance-type state variables (sleeve moves, book momentum, S&P vol) might. (c) A cutoff tuned on a 2017-2020 validation slice will not transfer. All thresholds are fixed in advance (z > 1 vs expanding history from 2015, gross x0.5 when a gate fires) and are not tuned, except the deliberately tuned `G1cal`. With 3-4 unwind months any positive result would be weak evidence.

## Research origin
docs/RESEARCH.md Part I sec 3.9 items 5-6 and sec 3.3 item 7 (When Alpha Breaks / Confidence Gate Theorem; dispersion and VIX as scaling inputs; Moreira-Muir vol management; the FIAM brief allows varying gross if predictable in advance) and docs/RESEARCH.md Part II sec 2.6 (r/quant 1u3n4x0: crowding is a variance shift, not an IC decay; r/algotrading 1sjicuf: a regime filter tuned on a validation slice zeroed 49.6% of validation dates and 0% of test dates).

## Implementation
The composite's `lc_t10` monthly net returns are rescaled ex post by the gate scale (a dollar-neutral book scales linearly; a cost of 7.5 bp x 2.0 gross x |change in scale| is charged). State variables at target month T use only data realised by the end of month T-1: G1 trailing-12-month composite IC < 0; G2 z-score of the 3-month mean absolute return of the seven factor-group sleeves (top-minus-bottom decile of each group) > 1; G3 z-score of last month's composite D10-D1 < -1; G4 z-score of the S&P 500 21-day realised volatility (`cache/SP500.csv` daily) > 1; G5 continuous vol scaling clip(median vol / vol, 0.5, 1.5); G6 any of G1-G3. z-scores use an expanding window from 2015 (>= 24 months). `G1cal`: the IC cutoff chosen among {-0.03..+0.03} to maximise Sharpe of the gated D10-D1 series on 2017-2020, then applied to the test window.

## Experimental setup
Common harness (a verbatim copy of the data / rank-transform / LP / cost / performance code of `experiments/largecap/largecap.py`, embedded so the script runs on its own; the LP functions `_solve_lp` / `build_portfolio` are redefined in the script with optional extra behaviours and reproduce the frozen LP when the options are off).

- **Signal**: the frozen composite (18 factors in 7 economic groups, equal weight, re-ranked in the universe each month; no fitting). It reproduces `experiments/largecap`: universe rank IC 0.0366 (t 1.91), `lc_t10` gross IR 0.615 / net IR 0.541, 2025 net return -12.4%.
- **Universe**: price >= $5, market cap >= $2,000M, 126-day dollar volume >= $10M, both betas observed (about 1,206 stocks per month); 68 test months 2021-01..2026-08.
- **Portfolio**: the frozen LP (dollar-neutral, neutral to `beta_60m` and `betabab_1260d`, net sector <= 5% of NAV, gross sector share <= 35%, gross 200%, per-name cap 1%, 10% one-way turnover cap = `lc_t10`), with tiered *assumed* trading and borrow costs (5/10/20 bp trading and 30/75/200 bp borrow by market-cap tier; not measured).
- **Caveat that applies to every number below**: a single 68-month path; the standard error of a single-path IR is about 0.46, so IR differences under ~0.5 are not distinguishable from noise on their own. Paired monthly net-return t-stats are reported next to IR; with 20+ variants per file, a few |t| > 1 are expected by chance.

## Baseline
`G0_ungated`: the frozen `lc_t10` composite book, net IR 0.541, 2025 return -12.4%, max drawdown -18.9%.

## Commands
```
.venv/bin/python experiments/crowding_gate/crowding_gate.py
```

## Results (68 months, single path)
| gate | IR net | d IR | CAGR % | max DD % | 2025 % | share gated | 4 unwind months, ungated % | 4 unwind months, gated % | mean net ret gated %/m | mean net ret not gated %/m | t (gated vs not) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G0_ungated | 0.541 | +0.000 | 11.0 | -18.9 | -12.4 | 0.00 | -0.7 | -0.7 |  | +0.95 |  |
| G1_trailing_IC12_lt_0 | 0.412 | -0.129 | 8.2 | -15.7 | -11.1 | 0.35 | -0.7 | -5.0 | +1.33 | +0.74 | +0.54 |
| G2_sleeve_moves_z_gt_1 | 0.349 | -0.192 | 7.8 | -20.9 | -14.6 | 0.26 | -0.7 | -4.9 | +1.88 | +0.61 | +1.17 |
| G3_book_momentum_z_lt_-1 | 0.505 | -0.036 | 9.8 | -15.4 | -9.2 | 0.18 | -0.7 | +1.4 | +1.00 | +0.94 | +0.04 |
| G4_spx_vol_z_gt_1 | 0.556 | +0.015 | 11.1 | -19.6 | -13.1 | 0.04 | -0.7 | -0.7 | -0.73 | +1.02 | -0.91 |
| G5_vol_scaled_(Moreira-Muir) | 0.382 | -0.159 | 8.1 | -21.8 | -15.8 | 0.65 | -0.7 | -3.4 | +1.33 | +0.24 | +1.17 |
| G6_any_of_G1_G2_G3 | 0.472 | -0.069 | 8.6 | -14.5 | -9.2 | 0.54 | -0.7 | -2.9 | +0.73 | +1.20 | -0.49 |
| G1cal_IC12_cutoff_tuned_on_2017_2020 | 0.471 | -0.070 | 9.0 | -15.7 | -11.1 | 0.32 | -0.7 | -5.0 | +1.07 | +0.88 | +0.17 |

Gross scale applied in the unwind windows (1.0 = not gated):

| gate | gross scale 2025-06 | gross scale 2025-07 | gross scale 2026-01 | gross scale 2026-07 | gross scale 2021-01 |
|---|---:|---:|---:|---:|---:|
| G0_ungated | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| G1_trailing_IC12_lt_0 | 1.000 | 1.000 | 0.500 | 0.500 | 0.500 |
| G2_sleeve_moves_z_gt_1 | 1.000 | 1.000 | 1.000 | 0.500 | 0.500 |
| G3_book_momentum_z_lt_-1 | 1.000 | 0.500 | 1.000 | 1.000 | 1.000 |
| G4_spx_vol_z_gt_1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| G5_vol_scaled_(Moreira-Muir) | 0.782 | 1.324 | 1.444 | 0.736 | 1.167 |
| G6_any_of_G1_G2_G3 | 1.000 | 0.500 | 0.500 | 0.500 | 0.500 |
| G1cal_IC12_cutoff_tuned_on_2017_2020 | 1.000 | 1.000 | 0.500 | 0.500 | 0.500 |

The validation-slice calibration chose an IC cutoff of -0.01 (Sharpe of the D10-D1 proxy by candidate cutoff: -0.159, -0.168, -0.154, -0.174, -0.171, -0.211, -0.198 for -0.03..+0.03; **negative for every candidate**: the composite's D10-D1 had no positive Sharpe in 2017-2020, so the "optimal" cutoff is chosen among losing options - the r/algotrading failure mode).

## Interpretation
**No gate improves the book.** Only G4 (S&P volatility) raises IR at all (+0.015) and it fires in 4% of months, none of the unwind months. G1 (trailing IC) and G2 fire in 35% / 26% of months but *not* in Jun/Jul-2025 (the IC-based gate misses the unwind, as hypothesis (a) predicts) and instead cut the Jul-2026 gain (+8.4% -> +4.2%), so the four unwind months as a group go from -0.7% ungated to -5.0% gated. G3 (own book momentum) is the only gate that helps the unwind windows (+1.4% vs -0.7%; it halves the Jul-2025 loss) and 2025 (-9.2% vs -12.4%) with an IR cost of 0.04, but it fires in 18% of months, only one of which (Jul-2025) is an unwind month, and gated months have the same mean return as non-gated months (t 0.04): it is one lucky month, not a detector. Vol scaling (G5) lowers IR (0.38) and deepens the drawdown (-21.8%). The tuned-on-validation gate `G1cal` behaves like G1 (IR 0.47; unwind sum -5.0%) - it transfers no information because the calibration slice had no signal. Cutting gross after variance shifts has a cost that a book with IR ~0.5 cannot afford. **Status: IMPLEMENTED_BUT_FAILED (no gate met the goal; the IC-gate failure to anticipate the unwinds and the non-transfer of the tuned cutoff both confirm the Reddit warning).**

## Limitations
- Four unwind months (Jan-2021 is a squeeze) and a single path; a gate's value is unidentifiable at this N.
- The ex-post linear rescaling ignores the extra turnover of changing gross beyond the assumed 7.5 bp cost.
- The sleeve / crowding variables are simple proxies computed from the panel, not manager-holdings overlap.

## Follow-up
None with this data. A real crowding measure needs holdings overlap (13F, quarterly and lagged; not implemented, see the blocked-data items in the final report).
