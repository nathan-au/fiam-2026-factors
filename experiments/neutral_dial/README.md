# Neutralisation depth as a dial, judged in the crowded-unwind windows (NEUTRAL_DIAL)

Implementation: `neutral_dial.py` (run: `.venv/bin/python experiments/neutral_dial/neutral_dial.py`, about 1 minute).

## Objective
Resolve the contradiction between docs/NEW.md (neutralise beyond beta: size, momentum, volatility, quality; sector-neutral ranking) and docs/REDDIT_RESEARCH.md (a 2026 multi-manager practitioner: tightly factor-neutral books did *worst*; hard neutrality to a common risk model converges on the "shared core") by measuring net IR, beta and P&L in the Jun-2025 / Jul-2025 / Jan-2026 / Jul-2026 unwind months as a function of neutralisation depth.

## Hypothesis (two-sided, stated in advance)
If NEW.md is right, deeper neutralisation lowers rolling beta and the unwind-month losses without costing IR. If Reddit is right, deeper neutralisation does not help (or hurts) stress-window P&L and costs IR. Decision rule (H3): weigh stress-window return at least as much as full-sample IR; N = 4 stress months (+ Jan-2021), so only the shape of the curve is informative.

## Research origin
docs/NEW.md sec 2 #6 and sec 3.9 items 1-2; docs/REDDIT_RESEARCH.md sec 2.4 (r/quant 1v30qx3, 1v02nb2 vs 1rvb4sm, 1ekpin6 Toraniko), H3.

## Implementation
LP with extra exact-neutrality equality constraints on columns of the preds frame (`extra_neutral`: log market cap, `ret_12_1` rank, `ivol_capm_21d` rank, `qmj` rank). Constraint dial: D0 beta-only (no sector limits) / D1 +sector limits (the current book) / D2 +size / D3 +size +momentum / D4 +size +momentum +volatility +quality. Alpha-projection dial: P2-P4 orthogonalise the composite month by month (OLS on the standardised columns) and keep the D1 constraints. S1/S2 rank the composite within GICS sector and drop / keep the sector limits. All use the 10% turnover cap (relaxed stepwise if infeasible; `cap relaxed` counts months).

## Experimental setup
Common harness (a verbatim copy of the data / rank-transform / LP / cost / performance code of `experiments/largecap/largecap.py`, embedded so the script runs on its own; the LP functions `_solve_lp` / `build_portfolio` are redefined in the script with optional extra behaviours and reproduce the frozen LP when the options are off).

- **Signal**: the frozen composite (18 factors in 7 economic groups, equal weight, re-ranked in the universe each month; no fitting). It reproduces `experiments/largecap`: universe rank IC 0.0366 (t 1.91), `lc_t10` gross IR 0.615 / net IR 0.541, 2025 net return -12.4%.
- **Universe**: price >= $5, market cap >= $2,000M, 126-day dollar volume >= $10M, both betas observed (about 1,206 stocks per month); 68 test months 2021-01..2026-08.
- **Portfolio**: the frozen LP (dollar-neutral, neutral to `beta_60m` and `betabab_1260d`, net sector <= 5% of NAV, gross sector share <= 35%, gross 200%, per-name cap 1%, 10% one-way turnover cap = `lc_t10`), with tiered *assumed* trading and borrow costs (5/10/20 bp trading and 30/75/200 bp borrow by market-cap tier; not measured).
- **Caveat that applies to every number below**: a single 68-month path; the standard error of a single-path IR is about 0.46, so IR differences under ~0.5 are not distinguishable from noise on their own. Paired monthly net-return t-stats are reported next to IR; with 20+ variants per file, a few |t| > 1 are expected by chance.

## Baseline
D1 = the frozen `lc_t10` book (beta + sector constraints).

## Commands
```
.venv/bin/python experiments/neutral_dial/neutral_dial.py
```

## Results (68 months, single path)
| variant | signal IC | IR gross | IR net | paired t vs D1 | beta | roll beta min | roll beta max | max DD % | 2025 net | cap relaxed (months) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D0_beta_only | 0.0366 | 0.655 | 0.586 | +0.64 | -0.025 | -0.414 | 0.414 | -20 | -0.187 | 0 |
| D1_beta_sector (current lc_t10) | 0.0366 | 0.615 | 0.541 |  | -0.061 | -0.598 | 0.322 | -19 | -0.124 | 0 |
| D2_plus_size | 0.0366 | 0.533 | 0.459 | -2.15 | -0.074 | -0.567 | 0.320 | -21 | -0.135 | 0 |
| D3_plus_size_mom | 0.0366 | 0.334 | 0.263 | -2.21 | -0.052 | -0.603 | 0.418 | -28 | -0.165 | 0 |
| D4_plus_size_mom_vol_qual | 0.0366 | 0.511 | 0.432 | -0.60 | -0.019 | -0.479 | 0.330 | -18 | -0.084 | 1 |
| P2_alpha_ortho_size | 0.0346 | 0.470 | 0.398 | -1.98 | -0.068 | -0.637 | 0.345 | -24 | -0.157 | 0 |
| P3_alpha_ortho_size_mom | 0.0340 | 0.326 | 0.256 | -1.86 | -0.029 | -0.556 | 0.490 | -27 | -0.175 | 0 |
| P4_alpha_ortho_all4 | 0.0244 | 0.538 | 0.463 | -0.36 | -0.021 | -0.561 | 0.416 | -18 | -0.084 | 0 |
| S1_sector_rank_no_sector_caps | 0.0310 | 0.506 | 0.415 | -1.20 | -0.005 | -0.287 | 0.258 | -18 | -0.142 | 0 |
| S2_sector_rank_plus_sector_caps | 0.0310 | 0.373 | 0.284 | -2.48 | -0.049 | -0.457 | 0.268 | -20 | -0.125 | 0 |

Net monthly return (%) in the unwind windows and the Jan-2021 squeeze:

| variant | Jun-2025 quant wobble | Jul-2025 quant wobble | Jan-2026 crowded unwind | Jul-2026 momentum unwind | Jan-2021 squeeze |
|---|---:|---:|---:|---:|---:|
| D0_beta_only | -4.4 | -3.4 | +0.5 | +7.3 | +1.3 |
| D1_beta_sector (current lc_t10) | -4.9 | -4.3 | +0.2 | +8.4 | +0.2 |
| D2_plus_size | -5.5 | -3.8 | +0.0 | +8.4 | -0.0 |
| D3_plus_size_mom | -4.6 | -4.9 | +0.1 | +7.2 | -0.1 |
| D4_plus_size_mom_vol_qual | -2.6 | -1.9 | +2.8 | +4.9 | +5.8 |
| P2_alpha_ortho_size | -5.6 | -3.8 | -0.3 | +7.8 | +0.6 |
| P3_alpha_ortho_size_mom | -5.4 | -4.7 | +0.2 | +6.6 | +0.5 |
| P4_alpha_ortho_all4 | -3.8 | -1.4 | +5.2 | +2.9 | +6.8 |
| S1_sector_rank_no_sector_caps | -4.3 | -4.5 | -0.9 | +6.9 | +1.7 |
| S2_sector_rank_plus_sector_caps | -3.7 | -3.0 | -0.5 | +8.0 | +1.9 |

Realised formation exposure to each neutraliser: maximum absolute monthly exposure, in units of the column (log size; ranks in [-1, 1]):

| variant | x_size max |exposure| | x_mom max |exposure| | x_vol max |exposure| | x_qual max |exposure| |
|---|---:|---:|---:|---:|
| D0_beta_only | 0.44 | 0.60 | 0.68 | 1.12 |
| D1_beta_sector (current lc_t10) | 0.29 | 0.54 | 0.67 | 1.14 |
| D2_plus_size | 0.00 | 0.52 | 0.67 | 1.12 |
| D3_plus_size_mom | 0.00 | 0.00 | 0.67 | 1.05 |
| D4_plus_size_mom_vol_qual | 0.00 | 0.00 | 0.00 | 0.00 |
| P2_alpha_ortho_size | 0.89 | 0.46 | 0.60 | 1.10 |
| P3_alpha_ortho_size_mom | 0.94 | 0.36 | 0.64 | 1.05 |
| P4_alpha_ortho_all4 | 0.39 | 0.32 | 0.18 | 0.20 |
| S1_sector_rank_no_sector_caps | 0.29 | 0.40 | 0.57 | 1.11 |
| S2_sector_rank_plus_sector_caps | 0.32 | 0.41 | 0.57 | 1.07 |

## Interpretation
Constraint depth trades IR for stress-window P&L, in the direction NEW.md predicts for the *full* dial and against it for the intermediate steps: D2/D3 (size, size+momentum) lose IR (0.46 / 0.26 net vs 0.54; paired t -2.1 / -2.2) with no stress benefit, but **D4 (size + momentum + volatility + quality neutral) cuts the 2025 loss from -12.4% to -8.4% and the unwind-window losses roughly in half (Jun-2025 -2.6% vs -4.9%, Jul-2025 -1.9% vs -4.3%, Jan-2026 +2.8% vs +0.2%, Jan-2021 +5.8% vs +0.2%) at a cost of 0.11 net IR** (0.43 vs 0.54, paired t -0.6, not significant); the alpha-projection version P4 gives the same stress protection at IC 0.024 (net IR 0.46). The composite carries a large quality and volatility exposure (max formation exposure 1.14 in `qmj` rank units in D1), which is the "shared core" Reddit describes; removing it protects the unwind months and costs some IR, so the two documents are each partly right: deep neutralisation to the *core factors* (volatility, quality) helps stress P&L, while neutralising size and momentum alone does not. Sector-neutral ranking (S1/S2) is no better than sector caps (IR 0.41 / 0.28). Beta-only (D0) has a slightly higher IR (0.59, t +0.6) and wider rolling beta (+/-0.41). With N = 4 unwind months and a single path, this is suggestive shape evidence, not proof. **Status: IMPLEMENTED_BUT_INCONCLUSIVE (tradeoff curve documented; not adopted).**

## Limitations
- Four unwind months (Jan-2021 is a squeeze, not a quant unwind); a single 68-month path; IR s.e. ~0.46.
- Exact-equality constraints can push the LP into relaxing the turnover cap (`cap relaxed`: 1 month at D4, 0 elsewhere), which changes turnover slightly.
- Exposures are neutralised at formation; realised exposure drifts.
- Neutraliser columns are my own simple choices (log size, `ret_12_1`, `ivol_capm_21d`, `qmj`), not a full risk-model (Barra/Toraniko) factor set.

## Follow-up
The most defensible use is D4 / P4 as a risk-oriented variant of the book (lower 2025 and unwind losses, ~0.1 lower IR); a longer sample would be needed to trust it. Not implemented as the default.
