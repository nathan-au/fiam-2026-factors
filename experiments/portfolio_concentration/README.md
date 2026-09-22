# Concentration / conviction: fewer, larger positions on the composite (PORTFOLIO_CONCENTRATION)

Implementation: `portfolio_concentration.py` (run: `.venv/bin/python experiments/portfolio_concentration/portfolio_concentration.py`, about 1 minute).

## Objective
Test whether raising the LP's per-name cap (0.5% / 1% / 1.5% / 2% / 3%, with and without the 10% turnover cap) concentrates the composite book into its highest-conviction names and raises IR, and check the 30-quantile-bucket table for whether the composite's extremes carry the alpha.

## Hypothesis
If the extremes carry the alpha (bucket returns steepest in the outer buckets), concentration raises IR; otherwise it only adds idiosyncratic risk. Pre-registered adoption: net IR >= the 1% control + 0.10, paired monthly net-return t >= 1, and >= 100 positions every month (FIAM position-count rule).

## Research origin
docs/RESEARCH.md Part I sec 3.9 item 7 (Quantitativo LTR: 30 quantiles beat 10/20/40, i.e. the top ~3%; the Russell-1000 agent's edge is concentrated in its top 20; "only pursue it if the decile table shows the extremes carry the alpha").

## Implementation
`quantile_table` (mean next-month excess return by within-month composite bucket, 30 buckets, universe rows, equal month weight) and the frozen LP with `max_weight` in {0.005, 0.01, 0.015, 0.02, 0.03}, each with `turnover = 10%` (`t10_*`) and without (`free_*`). Note the LP requires 1/max_weight names per side, so 3% gives about 33 per side.

## Experimental setup
Common harness (a verbatim copy of the data / rank-transform / LP / cost / performance code of `experiments/largecap/largecap.py`, embedded so the script runs on its own; the LP functions `_solve_lp` / `build_portfolio` are redefined in the script with optional extra behaviours and reproduce the frozen LP when the options are off).

- **Signal**: the frozen composite (18 factors in 7 economic groups, equal weight, re-ranked in the universe each month; no fitting). It reproduces `experiments/largecap`: universe rank IC 0.0366 (t 1.91), `lc_t10` gross IR 0.615 / net IR 0.541, 2025 net return -12.4%.
- **Universe**: price >= $5, market cap >= $2,000M, 126-day dollar volume >= $10M, both betas observed (about 1,206 stocks per month); 68 test months 2021-01..2026-08.
- **Portfolio**: the frozen LP (dollar-neutral, neutral to `beta_60m` and `betabab_1260d`, net sector <= 5% of NAV, gross sector share <= 35%, gross 200%, per-name cap 1%, 10% one-way turnover cap = `lc_t10`), with tiered *assumed* trading and borrow costs (5/10/20 bp trading and 30/75/200 bp borrow by market-cap tier; not measured).
- **Caveat that applies to every number below**: a single 68-month path; the standard error of a single-path IR is about 0.46, so IR differences under ~0.5 are not distinguishable from noise on their own. Paired monthly net-return t-stats are reported next to IR; with 20+ variants per file, a few |t| > 1 are expected by chance.

## Baseline
`t10_w010` (the frozen `lc_t10`, 1% cap) and `free_w010` for the uncapped variants.

## Commands
```
.venv/bin/python experiments/portfolio_concentration/portfolio_concentration.py
```

## Results (68 months, single path)
30-bucket table, mean next-month excess return in %/month, bucket 0 = lowest composite score ... 29 = highest: 0.11 0.29 0.63 0.54 0.58 0.64 0.95 0.99 0.75 0.73 0.66 0.76 0.73 0.82 0.80 0.82 0.65 0.54 0.88 0.78 0.80 0.72 0.89 0.75 0.86 0.70 0.98 0.76 0.84 0.87

The spread top bucket minus bottom bucket is +0.76%/month, but the gradient is not steepest at the top: the bottom two buckets (0.11, 0.29) are far below the rest (0.5-1.0, noisy, non-monotone), so most of the alpha is *avoiding the worst ~7%* (the short side), while the best buckets are not clearly better than the middle.

| variant | cap | IR gross | IR net | d IR net vs 1% | paired t vs 1% | CAGR net % | max DD % | 2025 net | worst month | avg pos | min pos | top-10 share | beta | roll beta min | roll beta max | >=100 pos every month |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| t10_w005 | 0.005 | 0.590 | 0.500 | -0.041 | -1.06 | 9.2 | -15 | -0.102 | -0.073 | 433 | 402 | 0.025 | -0.046 | -0.489 | 0.237 | True |
| free_w005 | 0.005 | 0.611 | 0.464 | +0.031 | -0.45 | 9.0 | -20 | -0.106 | -0.077 | 402 | 401 | 0.025 | -0.072 | -0.496 | 0.260 | True |
| t10_w010 | 0.010 | 0.615 | 0.541 | +0.000 |  | 11.0 | -19 | -0.124 | -0.089 | 228 | 202 | 0.050 | -0.061 | -0.598 | 0.322 | True |
| free_w010 | 0.010 | 0.572 | 0.433 | +0.000 |  | 9.5 | -25 | -0.164 | -0.086 | 202 | 201 | 0.050 | -0.120 | -0.677 | 0.261 | True |
| t10_w015 | 0.015 | 0.761 | 0.697 | +0.155 | +2.08 | 14.6 | -23 | -0.120 | -0.099 | 159 | 139 | 0.075 | -0.096 | -0.684 | 0.351 | True |
| free_w015 | 0.015 | 0.778 | 0.644 | +0.211 | +2.06 | 13.5 | -26 | -0.153 | -0.101 | 138 | 136 | 0.075 | -0.131 | -0.788 | 0.203 | True |
| t10_w020 | 0.020 | 0.783 | 0.724 | +0.183 | +1.85 | 16.0 | -32 | -0.216 | -0.102 | 124 | 106 | 0.100 | -0.090 | -0.668 | 0.376 | True |
| free_w020 | 0.020 | 0.895 | 0.767 | +0.334 | +2.36 | 16.7 | -30 | -0.188 | -0.113 | 105 | 103 | 0.100 | -0.179 | -0.931 | 0.215 | True |
| t10_w030 | 0.030 | 0.811 | 0.754 | +0.213 | +1.87 | 17.1 | -33 | -0.214 | -0.090 | 93 | 72 | 0.150 | -0.089 | -0.701 | 0.396 | False |
| free_w030 | 0.030 | 0.866 | 0.752 | +0.319 | +2.02 | 18.7 | -41 | -0.254 | -0.130 | 71 | 69 | 0.150 | -0.257 | -1.092 | 0.211 | False |

## Interpretation
Higher caps raise net IR (1.5%: 0.70, 2%: 0.72, 3%: 0.75 vs 0.54; paired t 1.8-2.1) but at a large cost in risk: max drawdown -23% / -32% / -33% vs -19% (and -41% for `free_w030`), 2025 return -12% to -21%, top-10 names' gross share 7.5-15% vs 5%, and the 3% cap gives only ~70-90 positions (violates the 100-position rule). Sub-1% caps (0.5%) lower IR (0.50) and drawdown (-15%). The rule is formally met for 1.5% and 2% (net IR +0.16 / +0.18, t 2.08 / 1.85, >= 100 positions), but (a) the 30-bucket table does not show the top extremes carrying the alpha, (b) 8 non-control variants were tested so a t of ~2 is unremarkable (Bonferroni-adjusted p about 0.3), (c) the improvement in IR comes with a worse worst-month and drawdown profile, i.e. it concentrates factor risk as docs/RESEARCH.md Part I warned. **Status: IMPLEMENTED_BUT_INCONCLUSIVE: a modest, unconfirmed IR gain with materially higher drawdown; not adopted.**

## Limitations
- Single path; the paired t of ~2 is one of many looks.
- The IR gain may reflect the LP loading more on the short-side extremes (where the alpha is) rather than "conviction" in the long book; not decomposed.
- Costs are assumed tiers; concentrated books hold fewer, larger positions whose impact costs are understated by the flat tiers.

## Follow-up
If concentration were pursued, test it out of sample with a longer history and with a drawdown / risk-based criterion rather than IR.
