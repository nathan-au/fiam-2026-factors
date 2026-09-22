# State-dependent borrow-cost stress and forced-cover shocks on the composite book (BORROW_STRESS)

Implementation: `borrow_stress.py` (run: `.venv/bin/python experiments/borrow_stress/borrow_stress.py`, about 30 seconds).

## Objective
Stress the composite book's assumed short-side costs: uniform borrow multipliers, borrow that is expensive exactly for the strongest short theses, a price screen on shorts, and forced covers; report IR and drawdown under each. The panel has no borrow data, so this stresses assumptions rather than measuring them.

## Hypothesis
The composite's net IR (0.54 at the assumed 30 / 75 / 200 bp borrow tiers) survives a 2-5x borrow multiplier and a forced cover of a few percent of the short book, because its shorts are large caps (median $7B), i.e. the short leg is not where its edge (or fragility) lives; it should NOT survive if the expensive-borrow state coincides with the strongest short theses. No adopt rule: robustness study; a book whose net IR turns negative under a plausible stress is flagged.

## Research origin
docs/RESEARCH.md Part II sec 2.10 (r/algotrading 1w9yp8g, r/quant 1lpzdwr: borrow cost is adversely selected; stress at 2x/5x/10x; add locate-reject / forced-cover states; screen shorts by price and ADV).

## Implementation
The frozen composite LP book `lc_t10`; costs decomposed from the harness's own return file (`port_excess_ret`, `trade_cost`, `borrow_cost`; asserted to sum to the harness net return). (1) borrow x 1/2/5/10/20 on all shorts; (2) the shorts in the lowest-score quartile of the held shorts each month (strongest thesis) pay x 2/5/10; (3) LP re-solved with no shorts priced below $10 / $20 (`short_min_price`), with base and x5 borrow; (4) forced cover: each month 4% of the short *names* (random) are covered at a +15% / +30% adverse move on their position, 500 Monte-Carlo draws; (5) the Jan-2021 squeeze month reported explicitly.

## Experimental setup
Common harness (a verbatim copy of the data / rank-transform / LP / cost / performance code of `experiments/largecap/largecap.py`, embedded so the script runs on its own; the LP functions `_solve_lp` / `build_portfolio` are redefined in the script with optional extra behaviours and reproduce the frozen LP when the options are off).

- **Signal**: the frozen composite (18 factors in 7 economic groups, equal weight, re-ranked in the universe each month; no fitting). It reproduces `experiments/largecap`: universe rank IC 0.0366 (t 1.91), `lc_t10` gross IR 0.615 / net IR 0.541, 2025 net return -12.4%.
- **Universe**: price >= $5, market cap >= $2,000M, 126-day dollar volume >= $10M, both betas observed (about 1,206 stocks per month); 68 test months 2021-01..2026-08.
- **Portfolio**: the frozen LP (dollar-neutral, neutral to `beta_60m` and `betabab_1260d`, net sector <= 5% of NAV, gross sector share <= 35%, gross 200%, per-name cap 1%, 10% one-way turnover cap = `lc_t10`), with tiered *assumed* trading and borrow costs (5/10/20 bp trading and 30/75/200 bp borrow by market-cap tier; not measured).
- **Caveat that applies to every number below**: a single 68-month path; the standard error of a single-path IR is about 0.46, so IR differences under ~0.5 are not distinguishable from noise on their own. Paired monthly net-return t-stats are reported next to IR; with 20+ variants per file, a few |t| > 1 are expected by chance.

## Baseline
The composite book at the assumed costs: net IR 0.541, average borrow cost 4.8 bp of NAV per month (about 57 bp per year), max drawdown -18.9%.

## Commands
```
.venv/bin/python experiments/borrow_stress/borrow_stress.py
```

## Results (68 months)
| scenario | ir | cagr_pct | max_dd_pct | worst_month_pct |
|---|---:|---:|---:|---:|
| baseline | 0.541 |  |  |  |
| borrow x1 | 0.541 | 11.0 | -18.9 | -8.9 |
| borrow x2 | 0.499 | 10.3 | -19.6 | -9.0 |
| borrow x5 | 0.373 | 8.5 | -21.8 | -9.1 |
| borrow x10 | 0.163 | 5.4 | -26.8 | -9.4 |
| borrow x20 | -0.256 | -0.4 | -37.0 | -9.8 |
| strong-thesis borrow x2 | 0.529 | 10.8 | -19.1 | -9.0 |
| strong-thesis borrow x5 | 0.493 | 10.2 | -19.7 | -9.0 |
| strong-thesis borrow x10 | 0.433 | 9.4 | -20.7 | -9.0 |
| short_px10 base | 0.566 | 11.2 | -17.8 | -8.1 |
| short_px20 base | 0.496 | 10.3 | -20.7 | -8.8 |
| short_px10 + borrow x5 | 0.397 | 8.8 | -21.4 | -8.3 |
| short_px20 + borrow x5 | 0.331 | 7.9 | -25.2 | -9.0 |
| forced cover +15% (mean of draws) | -0.005 |  | -31.2 |  |
| forced cover +30% (mean of draws) | -0.551 |  | -43.8 |  |

Jan-2021 (target month 2021-01): the book returned +0.17% net while its short leg lost -3.65% (the squeeze), offset by the long leg; this is the month that dominated the loose books in `docs/NEGATIVE_RESULT.md`.

Sanity: the cost decomposition reproduces the harness net returns to floating-point precision (asserted in the script); the x1 row equals the published net IR.

## Interpretation
**Uniform borrow stress is tolerable up to about 10x, not 20x**: net IR 0.54 -> 0.50 (x2) -> 0.37 (x5) -> 0.16 (x10) -> -0.26 (x20). **State-dependent stress hurts less than uniform stress** (highest-thesis shorts at x10: 0.43) because only a fifth of the short book is affected. A short-side price screen does not damage the book (no shorts below $10: net IR 0.57, below $20: 0.50; with x5 borrow 0.40 / 0.33). **A forced cover is the real fragility**: covering 4% of the short names each month at +15% is worth about 0.6% of NAV per month (roughly 7% per year), which drives IR to about 0 (mean -0.005, 5-95% band -0.02..+0.01) and the max drawdown to about -31%; at +30% IR is -0.55 and the drawdown -44%. That scenario is deliberately harsh (a permanent 4%-a-month squeeze rate) and is best read as a break-even: the book's roughly 7 percentage points a year of net excess return over the 4% hurdle is fully consumed by forced-cover losses of about 0.6% of NAV per month. The book is therefore robust to borrow *cost* assumptions but not to a persistent locate / forced-cover state, which is what the short-interest cap in `experiments/feat_si_followup` addresses. Status: IMPLEMENTED_AND_TESTED (robustness study; no adopt rule).

## Limitations
- No borrow data: everything here scales an assumption. Real HTB names can become unavailable (a discontinuity), and forced covers cluster in squeezes rather than occurring uniformly at random.
- 4% and +15% are illustrative (from the practitioner threads), not calibrated.
- Single path of 68 months.

## Follow-up
Add a squeeze-risk screen (short-interest cap) to the LP; done in `experiments/feat_si_followup`.
