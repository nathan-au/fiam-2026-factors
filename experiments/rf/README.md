# Random Forest — 3-Arm Feature Curation (Methodology and Results)

Implementation: `rf.py` (run with `.venv/bin/python rf.py`). Operationalizes
*Quant Convergence* (arXiv 2606.24575, Jun 2026 — `docs/PAPERS.md` §1).

## Fidelity to the source paper

The first version of this script only tested the model-family half of the
paper's finding (Random Forest in place of XGBoost, on the full
147-characteristic panel). The paper's verbatim abstract's actual headline
result was a three-way horse race across **feature sets**: a Random Forest
on **Graham value-rules-only** characteristics had the best return/Calmar,
and a **Graham + momentum combined** Random Forest had the best drawdown
control — "pure modern factors" was not their winning arm. This version now
runs all three arms:

1. **GRAHAM** (12 characteristics) — book-to-market, earnings yield,
   dividend yield, debt-to-market, book leverage, Altman Z-score, Piotroski
   F-score, market cap, earnings persistence, payout yield, cash position,
   firm age. Benjamin Graham's margin-of-safety criteria mapped onto this
   panel.
2. **MODERN** (15 characteristics) — momentum (3), quality-minus-junk
   composites (3), mispricing scores (2), earnings/revenue surprise (2),
   lottery/idiosyncratic-vol (2), betting-against-beta, asset growth,
   gross profitability. Deliberately non-overlapping with GRAHAM.
3. **COMBINED** (27 characteristics) — the union.

Each arm runs through the *same* walk-forward / validation-tuned RF / LP /
evaluation harness used everywhere else in this project. **The paper's own
backtest (buy-and-hold, one 4-year window) was deliberately not adopted** —
`docs/FIAM.md`'s monthly-rebalanced, market-neutral, LP-constrained mandate
is this project's fixed evaluation standard, so the paper's feature-set
question is tested within this project's methodology, not by replicating
theirs.

## Results

| Metric | Graham | Modern | Combined |
|---|---:|---:|---:|
| OOS R² | −0.44% | **+0.12%** | −0.21% |
| **Information Ratio** | **−0.23** | **1.05** | 0.81 |
| Sharpe ratio | −0.07 | **1.21** | 0.97 |
| CAGR | −4.6% | **30.0%** | 23.8% |
| Alpha t-stat | −0.10 | **2.84** | 2.12 |
| Realized beta (t-stat) | −0.06 (t=−0.29) | −0.04 (t=−0.21) | 0.14 (t=0.65) |
| Hit rate | 47.1% | **63.2%** | 64.7% |
| Max drawdown | **−45.9%** | −22.6% | −28.4% |
| Calmar ratio | −0.10 | **1.33** | 0.84 |

**This project's result directly contradicts the paper's finding.**
Graham-only characteristics, run through this project's market-neutral
long/short LP construction, produce a **losing** strategy (negative IR,
negative CAGR, worst drawdown of the three arms) — not the winning arm the
paper reports. Modern factors alone are the clear best performer here on
every metric except realized-beta magnitude. Combined sits between the two,
pulled down by Graham's negative contribution.

**Why the reversal is plausible, not just noise:** the paper's backtest is
buy-and-hold long-only over a single 4-year window (Mar 2022–Mar 2026);
this project's is a monthly-rebalanced, dollar-and-beta-neutral long/short
book evaluated over 68 months (2021–2026) including 2021's meme-stock
squeeze and 2022's rate-hike drawdown. A **short leg built from
low-quality/high-leverage Graham "cheap" stocks** is a very different bet
than a **long-only book of the same names** — cheap, high-leverage,
small/aging companies are exactly the kind of names that get squeezed hardest
in a short book during volatility spikes, which this project's construction
is directly exposed to and the paper's long-only design is not. This is a
plausible, economically coherent explanation for the divergence, not
confirmed by additional testing here.

Also notable: Random Forest's excellent realized-beta control from the
first version of this script (0.002, t=0.01) was specific to the full
147-characteristic panel — none of these three curated arms reproduces
that result (Modern comes closest at t=−0.21).

## Limitations

1. **Feature-set assignment (which characteristics count as "Graham" vs.
   "Modern") is this project's own mapping**, not a literal translation of
   the paper's exact rule set — a different mapping could plausibly shift
   results, especially for the smaller Graham arm (12 characteristics,
   more sensitive to any single addition/removal).
2. **The paper's buy-and-hold backtest and this project's market-neutral
   long/short backtest are different instruments answering different
   questions** — the contradiction reported here is a genuine, honest
   result of this project's methodology, not evidence the paper's own
   result is wrong on its own terms.
3. **Grid still narrow** (`max_depth ∈ {6, 10}`, fixed `min_samples_leaf=100`,
   `n_estimators=200`) for runtime reasons, same as the first version.
