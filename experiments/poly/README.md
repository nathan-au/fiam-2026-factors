# Interpretable Polynomial Model (Methodology and Results)

Implementation: `poly.py` (run with `.venv/bin/python experiments/poly/poly.py`).
Loosely inspired by *AlphaPortfolio: Goal-Oriented Investment Management
Through Deep Reinforcement Learning* (NBER WP 35195, May 2026, Cong-Tang-Wang
— `docs/RESEARCH.md` Part III §2).

## ⚠ A citation-accuracy error was caught and corrected during review

The original version of this doc described the paper as using "a
polynomial-network layer built specifically so the trained model's
decisions can be attributed back to characteristics ('economic
distillation')." **That description is wrong for this paper.** Reading the
actual NBER PDF (not just search-engine summaries) turned up the verbatim
abstract:

> "We adapt attention-based neural networks and reinforcement learning to
> direct portfolio construction... using Transformer encoder... cross-asset
> attention network... AlphaPortfolio yields superior out-of-sample
> performance (e.g., Sharpe ratio above two and risk-adjusted alpha over
> 13%...). We further demonstrate AlphaPortfolio's flexibility to
> incorporate transaction costs, state interactions, and alternative
> objectives, before developing a **polynomial-feature-sensitivity
> analysis** to uncover key drivers of performance, including their
> rotation and nonlinearity."

The paper's core architecture is a **Transformer + cross-asset-attention RL
system**. "Polynomial" refers to a **post-hoc sensitivity-analysis tool**
applied *after* training that model — not a standalone predictive
architecture, and not the paper's headline contribution. The "polynomial
network"/"economic distillation" language this doc originally used actually
belongs to a **different, older, pre-2026 working paper** by an overlapping
author set ("Direct Construction Through Deep Reinforcement Learning and
Interpretable AI") — the PDF's own acknowledgments section confirms NBER WP
35195 only "includes partial results" from that earlier paper, under a
different title. Citing that older paper's architecture as if it described
this 2026 paper directly contradicted this project's own explicit
2026-only filter (`docs/RESEARCH.md`'s Part III stated scope).

**What `poly.py` implements now, in two separate parts:**

1. **The standalone tradeable model** (unchanged from before the citation
   fix): an explicit degree-2 polynomial regression on a curated small
   subset of characteristics, directly and exactly attributable by
   construction. This is *inspired by the shape* of the paper's post-hoc
   polynomial-sensitivity idea, repurposed as the primary model itself
   rather than an analysis layered on top of a separately-trained
   Transformer/RL system — still not a test of the paper's actual method.
2. **A genuine reproduction of the paper's actual technique** (added after
   the citation-accuracy review — see `run_sensitivity_analysis()`): a
   real post-hoc "polynomial-feature-sensitivity analysis," applied to an
   *already-trained complex model's predictions*, exactly as the paper
   describes. Since this project has no Transformer/RL model, `xgb.py`'s
   already-computed, already-honest OOS predictions stand in as "the
   complex model": for each test year, a degree-2 polynomial surrogate is
   fit **in-sample on that year's realized characteristics to explain (not
   forecast) XGBoost's own predictions** — no walk-forward split needed
   for this part, since nothing is being forecast, only explained. This is
   a real, if partial, fidelity improvement: no Transformer or RL is built,
   but the *technique itself* — polynomial-sensitivity analysis of a
   complex model's output — now matches what the paper actually describes,
   which part 1 alone did not. See Results below.

No Transformer, attention, or RL is built anywhere in this script (no
RL/deep-learning infrastructure in this project).

## ⚠ A look-ahead bias bug was caught and fixed during development

The first version of this script selected its 20 characteristics by reading
`experiments/univariate/output/univariate_results.csv` — the file produced by `univariate.py`,
whose Information Coefficient was computed **over the 2021-01–2026-08 OOS
window**, the exact window this script is evaluated on. Using that file to
pick features here would be look-ahead bias in the selection step: the
"top 20 by IC" would have been chosen partly using each test fold's own
future data. Per `docs/FIAM.md` §10's zero-tolerance stance ("Python code is
submitted and will be checked for forward-looking information"), this was
caught and fixed before finalizing results — **feature selection is now
redone from scratch inside each walk-forward fold, using only that fold's
training data.** The 2021 test fold's selected-20, for example, only ever
saw characteristic months through 2018-12.

**The effect of the fix was large and worth reporting on its own:**

| | Leaky (pre-fix) | Fixed (train-only selection) |
|---|---:|---:|
| OOS R² | +0.0192% | −0.0218% |
| Information Ratio | 1.46 | **0.87** |
| Sharpe ratio | 1.75 | 1.08 |
| CAGR | 26.2% | 21.2% |
| Max drawdown | −9.7% | −21.4% |
| Calmar ratio | 2.70 | 0.99 |

The leaky version looked spectacular — by far the best result of any model
built in this project. After the fix, results come back down substantially
across every metric, though the model still performs *well* (see below) —
the leakage inflated, but was not the sole source of, the result.

## Methodology

Feature selection (the "curation" step): rather than use all 147
characteristics (a degree-2 expansion would be ~10,000+ columns), select the
**top 20 by |IC t-stat|**, using the same monthly-IC methodology as
`univariate.py`, recomputed independently inside each fold's training data.
Degree-2 polynomial expansion of the selected 20 (20 levels + 20 squares +
190 pairwise interactions + 1 intercept = 231 terms), fit via `Ridge`
(alpha tuned on validation, `{0.1, 0.3, 1, 3, 10, 30}`).

## Results (post-fix)

| Metric | Poly | OLS (baseline) | XGBoost |
|---|---:|---:|---:|
| OOS R² | −0.0218% | −0.0086% | −0.1265% |
| **Information Ratio** | **0.87** | 0.85 | 0.12 |
| **Sharpe ratio** | **1.08** | 0.98 | 0.37 |
| CAGR | 21.2% | 27.6% | 4.8% |
| Alpha t-stat | 2.74 | 2.41 | 0.96 |
| Realized beta (t-stat) | −0.18 (t=−1.12) | −0.17 (t=−0.67) | −0.06 (t=−0.46) |
| Hit rate | 52.9% | 61.8% | 50.0% |
| Max drawdown | −21.4% | −42.4% | −30.6% |

**Best IR and Sharpe ratio of any model in this batch of 8**, even after
removing the leakage — and by a meaningful margin on Sharpe (1.08 vs the
next-best e2e.py's 0.96). Max drawdown (−21.4%) is also the second-best in
the batch after Sparse (−19.6%, itself partly a degenerate-fold artifact —
see `experiments/sparse/README.md`). Hit rate (52.9%) is unremarkable, and alpha t-stat
(2.74) is the highest of any of the 8 models — a smoother, less
lottery-ticket-shaped return profile than most of the alternatives here,
which shows up in Sharpe more than in raw CAGR.

**Attribution — top terms selected per fold** (`experiments/poly/output/poly_top_terms.csv`,
for the standalone model): the dominant terms cluster into two groups
depending on the fold:
- **Early folds (2021–2022):** profitability interactions —
  `cop_atl1 × qmj_prof`, `cop_atl1 × op_at`, `op_atl1 × op_at`.
- **Later folds (2023, 2025, 2026):** idiosyncratic-volatility and
  lottery-behavior interactions — `ivol_capm_21d × ivol_hxz4_21d`,
  `ivol_ff3_21d × ivol_capm_21d`, `rmax5_21d × rmax1_21d`.

A caveat on the "distillation" itself: several of the largest-coefficient
pairs are interactions or squares among characteristics that are already
close near-twins by construction (`ivol_capm_21d`/`ivol_ff3_21d`/
`ivol_hxz4_21d` are three flavors of idiosyncratic volatility; `rmax1_21d`/
`rmax5_21d` are both extreme-daily-return measures — see `docs/FACTORS.md`
§15). The model may be partly re-discovering the same multicollinearity
`experiments/ols/README.md` §4.5 and `experiments/xgb/README.md` §3.6 already flagged, rather than
finding a genuinely new nonlinear relationship between economically distinct
signals.

## Part 2 results: sensitivity analysis of xgb.py's predictions

Per year, a degree-2 polynomial surrogate (fit in-sample on that year's
characteristics, explaining — not forecasting — that year's XGBoost
predictions):

| Year | Surrogate R² (explains XGBoost's ranking) | Top term |
|---|---:|---|
| 2021 | 17.0% | `at_me × eqnpo_me` (+0.008) |
| 2022 | 10.6% | `market_equity²` (+0.009) |
| 2023 | 19.4% | `lti_gr1a` (−0.001) |
| 2024 | **39.9%** | `div12m_me²` (+0.004) |
| 2025 | 12.0% | `lti_gr1a × pi_nix` (−0.0002) |
| 2026 | 23.9% | `op_atl1²` (−0.011) |

**Average surrogate R² = 20.5%.** A simple, fully interpretable degree-2
polynomial in a handful of named characteristics explains, on average, a
fifth of the cross-sectional variance in XGBoost's own predictions each
year — sometimes as much as 40% (2024). The characteristics that surface
most often across years — `div12m_me` (dividend yield), `age`, `eqpo_me`/
`eqnpo_me` (payout yield), `ni_me`/`ebit_sale` (profitability/margin),
`lti_gr1a` (long-term investment growth) — are predominantly classic
value/payout/quality signals, suggesting XGBoost's nonlinear predictions
are, to a substantial and explainable degree, tracking the same
value-and-quality structure a simple model would also pick up, not
something qualitatively alien to it.

This is the one piece of this whole batch that directly and faithfully
reproduces what NBER WP 35195 actually describes doing (applying a
polynomial-sensitivity analysis to explain a complex model's predictions),
even though the complex model being explained (XGBoost) is not the paper's
own (Transformer+RL).

## Limitations

1. **Only 20 characteristics are ever eligible**, chosen by a single
   linear-correlation criterion (IC) — a characteristic with real but
   nonlinear-only predictive power would never be selected in the first
   place.
2. **The selected-20 set changes completely fold to fold** (see printed
   output in `experiments/poly/output/poly_results.json`'s `fold_hyperparameters`) — there
   is no stable "these are the 20 that matter" story across the whole OOS
   period, consistent with the univariate analysis's own finding that no
   individual factor's standalone signal is very large.
3. **Not the paper's actual Transformer/RL architecture, and not even a
   reproduction of its polynomial-sensitivity component** (see the
   citation-accuracy correction above) — this is a much simpler, fully
   linear-algebra construction (Ridge on hand-built polynomial features)
   chosen specifically for its direct attributability, loosely inspired by
   but not a test of AlphaPortfolio's actual method.
4. **The leakage bug and its fix are the most important methodological
   note in this document** — the pre-fix numbers are reported here
   deliberately, as a demonstration of why per-fold reselection matters,
   not to imply the leaky version is a valid alternative result.
5. **Part 2's in-sample surrogate fit is intentional, not an oversight** —
   a sensitivity/attribution analysis explains an existing model's output
   on realized data, it doesn't forecast anything, so there is no
   walk-forward split to apply and no look-ahead risk (unlike part 1, which
   IS a forecast and does need one). Explaining XGBoost's predictions is
   also not the same as explaining "the market" — the ~20% average
   surrogate R² says the polynomial approximates XGBoost's own function
   reasonably well, not that XGBoost (or the polynomial) is right about
   future returns.
