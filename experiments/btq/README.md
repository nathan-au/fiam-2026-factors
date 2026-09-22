# Quantity-Conditioned Prediction Model — BTQ (Methodology and Results)

Implementation: `btq.py` (run with `.venv/bin/python experiments/btq/btq.py`). Operationalizes
*Quantity, Risk, and Return* — the Beta-Times-Quantity (BTQ) model (arXiv
2609.05162, Sep 2026 — `docs/RESEARCH.md` Part III §1).

## Fidelity to the source paper

The paper's "quantity" (q) is explicitly a **factor-level** construct — "the
factor's quantity fluctuations ... induced by trading flows" — how much
noise-trading flow a given factor's exposure has absorbed. The first version
of this script used a blanket **per-stock** liquidity proxy (mean of four
turnover/dollar-volume characteristics) interacted with *every one* of the
147 characteristics (294 total predictors) — a much simpler, structurally
different construction than the paper's factor-level mechanism.

This version builds a genuinely factor-level quantity variable:

1. **Named factors**: seven well-known priced factors, each represented by
   one characteristic — value (`be_me`), momentum (`ret_12_1`), size
   (`market_equity`), quality (`qmj`), profitability (`gp_at`), investment
   (`at_gr1`), betting-against-beta (`betabab_1260d`).
2. **Factor-level quantity**: for each named factor F and month t,
   `quantity_F,t = Σ|F_i,t| · turnover_i,t / Σ|F_i,t|` — a loading-weighted
   cross-sectional average of trading intensity across stocks exposed to F
   that month. **One number per factor per month** (not per stock) —
   directly operationalizing "the factor's quantity fluctuations induced by
   trading flows."
3. **BTQ interaction**: for each named factor, `F_i,t · quantity_F,t` — a
   stock's own loading on F, times F's own factor-level quantity that
   month — literally "beta times quantity" at the factor level.
4. **Model**: `[147 levels + 7 BTQ terms] = 154 predictors` (much smaller
   than the first version's blanket 294), fit via Ridge, walk-forward,
   alpha tuned on validation.

This still does not reproduce the paper's actual BTQ estimator (no
trading-flow-by-investor-type data exists in this panel), but it is a
structurally faithful analogue of their stated mechanism, not just a
same-spirit stand-in.

## Results

| Model | OOS R² |
|---|---:|
| Primary (levels + factor-level BTQ, 154 cols) | **−0.00237%** |
| Ablation (levels only, same Ridge family, 147 cols) | −0.00545% |

**The primary model now beats the ablation** — reversed from the first
version, where the (structurally different, blanket per-stock) quantity
interactions made pointwise prediction *worse* than levels alone. With the
factor-level construction, the 7 BTQ terms provide a small but genuine
improvement, isolated cleanly since both models share the same
regularization family and validation procedure. Both R² values remain
extremely close to zero in absolute terms — this is a small effect, not a
breakthrough.

| Metric | BTQ (primary, factor-level) | BTQ (first version, per-stock) | OLS (baseline) |
|---|---:|---:|---:|
| OOS R² | −0.0024% | −0.0404% | −0.0086% |
| Information Ratio | 0.67 | 0.77 | 0.85 |
| Sharpe ratio | 0.80 | 0.92 | 0.98 |
| CAGR | 20.9% | 22.4% | 27.6% |
| Alpha t-stat | 1.98 | 2.19 | 2.41 |
| Realized beta (t-stat) | −0.15 (t=−0.60) | −0.06 (t=−0.27) | −0.17 (t=−0.67) |
| Hit rate | 60.3% | 60.3% | 61.8% |

Interestingly, the more faithful factor-level construction has a **better
pointwise R² relationship to its ablation** (primary > ablation, the
"correct" direction) but a **slightly weaker portfolio-level result** than
the first version's cruder per-stock proxy (IR 0.67 vs 0.77). Per-fold alpha
selection (`experiments/btq/output/btq_results.json`) again mostly picked the strongest
regularization available (`alpha=300` in four of six folds), consistent
with heavy shrinkage being needed regardless of which quantity construction
is used.

## Limitations

1. **Named-factor selection (which 7 characteristics represent "the"
   priced factors) is this project's own choice**, not derived from the
   paper — a different or larger set of named factors could plausibly
   change results.
2. **Turnover alone, not a richer flow measure**, proxies "trading
   intensity" — the paper's actual quantity concept may involve
   directional flow (buying vs. selling pressure) that a simple turnover
   average cannot distinguish.
3. **Both R² values are still statistically indistinguishable from zero**
   in absolute terms — the primary-beats-ablation result, while directionally
   consistent with the paper's claim, is a small effect on a
   near-zero-signal panel, not strong evidence either way.
