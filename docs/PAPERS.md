# Candidate Papers — Beyond the OLS Baseline

Reading list of methods to build on top of the plain-OLS floor in `ols.py` /
`docs/OLS.md`. Scope: methods that operate on the 147 numeric characteristics
in `fiam/chars_final_with_names.parquet` — prediction models and
portfolio-construction methods only, no LLM/text/agentic approaches, and no
OLS baseline itself (that's the floor, not a candidate).

**Filtered to 2026-only.** An earlier version of this list included
foundational but older papers (Kelly-Malamud-Zhou 2024, Nagel 2025,
Kozak-Nagel-Santosh 2020, Freyberger-Neuhierl-Weber 2020, Bryzgalova-Pelger-Zhu
2025, Gu-Kelly-Xiu 2021, Feng-Giglio-Xiu 2020, Cakici et al. 2025, and an
older non-2026 cut of *AlphaPortfolio* and *Decision by Supervised Learning
with Deep Ensembles*) — all removed here per an explicit request to keep only
2026 work. Every paper below was confirmed dated 2026 (posting date checked
directly against NBER/arXiv, not inferred from search snippets).

Each entry: what it does, and which documented weakness in `docs/OLS.md` /
`docs/XGB.md` it would address.

---

## 1. Prediction layer — replace/augment plain OLS or XGBoost

### *The Virtue of Sparsity in Complexity* (Apr 2026)
Shows a sparse estimator continues to select a parsimonious pricing kernel as
model complexity rises, and **eventually overtakes dense ridge/random-feature
methods in Sharpe ratio**. Directly relevant if pursuing an
overparameterized/shrinkage prediction model: argues for sparsity over raw
dimensional expansion as the way to handle the 147 characteristics'
redundancy (the same multicollinearity problem `docs/OLS.md` §4.5 and
`docs/XGB.md` §3.6 both flag from opposite ends — plain OLS handles
correlated inputs badly, and XGBoost's feature importance ends up diffuse
across near-duplicate characteristics).
- Posted 2026-04.
- [arXiv PDF](https://arxiv.org/pdf/2604.17166)

### *Quantity, Risk, and Return* (Sep 2026)
Proposes the **Beta-Times-Quantity (BTQ) model**: factor premiums are
conditioned on trading-flow "quantity" information, not just risk exposure
(beta) — the unconditional risk-return relationship is flat, but becomes
predictive once quantity is added, and the quantity-conditioning also
attacks the factor-zoo problem by selecting a small subset of factors.
Numeric-only, no text/LLM component. Notable here specifically because the
panel already has quantity-adjacent liquidity variables
(`turnover_126d`, `turnover_var_126d`, `dolvol_126d`, `dolvol_var_126d` —
`docs/FACTORS.md` §16) that could proxy the paper's quantity signal without
sourcing new data.
- Posted 2026-09.
- [arXiv](https://arxiv.org/abs/2609.05162)

### *Quant Convergence: Bridging Classical Value Investing and Modern Factor Models for Systematic Equity Selection* (Jun 2026)
Empirical horse race of XGBoost, AutoGluon, and Random Forest across three
feature sets (classical Graham value rules, modern factors, and a hybrid) on
20 years of S&P 500 data. Finding: **plain Random Forest on the simplest
feature set had the best risk-adjusted result** (highest return, best Calmar
ratio), while the more complex AutoGluon ensemble had a larger drawdown for
similar return. A second independent data point — after this project's own
XGBoost result (`docs/XGB.md` §4) — that added model complexity doesn't
reliably buy better risk-adjusted performance on this style of tabular
factor data, and a concrete reason to try plain Random Forest as a cheap
bagging-based alternative to boosting before reaching for anything heavier.
- Posted 2026-06.
- [arXiv PDF](https://arxiv.org/pdf/2606.24575)

### *RankGLU: Residual Gated Score Formation for Cross-Sectional Stock Prediction* (Jun 2026)
A prediction-head architecture built specifically to solve the "how do I
turn a model's score into a stable ranking/weight" problem — a bounded,
gated nonlinear branch alongside a direct linear scoring path, designed so
the model doesn't overfit unstable return magnitudes while still capturing
some nonlinear interaction. Directly relevant to the calibration concern
`docs/NEXT.md` §3 already flags for tree-model conviction weighting ("z-score
or rank-transform the predictions first, rather than plugging the raw
predicted value directly into the weight") — this is a more structured,
learned version of that same fix. Only validated on Chinese equity indices
(CSI300/CSI800) in the paper; transfer to this panel is unverified.
- Posted 2026-06.
- [arXiv PDF](https://arxiv.org/pdf/2606.08930)

---

## 2. Portfolio-construction layer — predictions → constrained weights

### Wang, Gao, Harvey, Liu & Tao (2026), *Machine Learning Meets Markowitz*, NBER WP 34861
Argues the standard two-stage pipeline — forecast returns, then plug into an
optimizer — is "deeply problematic" because it treats prediction error as
equally costly for every stock, when the optimizer only cares about error in
the stocks that end up mattering to the final portfolio. Proposes fitting
the return model and the portfolio weights jointly instead. **Directly
describes the architecture both `ols.py` and `xgb.py` currently use**
(predict, then separately solve an LP) — worth citing in the deck's
methodology section as the documented limitation of that two-stage design,
whether or not it's rebuilt jointly before the deadline.
- Posted 2026-02-24.
- [NBER working paper (free PDF)](https://www.nber.org/system/files/working_papers/w34861/w34861.pdf)
- [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6290354)

### Cong, Tang & Wang (2026), *AlphaPortfolio: Goal-Oriented Investment Management Through Deep Reinforcement Learning*, NBER WP 35195
RL agent outputs portfolio weights directly from characteristics, trained
against the actual investment goal rather than an intermediate return
forecast, with a polynomial-network layer built specifically so the trained
model's decisions can be attributed back to characteristics ("economic
distillation"). Relevant to `docs/FIAM.md` §9's "explainability is a
deliverable" framing for portfolio construction, without needing an
LLM/agent.
- Posted 2026-05-19, revised 2026-06-01.
- [NBER working paper](https://www.nber.org/papers/w35195)
- [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6785198)

### *AlphaZeroBeta: Deep Reinforcement Learning for Market-Neutral Portfolios* (Jul 2026)
Trains position weights end-to-end under explicit market-neutrality /
exposure constraints via backpropagation, walk-forward evaluated. The most
literal upgrade path from the per-month linear program in `ols.py` §2.5/§3.5
(`docs/OLS.md`) or `xgb.py`'s identical LP (`docs/XGB.md`): instead of
re-solving a fresh LP from a fixed prediction each month, learn the
characteristics-to-constrained-weights mapping directly.
- Posted 2026-07-20.
- [arXiv PDF](https://arxiv.org/pdf/2607.18001)

---

## 3. Diagnostic / ruled-out — useful for the deck, not standalone methods

### *Quantum Kernels and the Cross-Section of Stock Returns: Anatomy of a Vanishing Advantage* (Jul 2026)
A controlled horse race (quantum fidelity kernel vs. projected quantum
kernel vs. a classical RBF kernel control, identical training data/solver/
tuning budget) finds the quantum kernels' apparent edge vanishes under fair
comparison. Not something to implement (no practical quantum hardware
access, and the result is negative), but useful as a citable example of due
diligence for `docs/FIAM.md` §14's "quality of reasoning" criterion — a
documented instance of an exotic method being checked and ruled out.
- Posted 2026-07.
- [arXiv abstract](https://arxiv.org/abs/2607.20168)

### Bernstein, Goldberg, Gunther, Kercheval, Lan, Lin & Yao, *Principal component error in high-dimensional factor models* (Sep 2026)
Decomposes PCA-based factor-estimation error into an "out-of-subspace"
component (distance from the estimated to the true factor subspace) and an
"in-subspace" component (finite-sample noise), with asymptotic bounds for
both as the number of characteristics grows relative to sample size. Not a
prediction method itself — a diagnostic tool. Relevant only if a
PCA/shrinkage-based prediction model is built on the 147 characteristics:
gives a way to check whether the estimated principal components are
themselves reliable before trusting a model built on top of them.
- Posted 2026-09.
- [arXiv](https://arxiv.org/abs/2609.20550)

---

## Suggested starting point

`ols.py` (floor) and `xgb.py` (nonlinear model, underperformed the floor —
see `docs/XGB.md` §4) are both already built. From this 2026-only list:

1. **Plain Random Forest**, per *Quant Convergence* — the cheapest next
   experiment: a small change from `xgb.py` (bagging instead of boosting),
   with an independent 2026 result suggesting it may handle this kind of
   weak-signal tabular data better than a tuned boosted-tree model did here.
2. **A sparse prediction model**, per *Virtue of Sparsity in Complexity* —
   directly targets the multicollinearity/redundancy problem documented in
   both `docs/OLS.md` and `docs/XGB.md`, more implementation effort than
   Random Forest but a stronger originality story for the deck.
3. **Machine Learning Meets Markowitz** is worth citing in the deck's
   methodology section regardless of which prediction model is chosen — it's
   a direct, 2026, named critique of the two-stage predict-then-optimize
   architecture both current scripts use.
