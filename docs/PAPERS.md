# Candidate Papers — Beyond the OLS Baseline

Reading list of methods to build on top of the plain-OLS floor in `ols.py` /
`experiments/ols/README.md`. Scope: methods that operate on the 147 numeric characteristics
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

Each entry: what it does, and which documented weakness in `experiments/ols/README.md` /
`experiments/xgb/README.md` it would address.

---

## 1. Prediction layer — replace/augment plain OLS or XGBoost

### *The Virtue of Sparsity in Complexity* (Apr 2026)
**Verbatim abstract (checked directly, not inferred):** distinguishes
*capacity sparsity* (dimensionality of the candidate feature space) from
*factor sparsity* (parsimonious structure of priced risks) and argues
they're complements — expanding capacity *enables* the discovery of factor
sparsity. Revisits Didisheim et al. (2025)'s benchmark design at higher
complexity and shows **nonlinear feature expansions combined with basis
pursuit** (an L1-minimization technique, not plain LASSO) yield portfolios
that dominate **ridgeless benchmarks** (near-zero-regularization ridge, the
Kelly-Malamud-Zhou-style overparameterized regime) beyond a critical
complexity threshold. The gains come from *enlarging* the feature space
first, not from directly penalizing the original small feature set.
- Posted 2026-04.
- [arXiv PDF](https://arxiv.org/pdf/2604.17166)
- **Implementation note (`sparse.py`):** what was actually built is plain
  LASSO on the original 147 characteristics, unexpanded — no nonlinear
  feature-expansion step, and benchmarked against OLS/XGBoost rather than a
  ridgeless overparameterized-ridge comparator. This tests a much narrower,
  simpler question ("does directly penalizing the raw features help") than
  the paper's actual claim (which is about sparsity *emerging from* an
  expanded feature space). See `experiments/sparse/README.md` for the full caveat.

### *Quantity, Risk, and Return* (Sep 2026)
**Verbatim abstract (checked directly):** expected stock return depends not
only on factor risk exposure (beta) but on the **factor's** own quantity
fluctuations (q) "induced by trading flows" — i.e. quantity is a
**factor-level** concept (how much noise-trading flow a given factor's
loading has absorbed), not a per-stock liquidity measure. "Sophisticated
investors should demand a higher factor premium when they have absorbed
noise trading flows of stocks with high loadings to that factor." The
cross-sectional risk-return relationship is flat unconditionally but strongly
depends on quantity once conditioned on it; the BTQ model also addresses the
factor-zoo problem by selecting a small number of factors.
- Posted 2026-09.
- [arXiv](https://arxiv.org/abs/2609.05162)
- **Implementation note (`btq.py`):** this panel has no trading-flow-by-type
  data to build the paper's actual factor-level quantity variable, so what
  was built is a much simpler per-*stock* liquidity proxy (mean of
  `turnover_126d`/`turnover_var_126d`/`dolvol_126d`/`dolvol_var_126d`,
  `docs/FACTORS.md` §16) interacted linearly with every characteristic —
  same general "condition on trading activity" spirit, structurally
  different from the paper's factor-level mechanism. See `experiments/btq/README.md`.

### *Quant Convergence: Bridging Classical Value Investing and Modern Factor Models for Systematic Equity Selection* (Jun 2026)
Empirical horse race of XGBoost, AutoGluon, and Random Forest across three
feature sets (classical Graham value rules, modern factors, and a hybrid) on
20 years of S&P 500 data. Finding: **plain Random Forest on the simplest
feature set had the best risk-adjusted result** (highest return, best Calmar
ratio), while the more complex AutoGluon ensemble had a larger drawdown for
similar return. A second independent data point — after this project's own
XGBoost result (`experiments/xgb/README.md` §4) — that added model complexity doesn't
reliably buy better risk-adjusted performance on this style of tabular
factor data, and a concrete reason to try plain Random Forest as a cheap
bagging-based alternative to boosting before reaching for anything heavier.
**Verbatim abstract confirms**: the *best* result was specifically the
**Graham-rules-only** Random Forest (highest return, 1.38 Calmar), and the
best-drawdown result was the **combined** (Graham + momentum) Random
Forest — "pure modern factors" alone was not the winning feature set in
their results, model family (Random Forest) was one part of the finding,
curated/simple features were the other.
- Posted 2026-06.
- [arXiv PDF](https://arxiv.org/pdf/2606.24575)
- **Implementation note (`rf.py`):** only the model-family swap was
  tested (Random Forest in place of XGBoost), on the *same full
  147-characteristic panel* as every other script here — the paper's
  actual best-performing feature-curation arm (Graham-only, or Graham +
  momentum) was not replicated. `rf.py`'s strong result is genuine but
  only directly confirms half of the paper's finding. See `experiments/rf/README.md`.

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
**Verbatim abstract confirms** the "direct linear scoring path + bounded
multiplicative branch" framing exactly, though exact layer formulas aren't
given in the abstract.
- Posted 2026-06.
- [arXiv PDF](https://arxiv.org/pdf/2606.08930)
- **Implementation note (`rankglu.py`):** this is the closest structural
  match of any of the 8 — `score = X@w_lin + b_lin + (tanh(z)*gate)@w_out
  + b_out` directly implements a linear path plus a bounded (tanh),
  gated (sigmoid) multiplicative branch, hand-coded in numpy since no
  deep-learning framework was available. Exact architecture details beyond
  what the abstract states are still unconfirmed. See `experiments/rankglu/README.md`.

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
- **Implementation note (`joint.py`):** the abstract gives the argument but
  no algorithmic detail (loss function, weighting scheme) — what's built is
  a much simpler approximation (train-loss weighted by realized-target
  rank extremity), not a reproduction of "unifies the expected return
  generation process and the final optimized portfolio." In practice,
  validation always selected the unweighted case, so the approximation
  ended up not even being tested against OLS in the primary result — see
  `experiments/joint/README.md`.

### Cong, Tang & Wang (2026), *AlphaPortfolio: Goal-Oriented Investment Management Through Deep Reinforcement Learning*, NBER WP 35195
**Verbatim abstract (checked directly — the PDF, not a search snippet):**
adapts attention-based neural networks and RL to direct portfolio
construction. Core architecture is a **Transformer encoder** for
long/short-range path dependence in firm and market states plus a
**cross-asset attention network**, trained end-to-end (not step-by-step) on
objectives that are non-additively-separable across periods — including the
Sharpe ratio directly. In U.S. equities: Sharpe above 2, risk-adjusted alpha
over 13% with monthly rebalancing, robust to excluding small/illiquid
stocks. Demonstrates flexibility to incorporate **transaction costs** and
state interactions, "before developing a **polynomial-feature-sensitivity
analysis**" — a *post-hoc interpretability tool applied to the trained
Transformer/RL model*, not a standalone predictive architecture — "to
uncover key drivers of performance, including their rotation and
nonlinearity." This paper was previously titled "Goal-Oriented Portfolio
Management Through Transformer-Based Reinforcement Learning" and includes
partial results from an earlier, separate, pre-2026 working paper ("Direct
Construction Through Deep Reinforcement Learning and Interpretable AI,"
jointly with Yang Zhang) — **the "polynomial network" / "economic
distillation" framing this project initially used to describe this entry
came from that older paper, not from this one**, and has been corrected.
- Posted 2026-05-19 (as "Working Paper 35195"), May 2026 per the PDF header.
- [NBER working paper](https://www.nber.org/papers/w35195)
- [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6785198)
- **Implementation note (`poly.py`):** no Transformer, no RL, no
  cross-asset attention, no transaction-cost term were built (no RL/deep-
  learning infrastructure in this project). What's implemented is a
  standalone degree-2 polynomial Ridge regression on a curated 20-
  characteristic subset — inspired by the *shape* of the paper's post-hoc
  polynomial-sensitivity idea (interaction/quadratic terms as an
  attribution tool), repurposed here as the primary predictive model
  itself rather than an analysis layered on top of a different model. This
  is a materially narrower and structurally different thing than what the
  paper built. See `experiments/poly/README.md` for the full correction and its effect
  on results.

### *AlphaZeroBeta: Deep Reinforcement Learning for Market-Neutral Portfolios* (Jul 2026)
**Verbatim abstract (checked directly):** a deep RL framework combining a
**composite reward function** (balances risk-adjusted excess return,
benchmark correlation, and **transaction costs**) with a **CNN-GRU policy
trained end-to-end via Recurrent PPO** (proximal policy optimization — a
specific policy-gradient RL algorithm, not generic backpropagation),
evaluated via rolling walk-forward across **seven equity indices,
2014–2024**. Achieves higher Sharpe than baselines with near-zero benchmark
correlation. The most literal upgrade path from the per-month linear
program in `ols.py` §2.5/§3.5 (`experiments/ols/README.md`) or `xgb.py`'s identical LP
(`experiments/xgb/README.md`): instead of re-solving a fresh LP from a fixed prediction
each month, learn the characteristics-to-constrained-weights mapping
directly.
- Posted 2026-07-20.
- [arXiv PDF](https://arxiv.org/pdf/2607.18001)
- **Implementation note (`e2e.py`):** no CNN, no GRU, no PPO, no RL of any
  kind, and no transaction-cost term were built. What's implemented is a
  linear scoring function trained by SPSA (a classical zeroth-order
  stochastic optimizer, unrelated to policy-gradient RL) against a smooth
  portfolio-Sharpe proxy. The only thing genuinely shared with the paper is
  the high-level idea "train weights end-to-end against a portfolio
  objective instead of a two-stage predict-then-optimize pipeline" — the
  actual mechanism is unrelated. See `experiments/e2e/README.md`.

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
**Verbatim abstract (checked directly):** decomposes PCA-based
factor-estimation error into "out-of-subspace error" (distance from the
estimate to the true population factor subspace — data-observable, an
estimable floor) and "in-subspace error" (arises from finite sample size of
latent factor returns, cannot be estimated from data alone), each with
almost-sure asymptotic limits as dimension grows with sample size bounded.
**Illustrated with a three-factor SIMULATION of the US public equity
market** (synthetic, low-dimensional) — their finding that "out-of-subspace
error dominates" is demonstrated in that controlled simulation, not on a
real 147-dimensional panel. Not a prediction method — a diagnostic tool.
Relevant only if a PCA/shrinkage-based prediction model is built on the 147
characteristics.
- Posted 2026-09.
- [arXiv](https://arxiv.org/abs/2609.20550)
- **Implementation note (`pca_diagnostic.py`):** doesn't reproduce their
  asymptotic estimator — uses split-half principal-angle subspace stability
  as a directly computable proxy, run on the real 147-characteristic panel
  (not a synthetic simulation). Related in spirit, answers a related but
  not identical question. See `experiments/pca_diagnostic/README.md`.

---

## Status: all 8 built, run, and fidelity-checked against verbatim abstracts

`ols.py` (floor) and `xgb.py` (nonlinear model, underperformed the floor —
`experiments/xgb/README.md` §4) were built first. All 8 entries above now have a working
implementation (`rf.py`, `sparse.py`, `joint.py`, `e2e.py`, `btq.py`,
`rankglu.py`, `poly.py`, `pca_diagnostic.py`), each corrected once against
its source paper's actual verbatim abstract (not a search summary) — see
each entry's "Implementation note" above and the corresponding `docs/*.md`
for what was fixed and why. Final OOS results, ranked by Information Ratio
(2021-01–2026-08, all vs. the same beta-neutral LP construction):

| Rank | Model | OOS R² | IR | Sharpe | CAGR | Realized β (t) |
|---|---|---:|---:|---:|---:|---:|
| 1 | **RF-Modern** (`experiments/rf/README.md`) | +0.12% | **1.05** | 1.21 | 30.0% | −0.04 (−0.21) |
| 2 | **Poly** (`experiments/poly/README.md`) | −0.02% | 0.87 | **1.08** | 21.2% | −0.18 (−1.12) |
| 3 | RF-Combined (`experiments/rf/README.md`) | −0.21% | 0.81 | 0.97 | 23.8% | 0.14 (0.65) |
| 4 | **E2E** (`experiments/e2e/README.md`) | **+0.07%** | 0.79 | 0.93 | 25.3% | −0.21 (−0.88) |
| 5 | Joint (`experiments/joint/README.md`) | −0.52% | 0.75 | 0.88 | 23.3% | −0.18 (−0.74) |
| 6 | BTQ (`experiments/btq/README.md`) | −0.002% | 0.67 | 0.80 | 20.9% | −0.15 (−0.60) |
| 7 | RankGLU (`experiments/rankglu/README.md`) | −0.34% | 0.57 | 0.77 | 14.0% | **−0.02 (−0.12)** |
| 8 | RF-Graham (`experiments/rf/README.md`) | −0.44% | −0.23 | −0.07 | −4.6% | −0.29 (−0.29) |
| 9 | Sparse-RFF (`experiments/sparse/README.md`) | −0.06% | −0.47 | −0.03 | −0.7% | −0.03 (−0.36) |
| 10 | Ridgeless-RFF (`experiments/sparse/README.md`) | −0.18% | −0.70 | −0.22 | −2.2% | 0.09 (1.37) |

*(vs. `ols.py`'s own baseline: R² −0.01%, IR 0.85, Sharpe 0.98, CAGR
27.6%, β −0.17 (t=−0.67) — still not beaten on IR/Sharpe/CAGR by any of the
8, though `RF-Modern`, `E2E`, and `Poly` all have better (less negative or
positive) OOS R², and `RankGLU` now has the best realized-beta neutrality
of any model in the whole project.)*

**Two genuinely positive R² results** (`RF-Modern`, `E2E`) and **one best-
in-project neutrality result** (`RankGLU`) survived rigorous fidelity
correction — these three are the strongest candidates for further work.
**RF-Graham directly contradicts its source paper** in this project's
market-neutral long/short setting (see `experiments/rf/README.md` for the likely
explanation) — a genuine, reportable negative result, not a bug.
