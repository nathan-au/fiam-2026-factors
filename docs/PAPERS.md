# Candidate Papers — Beyond the OLS Baseline

Reading list assembled while scoping what to build on top of the plain-OLS
floor in `ols.py` / `docs/OLS.md`. Scope: methods that operate on the 147
numeric characteristics in `fiam/chars_final_with_names.parquet` — prediction
models and portfolio-construction methods only. Excludes the OLS baseline
itself (that's the floor, not a candidate).

Each entry: what it does, and which documented weakness in `docs/OLS.md` it
would address.

---

## 1. Prediction layer — replace/augment plain OLS on the 147 characteristics

### Kelly, Malamud & Zhou (2024), *The Virtue of Complexity in Return Prediction*, Journal of Finance
Overparameterized ridge / random-Fourier-feature regression on the same
characteristics used here, run past the point where parameters exceed
observations, systematically outperforms low-dimensional OLS both in theory
and empirically. Near-drop-in replacement for the `LinearRegression` step in
`ols.py`.
- [Journal of Finance (published version)](https://onlinelibrary.wiley.com/doi/10.1111/jofi.13298)
- [NBER working paper (free PDF)](https://www.nber.org/system/files/working_papers/w30217/w30217.pdf)
- [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3984925)

### Nagel (2025/2026), *Seemingly Virtuous Complexity in Return Prediction*, NBER
Direct rebuttal of the above: argues the apparent gains from complexity partly
reflect embedded leverage / exposure to already-known factors rather than a
genuine complexity edge. Read alongside Kelly-Malamud-Zhou — worth citing in
the deck's methodology section regardless of which model is chosen, as
evidence the failure mode was considered (`docs/FIAM.md` §14: "quality of
reasoning around it").
- [NBER working paper (free PDF)](https://www.nber.org/system/files/working_papers/w34104/w34104.pdf)
- [Author's PDF](https://voices.uchicago.edu/stefannagel/files/2025/07/Complexity_2.pdf)

### Kozak, Nagel & Santosh (2020), *Shrinking the Cross-Section*, Journal of Financial Economics
Builds a stochastic discount factor from many correlated characteristic-based
factors using an economically motivated shrinkage prior over principal
components, rather than treating each characteristic as an independent
regressor. Directly addresses `docs/OLS.md` §4 point 3 ("no feature
deduplication... plain OLS handles correlated inputs badly").
- [Journal of Financial Economics (published version)](https://www.sciencedirect.com/science/article/abs/pii/S0304405X19301655)
- [Author's PDF (free)](https://cpb-us-w2.wpmucdn.com/voices.uchicago.edu/dist/f/575/files/2020/07/SCS.pdf)
- [NBER working paper](https://www.nber.org/papers/w24070)

### Freyberger, Neuhierl & Weber (2020), *Dissecting Characteristics Nonparametrically*, Review of Financial Studies
Adaptive group-LASSO selects which characteristics carry incremental
information for expected returns, then fits a flexible nonparametric
(non-linear) function of the selected ones — a middle ground between the
linear OLS baseline and an uninterpretable deep net. Directly useful for
defending which of the 147 characteristics were kept and why (`docs/FIAM.md`
§14.1: originality of the investment idea).
- [Review of Financial Studies (published version)](https://academic.oup.com/rfs/article-abstract/33/5/2326/5821383)
- [NBER working paper (free PDF)](https://www.nber.org/system/files/working_papers/w23227/w23227.pdf)
- [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2820700)

### Bryzgalova, Pelger & Zhu (2025), *Forest through the Trees: Building Cross-Sections of Stock Returns*, Journal of Finance
Decision trees recursively split stocks on characteristics to build
interpretable, diversified portfolios that span the SDF — reported up to 3x
higher OOS Sharpe/alpha than ML-prediction-sorted portfolios in their sample.
Natively produces the long/short groupings the deck needs (top holdings,
explainable splits) instead of a black-box score.
- [Journal of Finance (published version)](https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.13477)
- [SSRN](https://doi.org/10.2139/ssrn.3493458)
- [Free working-paper PDF](https://fass.nus.edu.sg/ecs/wp-content/uploads/sites/4/2020/09/Forest-Through-the-Trees-Bryzgalova-Pelger-and-Zhu.pdf)

### Gu, Kelly & Xiu (2021), *Autoencoder Asset Pricing Models*, Journal of Econometrics
Conditional autoencoder: characteristics drive time-varying factor loadings
(betas), a latent factor structure drives returns, fit jointly by a neural
net. A natural bridge between the linear baseline and a full deep-learning
model, reusing the same characteristic panel as input.
- [Journal of Econometrics (published version)](https://www.sciencedirect.com/science/article/abs/pii/S0304407620301998)
- [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3335536)

### Feng, Giglio & Xiu (2020), *Taming the Factor Zoo: A Test of New Factors*, Journal of Finance
Model-selection test for whether a candidate characteristic/factor adds
explanatory power beyond a high-dimensional set of existing ones, correcting
for the bias that naive variable selection introduces. Useful for defending
originality if the final model uses a hand-picked subset of the 147 rather
than all of them.
- [Journal of Finance (published version)](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12883)
- [NBER working paper (free PDF)](https://www.nber.org/system/files/working_papers/w25481/w25481.pdf)
- [Author's PDF (free)](https://dachxiu.chicagobooth.edu/download/ZOO.pdf)

---

## 2. Portfolio-construction layer — predictions → constrained weights

### *AlphaZeroBeta: Deep Reinforcement Learning for Market-Neutral Portfolios* (2024/2026)
Trains position weights end-to-end under explicit market-neutrality /
exposure constraints via backpropagation, walk-forward evaluated. The most
literal upgrade path from the per-month linear program in `ols.py` §2.5/§3.5
(`docs/OLS.md`): instead of re-solving a fresh LP from a fixed prediction each
month, learn the characteristics-to-constrained-weights mapping directly.
- [arXiv PDF](https://arxiv.org/pdf/2607.18001)

### Cong, Tang, Wang & Zhang, *AlphaPortfolio: Direct Construction Through Deep Reinforcement Learning and Interpretable AI*
RL agent outputs portfolio weights directly from characteristics, with a
polynomial-network layer built specifically so the trained model's decisions
can be attributed back to characteristics ("economic distillation"). Relevant
to `docs/FIAM.md` §9's "explainability is a deliverable" framing for
portfolio construction, without needing an LLM/agent.
- [SSRN](https://ssrn.com/abstract=3554486)
- [Free PDF (updated version)](https://acfr.aut.ac.nz/__data/assets/pdf_file/0008/573128/AlphaPortfolio-updated.pdf)
- [NBER working paper — 2026 follow-up, *Goal-Oriented Investment Management Through Deep Reinforcement Learning*](https://www.nber.org/papers/w35195)

### *Decision by Supervised Learning with Deep Ensembles: A Practical Framework for Robust Portfolio Optimization* (2025)
Trains directly against a decision-relevant objective (e.g. Sharpe/IR) rather
than pointwise return MSE, then derives weights from that. Relevant because
`docs/OLS.md` §3.1 reports OOS R² ≈ 0 for pointwise return prediction — a
model optimized for portfolio-level IR instead of per-stock accuracy may do
better with the same weak underlying signal.
- [arXiv PDF](https://arxiv.org/html/2503.13544v3)

---

## 3. Applied / survey reference

### Cakici, Fieberg, Osorio, Poddig & Zaremba (2025), *Picking Winners in Factorland: A Machine Learning Approach to Predicting Factor Returns*, Journal of Portfolio Management
Applies standard ML algorithms (not novel methodology) to predict which of
242 factor-style characteristics will do well next period, rather than
predicting individual stock returns directly — a useful sanity-check
reference for factor-timing framing versus the direct-return-prediction
approach used in `ols.py`.
- [Summary/coverage](https://alphaarchitect.com/predict-factor-returns/)

---

## 4. Very recent additions (web search, Sept 2026 — papers from roughly Jun–Sep 2026)

Found while updating this list after `ols.py` (floor, `docs/OLS.md`) and
`xgb.py` (nonlinear model, `docs/XGB.md`) were both built — XGBoost did *not*
beat the OLS floor (§4 of `docs/XGB.md`), so these were sourced specifically
to find methods addressing *why*, not just "another model to try." Same
scope as the rest of this document: numeric characteristics / factors only,
LLM/text/agentic papers excluded even where they turned up in search results
(e.g. *FactorEngine* and several 2026 "agentic factor mining" papers were
found and dropped for this reason).

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
- Posted 2026-02-24 (not summer, but recent enough and directly on-point to
  include).
- [NBER working paper (free PDF)](https://www.nber.org/system/files/working_papers/w34861/w34861.pdf)
- [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6290354)

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
- Posted 2026-09 (this month).
- [arXiv](https://arxiv.org/abs/2609.05162)

### *The Virtue of Sparsity in Complexity* (Apr 2026)
A third data point in the Kelly-Malamud-Zhou vs. Nagel debate already in §1
of this document: shows a sparse estimator continues to select a
parsimonious pricing kernel as model complexity rises and **eventually
overtakes dense ridge/random-feature methods in Sharpe ratio**. If the
Kelly-Malamud-Zhou overparameterized-ridge route is pursued, this is the
paper to cite alongside Nagel's rebuttal — both push toward sparsity/shrinkage
over raw dimensional expansion, from different angles.
- Posted 2026-04 (not summer, but a direct update to an already-cited debate).
- [arXiv PDF](https://arxiv.org/pdf/2604.17166)

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

### *Quantum Kernels and the Cross-Section of Stock Returns: Anatomy of a Vanishing Advantage* (Jul 2026)
A controlled horse race (quantum fidelity kernel vs. projected quantum
kernel vs. a classical RBF kernel control, identical training data/solver/
tuning budget) finds the quantum kernels' apparent edge vanishes under fair
comparison. Not something to implement (no practical quantum hardware
access, and the result is negative), but useful as a citable example of due
diligence — the same "considered and ruled out" framing `docs/PAPERS.md`
already uses for Nagel's rebuttal of Kelly-Malamud-Zhou.
- Posted 2026-07.
- [arXiv abstract](https://arxiv.org/abs/2607.20168)

### Bernstein, Goldberg, Gunther, Kercheval, Lan, Lin & Yao, *Principal component error in high-dimensional factor models* (Sep 2026)
Decomposes PCA-based factor-estimation error into an "out-of-subspace"
component (distance from the estimated to the true factor subspace) and an
"in-subspace" component (finite-sample noise), with asymptotic bounds for
both as the number of characteristics grows relative to sample size. Not a
prediction method itself — a diagnostic tool. Relevant only if the
Kozak-Nagel-Santosh PCA-shrinkage route (§1) is pursued: this gives a way to
check whether the estimated principal components of the 147 characteristics
are themselves reliable before trusting a shrinkage-based model built on
top of them.
- Posted 2026-09 (this month).
- [arXiv](https://arxiv.org/abs/2609.20550)

---

## Suggested starting point

`ols.py` (floor) and `xgb.py` (nonlinear model, underperformed the floor —
see `docs/XGB.md` §4) are both already built. A high-leverage pair for the
next model:
1. **Kozak-Nagel-Santosh shrinkage** or **Freyberger-Neuhierl-Weber** for the
   prediction step — cheap to implement, directly targets the documented
   multicollinearity problem in the 147 characteristics, easy to defend to
   judges. The *Virtue of Sparsity* paper above is a live 2026 argument for
   this general direction over the dense-ridge alternative.
2. **Bryzgalova-Pelger-Zhu asset-pricing trees** for portfolio construction —
   interpretable by design, satisfies both the neutrality-reporting and
   explainability asks in `docs/FIAM.md` without extra tooling.
3. **Plain Random Forest**, per *Quant Convergence* above, is now the
   cheapest thing worth trying before either of the above — it's a small
   change from `xgb.py` (bagging instead of boosting) and has an independent
   2026 result suggesting it may handle this kind of weak-signal tabular data
   better than a tuned boosted-tree model did here.
