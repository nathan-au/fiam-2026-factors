# Candidate Papers — Beyond the OLS Baseline

Reading list assembled while scoping what to build on top of the plain-OLS
floor in `ols.py` / `docs/OLS.md`. Scope: methods that operate on the 147
numeric characteristics in `fiam/chars_final_with_names.parquet` — prediction
models and portfolio-construction methods. Deliberately **excludes**
LLM/text/agentic approaches (already scoped separately in `docs/LLM.md`,
owned by a teammate) and excludes the OLS baseline itself (that's the floor,
not a candidate).

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

## Suggested starting point

Given the team split (teammate owns text/LLM/agentic per `docs/LLM.md`), a
high-leverage pair to start with:
1. **Kozak-Nagel-Santosh shrinkage** or **Freyberger-Neuhierl-Weber** for the
   prediction step — cheap to implement, directly targets the documented
   multicollinearity problem in the 147 characteristics, easy to defend to
   judges.
2. **Bryzgalova-Pelger-Zhu asset-pricing trees** for portfolio construction —
   interpretable by design, satisfies both the neutrality-reporting and
   explainability asks in `docs/FIAM.md` without extra tooling.
