# Next Steps — Factor-Only Track

Scope: continuing to work with `chars_final_with_names.parquet` / the 147
characteristics only. Not touching the 8-K text data, LLMs, or agentic
pipelines yet, and not touching `ols.py` again — it stays as the linear
benchmark, frozen.

Target: still the main prediction target, `ret_exc_lead1m` (next-month
excess return), not an alternative fundamentals target.

---

## 1. Nonlinear ML is the right next step

Linear models (plain OLS, and LASSO/Ridge/ElasticNet) are explicitly ruled
out for this phase. With the target staying as direct next-month return, the
standard next layer is a **nonlinear model over the same 147 ranked
characteristics**:

- Tree ensembles — **LightGBM or XGBoost**. These pick up interaction
  effects no linear model can express (e.g. momentum conditional on size,
  accruals conditional on leverage) and handle correlated/redundant factors
  natively via split selection — a free fix for the "no feature
  deduplication" issue noted in `docs/OLS.md` §4.3, without doing manual
  dedup.
- A small feedforward net is the other standard option, but trees are the
  more natural first move: less tuning surface, easier to reason about
  feature importance, and the walk-forward/validation harness already built
  for OLS carries over almost unchanged (same expanding-window annual
  refit, but now the validation fold gets **real use** — tuning tree depth /
  learning rate / n_estimators / num_leaves — unlike OLS's placeholder
  validation window).

Reuse from `ols.py`: data loading, cross-sectional median-fill + rank
transform to `[-1, 1]`, the target-month assignment logic, and the
walk-forward fold schedule. Swap only the model-fitting step.

---

## 2. Beta neutralization: switch `beta_60m` → `betabab_1260d`

Current state (`ols.py`, beta-neutral LP): the per-month constraint is
`Σ w_i · beta_60m_i = 0`. `docs/OLS.md` §3.5 documents the consequence:
formation beta is exact (zero to float precision), but **realized** beta
over the OOS window only fell to −0.17 (t = −0.67), not all the way to
zero — because `beta_60m` is a plain 5-year trailing OLS beta, a slow,
noisy estimate of the stock's *future* beta, not its true forward beta.

Recommendation: for the new (nonlinear) model's beta-neutral construction,
swap the LP's neutralization column from `beta_60m` to **`betabab_1260d`**
(Frazzini–Pedersen beta) and compare.

Why this one and not the other beta-family columns in `docs/FACTORS.md`:

| Column | Verdict | Why |
|---|---|---|
| `betabab_1260d` | **Use this** | Estimates correlation and volatility over different windows and shrinks each stock's beta toward 1 (Vasicek-style) — built specifically to track a stock's *actual forward* beta more reliably than a raw single-regression beta. Directly targets the formation-vs-realized gap above. |
| `beta_60m` | Current choice, keep for reporting | Fine as the input to the required CAPM alpha/beta *reporting* regression (portfolio returns vs. S&P 500) — that's a different, unrelated regression from the LP constraint. Not the best choice for the LP constraint itself. |
| `beta_dimson_21d` | Don't use | Only 1 month of data — too short/noisy, would make the constraint itself unstable month to month. |
| `betadown_252d` | Don't substitute; could add | Different economic quantity (downside/crash beta), not a stand-in for overall market beta. Could be layered on as an *additional* constraint (downside-neutral) later, not a replacement. |
| `corr_1260d` | Can't use directly | Not a beta — no volatility scaling, doesn't fit the linear constraint as-is. |

Action: build both variants (LP constrained on `beta_60m` vs. on
`betabab_1260d`) for the new model and report formation beta (should be ~0
exactly for both, by construction) alongside realized beta (the number that
should actually improve). This is a clean, cheap robustness comparison
worth including in the deck regardless of which nonlinear model is used.

---

## 3. Conviction-weighted sizing is model-agnostic

Position sizing (equal-weight top/bottom-N vs. conviction-weighted vs.
volatility-scaled) is a **portfolio-construction-layer** choice, independent
of which model produced the predicted return. It is not specific to OLS or
linear models — any model's per-stock, per-month predicted return can drive
conviction weights inside the same LP (or a simpler proportional scheme),
subject to the same gross/net/position-count constraints.

One caveat specific to nonlinear models: tree-ensemble prediction
*magnitudes* aren't always well-calibrated the way a linear model's are —
the model can be good at cross-sectional ranking while having an arbitrary
looking prediction scale. Before using raw predicted values as conviction
weights, z-score or rank-transform the predictions first, rather than
plugging the raw predicted return directly into the weight.

---

## 4. Concrete next step

Build a LightGBM (or XGBoost) model:

1. Same data load / rank-transform / target-month assignment as `ols.py`.
2. Same expanding-window annual refit schedule, but now actually use the
   validation fold to tune hyperparameters (tree depth, learning rate,
   n_estimators, regularization).
3. Score OOS months, run through the same investability screen
   (`dolvol_126d` ≥ $10M).
4. Beta-neutral LP, built twice — once constrained on `beta_60m`, once on
   `betabab_1260d` — to get a direct before/after comparison of the
   neutralization fix.
5. Compare OOS R², IR, Sharpe, realized beta, and turnover against the OLS
   benchmark in `docs/OLS.md`.

Keep this as a new script + new doc (e.g. `lgbm.py` / `docs/LGBM.md`) rather
than modifying `ols.py` or `docs/OLS.md` — those stay frozen as the linear
baseline.

---

## Other candidates, lower priority for now

- Turnover/transaction-cost model (both OLS legs run ~40% one-way monthly
  turnover, uncosted — see `docs/OLS.md` §4.4).
- Sector/factor neutrality as a refinement beyond beta (FIAM.md §2,
  optional).
- Daily risk measurement between monthly rebalances (FIAM.md §11,
  encouraged, not required).
- Alternative fundamentals-based targets (FIAM.md §7) — explicitly deferred
  per this conversation; direct return prediction stays the target for now.
