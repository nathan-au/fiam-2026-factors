# Nonlinear ML Strategy — XGBoost (Methodology and Results)

Implementation: `xgb.py` (single file, run with `.venv/bin/python experiments/xgb/xgb.py`).
Self-contained: the data loading, rank transform, portfolio construction, and
evaluation code is **ported from `ols.py`, not imported from it** — the two
scripts share no code at runtime, by design, so each stands alone and either
can be edited without touching the other. `ols.py` / `experiments/ols/README.md` stay
frozen as the linear baseline this compares against.

**Naming note:** the script is `xgb.py`, not `xgboost.py`. Naming it
`xgboost.py` would shadow the installed `xgboost` package — when the script's
own directory is first on `sys.path`, `import xgboost` inside a file called
`xgboost.py` resolves to itself instead of the library, silently breaking
every `xgb.XGBRegressor` call. Confirmed with a throwaway repro before
writing this file.

Outputs (all in `output/`): `oos_predictions_xgb.csv`,
`portfolio_holdings_beta_neutral_xgb.csv`,
`portfolio_returns_beta_neutral_xgb.csv`, `xgb_feature_importance.csv`,
`xgb_results.json`. Distinct filenames from `ols.py`'s outputs so both can
sit in `output/` at once.

---

## 1. What changed vs. the OLS baseline, and what didn't

Per `docs/NEXT.md` §1 and §4: swap the model-fitting step from plain OLS to
gradient-boosted trees (XGBoost), and — unlike OLS, which had no
hyperparameters — actually use the validation fold to tune them. Everything
else is unchanged from `experiments/ols/README.md`:

- Same data (`fiam/chars_final_with_names.parquet`, 147 characteristics),
  same target (`ret_exc_lead1m`), same target-month alignment.
- Same cross-sectional median-fill + dense-rank-to-`[-1,1]` preprocessing,
  per characteristic month.
- Same expanding-window / 2-year-validation / 1-year-test walk-forward
  schedule, refit annually, split by target month (`docs/FIAM.md` §5).
- Same investability screen ($10M `dolvol_126d`), same beta-neutral LP
  portfolio construction (dollar-neutral, gross 200%, 1% per-name cap),
  **still constrained on `beta_60m`** — `docs/NEXT.md` §2 recommends
  eventually swapping this to `betabab_1260d`, but per direction for this
  script that swap is deferred, not done here. Both scripts currently
  optimize and report against the same beta proxy, which is what makes the
  R²/IR/beta comparison below apples-to-apples.
- Same benchmark, alpha/beta regression, drawdown, turnover, and short-book
  reporting code.

## 2. What's new: validation-tuned walk-forward XGBoost

For each of the six annual folds (2021–2026, same boundaries as
`experiments/ols/README.md` §2.3), a small grid of `(max_depth, learning_rate)` —
`{3, 4, 5} × {0.01, 0.03}`, 6 combinations — is fit on the training fold with
early stopping (50 rounds, cap 2000 trees) evaluated against that fold's
validation set. The combination with the lowest validation MSE is kept, and
**its trees are the ones actually estimated on training data only** — the
validation fold picks hyperparameters and the early-stopping iteration count,
it never contributes to a tree split, matching `docs/FIAM.md` §5's explicit
split of duties ("tune on validation, estimate on training").

The grid is deliberately narrow (shallow trees, low learning rates) rather
than exhaustive. `experiments/ols/README.md` §3.1 already showed these characteristics
carry an OOS R² indistinguishable from zero under a linear model; with a
signal this weak, the risk from a nonlinear model is overfitting noise, not
underfitting — so the grid favors regularization over capacity.

Chosen hyperparameters per fold (`experiments/xgb/output/xgb_results.json`,
`fold_hyperparameters`):

| Test year | max_depth | learning_rate | n_trees (early-stopped) | Val MSE |
|---|---:|---:|---:|---:|
| 2021 | 5 | 0.03 | 96 | 0.0597 |
| 2022 | 5 | 0.03 | 38 | 0.0645 |
| 2023 | 5 | 0.03 | 10 | 0.0480 |
| 2024 | 3 | 0.03 | 36 | 0.0507 |
| 2025 | 3 | 0.01 | **4** | 0.0796 |
| 2026 | 3 | 0.01 | 254 | 0.0933 |

Two things stand out: (1) depth and learning rate both trend down as the
training window grows — the search increasingly prefers regularization, not
capacity, as more (noisy) data accumulates; (2) the 2025 fold's early
stopping triggered after only 4 trees, meaning validation loss stopped
improving almost immediately — a strong indicator that fold's validation
window saw essentially no learnable signal beyond a handful of splits.

Model settings held fixed across the grid (not searched): `subsample=0.8`,
`colsample_bytree=0.8`, `reg_lambda=1.0`, `tree_method="hist"`,
`objective="reg:squarederror"`, `random_state=42` (deterministic reruns).

## 3. Results

All figures are for 2021-01 through 2026-08 (68 months), gross of trading
costs. Source: `experiments/xgb/output/xgb_results.json`. OLS comparison figures are from
`experiments/ols/README.md` §3.

### 3.1 Predictive power

**OOS R² = −0.1265%** (pooled across 268,733 OOS predictions) — *worse* (more
negative) than plain OLS's −0.0086%. Tuned, regularized gradient boosting on
these 147 ranked characteristics does not recover pointwise predictive power
that linear regression lacks; if anything, this run's pooled fit is slightly
further from zero. This is consistent with the 2025 fold's near-instant early
stopping above: the validation-driven tuning is correctly detecting that
there isn't much of a nonlinear pattern to fit, not failing to find one.

### 3.2 Returns

| Metric | XGBoost | OLS (baseline) |
|---|---:|---:|
| Avg monthly return | 0.50% | 2.48% |
| Annualized return, arithmetic | 6.0% | 29.8% |
| Annualized return, geometric (CAGR) | 4.8% | 27.6% |
| Cumulative return (68 months) | 30.5% | 298% |
| Hit rate (active return > 0) | 50.0% | 61.8% |
| Best month | +11.9% (Jul 2026) | +19.2% (Jan 2022) |
| Worst month | −14.2% (Jan 2021) | −42.4% (Jan 2021) |
| Long leg CAGR | 0.3% | 11.3% |
| Short leg CAGR | −0.02% | 10.2% |

Calendar-year returns:

| | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 (through Aug) |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | −22.1% | −1.8% | +18.1% | −7.9% | +8.0% | +45.1% |
| Benchmark (T-bill+4%) | +4.1% | +6.2% | +9.5% | +9.3% | +8.4% | +5.2% |
| S&P 500 | +26.9% | −19.4% | +24.2% | +23.3% | +16.4% | +12.3% |

Unlike OLS (every calendar year positive), XGBoost posts **three losing
years out of six** (2021, 2022, 2024) and a 50% hit rate — a coin flip on
whether any given month beats the benchmark. Both legs are close to flat over
the full window (long +0.3% CAGR, short −0.02% CAGR): the portfolio's modest
positive return comes from month-to-month variation around roughly zero, not
from either leg contributing durable performance, which matches the
near-zero OOS R² directly.

### 3.3 Risk-adjusted performance and neutrality

| Metric | XGBoost | OLS (baseline) |
|---|---:|---:|
| **Information Ratio** | **0.12** | **0.85** |
| Sharpe ratio | 0.37 | 0.98 |
| Alpha, annualized (t-stat) | 6.6% (t=0.96) | 31.5% (t=2.41) |
| Beta at formation (every month) | ~0 (≤2.3×10⁻¹⁵) | ~0 (≤3.4×10⁻¹⁵) |
| **Realized beta vs S&P 500** (SE, t-stat) | **−0.06 (SE 0.13, t=−0.46)** | −0.17 (SE 0.25, t=−0.67) |
| Correlation with S&P 500 | −0.05 | −0.08 |

Formation beta is exact for both models (same LP, same `beta_60m`
constraint, by construction). **Realized beta is closer to zero for XGBoost**
(−0.06 vs OLS's −0.17), though neither is statistically distinguishable from
zero and the standard error is tighter for XGBoost too. This is a
side-effect of XGBoost's flatter, lower-turnover-of-conviction return
profile, not a deliberate improvement — the beta-neutralization mechanism
(the LP + `beta_60m`) is identical between the two scripts; only the
predictions feeding it differ. Alpha is neither economically nor
statistically significant here (t=0.96, vs OLS's t=2.41): unlike the OLS
result, there isn't a clean "market exposure is flat, alpha explains
everything" story — with returns this close to the benchmark, there's not
much return left for alpha to explain in the first place.

### 3.4 Drawdowns

| | XGBoost | OLS (baseline) | S&P 500 |
|---|---:|---:|---:|
| Max drawdown | −30.6% | −42.4% | −24.8% |
| Peak → trough | Dec 2020 → May 2021 (5 mo) | Dec 2020 → Jan 2021 (1 mo) | Dec 2021 → Sep 2022 (9 mo) |
| Recovered | Feb 2026 (57 mo after trough) | Dec 2021 (11 mo) | Dec 2023 (15 mo) |
| Longest underwater spell | 61 months | 11 months | 23 months |
| Calmar ratio | 0.16 | 0.65 | 0.54 |

The single-month meme-stock-squeeze shock that dominates OLS's drawdown
(§3.4 of `experiments/ols/README.md`) is smaller here (XGBoost's short book differs
month-to-month from OLS's), but XGBoost's drawdown is **far more
persistent**: 61 of 68 months spent underwater, recovering only in the
second-to-last month of the OOS window. A shallower single-month loss but a
much longer, shallower drag is the signature of a model with close-to-zero
edge rather than one strong loss event — consistent with §3.1's OOS R².

### 3.5 Exposure, concentration, and implementation

| Metric | XGBoost | OLS (baseline) |
|---|---:|---:|
| Avg positions / month (long / short) | 201 (100.6 / 100.4) | 201 (100.3 / 100.7) |
| Gross / net exposure | 200% / 0% | 200% / 0% |
| Avg monthly turnover (one-way) | 44.2% | 39.7% |
| Turnover range | 3.1% – 93.5% | 19.9% – 57.1% |
| Short book avg / median market cap | $29,933M / $4,129M | $15,949M / $1,764M |
| Short book share < $1,000M cap | 24.4% | 38.3% |

Position count, gross, and net are identical by construction (same LP, same
caps, same $10M liquidity screen). Turnover is both higher on average and
far more variable (3%–94% vs. OLS's tighter 20%–57% band) — month-to-month
XGBoost predictions reorder the cross-section more erratically than OLS's
stable linear ranking, which would matter more once a transaction-cost model
is added (neither script has one yet — see `experiments/ols/README.md` §4.6). The short
book skews toward larger, more liquid names than OLS's (median $4.1B vs.
$1.8B market cap), which is a lower-borrow-risk book on the stated
market-cap proxy, but is incidental to how the LP happened to fill the cap
this run, not a deliberate design choice.

### 3.6 Feature importance

Average XGBoost gain-based importance across the six folds
(`experiments/xgb/output/xgb_feature_importance.csv`), top 15:

| Rank | Feature | Category (`docs/FACTORS.md`) |
|---:|---|---|
| 1 | `at_turnover` | Profitability |
| 2 | `lti_gr1a` | Accruals & balance-sheet changes |
| 3 | `ebitda_mev` | Value |
| 4 | `rd5_at` | Investment & asset growth |
| 5 | `dolvol_126d` | Liquidity & trading activity |
| 6 | `ivol_capm_21d` | Volatility & lottery-like behaviour |
| 7 | `rd_sale` | Investment & asset growth |
| 8 | `div12m_me` | Value |
| 9 | `bidaskhl_21d` | Liquidity & trading activity |
| 10 | `pi_nix` | Profitability |
| 11 | `capx_gr2` | Investment & asset growth |
| 12 | `cash_at` | Leverage & financial health |
| 13 | `aliq_mat` | Leverage & financial health |
| 14 | `ivol_hxz4_21d` | Volatility & lottery-like behaviour |
| 15 | `tangibility` | Leverage & financial health |

No single feature dominates (top feature carries only 3.7% of total gain,
vs. 147 features that would average 0.68% each under no signal at all) — a
diffuse importance profile is itself weak evidence against a strong
nonlinear pattern concentrated in a few characteristics. Two liquidity
measures (`dolvol_126d`, `bidaskhl_21d`) and one leverage-liquidity measure
(`aliq_mat`) appear in the top 15, alongside `dolvol_126d` also being the
investability-screen variable — worth flagging as a caveat rather than a
finding: the model may be partly picking up liquidity-related structure
that correlates with, rather than causes, next-month returns, and this
hasn't been disentangled here.

## 4. Known limitations / next steps

1. **XGBoost did not improve on the OLS floor by either metric measured
   here.** Pooled OOS R² is more negative and every portfolio-level
   risk-adjusted statistic (IR, Sharpe, alpha, hit rate) is lower than the
   linear baseline. This is a legitimate, reportable result
   (`docs/FIAM.md` §9's own framing: "a candid account of an agent that did
   not work is worth more than a polished account of one that supposedly
   did" applies just as well to a model that didn't work).
2. **The OLS result itself was flagged as hard to trust** (`experiments/ols/README.md`
   §4.1: strong portfolio returns from a near-zero-R² model). XGBoost's
   weaker portfolio result from a similarly near-zero R² is, if anything,
   more internally consistent — but that consistency doesn't resolve
   whether either model's portfolio-level performance reflects real
   predictive signal or noise concentrated by the ranking-based
   construction.
3. **Grid is narrow by design, not by evidence of being sufficient.** Only
   6 `(depth, learning_rate)` combinations were tried, subsample/
   colsample/`reg_lambda` were fixed rather than searched, and no
   feature-subset or dimensionality-reduction step was applied ahead of
   the trees. A wider search might close some of the gap to OLS, though
   given how weak the underlying signal already is (§3.1), there's no
   strong prior that more search capacity finds it.
4. **`beta_60m` is still the LP's neutralization column**, per this
   iteration's explicit scope (deferred, not dropped — `docs/NEXT.md` §2
   still recommends the `betabab_1260d` swap as a follow-up robustness
   check, applicable to either model).
5. **Turnover is measured but not costed**, same caveat as
   `experiments/ols/README.md` §4.6 — and XGBoost's turnover is both higher and more
   variable, so a transaction-cost model would likely narrow the gap to
   OLS further (OLS's smaller, steadier turnover is relatively cheaper to
   trade).
6. **Liquidity-adjacent features rank unexpectedly high** (§3.6) — worth a
   direct check (e.g. refit excluding `dolvol_126d`/`bidaskhl_21d`/
   `aliq_mat`, or partial dependence on them) before reading anything into
   the top-15 list as an investment thesis.

This script's job was to test whether nonlinear structure in the 147
characteristics beats the linear floor — for direct next-month return
prediction, under this grid and this construction, it does not. The next
candidates worth trying are the ones in `docs/PAPERS.md` that target the
documented weaknesses more directly: Kozak-Nagel-Santosh shrinkage or
Freyberger-Neuhierl-Weber selection for the prediction step (both aimed at
the multicollinearity/redundancy problem neither OLS nor XGBoost's
feature-importance diffuseness resolves), or Bryzgalova-Pelger-Zhu trees for
a fundamentally different portfolio-construction approach rather than a
different prediction model feeding the same LP.
