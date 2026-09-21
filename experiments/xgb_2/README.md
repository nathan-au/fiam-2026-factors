# XGBoost, Round 2 — Fixing the Tuning and the Regularization

Implementation: `xgb_2.py` (run with `.venv/bin/python experiments/xgb_2/xgb_2.py`). Follows `experiments/xgb/README.md`; `xgb.py` and its outputs are untouched.

**What was wrong in `experiments/xgb/README.md`.** Selection and early stopping on validation **MSE** (the 2025 fold stopped after 4 trees — validation MSE cannot see this signal), 0.8 row/column sampling, no leaf-size floor, `reg_lambda=1`, a 6-config (`max_depth` × `learning_rate`) grid, all 147 characteristics including momentum, and an unwinsorized label. Result: IR 0.12, OOS R² −0.13%, below the OLS floor.

**What changed here.**

| Change | Setting |
|---|---|
| No early stopping; tree count chosen on **validation rank IC** | One 800-tree, learning-rate-0.02 run per config; candidates at {50, 100, 200, 400, 800} trees via `iteration_range` |
| Forest-style sampling | `subsample=0.5`, `colsample_bytree=0.5`, `max_bin=64` |
| Big leaves and strong shrinkage | `min_child_weight` {500, 5000} (≈ rows per leaf), `reg_lambda=100`, `max_depth` {2, 4} |
| Winsorized fit label | 1st/99th training-fold percentile |
| Feature arms | `modern_nomom` (12), `modern` (15), and **`all_nomom` (139)** — run so the comparison to `xgb.py` (147 features) is close to like-for-like |
| Portfolio layer | as `experiments/rf_3/README.md` |

**Bottom line.**
1. **The XGBoost failure was tuning and regularization, not "trees don't work here".** On the same legacy LP as `experiments/xgb/README.md`, XGB_2 goes from IR 0.12 to 1.25 on the 139-feature arm (0.90 on 12 features, 1.04 on 15). Several changes were made together (the feature set also differs by the 8 momentum columns), so no single fix is credited.
2. **Same tradeability problem as every other model:** on the PM LP the 139-feature model is worse than the small-feature arms (`pm_t10` IR -0.54 gross), i.e. more characteristics fit more of the small-cap structure the PM screens remove.

## Portfolio layer (shared by the five scripts of this batch)

Two portfolios are built from the same predictions each month and every result is reported **gross and net of costs**:

| Name | What it is |
|---|---|
| `legacy` | The exact LP of every earlier script (`experiments/ols/README.md` §2.5): $10M `dolvol_126d` screen, dollar-neutral, `beta_60m`-neutral, gross 200%, 1% per-name cap. It reproduces `rf_2.py`'s recorded IR (1.0373) to four decimals on `rf_2.py`'s own predictions, so it is directly comparable to `experiments/rf/README.md`, `experiments/rf_2/README.md`, `experiments/xgb/README.md`, etc. |
| `pm_free` / `pm_t20` / `pm_t10` | A "portfolio-manager" LP built from *Valentino's FIAM Tips* (`Valentino_FIAM_Tips.pdf`) and the financial-engineer notes: price ≥ $5 and market cap ≥ $500M screens (plus the $10M dollar-volume screen); neutral to **both** `beta_60m` and `betabab_1260d`; net sector exposure ≤ 5% of NAV and sector share ≤ 35% of the gross book (2-digit GICS); and a **hard one-way turnover budget** as an LP constraint — none (`pm_free`), 20% (`pm_t20`) or 10% (`pm_t10`, the headline, following the tips' "around 10% per month"). |

- **Turnover convention:** one-way, as a share of the 200% gross book, from drift-adjusted trades (last month's weights are drifted by that month's realized returns, which are known at the rebalance date). This is the same convention as this project's `avg_monthly_turnover`. If the cap is infeasible in a month (names forced out of the universe), it is loosened stepwise ×1.5, 2, 3, 5 and the month is counted in "Months cap relaxed".
- **Costs are assumptions, not measurements** (the panel has no borrow or spread data by name): one-way trading cost 5 / 10 / 20 bp and annual borrow 30 / 75 / 200 bp on short notional for market caps ≥ $10B / $2–10B / < $2B. Full derivation in `experiments/pm_ablation/README.md`.
- **Pre-specified, not tuned.** The screens, sector limits and the 10% headline cap were fixed from the tips before any result was seen. The `pm_free`/`pm_t20`/`pm_t10` rows are a sensitivity sweep, not a search.
- **Selection:** hyperparameters (and tree counts for boosted models) are picked per fold on **validation mean monthly rank IC**, never on validation MSE and never on test data. The fitting label is `ret_exc_lead1m` winsorized at the training fold's 1st/99th percentiles (`--raw-target` disables); predictions remain in next-month-return units and OOS R² is scored on the raw return.

## 1. Like-for-like comparison with `xgb.py` (`legacy` LP, gross)

| Metric (2021-01 – 2026-08, `legacy` LP, gross) | `xgb.py` (`experiments/xgb/README.md`), 147 features | `xgb_2.py` `all_nomom`, 139 features | `xgb_2.py` `modern`, 15 | `xgb_2.py` `modern_nomom`, 12 |
|---|---:|---:|---:|---:|
| OOS R² | −0.127% | -0.014% | +0.098% | +0.140% |
| Information ratio | 0.12 | 1.25 | 1.04 | 0.90 |
| Sharpe | 0.37 | 1.43 | 1.17 | 1.03 |
| CAGR | 4.8% | 32.9% | 35.9% | 30.2% |
| Alpha t-stat | 0.96 | 3.36 | 2.81 | 2.47 |
| Realized β (t) | −0.06 (−0.46) | -0.04 (-0.23) | -0.11 | -0.12 |
| Max drawdown | −30.6% | -27.8% | -41.0% | -43.8% |
| One-way turnover (project convention) | 44.2% | 39.5% | 44.7% | 41.2% |

`xgb.py` numbers are copied from `experiments/xgb/README.md` §3. The LP, benchmark, folds and OOS window are identical; `xgb_2.py` additionally uses the momentum-free / smaller feature sets and the winsorized label.

## 2. Selection (headline arm `modern_nomom`)

| Test year | Chosen (on validation rank IC) | Val IC | Val IC range over candidates |
|---|---|---:|---|
| 2021 | {'max_depth': 2, 'min_child_weight': 500, 'n_trees': 200} | 0.0534 | 0.0441 … 0.0534 |
| 2022 | {'max_depth': 2, 'min_child_weight': 5000, 'n_trees': 100} | 0.0940 | 0.0526 … 0.0940 |
| 2023 | {'max_depth': 2, 'min_child_weight': 500, 'n_trees': 50} | 0.1381 | 0.0589 … 0.1381 |
| 2024 | {'max_depth': 2, 'min_child_weight': 500, 'n_trees': 100} | 0.1507 | 0.1266 … 0.1507 |
| 2025 | {'max_depth': 2, 'min_child_weight': 500, 'n_trees': 50} | 0.1520 | 0.1151 … 0.1520 |
| 2026 | {'max_depth': 2, 'min_child_weight': 500, 'n_trees': 50} | 0.1383 | 0.1203 … 0.1383 |

Depth 2 (the shallower option) is chosen in every fold, and 50–200 trees — small, smooth models again, with the smallest tree count (50) chosen in three of six folds (grid edge). `min_child_weight` = 500 (the smaller value) wins in five of six.

| Feature | Avg. importance |
|---|---:|
| ivol_capm_21d | 0.155 |
| mispricing_perf | 0.144 |
| betabab_1260d | 0.124 |
| qmj | 0.104 |
| rmax5_21d | 0.091 |
| qmj_growth | 0.086 |
| qmj_prof | 0.061 |
| saleq_su | 0.060 |
| niq_su | 0.053 |
| mispricing_mgmt | 0.041 |

## 3. Results

### 3.1 Model-level statistics

| Arm | Features | Mean val rank IC | Test rank IC (% months > 0) | OOS R² | Pred. rank autocorr |
|---|---:|---:|---:|---:|---:|
| modern_nomom | 12 | 0.1211 | 0.1480 (82%) | +0.140% | 0.84 |
| modern | 15 | 0.1213 | 0.1518 (81%) | +0.098% | 0.81 |
| modern_nomom_trad | 12 | 0.0358 | 0.1218 (81%) | +0.035% | 0.68 |
| modern_trad | 15 | 0.0336 | 0.1208 (81%) | +0.028% | 0.67 |

`Mean val rank IC` is the selection metric (mean over the six validation windows). `Test rank IC` is the mean monthly Spearman correlation between prediction and realized next-month return over 2021-01…2026-08 across **all** stocks. `Pred. rank autocorr` is the month-over-month rank correlation of the prediction on the liquid universe (higher = steadier book). `_trad` arms train and validate only on tradeable-universe rows (§3.4).

### 3.2 Portfolios, full-universe models (2021-01 – 2026-08, 68 months)

| Arm | Portfolio | IR gross | IR net | Sharpe net | CAGR gross / net | α t (net) | β (t), gross | Rolling-12m β min / max (months >1) | One-way turnover, % of gross | Max DD net | Short-book median mcap | Positions |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| modern_nomom | legacy | 0.90 | 0.76 | 0.88 | 30.2% / 24.6% | 2.14 | -0.12 (-0.47) | -1.16 / +2.07 (1) | 46% | -44% | $882M | 201–201 |
| modern_nomom | pm_free | -0.24 | -0.44 | -0.24 | -2.7% / -6.5% | -0.60 | +0.03 (+0.22) | -0.80 / +1.06 (1) | 48% | -48% | $2,247M | 201–206 |
| modern_nomom | pm_t20 | -0.03 | -0.15 | 0.04 | 1.4% / -1.2% | 0.04 | +0.05 (+0.30) | -0.85 / +1.18 (1) | 20% | -37% | $2,204M | 203–244 |
| modern_nomom | pm_t10 | -0.17 | -0.27 | -0.06 | -1.0% / -2.9% | -0.26 | +0.09 (+0.61) | -0.67 / +1.12 (1) | 10% | -36% | $2,493M | 203–272 |
| modern | legacy | 1.04 | 0.89 | 1.02 | 35.9% / 29.9% | 2.46 | -0.11 (-0.44) | -1.19 / +1.89 (1) | 49% | -42% | $890M | 201–201 |
| modern | pm_free | -0.10 | -0.31 | -0.11 | -0.3% / -4.4% | -0.34 | +0.06 (+0.38) | -0.80 / +1.26 (1) | 51% | -41% | $2,251M | 202–206 |
| modern | pm_t20 | -0.11 | -0.23 | -0.04 | -0.4% / -2.9% | -0.19 | +0.08 (+0.49) | -0.90 / +1.35 (1) | 20% | -39% | $2,296M | 203–247 |
| modern | pm_t10 | -0.29 | -0.39 | -0.18 | -3.5% / -5.4% | -0.46 | +0.04 (+0.22) | -0.67 / +1.02 (1) | 10% | -38% | $2,568M | 203–283 |

Costs: tiered trading and borrow assumptions from `experiments/pm_ablation/README.md` §1; net = gross − trading cost − borrow cost. `Positions` is the min–max count of holdings per month (limit 100–500). Rolling-12m β: the parenthesis is the number of 12-month windows with β > 1.

### 3.3 Legs, costs and trading, headline arm (`modern_nomom`)

| Portfolio | Long-leg CAGR | Short-leg CAGR | Avg trade cost (bp NAV/mo) | Avg borrow cost (bp NAV/mo) | Traded notional (x capital/mo) | Months cap relaxed |
|---|---:|---:|---:|---:|---:|---:|
| legacy | 10.4% | 12.2% | 23.5 | 13.8 | 1.83 | 0 |
| pm_free | 9.1% | -14.8% | 22.6 | 10.6 | 1.93 | 0 |
| pm_t20 | 12.0% | -13.6% | 10.7 | 10.7 | 0.80 | 0 |
| pm_t10 | 10.7% | -14.2% | 5.8 | 10.3 | 0.40 | 1 |

Leg CAGRs are gross and in excess of the risk-free rate; a *negative* short-leg CAGR means the shorted names rose.

Calendar-year returns, `modern_nomom`:

| Portfolio (net of costs) | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 (8 mo) |
|---|---:|---:|---:|---:|---:|---:|
| legacy | -0.9% | +43.9% | +27.1% | +39.0% | +31.8% | +4.4% |
| pm_t10 | +3.7% | +2.5% | -6.0% | +2.7% | -17.5% | -0.1% |
| T-bill + 4% | +4.1% | +6.2% | +9.5% | +9.3% | +8.4% | +5.2% |
| S&P 500 | +26.9% | -19.4% | +24.2% | +23.3% | +16.4% | +12.3% |

### 3.4 `_trad` arms (trained and validated on the tradeable universe only)

| Arm | Portfolio | IR gross | IR net | Sharpe net | CAGR gross / net | α t (net) | β (t), gross | Rolling-12m β min / max (months >1) | One-way turnover, % of gross | Max DD net | Short-book median mcap | Positions |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| modern_nomom_trad | legacy | 1.13 | 0.89 | 1.10 | 26.5% / 20.9% | 2.61 | -0.04 (-0.27) | -0.58 / +0.82 (0) | 56% | -23% | $1,742M | 201–201 |
| modern_nomom_trad | pm_free | -0.27 | -0.56 | -0.30 | -1.4% / -5.6% | -0.56 | -0.08 (-0.64) | -0.70 / +0.74 (0) | 59% | -38% | $2,621M | 201–204 |
| modern_nomom_trad | pm_t20 | -0.29 | -0.45 | -0.19 | -1.8% / -4.1% | -0.23 | -0.13 (-1.05) | -0.70 / +0.67 (0) | 20% | -40% | $2,718M | 204–251 |
| modern_nomom_trad | pm_t10 | -0.37 | -0.48 | -0.22 | -2.7% / -4.3% | -0.23 | -0.17 (-1.40) | -0.81 / +0.50 (0) | 10% | -43% | $2,992M | 204–270 |
| modern_trad | legacy | 1.39 | 1.12 | 1.35 | 29.9% / 24.2% | 3.28 | -0.09 (-0.64) | -0.52 / +0.34 (0) | 56% | -15% | $1,700M | 201–201 |
| modern_trad | pm_free | -0.18 | -0.48 | -0.20 | 0.4% / -3.9% | -0.26 | -0.13 (-1.11) | -0.64 / +0.18 (0) | 59% | -46% | $2,646M | 201–205 |
| modern_trad | pm_t20 | 0.04 | -0.13 | 0.16 | 3.7% / 1.3% | 0.59 | -0.12 (-1.04) | -0.54 / +0.20 (0) | 20% | -36% | $2,688M | 202–243 |
| modern_trad | pm_t10 | 0.07 | -0.06 | 0.25 | 4.2% / 2.4% | 0.73 | -0.08 (-0.78) | -0.65 / +0.17 (0) | 10% | -39% | $3,009M | 202–272 |

### 3.5 All-feature arm (`all_nomom`, 139 characteristics)

Model-level: mean val rank IC 0.1132, test rank IC 0.1163, OOS R² -0.014%, prediction rank autocorr 0.79.

| Arm | Portfolio | IR gross | IR net | Sharpe net | CAGR gross / net | α t (net) | β (t), gross | Rolling-12m β min / max (months >1) | One-way turnover, % of gross | Max DD net | Short-book median mcap | Positions |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| all_nomom | legacy | 1.25 | 1.06 | 1.25 | 32.9% / 27.8% | 2.94 | -0.04 (-0.23) | -1.03 / +0.89 (0) | 44% | -29% | $1,180M | 201–201 |
| all_nomom | pm_free | -0.57 | -0.88 | -0.52 | -3.0% / -6.3% | -0.97 | -0.11 (-1.22) | -0.77 / +0.16 (0) | 49% | -38% | $4,212M | 201–204 |
| all_nomom | pm_t20 | -0.74 | -0.93 | -0.56 | -4.6% / -6.6% | -1.11 | -0.09 (-0.99) | -0.66 / +0.16 (0) | 20% | -40% | $4,425M | 202–236 |
| all_nomom | pm_t10 | -0.54 | -0.68 | -0.31 | -2.4% / -3.8% | -0.51 | -0.09 (-1.04) | -0.63 / +0.21 (0) | 10% | -32% | $5,003M | 202–271 |

## 4. Findings

1. **Rank IC and R² are in line with the other tree models** on the small arms (test IC 0.148, R² +0.140% on `modern_nomom`); the 139-feature arm has a slightly lower IC and an R² of about zero. XGBoost is not worse than RF here once tuned on the right metric.
2. **Legacy portfolio:** IR 0.90 / 1.04 / 1.25 (12 / 15 / 139 features). The 139-feature model has the best legacy book of the XGB arms and the worst PM book.
3. Under the PM constraints every arm is at or below 0 (see §3); `experiments/pm_ablation/README.md` explains it.
4. Rolling-12m β exceeds 1 in at most one window (the first, ending Dec 2021) on every book; the 139-feature books never exceed 0.9.

## 5. Limitations

- Depth 2 and the smallest tree counts sit on the grid edge (see §2); depth-1 stumps or < 50 trees were not tried.
- The `xgb.py` → `xgb_2.py` gain bundles several changes, and the feature set is not identical (139 vs 147).
- Single seed; assumed cost tiers. Same PM caveats as `experiments/rf_3/README.md` §5.

## Reproduce

```
.venv/bin/python experiments/xgb_2/xgb_2.py                                              # modern_nomom, modern  -> xgb2_results.json / xgb2_summary.csv
.venv/bin/python experiments/xgb_2/xgb_2.py --arms modern_nomom_trad,modern_trad --suffix _trad   # tradeable-universe training -> xgb2_results_trad.json
```
Outputs are in `output/` (`oos_predictions_xgb2_<arm>.csv`, `portfolio_holdings_<portfolio>_xgb2_<arm>.csv`, `portfolio_returns_<portfolio>_xgb2_<arm>.csv`, `xgb2_feature_importance_<arm>.csv`).

(The 139-feature arm: `.venv/bin/python experiments/xgb_2/xgb_2.py --arms all_nomom --suffix _all` → `xgb2_results_all.json`.)
