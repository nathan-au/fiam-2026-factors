# Extra-Trees (Extremely Randomized Trees) — Methodology and Results

Implementation: `et.py` (run with `.venv/bin/python et.py`). Sibling of `rf_3.py` (`docs/RF_3.md`); new file, no existing file edited.

**Why this model.** Extra-Trees draw split thresholds at random and grow every tree on all rows (no bootstrap): more randomization per tree, less variance in the ensemble — a natural candidate when the label is close to pure noise (`docs/OLS.md` §3.1). Grid: `max_features` {sqrt, 0.33} × `min_samples_leaf` {100, 300, 1000}, 300 trees. Features: `modern_nomom` (12, headline), `modern` (15, reference). Everything else — data, folds, selection on validation rank IC, winsorized label, portfolio layer — is identical to `docs/RF_3.md`.

**Bottom line.** ET matches RF on rank IC (0.15 vs 0.14) but is no better on the portfolio: legacy IR 0.96 / 1.13, and it fails the tradeable-universe test exactly like the others (`pm_t10` IR −0.28 gross). It also selected the **largest** allowed leaf (1000) in five of six folds, so the grid edge, not the optimum, was picked.

## Portfolio layer (shared by the five scripts of this batch)

Two portfolios are built from the same predictions each month and every result is reported **gross and net of costs**:

| Name | What it is |
|---|---|
| `legacy` | The exact LP of every earlier script (`docs/OLS.md` §2.5): $10M `dolvol_126d` screen, dollar-neutral, `beta_60m`-neutral, gross 200%, 1% per-name cap. It reproduces `rf_2.py`'s recorded IR (1.0373) to four decimals on `rf_2.py`'s own predictions, so it is directly comparable to `docs/RF.md`, `docs/RF_2.md`, `docs/XGB.md`, etc. |
| `pm_free` / `pm_t20` / `pm_t10` | A "portfolio-manager" LP built from *Valentino's FIAM Tips* (`Valentino_FIAM_Tips.pdf`) and the financial-engineer notes: price ≥ $5 and market cap ≥ $500M screens (plus the $10M dollar-volume screen); neutral to **both** `beta_60m` and `betabab_1260d`; net sector exposure ≤ 5% of NAV and sector share ≤ 35% of the gross book (2-digit GICS); and a **hard one-way turnover budget** as an LP constraint — none (`pm_free`), 20% (`pm_t20`) or 10% (`pm_t10`, the headline, following the tips' "around 10% per month"). |

- **Turnover convention:** one-way, as a share of the 200% gross book, from drift-adjusted trades (last month's weights are drifted by that month's realized returns, which are known at the rebalance date). This is the same convention as this project's `avg_monthly_turnover`. If the cap is infeasible in a month (names forced out of the universe), it is loosened stepwise ×1.5, 2, 3, 5 and the month is counted in "Months cap relaxed".
- **Costs are assumptions, not measurements** (the panel has no borrow or spread data by name): one-way trading cost 5 / 10 / 20 bp and annual borrow 30 / 75 / 200 bp on short notional for market caps ≥ $10B / $2–10B / < $2B. Full derivation in `docs/PM_ABLATION.md`.
- **Pre-specified, not tuned.** The screens, sector limits and the 10% headline cap were fixed from the tips before any result was seen. The `pm_free`/`pm_t20`/`pm_t10` rows are a sensitivity sweep, not a search.
- **Selection:** hyperparameters (and tree counts for boosted models) are picked per fold on **validation mean monthly rank IC**, never on validation MSE and never on test data. The fitting label is `ret_exc_lead1m` winsorized at the training fold's 1st/99th percentiles (`--raw-target` disables); predictions remain in next-month-return units and OOS R² is scored on the raw return.

## 1. Selection (headline arm `modern_nomom`)

| Test year | Chosen (on validation rank IC) | Val IC | Val IC range over candidates |
|---|---|---:|---|
| 2021 | {'max_features': 'sqrt', 'min_samples_leaf': 1000} | 0.0533 | 0.0509 … 0.0533 |
| 2022 | {'max_features': 'sqrt', 'min_samples_leaf': 300} | 0.0989 | 0.0977 … 0.0989 |
| 2023 | {'max_features': 'sqrt', 'min_samples_leaf': 1000} | 0.1559 | 0.1349 … 0.1559 |
| 2024 | {'max_features': 'sqrt', 'min_samples_leaf': 1000} | 0.1555 | 0.1439 … 0.1555 |
| 2025 | {'max_features': 'sqrt', 'min_samples_leaf': 1000} | 0.1531 | 0.1476 … 0.1531 |
| 2026 | {'max_features': 'sqrt', 'min_samples_leaf': 1000} | 0.1390 | 0.1335 … 0.1390 |

`max_features=sqrt` wins in every fold, and `min_samples_leaf=1000` in five of six — the search hit the edge of the grid, so a larger leaf might do better still (untested).

| Feature | Avg. importance |
|---|---:|
| rmax5_21d | 0.218 |
| ivol_capm_21d | 0.217 |
| mispricing_perf | 0.179 |
| qmj_prof | 0.100 |
| gp_at | 0.058 |
| qmj | 0.045 |
| at_gr1 | 0.040 |
| saleq_su | 0.040 |
| mispricing_mgmt | 0.026 |
| qmj_growth | 0.026 |

## 2. Results

### 3.1 Model-level statistics

| Arm | Features | Mean val rank IC | Test rank IC (% months > 0) | OOS R² | Pred. rank autocorr |
|---|---:|---:|---:|---:|---:|
| modern_nomom | 12 | 0.1259 | 0.1516 (82%) | +0.154% | 0.81 |
| modern | 15 | 0.1258 | 0.1516 (82%) | +0.157% | 0.83 |
| modern_nomom_trad | 12 | 0.0251 | 0.1294 (79%) | +0.023% | 0.75 |
| modern_trad | 15 | 0.0256 | 0.1348 (81%) | +0.038% | 0.76 |

`Mean val rank IC` is the selection metric (mean over the six validation windows). `Test rank IC` is the mean monthly Spearman correlation between prediction and realized next-month return over 2021-01…2026-08 across **all** stocks. `Pred. rank autocorr` is the month-over-month rank correlation of the prediction on the liquid universe (higher = steadier book). `_trad` arms train and validate only on tradeable-universe rows (§3.4).

### 3.2 Portfolios, full-universe models (2021-01 – 2026-08, 68 months)

| Arm | Portfolio | IR gross | IR net | Sharpe net | CAGR gross / net | α t (net) | β (t), gross | Rolling-12m β min / max (months >1) | One-way turnover, % of gross | Max DD net | Short-book median mcap | Positions |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| modern_nomom | legacy | 0.96 | 0.81 | 0.94 | 32.1% / 26.5% | 2.47 | -0.31 (-1.28) | -1.51 / +1.59 (1) | 45% | -31% | $916M | 201–201 |
| modern_nomom | pm_free | -0.11 | -0.30 | -0.11 | -0.6% / -4.5% | -0.20 | -0.05 (-0.30) | -0.87 / +1.32 (1) | 49% | -45% | $2,236M | 201–205 |
| modern_nomom | pm_t20 | -0.15 | -0.27 | -0.08 | -1.5% / -4.0% | -0.19 | +0.01 (+0.03) | -0.92 / +1.48 (1) | 20% | -45% | $2,263M | 203–240 |
| modern_nomom | pm_t10 | -0.28 | -0.37 | -0.18 | -3.7% / -5.6% | -0.45 | +0.03 (+0.18) | -0.82 / +1.36 (1) | 10% | -47% | $2,411M | 203–280 |
| modern | legacy | 1.13 | 0.97 | 1.11 | 37.7% / 31.8% | 2.77 | -0.22 (-0.92) | -1.12 / +1.50 (1) | 45% | -23% | $860M | 201–201 |
| modern | pm_free | -0.08 | -0.28 | -0.08 | 0.4% / -3.6% | -0.12 | -0.05 (-0.32) | -0.77 / +1.21 (1) | 49% | -42% | $2,160M | 202–205 |
| modern | pm_t20 | -0.14 | -0.27 | -0.07 | -0.9% / -3.4% | -0.20 | +0.03 (+0.18) | -0.86 / +1.34 (1) | 20% | -39% | $2,222M | 204–236 |
| modern | pm_t10 | -0.32 | -0.42 | -0.21 | -3.9% / -5.8% | -0.48 | -0.01 (-0.04) | -0.82 / +1.22 (1) | 10% | -45% | $2,432M | 204–272 |

Costs: tiered trading and borrow assumptions from `docs/PM_ABLATION.md` §1; net = gross − trading cost − borrow cost. `Positions` is the min–max count of holdings per month (limit 100–500). Rolling-12m β: the parenthesis is the number of 12-month windows with β > 1.

### 3.3 Legs, costs and trading, headline arm (`modern_nomom`)

| Portfolio | Long-leg CAGR | Short-leg CAGR | Avg trade cost (bp NAV/mo) | Avg borrow cost (bp NAV/mo) | Traded notional (x capital/mo) | Months cap relaxed |
|---|---:|---:|---:|---:|---:|---:|
| legacy | 9.3% | 16.4% | 22.9 | 13.8 | 1.81 | 0 |
| pm_free | 12.8% | -16.0% | 22.6 | 10.6 | 1.95 | 0 |
| pm_t20 | 10.9% | -15.1% | 10.5 | 10.6 | 0.80 | 0 |
| pm_t10 | 10.5% | -16.8% | 5.8 | 10.4 | 0.40 | 1 |

Leg CAGRs are gross and in excess of the risk-free rate; a *negative* short-leg CAGR means the shorted names rose.

Calendar-year returns, `modern_nomom`:

| Portfolio (net of costs) | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 (8 mo) |
|---|---:|---:|---:|---:|---:|---:|
| legacy | +22.7% | +44.2% | +14.4% | +44.0% | +20.8% | +7.5% |
| pm_t10 | +1.9% | -2.3% | +1.8% | -1.1% | -22.2% | -7.5% |
| T-bill + 4% | +4.1% | +6.2% | +9.5% | +9.3% | +8.4% | +5.2% |
| S&P 500 | +26.9% | -19.4% | +24.2% | +23.3% | +16.4% | +12.3% |

### 3.4 `_trad` arms (trained and validated on the tradeable universe only)

| Arm | Portfolio | IR gross | IR net | Sharpe net | CAGR gross / net | α t (net) | β (t), gross | Rolling-12m β min / max (months >1) | One-way turnover, % of gross | Max DD net | Short-book median mcap | Positions |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| modern_nomom_trad | legacy | 0.76 | 0.60 | 0.75 | 21.9% / 17.1% | 1.97 | -0.22 (-1.06) | -1.26 / +1.47 (1) | 50% | -30% | $1,603M | 201–201 |
| modern_nomom_trad | pm_free | -0.04 | -0.25 | -0.04 | 1.4% / -2.6% | 0.14 | -0.18 (-1.17) | -0.89 / +1.10 (1) | 55% | -47% | $2,648M | 201–205 |
| modern_nomom_trad | pm_t20 | -0.04 | -0.16 | 0.05 | 1.4% / -0.9% | 0.38 | -0.20 (-1.32) | -0.97 / +1.19 (1) | 20% | -48% | $2,741M | 203–235 |
| modern_nomom_trad | pm_t10 | -0.14 | -0.24 | -0.02 | -0.4% / -2.1% | 0.12 | -0.12 (-0.82) | -0.95 / +1.14 (1) | 10% | -46% | $2,974M | 203–273 |
| modern_trad | legacy | 1.05 | 0.86 | 1.04 | 28.7% / 23.4% | 2.75 | -0.28 (-1.51) | -1.22 / +0.90 (0) | 51% | -25% | $1,431M | 201–201 |
| modern_trad | pm_free | 0.25 | -0.01 | 0.25 | 7.0% / 2.7% | 1.00 | -0.25 (-2.03) | -0.77 / +0.54 (0) | 55% | -36% | $2,556M | 201–205 |
| modern_trad | pm_t20 | 0.04 | -0.10 | 0.15 | 3.5% / 1.1% | 0.76 | -0.26 (-2.04) | -0.86 / +0.50 (0) | 20% | -45% | $2,667M | 203–237 |
| modern_trad | pm_t10 | 0.05 | -0.06 | 0.19 | 3.5% / 1.7% | 0.64 | -0.13 (-1.01) | -0.99 / +0.60 (0) | 10% | -48% | $2,962M | 203–277 |

## 3. Findings

1. Rank IC 0.152 (`modern_nomom`), OOS R² +0.154% — the second-highest rank IC of the batch (LightGBM 0.155) and the third-highest OOS R² (after RF and CatBoost), but the differences between models (IC 0.143–0.155) are far inside noise.
2. Legacy IR: 0.96 (`modern_nomom`) vs 1.13 (`modern`); Random Forest reaches 1.26/1.33 on the same LP. ET's importance is the most concentrated in the lottery/volatility pair (`rmax5_21d`, `ivol_capm_21d`, 44% together), i.e. ET leans hardest on the effect that is untradeable.
3. Under the PM constraints: `pm_free` −0.11, `pm_t10` −0.28 gross (net −0.30 / −0.37). The turnover cap itself is cheap on the legacy book (`+turnover_10pct`: IR 0.96 → 0.91) — see `docs/PM_ABLATION.md`.
4. Beta: legacy realized β −0.31 (t −1.3 in this window); PM books ≈ 0. Rolling-12m β exceeds 1 only in the first window.

## 4. Limitations

- Grid-edge selection (leaf 1000, `max_features=sqrt`) — the search space, not the model, limited this run.
- Single seed; no bootstrap, so `max_samples` was not used. Costs are assumed tiers. `all_nomom` not run.
- Same caveats on the PM constraints and cost tiers as `docs/RF_3.md` §5.

## Reproduce

```
.venv/bin/python et.py                                              # modern_nomom, modern  -> et_results.json / et_summary.csv
.venv/bin/python et.py --arms modern_nomom_trad,modern_trad --suffix _trad   # tradeable-universe training -> et_results_trad.json
```
Outputs are in `output/` (`oos_predictions_et_<arm>.csv`, `portfolio_holdings_<portfolio>_et_<arm>.csv`, `portfolio_returns_<portfolio>_et_<arm>.csv`, `et_feature_importance_<arm>.csv`).
