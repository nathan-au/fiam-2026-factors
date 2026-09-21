# CatBoost — Methodology and Results

Implementation: `cat.py` (run with `.venv/bin/python cat.py`). New file; no existing file edited.

**Why this model.** CatBoost grows **oblivious (symmetric) trees** — one split per level for the whole tree — and uses ordered boosting, which caps per-tree capacity: the most heavily regularized boosting variant, so the a-priori best bet for a near-noise label. Grid: `depth` {4, 6} × `l2_leaf_reg` {30, 300}, learning rate 0.03, `rsm=0.5` (feature sampling), Bernoulli `subsample=0.5`, 64 borders, 800 iterations; the tree count {100, 200, 400, 800} is chosen per fold on validation rank IC (`ntree_end`). Features `modern_nomom` (12, headline) and `modern` (15). Everything else as `docs/RF_3.md`.

**Bottom line.** CatBoost's validation picks were the most extreme of the batch — **100 trees** (the smallest option) in all six folds, depth 4 (the smaller depth) in five of six and `l2_leaf_reg=300` (the larger L2) in five of six — i.e. the grid edges on all three axes. Portfolio results look like the others: legacy IR 0.92 / 1.19; PM book about 0 or negative.

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
| 2021 | {'depth': 4, 'l2_leaf_reg': 30.0, 'n_trees': 100} | 0.0525 | 0.0451 … 0.0525 |
| 2022 | {'depth': 4, 'l2_leaf_reg': 300.0, 'n_trees': 100} | 0.0777 | 0.0452 … 0.0777 |
| 2023 | {'depth': 4, 'l2_leaf_reg': 300.0, 'n_trees': 100} | 0.1150 | 0.0559 … 0.1150 |
| 2024 | {'depth': 4, 'l2_leaf_reg': 300.0, 'n_trees': 100} | 0.1501 | 0.1287 … 0.1501 |
| 2025 | {'depth': 4, 'l2_leaf_reg': 300.0, 'n_trees': 100} | 0.1427 | 0.1219 … 0.1427 |
| 2026 | {'depth': 6, 'l2_leaf_reg': 300.0, 'n_trees': 100} | 0.1355 | 0.1207 … 0.1355 |

Because nearly every selected value sits on the edge of the grid, this run says "the simplest model in the grid is best", not "these are the best hyperparameters". A shallower/stronger-L2/shorter run (depth 2–3, L2 ≥ 1000, < 100 trees) was not tested.

| Feature | Avg. importance |
|---|---:|
| betabab_1260d | 0.299 |
| qmj | 0.191 |
| qmj_growth | 0.122 |
| mispricing_perf | 0.069 |
| ivol_capm_21d | 0.061 |
| saleq_su | 0.059 |
| rmax5_21d | 0.054 |
| gp_at | 0.039 |
| niq_su | 0.030 |
| qmj_prof | 0.027 |

`betabab_1260d` carries 30% of CatBoost's importance (vs 11–12% in RF/XGB) — CatBoost leans on the Frazzini-Pedersen beta, i.e. on a low-beta/high-beta ordering; the book is neutral to that beta at formation, so the exposure shows up as a stock-selection tilt, not as market beta.

## 2. Results

### 3.1 Model-level statistics

| Arm | Features | Mean val rank IC | Test rank IC (% months > 0) | OOS R² | Pred. rank autocorr |
|---|---:|---:|---:|---:|---:|
| modern_nomom | 12 | 0.1122 | 0.1442 (82%) | +0.176% | 0.84 |
| modern | 15 | 0.1150 | 0.1445 (82%) | +0.173% | 0.84 |
| modern_nomom_trad | 12 | 0.0270 | 0.1108 (84%) | +0.012% | 0.71 |
| modern_trad | 15 | 0.0271 | 0.1076 (81%) | +0.067% | 0.74 |

`Mean val rank IC` is the selection metric (mean over the six validation windows). `Test rank IC` is the mean monthly Spearman correlation between prediction and realized next-month return over 2021-01…2026-08 across **all** stocks. `Pred. rank autocorr` is the month-over-month rank correlation of the prediction on the liquid universe (higher = steadier book). `_trad` arms train and validate only on tradeable-universe rows (§3.4).

### 3.2 Portfolios, full-universe models (2021-01 – 2026-08, 68 months)

| Arm | Portfolio | IR gross | IR net | Sharpe net | CAGR gross / net | α t (net) | β (t), gross | Rolling-12m β min / max (months >1) | One-way turnover, % of gross | Max DD net | Short-book median mcap | Positions |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| modern_nomom | legacy | 0.92 | 0.78 | 0.91 | 31.0% / 25.4% | 2.24 | -0.16 (-0.62) | -1.14 / +2.06 (1) | 45% | -46% | $923M | 201–201 |
| modern_nomom | pm_free | -0.03 | -0.23 | -0.03 | 1.2% / -2.7% | -0.09 | +0.02 (+0.12) | -0.80 / +1.23 (1) | 47% | -39% | $2,261M | 202–205 |
| modern_nomom | pm_t20 | -0.17 | -0.30 | -0.10 | -1.6% / -4.1% | -0.30 | +0.05 (+0.31) | -0.78 / +1.37 (1) | 20% | -41% | $2,275M | 204–240 |
| modern_nomom | pm_t10 | -0.31 | -0.41 | -0.21 | -4.2% / -6.1% | -0.54 | +0.05 (+0.29) | -0.74 / +1.15 (1) | 10% | -43% | $2,430M | 204–279 |
| modern | legacy | 1.19 | 1.03 | 1.18 | 37.6% / 31.8% | 2.84 | -0.12 (-0.56) | -1.19 / +1.61 (1) | 45% | -27% | $899M | 201–201 |
| modern | pm_free | -0.17 | -0.37 | -0.17 | -1.4% / -5.3% | -0.42 | +0.02 (+0.14) | -0.95 / +1.18 (1) | 48% | -45% | $2,232M | 201–205 |
| modern | pm_t20 | -0.18 | -0.31 | -0.11 | -1.7% / -4.2% | -0.33 | +0.06 (+0.37) | -0.92 / +1.22 (1) | 20% | -41% | $2,308M | 203–246 |
| modern | pm_t10 | -0.24 | -0.34 | -0.13 | -2.4% / -4.3% | -0.37 | +0.05 (+0.34) | -0.75 / +1.06 (1) | 10% | -37% | $2,471M | 203–279 |

Costs: tiered trading and borrow assumptions from `docs/PM_ABLATION.md` §1; net = gross − trading cost − borrow cost. `Positions` is the min–max count of holdings per month (limit 100–500). Rolling-12m β: the parenthesis is the number of 12-month windows with β > 1.

### 3.3 Legs, costs and trading, headline arm (`modern_nomom`)

| Portfolio | Long-leg CAGR | Short-leg CAGR | Avg trade cost (bp NAV/mo) | Avg borrow cost (bp NAV/mo) | Traded notional (x capital/mo) | Months cap relaxed |
|---|---:|---:|---:|---:|---:|---:|
| legacy | 13.2% | 9.9% | 23.3 | 13.7 | 1.80 | 0 |
| pm_free | 14.0% | -15.4% | 22.0 | 10.6 | 1.88 | 0 |
| pm_t20 | 12.1% | -16.4% | 10.6 | 10.7 | 0.80 | 0 |
| pm_t10 | 9.1% | -16.1% | 5.8 | 10.3 | 0.40 | 1 |

Leg CAGRs are gross and in excess of the risk-free rate; a *negative* short-leg CAGR means the shorted names rose.

Calendar-year returns, `modern_nomom`:

| Portfolio (net of costs) | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 (8 mo) |
|---|---:|---:|---:|---:|---:|---:|
| legacy | -2.6% | +50.1% | +16.5% | +33.4% | +34.1% | +18.6% |
| pm_t10 | -1.4% | -1.2% | -6.5% | +0.1% | -20.5% | -3.2% |
| T-bill + 4% | +4.1% | +6.2% | +9.5% | +9.3% | +8.4% | +5.2% |
| S&P 500 | +26.9% | -19.4% | +24.2% | +23.3% | +16.4% | +12.3% |

### 3.4 `_trad` arms (trained and validated on the tradeable universe only)

| Arm | Portfolio | IR gross | IR net | Sharpe net | CAGR gross / net | α t (net) | β (t), gross | Rolling-12m β min / max (months >1) | One-way turnover, % of gross | Max DD net | Short-book median mcap | Positions |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| modern_nomom_trad | legacy | 1.36 | 1.09 | 1.33 | 29.1% / 23.3% | 3.28 | -0.13 (-0.92) | -0.67 / +0.50 (0) | 55% | -14% | $1,636M | 201–201 |
| modern_nomom_trad | pm_free | -0.24 | -0.51 | -0.26 | -1.3% / -5.5% | -0.36 | -0.17 (-1.28) | -0.77 / +0.72 (0) | 57% | -46% | $2,599M | 201–205 |
| modern_nomom_trad | pm_t20 | -0.26 | -0.41 | -0.16 | -1.6% / -3.9% | -0.11 | -0.18 (-1.34) | -0.84 / +0.71 (0) | 20% | -45% | $2,647M | 203–244 |
| modern_nomom_trad | pm_t10 | -0.34 | -0.45 | -0.20 | -2.5% / -4.3% | -0.18 | -0.18 (-1.41) | -0.77 / +0.59 (0) | 10% | -45% | $2,908M | 203–274 |
| modern_trad | legacy | 1.11 | 0.87 | 1.08 | 25.8% / 20.3% | 2.88 | -0.24 (-1.61) | -0.72 / +0.34 (0) | 54% | -20% | $1,598M | 201–201 |
| modern_trad | pm_free | -0.08 | -0.37 | -0.10 | 1.6% / -2.6% | 0.22 | -0.28 (-2.33) | -0.84 / +0.25 (0) | 57% | -47% | $2,595M | 201–204 |
| modern_trad | pm_t20 | -0.10 | -0.26 | 0.01 | 1.3% / -1.1% | 0.41 | -0.24 (-2.00) | -0.83 / +0.22 (0) | 20% | -48% | $2,696M | 202–241 |
| modern_trad | pm_t10 | -0.09 | -0.21 | 0.07 | 1.7% / -0.1% | 0.50 | -0.20 (-1.72) | -0.90 / +0.15 (0) | 10% | -47% | $2,932M | 202–279 |

## 3. Findings

1. Legacy IR 0.92 (`modern_nomom`) / 1.19 (`modern`) vs RF 1.26/1.33; OOS R² +0.176% / +0.173%; test rank IC 0.144.
2. The `modern_nomom` legacy book has a −46% net max drawdown (vs −27% for RF); the momentum-free boosted books are all in the same range on the legacy LP (LightGBM −49%, CatBoost −46%, XGB_2 −44%), while the same models with the 3 momentum factors (`modern`) drop to −27% to −42%. That is the one place the momentum-free arm looks clearly worse — on drawdown, not IR — and in every model the trough is Feb 2021 (peak Dec 2020), i.e. the meme-stock squeeze window of the first two OOS months — not verified name by name.
3. PM books: `pm_free` −0.03 gross / −0.23 net, `pm_t10` −0.31 / −0.41.
4. Rolling-12m β peaks at +2.06 (first window) on legacy; PM books peak at 1.15–1.37; only the first window exceeds 1.

## 4. Limitations

- All-edges selection (see §1). Single seed; assumed costs; `all_nomom` not run. Same PM caveats as `docs/RF_3.md` §5.

## Reproduce

```
.venv/bin/python cat.py                                              # modern_nomom, modern  -> cat_results.json / cat_summary.csv
.venv/bin/python cat.py --arms modern_nomom_trad,modern_trad --suffix _trad   # tradeable-universe training -> cat_results_trad.json
```
Requires `pip install lightgbm==4.7.0 catboost==1.2.10` for the LightGBM/CatBoost scripts (`requirements.txt` is untouched).

Outputs are in `output/` (`oos_predictions_cat_<arm>.csv`, `portfolio_holdings_<portfolio>_cat_<arm>.csv`, `portfolio_returns_<portfolio>_cat_<arm>.csv`, `cat_feature_importance_<arm>.csv`).
