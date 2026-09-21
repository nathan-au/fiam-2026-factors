# LightGBM, Forest-Style — Methodology and Results

Implementation: `lgbm.py` (run with `.venv/bin/python lgbm.py`). `docs/NEXT.md` §1 named LightGBM as the natural next step; `docs/XGB.md` showed ordinary boosting overfits this panel. New file; no existing file edited.

**Configuration ("forest-style").** Learning rate 0.02, shallow trees (`num_leaves` 7 or 31), huge leaves (`min_child_samples` 1000 or 5000), 30% feature sampling per tree, 50% row bagging every round, randomized thresholds (`extra_trees=True`), L2 = 100, 63 bins. Up to 600 trees; the tree count {50, 100, 200, 400, 600} is selected per fold on **validation rank IC** (`num_iteration`). One extra candidate per fold: LightGBM's bagged-forest mode (`boosting='rf'`, 63 leaves, 300 trees). Features `modern_nomom` (12, headline) and `modern` (15). Everything else as `docs/RF_3.md`.

**Bottom line.** Similar to the other tree models on rank IC (0.155 — nominally the highest) and worse on the legacy portfolio (IR 0.92 / 1.07). Validation picked **50 trees** (the smallest option) in five of six folds and **never** picked `rf` mode. The tradeable-universe result is the same as everywhere else.

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
| 2021 | {'mode': 'gbdt', 'num_leaves': 31, 'min_child_samples': 5000, 'n_trees': 50} | 0.0563 | 0.0490 … 0.0563 |
| 2022 | {'mode': 'gbdt', 'num_leaves': 31, 'min_child_samples': 5000, 'n_trees': 200} | 0.0959 | 0.0797 … 0.0959 |
| 2023 | {'mode': 'gbdt', 'num_leaves': 31, 'min_child_samples': 5000, 'n_trees': 50} | 0.1674 | 0.0925 … 0.1674 |
| 2024 | {'mode': 'gbdt', 'num_leaves': 7, 'min_child_samples': 1000, 'n_trees': 50} | 0.1650 | 0.1339 … 0.1650 |
| 2025 | {'mode': 'gbdt', 'num_leaves': 7, 'min_child_samples': 5000, 'n_trees': 50} | 0.1553 | 0.1407 … 0.1553 |
| 2026 | {'mode': 'gbdt', 'num_leaves': 7, 'min_child_samples': 1000, 'n_trees': 50} | 0.1471 | 0.1296 … 0.1471 |

The smallest tree count is chosen in 5 of 6 folds: validation IC is peaked at very few, very smooth trees, i.e. the exploitable pattern is close to a low-order function of a few volatility/quality ranks. As with the other boosted models, the tree-count grid edge (50) was hit; fewer trees were not tried.

| Feature | Avg. importance |
|---|---:|
| ivol_capm_21d | 0.245 |
| mispricing_perf | 0.208 |
| qmj | 0.129 |
| rmax5_21d | 0.072 |
| qmj_growth | 0.064 |
| betabab_1260d | 0.062 |
| saleq_su | 0.056 |
| gp_at | 0.048 |
| qmj_prof | 0.036 |
| at_gr1 | 0.034 |

## 2. Results

### 3.1 Model-level statistics

| Arm | Features | Mean val rank IC | Test rank IC (% months > 0) | OOS R² | Pred. rank autocorr |
|---|---:|---:|---:|---:|---:|
| modern_nomom | 12 | 0.1312 | 0.1545 (81%) | +0.109% | 0.85 |
| modern | 15 | 0.1304 | 0.1540 (79%) | +0.124% | 0.84 |
| modern_nomom_trad | 12 | 0.0388 | 0.1428 (78%) | +0.002% | 0.78 |
| modern_trad | 15 | 0.0329 | 0.1306 (81%) | +0.015% | 0.77 |

`Mean val rank IC` is the selection metric (mean over the six validation windows). `Test rank IC` is the mean monthly Spearman correlation between prediction and realized next-month return over 2021-01…2026-08 across **all** stocks. `Pred. rank autocorr` is the month-over-month rank correlation of the prediction on the liquid universe (higher = steadier book). `_trad` arms train and validate only on tradeable-universe rows (§3.4).

### 3.2 Portfolios, full-universe models (2021-01 – 2026-08, 68 months)

| Arm | Portfolio | IR gross | IR net | Sharpe net | CAGR gross / net | α t (net) | β (t), gross | Rolling-12m β min / max (months >1) | One-way turnover, % of gross | Max DD net | Short-book median mcap | Positions |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| modern_nomom | legacy | 0.92 | 0.80 | 0.91 | 33.4% / 27.7% | 2.30 | -0.25 (-0.87) | -1.29 / +2.20 (1) | 44% | -49% | $851M | 201–201 |
| modern_nomom | pm_free | 0.04 | -0.16 | 0.04 | 2.8% / -1.4% | 0.13 | -0.04 (-0.23) | -0.92 / +1.22 (1) | 49% | -34% | $2,014M | 202–206 |
| modern_nomom | pm_t20 | -0.12 | -0.24 | -0.05 | -0.5% / -3.2% | -0.09 | -0.02 (-0.13) | -0.97 / +1.39 (1) | 20% | -40% | $1,997M | 204–240 |
| modern_nomom | pm_t10 | -0.12 | -0.22 | -0.02 | -0.6% / -2.6% | -0.11 | +0.05 (+0.29) | -0.94 / +1.26 (1) | 10% | -40% | $2,215M | 204–273 |
| modern | legacy | 1.07 | 0.92 | 1.06 | 35.7% / 29.9% | 2.68 | -0.25 (-1.06) | -1.36 / +1.63 (1) | 45% | -28% | $847M | 201–201 |
| modern | pm_free | -0.12 | -0.31 | -0.12 | -0.7% / -4.7% | -0.14 | -0.12 (-0.71) | -0.76 / +1.15 (1) | 49% | -47% | $1,994M | 202–205 |
| modern | pm_t20 | -0.10 | -0.22 | -0.03 | -0.2% / -2.8% | -0.05 | -0.01 (-0.09) | -0.93 / +1.34 (1) | 20% | -42% | $2,004M | 204–236 |
| modern | pm_t10 | -0.33 | -0.43 | -0.23 | -4.7% / -6.6% | -0.53 | +0.00 (+0.02) | -0.81 / +1.32 (1) | 10% | -48% | $2,230M | 204–272 |

Costs: tiered trading and borrow assumptions from `docs/PM_ABLATION.md` §1; net = gross − trading cost − borrow cost. `Positions` is the min–max count of holdings per month (limit 100–500). Rolling-12m β: the parenthesis is the number of 12-month windows with β > 1.

### 3.3 Legs, costs and trading, headline arm (`modern_nomom`)

| Portfolio | Long-leg CAGR | Short-leg CAGR | Avg trade cost (bp NAV/mo) | Avg borrow cost (bp NAV/mo) | Traded notional (x capital/mo) | Months cap relaxed |
|---|---:|---:|---:|---:|---:|---:|
| legacy | 14.8% | 11.3% | 22.9 | 14.1 | 1.77 | 0 |
| pm_free | 13.0% | -13.4% | 23.2 | 11.1 | 1.95 | 0 |
| pm_t20 | 11.3% | -14.8% | 10.9 | 11.2 | 0.80 | 0 |
| pm_t10 | 11.9% | -15.3% | 5.9 | 10.7 | 0.40 | 1 |

Leg CAGRs are gross and in excess of the risk-free rate; a *negative* short-leg CAGR means the shorted names rose.

Calendar-year returns, `modern_nomom`:

| Portfolio (net of costs) | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 (8 mo) |
|---|---:|---:|---:|---:|---:|---:|
| legacy | -6.4% | +48.8% | +17.1% | +53.6% | +43.6% | +11.0% |
| pm_t10 | +4.7% | +1.6% | -1.5% | +4.5% | -19.8% | -1.7% |
| T-bill + 4% | +4.1% | +6.2% | +9.5% | +9.3% | +8.4% | +5.2% |
| S&P 500 | +26.9% | -19.4% | +24.2% | +23.3% | +16.4% | +12.3% |

### 3.4 `_trad` arms (trained and validated on the tradeable universe only)

| Arm | Portfolio | IR gross | IR net | Sharpe net | CAGR gross / net | α t (net) | β (t), gross | Rolling-12m β min / max (months >1) | One-way turnover, % of gross | Max DD net | Short-book median mcap | Positions |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| modern_nomom_trad | legacy | 0.66 | 0.52 | 0.66 | 20.3% / 15.4% | 1.79 | -0.28 (-1.25) | -1.22 / +1.55 (1) | 48% | -30% | $1,259M | 201–201 |
| modern_nomom_trad | pm_free | -0.19 | -0.44 | -0.20 | -0.6% / -4.6% | -0.27 | -0.12 (-0.92) | -0.91 / +0.89 (0) | 56% | -45% | $2,466M | 201–205 |
| modern_nomom_trad | pm_t20 | -0.01 | -0.15 | 0.09 | 2.6% / 0.2% | 0.37 | -0.11 (-0.78) | -0.88 / +0.69 (0) | 20% | -41% | $2,426M | 204–237 |
| modern_nomom_trad | pm_t10 | -0.00 | -0.12 | 0.14 | 2.7% / 0.9% | 0.33 | -0.01 (-0.10) | -0.88 / +0.70 (0) | 10% | -41% | $2,681M | 204–267 |
| modern_trad | legacy | 0.79 | 0.62 | 0.78 | 22.6% / 17.6% | 1.89 | -0.09 (-0.45) | -0.88 / +1.40 (1) | 52% | -27% | $1,460M | 201–201 |
| modern_trad | pm_free | -0.35 | -0.57 | -0.36 | -4.6% / -8.5% | -0.69 | -0.11 (-0.72) | -0.80 / +1.25 (1) | 57% | -49% | $2,542M | 202–205 |
| modern_trad | pm_t20 | -0.30 | -0.42 | -0.21 | -3.6% / -5.8% | -0.39 | -0.08 (-0.51) | -0.77 / +1.28 (1) | 20% | -42% | $2,686M | 203–236 |
| modern_trad | pm_t10 | -0.35 | -0.44 | -0.22 | -4.1% / -5.8% | -0.48 | -0.02 (-0.15) | -0.73 / +1.25 (1) | 10% | -43% | $2,896M | 203–269 |

## 3. Findings

1. **No gain from boosting over bagging here.** Legacy IR 0.92 (`modern_nomom`) / 1.07 (`modern`) vs Random Forest 1.26/1.33; OOS R² +0.109% / +0.124% is positive but smaller than RF's +0.225%.
2. **PM book:** `pm_t10` IR −0.12 gross / −0.22 net; `pm_free` +0.04 / −0.16.
3. LightGBM has the steadiest prediction ranking of the batch (autocorr 0.85) but that does not translate to lower legacy turnover (44% one-way).
4. Legacy rolling-12m β peaks at +2.20 (first window, Dec 2021) — the highest of the batch; `pm_t10` peaks at +1.26. Only the first window exceeds 1.

## 4. Limitations

- Tree-count and depth grid-edge picks (see §1); `rf` mode was included but got no validation wins.
- Single seed; assumed costs; `all_nomom` not run. Same PM caveats as `docs/RF_3.md` §5.

## Reproduce

```
.venv/bin/python lgbm.py                                              # modern_nomom, modern  -> lgbm_results.json / lgbm_summary.csv
.venv/bin/python lgbm.py --arms modern_nomom_trad,modern_trad --suffix _trad   # tradeable-universe training -> lgbm_results_trad.json
```
Requires `pip install lightgbm==4.7.0 catboost==1.2.10` for the LightGBM/CatBoost scripts (`requirements.txt` is untouched).

Outputs are in `output/` (`oos_predictions_lgbm_<arm>.csv`, `portfolio_holdings_<portfolio>_lgbm_<arm>.csv`, `portfolio_returns_<portfolio>_lgbm_<arm>.csv`, `lgbm_feature_importance_<arm>.csv`).
