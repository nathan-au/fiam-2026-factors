# Random Forest, Round 3 — Momentum-Free Features, Rank-IC Selection, Portfolio-Manager LP

Implementation: `rf_3.py` (run with `.venv/bin/python rf_3.py`). Follows `docs/RF.md` / `docs/RF_2.md`; `rf.py` and `rf_2.py` are untouched.
Same data, target (`ret_exc_lead1m`), rank transform, walk-forward schedule (`docs/OLS.md` §2) as every earlier script; self-contained (ported, not imported).

**Bottom line.**
1. On the project's legacy LP the retrained Random Forest is the best model of the batch (IR 1.26 momentum-free, 1.33 with the original 15 factors, vs. 1.04 in `docs/RF_2.md`).
2. **That edge does not survive a tradeable universe.** Under the portfolio-manager LP (price ≥ $5, market cap ≥ $500M, dual-beta, sector limits, 10% turnover) IR falls to about 0 gross and about −0.3 net, and the short leg loses money. `docs/PM_ABLATION.md` shows why: the alpha lives in the short leg, in sub-$5 / sub-$500M names, concentrated in Health Care / biotech.
3. Momentum is not what the result rests on: removing the three momentum factors moves legacy IR 1.33 → 1.26, inside the ±0.11 noise band of `docs/RF_2.md` §1.

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

## 1. What changed vs. `rf_2.py`

| Change | Why |
|---|---|
| **Momentum removed** (`modern_nomom` = Modern minus `ret_12_1`, `ret_6_1`, `resff3_12_1`; `all_nomom` = all 147 minus the 8 momentum characteristics of `docs/FACTORS.md` §4). `modern` (the original 15) is kept as an in-harness reference. | The financial engineer flagged momentum as suspect; `docs/RF_2.md` §2 had already found dropping it did not hurt. Caveat: `mispricing_perf` is a composite that includes a momentum component (`docs/FACTORS.md` §13) and stays in every arm — it is the most load-bearing single factor in `docs/RF_2.md`'s leave-one-out. |
| Hyperparameters chosen on **validation rank IC** instead of MSE. | `docs/RF_2.md` §1: validation MSE was identical to four decimals across 15 configs. |
| Grid: `max_depth` {6, 8, 12} × `min_samples_leaf` {200, 500} (was leaf 100/50/200), 300 trees, `max_samples=0.5`, `max_features=sqrt`. | Bigger leaves = more averaging on an almost-pure-noise label; subsampling decorrelates trees and halves fit time. |
| Winsorized training label (1st/99th percentile of the training fold). | Heavy-tailed microcap returns dominate an MSE forest. |
| New portfolio layer (above). | Valentino's tips; financial-engineer notes. |

These changes were made together, so their effects are **not separately attributed**; the `rf_2.py` → `rf_3.py` IR gain (1.04 → 1.26/1.33 on the legacy LP) is a single-seed, single-run comparison.

## 2. Selection (headline arm `modern_nomom`)

| Test year | Chosen (on validation rank IC) | Val IC | Val IC range over candidates |
|---|---|---:|---|
| 2021 | {'max_depth': 8, 'min_samples_leaf': 200} | 0.0523 | 0.0507 … 0.0523 |
| 2022 | {'max_depth': 6, 'min_samples_leaf': 500} | 0.0679 | 0.0628 … 0.0679 |
| 2023 | {'max_depth': 6, 'min_samples_leaf': 200} | 0.0971 | 0.0778 … 0.0971 |
| 2024 | {'max_depth': 6, 'min_samples_leaf': 500} | 0.1423 | 0.1294 … 0.1423 |
| 2025 | {'max_depth': 6, 'min_samples_leaf': 200} | 0.1465 | 0.1371 … 0.1465 |
| 2026 | {'max_depth': 6, 'min_samples_leaf': 500} | 0.1348 | 0.1287 … 0.1348 |

Validation IC barely separates the six configs (range in the last column), and the picks jump between leaf 200 and 500 — the same "tuning is close to noise" finding as `docs/RF_2.md` §1, now on a metric that is at least the right one. Depth 6 is chosen in five of six folds.

Average impurity importance (fold-averaged):

| Feature | Avg. importance |
|---|---:|
| qmj | 0.175 |
| ivol_capm_21d | 0.138 |
| mispricing_perf | 0.129 |
| rmax5_21d | 0.125 |
| betabab_1260d | 0.113 |
| qmj_growth | 0.096 |
| qmj_prof | 0.063 |
| saleq_su | 0.046 |
| gp_at | 0.043 |
| at_gr1 | 0.028 |

The model leans on quality (`qmj*`), lottery/volatility (`ivol_capm_21d`, `rmax5_21d`) and `mispricing_perf`/`betabab_1260d` — i.e. it ranks stocks largely by volatility and low quality. That is consistent with where the edge lives (small, volatile, distressed shorts; `docs/PM_ABLATION.md`).

## 3. Results

### 3.1 Model-level statistics

| Arm | Features | Mean val rank IC | Test rank IC (% months > 0) | OOS R² | Pred. rank autocorr |
|---|---:|---:|---:|---:|---:|
| modern_nomom | 12 | 0.1068 | 0.1434 (81%) | +0.225% | 0.81 |
| modern | 15 | 0.1088 | 0.1456 (81%) | +0.195% | 0.81 |
| modern_nomom_trad | 12 | 0.0266 | 0.1168 (84%) | +0.083% | 0.72 |
| modern_trad | 15 | 0.0254 | 0.1273 (82%) | +0.079% | 0.74 |

`Mean val rank IC` is the selection metric (mean over the six validation windows). `Test rank IC` is the mean monthly Spearman correlation between prediction and realized next-month return over 2021-01…2026-08 across **all** stocks. `Pred. rank autocorr` is the month-over-month rank correlation of the prediction on the liquid universe (higher = steadier book). `_trad` arms train and validate only on tradeable-universe rows (§3.4).

### 3.2 Portfolios, full-universe models (2021-01 – 2026-08, 68 months)

| Arm | Portfolio | IR gross | IR net | Sharpe net | CAGR gross / net | α t (net) | β (t), gross | Rolling-12m β min / max (months >1) | One-way turnover, % of gross | Max DD net | Short-book median mcap | Positions |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| modern_nomom | legacy | 1.26 | 1.09 | 1.24 | 40.5% / 34.6% | 3.06 | -0.19 (-0.85) | -1.08 / +1.53 (1) | 43% | -27% | $932M | 201–201 |
| modern_nomom | pm_free | 0.07 | -0.13 | 0.08 | 3.5% / -0.4% | 0.17 | +0.01 (+0.03) | -0.57 / +1.14 (1) | 46% | -34% | $2,363M | 201–206 |
| modern_nomom | pm_t20 | -0.04 | -0.17 | 0.04 | 1.4% / -1.2% | 0.03 | +0.05 (+0.30) | -0.75 / +1.36 (1) | 20% | -38% | $2,365M | 205–242 |
| modern_nomom | pm_t10 | -0.21 | -0.31 | -0.10 | -1.5% / -3.5% | -0.31 | +0.06 (+0.43) | -0.58 / +1.26 (1) | 10% | -41% | $2,591M | 205–287 |
| modern | legacy | 1.33 | 1.17 | 1.31 | 44.3% / 38.2% | 3.15 | -0.12 (-0.53) | -1.01 / +1.55 (1) | 44% | -28% | $885M | 201–201 |
| modern | pm_free | 0.05 | -0.15 | 0.05 | 3.0% / -1.0% | 0.06 | +0.05 (+0.29) | -0.53 / +1.18 (1) | 47% | -34% | $2,271M | 201–205 |
| modern | pm_t20 | -0.05 | -0.17 | 0.02 | 0.8% / -1.7% | -0.02 | +0.07 (+0.39) | -0.86 / +1.19 (1) | 20% | -43% | $2,358M | 205–250 |
| modern | pm_t10 | -0.15 | -0.26 | -0.05 | -0.8% / -2.8% | -0.20 | +0.07 (+0.44) | -0.58 / +1.04 (1) | 10% | -38% | $2,375M | 205–282 |

Costs: tiered trading and borrow assumptions from `docs/PM_ABLATION.md` §1; net = gross − trading cost − borrow cost. `Positions` is the min–max count of holdings per month (limit 100–500). Rolling-12m β: the parenthesis is the number of 12-month windows with β > 1.

### 3.3 Legs, costs and trading, headline arm (`modern_nomom`)

| Portfolio | Long-leg CAGR | Short-leg CAGR | Avg trade cost (bp NAV/mo) | Avg borrow cost (bp NAV/mo) | Traded notional (x capital/mo) | Months cap relaxed |
|---|---:|---:|---:|---:|---:|---:|
| legacy | 13.4% | 18.2% | 22.5 | 13.7 | 1.71 | 0 |
| pm_free | 14.3% | -13.8% | 21.5 | 10.4 | 1.84 | 0 |
| pm_t20 | 12.2% | -13.8% | 10.6 | 10.5 | 0.80 | 0 |
| pm_t10 | 10.5% | -15.0% | 5.9 | 10.2 | 0.40 | 1 |

Leg CAGRs are gross and in excess of the risk-free rate; a *negative* short-leg CAGR means the shorted names rose.

Calendar-year returns, `modern_nomom`:

| Portfolio (net of costs) | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 (8 mo) |
|---|---:|---:|---:|---:|---:|---:|
| legacy | +22.4% | +48.0% | +21.6% | +49.4% | +38.7% | +18.3% |
| pm_t10 | +0.6% | +2.3% | -0.6% | +4.9% | -19.1% | -5.6% |
| T-bill + 4% | +4.1% | +6.2% | +9.5% | +9.3% | +8.4% | +5.2% |
| S&P 500 | +26.9% | -19.4% | +24.2% | +23.3% | +16.4% | +12.3% |

### 3.4 `_trad` arms (trained and validated on the tradeable universe only)

| Arm | Portfolio | IR gross | IR net | Sharpe net | CAGR gross / net | α t (net) | β (t), gross | Rolling-12m β min / max (months >1) | One-way turnover, % of gross | Max DD net | Short-book median mcap | Positions |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| modern_nomom_trad | legacy | 1.15 | 0.93 | 1.12 | 28.7% / 23.1% | 2.66 | -0.05 (-0.30) | -0.89 / +0.97 (0) | 54% | -20% | $1,557M | 201–201 |
| modern_nomom_trad | pm_free | 0.03 | -0.24 | 0.02 | 3.3% / -0.9% | 0.25 | -0.13 (-1.05) | -0.68 / +0.63 (0) | 56% | -40% | $2,586M | 201–205 |
| modern_nomom_trad | pm_t20 | -0.03 | -0.17 | 0.07 | 2.2% / -0.2% | 0.33 | -0.11 (-0.86) | -0.82 / +0.80 (0) | 20% | -49% | $2,701M | 203–243 |
| modern_nomom_trad | pm_t10 | -0.06 | -0.18 | 0.10 | 2.1% / 0.3% | 0.34 | -0.07 (-0.60) | -0.64 / +0.73 (0) | 10% | -43% | $2,960M | 203–270 |
| modern_trad | legacy | 1.17 | 0.97 | 1.15 | 32.0% / 26.2% | 2.78 | -0.11 (-0.58) | -0.97 / +1.09 (1) | 53% | -24% | $1,373M | 201–201 |
| modern_trad | pm_free | -0.03 | -0.30 | -0.04 | 2.3% / -1.9% | 0.11 | -0.14 (-1.07) | -0.74 / +0.73 (0) | 56% | -46% | $2,520M | 202–205 |
| modern_trad | pm_t20 | 0.01 | -0.13 | 0.11 | 2.8% / 0.4% | 0.34 | -0.07 (-0.47) | -0.57 / +0.71 (0) | 20% | -43% | $2,623M | 203–241 |
| modern_trad | pm_t10 | 0.02 | -0.09 | 0.16 | 3.1% / 1.3% | 0.41 | -0.03 (-0.21) | -0.67 / +0.69 (0) | 10% | -41% | $2,823M | 203–270 |

## 4. Findings

1. **Full-universe skill is high, tradeable skill is not.** Test rank IC is 0.14 over all stocks, but the ablation script measures it at about 0.05 inside price ≥ $5 / market cap ≥ $500M, 0.035 once names without beta history are also excluded (the universe the LP can hold), and 0.20 below $250M (`docs/PM_ABLATION.md` §3).
2. **Momentum:** legacy IR 1.33 (`modern`) vs 1.26 (`modern_nomom`); the same direction (small drop when momentum is removed) appears in all five models of this batch (`docs/PM_ABLATION.md` §5), so the data do not support "momentum is hurting" — the case for excluding it is defensibility, not performance. Note the five models share data and features, so this is not five independent confirmations.
3. **Turnover.** The legacy LP trades 1.7× capital per month (43% one-way of gross) and pays about 22 bp/month of NAV in trading cost plus 14 bp of borrow. A 10% cap on its own (ablation row `+turnover_10pct`) costs little IR (1.26 → 1.00 gross) — the cheapest of the constraints. In the full `pm_t10` book the turnover is 10% by construction but the book has no edge, so the cap saves cost without producing return.
4. **Beta consistency.** Formation beta is exactly zero on both betas in every PM month (LP constraint). Realized beta over 68 months is −0.19 (t −0.85) for `legacy` and +0.06 (t +0.43) for `pm_t10`. Rolling-12m β exceeds 1 in exactly **one** window — the first, ending Dec 2021, which contains the Jan-2021 squeeze — and is roughly within ±0.6 afterwards (legacy min −1.08 in the last window). The financial engineer's "no peaks over 1 in the middle" is met after the first window; the first window is 12 noisy observations (s.e. ≈ 0.4).
5. **Positions:** legacy 201; PM books 201–287 (partial fills under the turnover cap add names); all within the 100–500 rule.

## 5. Limitations

- Single seed (42); `docs/RF_2.md` §1 puts the seed noise at sd ≈ 0.055 IR (±0.11 at 2 sd), and that estimate is not refreshed for this configuration.
- Costs and borrow are assumed tiers, not measured; real borrow on the legacy shorts (IOVA, NTLA, AMC, MULN, NKLA …) would be far higher than 200 bp.
- The PM constraint set and 10% cap were fixed a priori (not tuned); a different set could give a different — but, per the ablation, not a qualitatively different — answer.
- `all_nomom` (139 features) was **not run** for this model (about 10× slower); only 12/15-feature arms are reported.
- The `_trad` universe screen uses characteristic-month values of price, market cap and dollar volume (known at the rebalance date); names without 5 years of beta history are excluded from every portfolio, as in all earlier scripts.

## Reproduce

```
.venv/bin/python rf_3.py                                              # modern_nomom, modern  -> rf3_results.json / rf3_summary.csv
.venv/bin/python rf_3.py --arms modern_nomom_trad,modern_trad --suffix _trad   # tradeable-universe training -> rf3_results_trad.json
```
Outputs are in `output/` (`oos_predictions_rf3_<arm>.csv`, `portfolio_holdings_<portfolio>_rf3_<arm>.csv`, `portfolio_returns_<portfolio>_rf3_<arm>.csv`, `rf3_feature_importance_<arm>.csv`).
