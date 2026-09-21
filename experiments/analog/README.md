# Analog Forecasting (Lookalike Stocks) — Methodology and Results

Implementation: `analog.py` (run with `.venv/bin/python analog.py`, about 15 minutes). New file; nothing existing is edited. Data, target (`ret_exc_lead1m`), rank transform, walk-forward schedule and the portfolio layer are the same as `docs/RF_3.md` (ported, not imported).

## The idea

Instead of fitting a function from characteristics to returns, **remember** every historical stock-month whose next-month return is already known, and forecast a stock by what happened next to its **nearest lookalikes** — the K historical stock-months closest to it in characteristic space, from any stock and any earlier month. It is memory-based ("case-based") forecasting: no coefficients, no trees, and every forecast can be explained by pointing at the lookalikes it used.

## How it is built

| Element | Choice |
|---|---|
| Space | The cross-sectional ranks in [−1, 1] used throughout this project; Euclidean distance; equal weight per factor. Arms: `modern_nomom` (12 factors, headline), `modern` (15, with momentum — reference). No 139-factor arm: nearest neighbours in 139 dimensions are meaningless. |
| Memory bank | Stock-months whose **target month is before the fold's validation window** (the `tr` set of every other script). Validation and test rows are never in the bank. |
| Forecast | (Weighted) mean next-month return of the K lookalikes, each measured **relative to its own month's cross-sectional median across all stocks** (so market-wide moves in a neighbour's month don't leak in), winsorized at the bank's 1st/99th percentiles. Predictions are therefore relative returns; OOS R² is still scored on raw returns. |
| Hyperparameters | K ∈ {25, 100, 400, 1600} × {uniform, inverse-distance} weights, chosen per fold on **validation mean monthly rank IC**. |
| `_inv` arm | `modern_nomom_inv`: the bank contains only LP-investable stock-months (price ≥ $5, market cap ≥ $500M, $10M dollar volume, both betas observed), so a tradeable stock's lookalikes are tradeable. Validation queries are also investable only. Test rows are still scored for every stock. |
| Portfolios | `legacy`, `pm_free`, `pm_t20`, `pm_t10` exactly as `docs/RF_3.md`. |

## Bottom line

1. **As a forecaster it performs like the tree models.** Rank IC 0.139 on all stocks, OOS R² +0.142%, legacy IR 1.04 gross / 0.85 net (`modern`: 1.31 / 1.11) — inside the range of the five tree models (IC 0.143–0.155, legacy IR 0.90–1.33).
2. **It hits the same wall.** Rank IC on the universe the portfolio can hold is 0.032 (trees: 0.034–0.037), and the tradeable books are negative: `pm_t10` gross IR -0.48, `pm_free` -0.32 (net -0.58 / -0.54).
3. **The analog premise — that *close* lookalikes are informative — is not supported.** Validation IC rises monotonically with K (§2): K = 25 has the lowest mean validation IC of the four K values (and is far below the chosen setting in every fold), and the **largest** K in the grid (1600) was chosen in all six folds of both full-memory arms. What predicts is broad averaging over a loose neighbourhood, i.e. this behaves like a smooth kernel regression, not case-based reasoning. (K = 1600 sits on the grid edge; larger K was not tried.)
4. **Searching for lookalikes among tradeable stocks only removes most of the skill.** `modern_nomom_inv`: LP-investable IC 0.016 (t 2.0), OOS R² -0.148%, and validation IC ≤ 0 in two of six folds. Its `pm_t10` gross IR is -0.11 (net -0.25) and `pm_free` -0.00 — the least negative of the analog books, but on a weaker signal and not distinguishable from zero.
5. **It is not a genuinely different signal.** Mean monthly rank correlation between the analog and the tree predictions (computed from the saved prediction files, not by `analog.py`): 0.77 with RF, 0.86 with ET, 0.71 with XGB_2 on all stocks; 0.63, 0.72 and 0.48 on the investable universe.

## Portfolio layer

Same as `docs/RF_3.md` (see its "Portfolio layer" section): `legacy` is the project's original LP (it reproduces `rf_2.py`'s IR to four decimals); `pm_free` / `pm_t20` / `pm_t10` add price ≥ $5, market cap ≥ $500M, dual-beta neutrality, 5% net / 35% gross sector limits and a hard turnover budget (none / 20% / 10% one-way of gross), all reported gross and net of assumed market-cap-tiered costs. The constraint set and the 10% headline were fixed from the tips before any result was seen.

## 1. Model-level results

| Arm | Factors | Mean val rank IC | Test rank IC, all stocks (% months > 0) | Test rank IC, LP-investable (t) | OOS R² | Pred. rank autocorr |
|---|---:|---:|---:|---:|---:|---:|
| `modern_nomom` | 12 | 0.1159 | 0.1395 (82%) | 0.0319 (2.6) | +0.142% | 0.74 |
| `modern` | 15 | 0.1148 | 0.1392 (84%) | 0.0296 (2.4) | +0.140% | 0.72 |
| `modern_nomom_inv` | 12 | 0.0100 | 0.0739 (81%) | 0.0158 (2.0) | -0.148% | 0.57 |

`Test rank IC, LP-investable` is measured on the same universe as the tradeable portfolios (screens plus both betas observed); the trees' figure for that universe is 0.034–0.037 (`docs/PM_ABLATION.md` §3). Diagnostics from `analog_results.json` (`modern_nomom`): the 100 nearest lookalikes are on average 0.76 away for investable queries, against ≈ 2.8 between two random stock-months in 12 rank dimensions (my arithmetic, assuming uniform ranks); 58% of an investable stock's 100 nearest lookalikes are themselves investable (investable stock-months are about 36% of the bank), against 19% for non-investable stocks.

## 2. Selection: bigger neighbourhoods are always better

Mean validation rank IC across the six folds, by K and weighting (`modern_nomom`, full memory):

| K (lookalikes averaged) | Uniform weights | Inverse-distance weights |
|---|---:|---:|
| 25 | 0.0469 | 0.0455 |
| 100 | 0.0749 | 0.0737 |
| 400 | 0.1007 | 0.0997 |
| 1600 | 0.1154 | 0.1157 |

Per fold (`modern_nomom`):

| Test year | Chosen (validation rank IC) | Val IC | Val IC of K=25 (best weighting) |
|---|---|---:|---:|
| 2021 | K=1600, uniform | 0.0501 | 0.0196 |
| 2022 | K=1600, uniform | 0.0911 | 0.0360 |
| 2023 | K=1600, inverse-distance | 0.1386 | 0.0420 |
| 2024 | K=1600, inverse-distance | 0.1482 | 0.0603 |
| 2025 | K=1600, uniform | 0.1375 | 0.0649 |
| 2026 | K=1600, uniform | 0.1298 | 0.0591 |

For the `_inv` arm (memory = tradeable stocks only) the same table is much flatter and around zero:

| K (lookalikes averaged) | Uniform weights | Inverse-distance weights |
|---|---:|---:|
| 25 | -0.0001 | -0.0013 |
| 100 | 0.0020 | 0.0008 |
| 400 | 0.0043 | 0.0039 |
| 1600 | 0.0065 | 0.0063 |

| Test year | Chosen (validation rank IC) | Val IC | Val IC of K=25 (best weighting) |
|---|---|---:|---:|
| 2021 | K=25, uniform | -0.0091 | -0.0091 |
| 2022 | K=1600, inverse-distance | 0.0182 | 0.0056 |
| 2023 | K=25, inverse-distance | -0.0066 | -0.0066 |
| 2024 | K=1600, inverse-distance | 0.0258 | 0.0111 |
| 2025 | K=1600, uniform | 0.0225 | 0.0066 |
| 2026 | K=1600, uniform | 0.0091 | -0.0064 |

## 3. Portfolios (2021-01 – 2026-08, 68 months)

| Arm | Portfolio | IR gross | IR net | Sharpe net | CAGR gross / net | α t (net) | β (t), gross | Rolling-12m β min / max (months >1) | One-way turnover, % of gross | Max DD net | Short-book median mcap | Positions |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| modern_nomom | legacy | 1.04 | 0.85 | 1.01 | 30.6% / 24.7% | 2.48 | -0.13 (-0.64) | -0.93 / +1.15 (1) | 49% | -23% | $1,091M | 201–201 |
| modern_nomom | pm_free | -0.32 | -0.54 | -0.33 | -3.9% / -7.9% | -0.75 | -0.02 (-0.11) | -0.80 / +1.02 (1) | 52% | -50% | $2,328M | 202–205 |
| modern_nomom | pm_t20 | -0.24 | -0.37 | -0.16 | -2.4% / -4.8% | -0.37 | -0.00 (-0.02) | -0.87 / +1.25 (1) | 20% | -36% | $2,336M | 205–234 |
| modern_nomom | pm_t10 | -0.48 | -0.58 | -0.36 | -6.5% / -8.3% | -0.76 | -0.06 (-0.42) | -0.74 / +1.17 (1) | 10% | -49% | $2,609M | 205–270 |
| modern_nomom_inv | legacy | 1.14 | 0.81 | 1.09 | 21.3% / 15.7% | 2.69 | -0.09 (-0.78) | -0.79 / +0.41 (0) | 60% | -14% | $2,112M | 201–201 |
| modern_nomom_inv | pm_free | -0.00 | -0.36 | -0.04 | 3.2% / -1.3% | 0.17 | -0.13 (-1.30) | -0.63 / +0.21 (0) | 62% | -32% | $3,169M | 201–204 |
| modern_nomom_inv | pm_t20 | -0.00 | -0.17 | 0.13 | 3.2% / 0.9% | 0.58 | -0.14 (-1.32) | -0.79 / +0.13 (0) | 20% | -38% | $3,284M | 202–240 |
| modern_nomom_inv | pm_t10 | -0.11 | -0.25 | 0.08 | 1.9% / 0.2% | 0.37 | -0.09 (-0.93) | -0.70 / +0.26 (0) | 10% | -42% | $3,455M | 202–278 |
| modern | legacy | 1.31 | 1.11 | 1.27 | 38.9% / 32.4% | 3.00 | -0.05 (-0.27) | -0.82 / +1.20 (1) | 53% | -21% | $1,022M | 201–201 |
| modern | pm_free | -0.20 | -0.45 | -0.22 | -1.1% / -5.4% | -0.58 | +0.05 (+0.36) | -0.46 / +1.20 (1) | 56% | -40% | $2,257M | 201–205 |
| modern | pm_t20 | -0.26 | -0.40 | -0.18 | -2.4% / -4.9% | -0.49 | +0.04 (+0.31) | -0.77 / +1.13 (1) | 20% | -36% | $2,259M | 205–238 |
| modern | pm_t10 | -0.47 | -0.58 | -0.35 | -5.6% / -7.4% | -0.91 | +0.08 (+0.53) | -0.54 / +1.09 (1) | 10% | -43% | $2,619M | 205–283 |

Costs: tiered assumptions from `docs/PM_ABLATION.md` §1; net = gross − trading cost − borrow cost. Rolling-12m β: the parenthesis is the number of 12-month windows with β > 1 (at most one, the first window ending Dec 2021, for every full-memory book; none for the `_inv` books).

### Legs and costs, `modern_nomom`

| Portfolio | Long-leg CAGR | Short-leg CAGR | Avg trade cost (bp NAV/mo) | Avg borrow cost (bp NAV/mo) | Traded notional (x capital/mo) | Months cap relaxed |
|---|---:|---:|---:|---:|---:|---:|
| legacy | 11.9% | 10.7% | 25.4 | 13.3 | 1.98 | 0 |
| pm_free | 11.6% | -18.1% | 24.4 | 10.5 | 2.09 | 0 |
| pm_t20 | 11.0% | -16.3% | 10.5 | 10.5 | 0.80 | 0 |
| pm_t10 | 8.0% | -17.1% | 5.6 | 10.0 | 0.40 | 1 |

## 4. Comparison with the tree models (headline arm)

| Model (`modern_nomom`) | Test IC, all stocks | Test IC, LP-investable | OOS R² | Legacy IR gross / net | `pm_t10` IR gross / net |
|---|---:|---:|---:|---:|---:|
| rf3 | 0.1434 | 0.0356 | +0.225% | 1.26 / 1.09 | -0.21 / -0.31 |
| et | 0.1516 | 0.0346 | +0.154% | 0.96 / 0.81 | -0.28 / -0.37 |
| lgbm | 0.1545 | 0.0374 | +0.109% | 0.92 / 0.80 | -0.12 / -0.22 |
| cat | 0.1442 | 0.0341 | +0.176% | 0.92 / 0.78 | -0.31 / -0.41 |
| xgb2 | 0.1480 | 0.0343 | +0.140% | 0.90 / 0.76 | -0.17 / -0.27 |
| **analog** | **0.1395** | **0.0319** | **+0.142%** | **1.04 / 0.85** | **-0.48 / -0.58** |

Tree rows are copied from the model docs (`docs/RF_3.md`, `ET.md`, `LGBM.md`, `CAT.md`, `XGB_2.md`) and the LP-investable IC column from `docs/PM_ABLATION.md` §3.

## 5. Findings

- **Nothing new is learned about tradeable alpha.** A model with no functional form — pure memory — reaches the same all-stock skill (0.14) and the same tradeable skill (0.03) as trees, and the same negative tradeable books. That is a useful robustness result: the ceiling is set by the information in these 12–15 factors in this universe, not by the model class.
- **The "analog" story does not survive the K sweep.** K = 25 has mean validation IC 0.046 vs 0.116 at K = 1600; the method works by averaging over a large neighbourhood. Presenting it as "we found the stock's twins" would overstate what the data show.
- **Restricting memory to tradeable stocks is the honest version, and it is weaker.** The all-stock skill (IC 0.14) is carried by lookalikes drawn from the whole universe, most of whose stock-months are small or low-priced. When the memory is the tradeable universe alone, LP-investable IC falls to about 0.016.
- **Explainability is the one genuine advantage.** Every forecast can be justified by listing its nearest lookalikes and what happened to them next. This was not built or shown here.

## 6. Limitations

- K = 1600 (the top of the grid) was selected in all six folds of both full-memory arms; the optimum may be larger, which would only reinforce finding 3.
- Equal-weight Euclidean distance on 12–15 rank-transformed factors; no learned or IC-weighted metric, no sector or size matching, no PCA-reduced space. A metric tuned on validation could behave differently.
- The memory bank is the training window only (never updated with validation rows for test predictions), consistent with every other script; a longer memory for the later folds was not tried.
- Single run (the method is deterministic), no confidence intervals: 68 test months; an IC of 0.03 has t ≈ 2.4–2.6 and the differences among models are far inside noise.
- Same assumed cost tiers and PM caveats as `docs/RF_3.md` §5. The `_inv` arm and the K = 1600 extension were added after a first run with K ≤ 400 showed grid-edge selection; that run's numbers were discarded and not used.

## Reproduce

```
.venv/bin/python analog.py                                  # modern_nomom, modern_nomom_inv, modern -> analog_results.json / analog_summary.csv
.venv/bin/python analog.py --arms modern_nomom --suffix _x  # subset
```
Outputs (`output/`): `oos_predictions_analog_<arm>.csv`, `portfolio_holdings_<portfolio>_analog_<arm>.csv`, `portfolio_returns_<portfolio>_analog_<arm>.csv`, `analog_results.json`, `analog_summary.csv`.
