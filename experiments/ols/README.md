# Baseline OLS Strategy — Methodology and Results

Implementation: `ols.py` (single file, run with `.venv/bin/python experiments/ols/ols.py`).
Downloaded benchmark data is cached in `cache/` (`TB3MS.csv`, `SP500.csv`) —
`cache/` holds only external downloads. Everything the script produces
(`oos_predictions.csv`, `portfolio_holdings_beta_neutral.csv`,
`portfolio_returns_beta_neutral.csv`, `ols_results.json`) is written to
`output/`.

This is the first-pass linear baseline the competition brief asks for before
building anything more sophisticated (§14: "Purely replicating the provided
templates is not sufficient: use them to establish a linear baseline, then
build on it.").

---

## 1. What it does

Plain (unregularized) OLS regresses next-month excess stock return on all 147
characteristics, refit once a year on an expanding window. Its predictions
rank stocks into a monthly long/short portfolio. No LASSO/Ridge/ElasticNet
penalty, no feature selection, no dimensionality reduction — the simplest
possible model, to establish a floor.

The portfolio is **investability-screened and beta-neutral**: a liquid
universe only, with position weights solved each month by a small linear
program that holds net exposure and beta exposure at zero when the portfolio
is formed (§2.5). This is the only portfolio the script builds and reports.

## 2. Pipeline

### 2.1 Data and target alignment
- Source: `fiam/chars_final_with_names.parquet` (529,082 stock-months), using
  the 147 columns listed in `fiam/factor_char_list.csv` as predictors.
- Target: `ret_exc_lead1m` (already the *next*-month excess return for a row
  labeled `eom`). Rows with a missing target (terminal observations, gaps) are
  dropped, leaving 523,125 rows.
- Every row is tagged with its **target month** (`eom` + 1 month) — this, not
  the characteristic month, is what the train/validate/test split is keyed on,
  per `docs/FIAM.md` §5.

### 2.2 Cross-sectional preprocessing
For each characteristic, within each characteristic month (`eom`):
1. Fill missing values with that month's cross-sectional median (a handful of
   months where a factor is entirely missing fall back to 0, i.e. a neutral
   rank).
2. Dense-rank all stocks and rescale ranks to `[-1, 1]`.

This matches `fiam/penalized_linear_hackathon.py`'s convention. It makes every
factor's value mean "how extreme is this stock relative to its peers this
month," not an absolute unit. That is robust to outliers and keeps factors of
very different scales comparable (ratios vs. growth rates vs. volatilities).

### 2.3 Walk-forward schedule
Expanding training window, rolling 2-year validation window, one calendar
year of test, refit annually — exactly the schedule in `docs/FIAM.md` §5:

| Test year | Train (target months) | Validate | Test |
|---|---|---|---|
| 2021 | 2015-02 – 2018-12 | 2019 – 2020 | 2021 |
| 2022 | 2015-02 – 2019-12 | 2020 – 2021 | 2022 |
| 2023 | 2015-02 – 2020-12 | 2021 – 2022 | 2023 |
| 2024 | 2015-02 – 2021-12 | 2022 – 2023 | 2024 |
| 2025 | 2015-02 – 2022-12 | 2023 – 2024 | 2025 |
| 2026 | 2015-02 – 2023-12 | 2024 – 2025 | 2026-01 – 2026-08 |

**Note on the validation fold:** plain OLS has no hyperparameters to tune, so
the validation window isn't actually used for anything in this script. It is
computed and logged so the schedule matches the LASSO/Ridge/ElasticNet models
that *do* need it, letting all models be compared on an identical
walk-forward design later. Coefficients are estimated on the training fold
only, per `docs/FIAM.md` §5's explicit split of duties ("tune on validation,
estimate on training, test purely for OOS evaluation").

Six models are fit in total — one `LinearRegression` per test year — never one
model fit on the whole sample.

### 2.4 Out-of-sample evaluation
```
R²_oos = 1 − Σ(actual − predicted)² / Σ(actual)²
```
computed across all 268,733 OOS stock-month predictions (2021-01 through
2026-08 target months), pooled.

### 2.5 Portfolio construction

**Investability screen.** Each month, only stocks with a 6-month average
daily dollar volume (`dolvol_126d`) of at least **$10M** are eligible. The
screen uses values known as of the characteristic month and is applied only
when the portfolio is formed, never during model training.

**Beta-neutral LP.** Over the screened universe, weights `w_i` are chosen by
solving
```
maximize   Σ w_i · pred_i
subject to Σ w_i = 0                  (dollar/net neutral)
           Σ w_i · beta_60m_i = 0     (beta neutral, per docs/FIAM.md §2's
                                        definition: beta-weighted long
                                        exposure = beta-weighted short
                                        exposure)
           Σ |w_i| = 2.0              (gross = 200%)
           |w_i| ≤ 1%                 (per-name cap, forces diversification)
```
The weights are split as `w_i = w_i^+ − w_i^-` (both ≥ 0) so every constraint
above is linear. The LP is solved each month with `scipy.optimize.linprog`
(HiGHS). Stocks with a missing `beta_60m` (~11% of the liquid universe, mostly
recently listed names without 5 years of return history) are excluded, since
they can't enter the beta constraint. The 1% per-name cap forces the solution
to spread across roughly 200 names to reach 200% gross.

**Why both constraints exist.** Earlier versions of this script (through
commit `addbca6`) also built simpler equal-weight top-100/bottom-100
portfolios, and those showed why each constraint is needed:
- *Without the liquidity screen*, the book shorted illiquid microcaps
  (short-leg median market cap $44M, median price $2.47). That produced an
  apparent 83% CAGR from a model with near-zero OOS R² — returns that could
  not actually be traded. This was found by inspecting that run's holdings
  against raw market cap, price and volume.
- *With the screen but equal weights*, realized beta was −0.37 (t = −1.36),
  because the short leg happened to carry higher beta than the long leg.
  Equal weighting has no way to correct for that, and `docs/FIAM.md` §14
  checks neutrality before anything else.

### 2.6 Benchmark and risk-adjusted metrics
- **Benchmark**: `TB3MS` (FRED) / 100 / 12 + 0.04/12, a genuine monthly time
  series, downloaded to `cache/TB3MS.csv`.
- **S&P 500**: FRED's daily `SP500` series, resampled to month-end close,
  downloaded to `cache/SP500.csv`.
- **Return convention (stated explicitly, per `docs/FIAM.md`'s requirement
  not to silently assume RF series cancel):** the long/short spread return is
  treated as already being a return in excess of a risk-free rate, and that
  rate is assumed close enough to TB3MS to use directly — i.e.
  `active_return = spread_return − 0.04/12`. This is a stated approximation,
  not a verified identity: the pipeline's actual embedded RF (used to build
  `ret_exc`/`ret_exc_lead1m`) was not disclosed and may differ from TB3MS by a
  small basis. Sharpe ratio is computed on the spread return directly
  (already excess-of-risk-free under this convention). Alpha/beta use
  `TB3MS/100/12` as `r_f` in a CAPM-style regression against the S&P 500.
- **Drawdown**: the simple (not log) decline of compounded wealth from its
  running peak, `W_t / max(W_0..W_t) − 1`, with starting capital `W_0 = 1`
  counted as the initial peak. This differs from
  `fiam/portfolio_analysis_hackathon.py`, which takes the `cummax` of
  cumulative *log* returns without the starting point. That version reports
  log rather than simple drawdown and would miss a loss in the very first
  month — which matters here, because the worst month is the first OOS month
  (Jan 2021). Drawdowns are measured on monthly returns only, so intra-month
  drawdowns are not captured (see `docs/FIAM.md`'s note on daily marking).

---

## 3. Results

All figures are for 2021-01 through 2026-08 (68 months) and are gross of
trading costs. Source: `experiments/ols/output/ols_results.json`.

### 3.1 Predictive power
**OOS R² = −0.0086%** (pooled across all 268,733 OOS predictions).

This is indistinguishable from zero: direct next-month return prediction from
these characteristics, with no regularization or dimensionality reduction,
carries essentially no information here. **This model's forecasts are not
meaningfully better than predicting zero return for every stock every
month.** The portfolio results below should be read with that in mind (see
§4).

### 3.2 Returns

| Metric | Value |
|---|---:|
| Avg monthly return | 2.48% |
| Annualized return, arithmetic | 29.8% |
| Annualized return, geometric (CAGR) | 27.6% |
| Cumulative return (68 months) | 298% |
| Hit rate (active return > 0) | 61.8% |
| Best month | +19.2% (Jan 2022) |
| Worst month | −42.4% (Jan 2021) |
| Long leg CAGR | 11.3% |
| Short leg CAGR | 10.2% |

Calendar-year returns vs. the benchmark (TB3MS+4%) and the S&P 500:

| | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 (through Aug) |
|---|---:|---:|---:|---:|---:|---:|
| Strategy | +10.6% | +57.0% | +9.8% | +18.0% | +40.8% | +25.7% |
| Benchmark (T-bill+4%) | +4.1% | +6.2% | +9.5% | +9.3% | +8.4% | +5.2% |
| S&P 500 | +26.9% | −19.4% | +24.2% | +23.3% | +16.4% | +12.3% |

Every calendar year is positive and beats the benchmark, though 2023 only
barely (+9.8% vs. +9.5%). The long and short legs contribute roughly equally
(11.3% vs. 10.2% CAGR), which is what removing the beta tilt should produce.

### 3.3 Risk-adjusted performance and neutrality

| Metric | Value |
|---|---:|
| **Information Ratio** | **0.85** |
| Sharpe ratio | 0.98 |
| Alpha, annualized (t-stat) | 31.5% (t=2.41) |
| **Beta at formation** (mean, max\|·\|, every month) | **~0 (≤3.4×10⁻¹⁵)** |
| **Realized beta vs S&P 500** (SE, t-stat) | **−0.17 (SE 0.25, t=−0.67)** |
| Correlation with S&P 500 | −0.08 |

Two different "beta" numbers are reported and they answer different
questions:
- **Beta at formation** is exact by construction — the LP forces
  `Σ w_i · beta_60m_i = 0` every month (verified: mean 5.6×10⁻¹⁷, max
  absolute value 3.4×10⁻¹⁵ across all 68 months, i.e. zero to floating-point
  precision).
- **Realized beta** comes from regressing the portfolio's actual monthly
  returns on the S&P 500 over the whole OOS window — this is what
  `docs/FIAM.md` §14 actually checks. At −0.17 (t=−0.67) it is not
  statistically distinguishable from zero, but it is not exactly zero either.

The gap exists because `beta_60m` is a **trailing 5-year estimate**, not the
stock's true beta over the next month. Constraining against a lagging, noisy
proxy reduces realized beta risk but can't cancel it exactly. This is
expected, not a bug: holding formation-time exposure at zero is the only
thing that can be enforced at trade time.

### 3.4 Drawdowns

| | Strategy | S&P 500 |
|---|---:|---:|
| Max drawdown | −42.4% | −24.8% |
| Peak → trough | Dec 2020 → Jan 2021 (1 mo) | Dec 2021 → Sep 2022 (9 mo) |
| Recovered | Dec 2021 (11 mo after trough) | Dec 2023 (15 mo) |
| Longest underwater spell | 11 months | 23 months |
| Avg drawdown while underwater | −10.3% | −8.6% |
| Drawdown at end of window (Aug 2026) | −4.2% | 0.0% |
| Calmar ratio (CAGR / \|max DD\|) | 0.65 | 0.54 |

The worst loss is essentially **one month**: −42.4% in January 2021, the month
of the meme-stock short squeeze, when heavily shorted names spiked. A −42%
single-month drawdown on a dollar-neutral book is severe, and it would likely
have triggered margin calls well before month-end. Neither beta neutrality
nor the $10M liquidity screen protects against it, because it was a
short-crowding event, not a market-beta event. The strategy spends far less
time underwater than the S&P 500 (11 vs. 23 months), but its worst drawdown
is much deeper and happens much faster. The per-month `drawdown` and
`sp500_drawdown` columns in `experiments/ols/output/portfolio_returns_beta_neutral.csv` are
the data for the required underwater chart.

### 3.5 Exposure, concentration, and implementation

`docs/FIAM.md`'s deck-reporting list also asks for exposure, turnover,
position concentration and short-book characteristics. These are computed
directly from the holdings and return series.

| Metric | Value |
|---|---:|
| Avg positions / month (long / short) | 201 (100.3 / 100.7) |
| Gross / net exposure | 200% / 0% (exact, to float precision) |
| Avg monthly turnover (one-way, % of gross) | 39.7% |
| Turnover range | 19.9% – 57.1% |
| Avg / max position weight | 1.0% / 1.0% |
| Avg share of book in top 10 names | 5.0% |
| Short book avg / median market cap | $15,949M / $1,764M |
| Short book avg daily dollar volume | $151M |
| Short book share with market cap < $2,000M | 52.6% |
| Short book share with market cap < $1,000M | 38.3% |

Notes on what these numbers actually show:
- **Turnover is high (~40% one-way per month).** The LP is re-solved
  independently each month with no penalty for trading away from last
  month's book. This would materially affect any net-of-cost figures (not
  computed here — see §4).
- **Position weights and top-10 concentration are trivial by construction**
  (1% cap, ~200 names ⇒ ~5% in the top 10). This reflects the deliberately
  flat weighting scheme, not something discovered from the data.
- **No hard-to-borrow data exists in this dataset.** Small market cap is used
  only as the stated proxy `docs/FIAM.md` itself suggests ("small-cap or
  plausibly hard-to-borrow"); it is not a real borrow-cost or
  short-availability signal. Over a third of shorts are still below $1B
  market cap despite the liquidity screen.

---

## 4. Known limitations / next steps

1. **The returns are hard to reconcile with the model's forecasting power.**
   An OOS R² of essentially zero (§3.1) sits alongside a 27.6% CAGR and an
   alpha t-stat of 2.41. Ranking can carry signal that pooled R² doesn't
   show, but a result this strong from a model this weak deserves real
   caution rather than being taken at face value.
2. **Beta neutrality is enforced at formation but not perfectly realized.**
   Realized beta is −0.17 (t=−0.67), not exactly zero, because `beta_60m` is
   a lagging proxy for each stock's actual forward beta. Constraining against
   a shorter-window beta, or re-estimating beta more often, could close more
   of that gap.
3. **The construction has its own untuned choices.** The 1% per-name cap and
   the $10M dolvol screen were picked to give a reasonable scale and position
   count (~200 names), not optimized. Stocks missing `beta_60m` (~11% of the
   liquid universe, mostly younger listings) are excluded entirely, a further
   universe restriction beyond the dolvol screen; its 5-year lookback also
   means very recently listed companies can never be included even once
   they're liquid.
4. **Short-crowding risk is unmanaged.** The January 2021 drawdown (§3.4) came
   from a short squeeze that neither constraint addresses. Short-interest or
   borrow data, or a cap on exposure to heavily shorted names, would be
   needed to control it.
5. **No feature deduplication.** All 147 factors are fed to OLS as-is. Some
   are constructed similarly (e.g. several liquidity measures, or `_gr1`/
   `_gr1a`/`_gr3` variants of the same growth concept), and plain OLS handles
   correlated inputs badly, assigning large offsetting weights to near-twins.
   Dropping or combining redundant columns (or using PCA/ridge instead of
   plain OLS) is a natural next step — not yet done here.
6. **Turnover is measured but not costed.** ~40% one-way per month (§3.5) is
   real and not trivial, but no transaction-cost model converts it into a
   net-of-cost return. All returns in this document are gross of trading
   costs.
7. **RF convention is stated but unverified** (§2.6). If precision matters
   later, the pipeline's actual embedded risk-free series should be recovered
   or bounded rather than assumed equal to TB3MS.

This baseline's job is to be a floor to build on (§14 of `docs/FIAM.md`), not
a submission candidate — LASSO/Ridge/ElasticNet and the 8-K text signal are
the logical next layers.
