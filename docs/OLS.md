# Baseline OLS Strategy — Methodology and Results

Implementation: `ols.py` (single file, run with `.venv/bin/python ols.py`).
Downloaded benchmark data is cached in `cache/` (`TB3MS.csv`, `SP500.csv`) —
`cache/` holds only external downloads. Everything the script produces
(`oos_predictions.csv`, `portfolio_holdings_*.csv`, `portfolio_returns_*.csv`,
`ols_results.json`) is written to `output/`.

This is the first-pass linear baseline the competition brief asks for before
building anything more sophisticated (§14: "Purely replicating the provided
templates is not sufficient: use them to establish a linear baseline, then
build on it.").

---

## 1. What it does

Plain (unregularized) OLS regresses next-month excess stock return on all 147
characteristics, refit once a year on an expanding window, and used to rank
stocks into a monthly long/short portfolio. No LASSO/Ridge/ElasticNet penalty,
no feature selection, no dimensionality reduction — the simplest possible
model, to establish a floor.

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

This matches `fiam/penalized_linear_hackathon.py`'s convention and makes every
factor's value mean "how extreme is this stock relative to its peers this
month," not an absolute unit — robust to outliers, comparable across factors
of very different scales (ratios vs. growth rates vs. volatilities).

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
the validation window isn't actually used for anything in this script — it's
computed and logged for schedule parity with the LASSO/Ridge/ElasticNet
models that *do* need it (so all models can be compared on an identical
walk-forward design later). Coefficients are estimated on the training fold
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
Each month, rank stocks by predicted return; go long the top 100 and short
the bottom 100, equal-weighted within each leg (`1/100` and `−1/100`). This
is dollar-neutral and self-financing by construction: 100% long, 100% short,
200 positions, gross exposure exactly 200%, net exposure exactly 0% every
month — comfortably inside the 100–500 position / ≤200% gross / ±50% net
constraints.

### 2.6 Benchmark and risk-adjusted metrics
- **Benchmark**: `TB3MS` (FRED) / 100 / 12 + 0.04/12, a genuine monthly time
  series, downloaded to `cache/TB3MS.csv`.
- **S&P 500**: FRED's daily `SP500` series, resampled to month-end close,
  downloaded to `cache/SP500.csv`.
- **Return convention (stated explicitly, per `docs/FIAM.md`'s requirement
  not to silently assume RF series cancel):** the long/short spread return is
  treated as already representing a return earned in excess of a risk-free
  rate, and that risk-free rate is assumed close enough to TB3MS to use
  directly — i.e. `active_return = spread_return − 0.04/12`. This is a stated
  approximation, not a verified identity: the pipeline's actual embedded RF
  (used to build `ret_exc`/`ret_exc_lead1m`) was not disclosed and may differ
  from TB3MS by a small basis. Sharpe ratio is computed on the spread return
  directly (already excess-of-risk-free under this convention); alpha/beta
  use `TB3MS/100/12` as `r_f` in the CAPM-style regression against the S&P
  500.

---

## 3. Results

### 3.1 Predictive power
**OOS R² = −0.0086%** (pooled across all 268,733 OOS predictions).

This is indistinguishable from zero: direct next-month return prediction from
these characteristics, with no regularization or dimensionality reduction,
carries essentially no information here. **This model's forecasts are not
meaningfully better than predicting zero return for every stock every
month.**

### 3.2 A critical artifact: unscreened portfolio returns are fake alpha

Building the top-100/bottom-100 portfolio directly from raw predictions
(no liquidity screen) produced:

| Metric | Value |
|---|---:|
| Annualized return (CAGR) | 82.6% |
| Information Ratio | 1.69 |
| Sharpe ratio | 1.79 |
| Alpha (annualized, t-stat) | 71.2% (t=4.12) |

This looks spectacular and is **not real**. Checking the holdings against
raw market cap/price/volume:

| | Median market cap | Median price | Median $ volume/day |
|---|---:|---:|---:|
| Long leg | $251M | $9.41 | $1.29M |
| Short leg | **$44M** | **$2.47** | $4.78M |
| Full universe | $676M | — | — |

An OOS R² of essentially zero cannot honestly produce an 83% CAGR; a model
with no real forecasting power is instead exploiting noisy, occasionally-huge
returns in illiquid microcap/penny stocks that a real fund could not actually
execute a short book in at this size. **This result must not be reported as
the strategy's performance.**

(Note: an earlier draft of this document attributed this diagnosis to a
specific warning in `docs/FACTORS.md`, including a quoted "78% annual return /
$18M market cap" anecdote and an IC/t-statistic table. That citation was
fabricated — no such content exists in `docs/FACTORS.md`, which is a plain
factor glossary with no backtest results in it. The diagnosis itself was
reached independently, by directly inspecting this run's holdings against raw
market cap/price/volume, as shown below — that part is real and reproducible
from `output/portfolio_holdings_unfiltered.csv`.)

### 3.3 Investability-screened results (the honest number)

An investability screen — applied only at portfolio-formation time, using
values known as of the characteristic month, never during model training —
requires every eligible stock to have a 6-month average daily dollar volume
(`dolvol_126d`) ≥ **$10M**. This is a single threshold, chosen as the most
direct available measure of whether a position is actually tradable (rather
than layering price and market-cap cutoffs on top of it).

Checking this screen's own holdings confirms it removes the microcap
artifact from §3.2:

| | Median market cap | Median price | Median $ volume/day |
|---|---:|---:|---:|
| Long leg | $3,557M | $37.62 | $33.8M |
| Short leg | $1,114M | $11.66 | $32.6M |

Both legs are now solidly liquid, investment-grade names — a real change
from the unfiltered version's $44M-median-market-cap short leg.

| Metric | Unfiltered | Investability-screened ($10M dolvol) |
|---|---:|---:|
| Avg monthly return | 5.97% | **3.07%** |
| Annualized return (CAGR) | 82.6% | **35.3%** |
| Cumulative return (68 months) | 2,934% | **355%** |
| Hit rate (active return > 0) | 76.5% | 63.2% |
| Information Ratio | 1.69 | **0.98** |
| Sharpe ratio | 1.79 | 1.10 |
| Alpha, annualized (t-stat) | 71.2% (t=4.12) | 40.7% (t=2.85) |
| Beta vs S&P 500 (t-stat) | 0.045 (t=0.14) | **−0.37 (t=−1.36)** |
| Correlation with S&P 500 | 0.02 | −0.17 |
| Best month | +22.0% (Dec 2025) | +21.2% (Jul 2026) |
| Worst month | −57.1% (Jan 2021) | −42.9% (Jan 2021) |

Calendar-year returns, investability-screened portfolio:

| 2021 | 2022 | 2023 | 2024 | 2025 | 2026 (through Aug) |
|---:|---:|---:|---:|---:|---:|
| +6.7% | +98.7% | +21.3% | +12.4% | +46.8% | +30.8% |

Every calendar year in the OOS window is positive, with 2022 as a standout —
unlike the concentrated "one good year, flat after" shape sometimes seen in
this kind of baseline. A result this strong from a single liquidity
threshold and no other change is worth treating with real caution rather
than taken at face value (see §4).

### 3.4 Neutrality check — this baseline still doesn't clear it

`docs/FIAM.md` §14 is explicit that **neutrality is checked before anything
else**. The screened portfolio's beta is **−0.37 (t = −1.36)** — not
statistically significant at conventional levels, but the point estimate is
still sizeable and negative, in the wrong direction for a strategy the deck
would want to call "neutral." This baseline, as implemented, would not clear
the neutrality bar with confidence. (The unfiltered version's beta looks
near zero, but that's not meaningful either — its return doesn't reflect an
executable strategy in the first place.)

---

## 4. Known limitations / next steps

1. **Not beta-neutral.** The top/bottom-N selection has no beta constraint;
   the resulting −0.37 beta (t=−1.36) needs to be fixed before this is a
   legitimate submission candidate — e.g. neutralize against `beta_60m`
   explicitly, or optimize position weights subject to a beta constraint
   instead of raw equal weighting.
2. **No feature deduplication.** All 147 factors are fed to OLS as-is. Some
   are constructed similarly (e.g. several liquidity measures, or `_gr1`/
   `_gr1a`/`_gr3` variants of the same growth concept) and plain OLS handles
   correlated inputs badly, assigning large offsetting weights to near-twins.
   Checking the actual pairwise correlations and dropping or combining
   redundant columns (or using PCA/ridge instead of plain OLS) is a natural
   next step — not yet done here.
3. **Short-book quality untested beyond the basic screen.** `docs/FIAM.md`
   asks for average market cap, dollar volume, and share of hard-to-borrow
   names in the short book specifically — not computed here yet.
4. **No turnover or trading-cost accounting.** Monthly full-rebalance
   turnover on a 200-name book is likely substantial; not measured.
5. **The $10M dollar-volume threshold is a simple heuristic**, not tuned or
   justified beyond "removes the microcap artifact and produces plausible
   long/short leg market caps." Whether it's the right cutoff, or whether a
   proper investable universe definition (e.g. NYSE size-percentile
   breakpoints, as in the academic factor literature) would materially change
   the result, has not been checked.
6. **RF convention is stated but unverified** (§2.6) — if precision matters
   later, the pipeline's actual embedded risk-free series should be recovered
   or bounded rather than assumed equal to TB3MS.

This baseline's job is to be a floor to build on (§14 of `docs/FIAM.md`), not
a submission candidate — LASSO/Ridge/ElasticNet, the 8-K text signal, and
explicit neutrality constraints are the logical next layers.
