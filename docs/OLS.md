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

Three portfolio variants are built from the same predictions, in increasing
order of realism:
1. **Unfiltered** (§3.2) — no liquidity screen. Kept only to demonstrate a
   failure mode; its return is not real.
2. **Investability-screened** (§3.3) — liquid universe, equal-weight
   top-100/bottom-100. Tradable, but not beta-neutral.
3. **Beta-neutral** (§3.5) — same liquid universe, but position weights are
   solved by a small linear program that constrains beta exposure to zero at
   formation, not just approximated by equal weighting. This is the
   recommended result of the three.

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

**Equal-weight top/bottom-N (§3.2, §3.3):** each month, rank stocks by
predicted return; go long the top 100 and short the bottom 100,
equal-weighted within each leg (`1/100` and `−1/100`). This is dollar-neutral
and self-financing by construction: 100% long, 100% short, 200 positions,
gross exposure exactly 200%, net exposure exactly 0% every month —
comfortably inside the 100–500 position / ≤200% gross / ±50% net
constraints. It has no beta control, however — see §3.4.

**Beta-neutral, LP-optimized (§3.5):** each month, instead of a fixed
top/bottom-100 selection, weights `w_i` over the investability-screened
universe are chosen by solving
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
implemented as a linear program (`w_i = w_i^+ − w_i^-`, both ≥ 0, so every
constraint above is linear) and solved per month with
`scipy.optimize.linprog` (HiGHS). Stocks with a missing `beta_60m` (~11% of
the liquid universe, mostly recently-listed names without 5 years of return
history) are excluded from this construction's eligible universe, since they
can't enter the beta constraint. The 1% per-name cap forces the solution to
spread across roughly 200 names to reach 200% gross, keeping the position
count comparable to the equal-weight version rather than concentrating in a
handful of names.

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

Calendar-year returns, investability-screened portfolio, vs. the benchmark
(TB3MS+4%) and the S&P 500 for the same years:

| | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 (through Aug) |
|---|---:|---:|---:|---:|---:|---:|
| Strategy | +6.7% | +98.7% | +21.3% | +12.4% | +46.8% | +30.8% |
| Benchmark (T-bill+4%) | +4.1% | +6.2% | +9.5% | +9.3% | +8.4% | +5.2% |
| S&P 500 | +26.9% | −19.4% | +24.2% | +23.3% | +16.4% | +12.3% |

Every calendar year in the OOS window is positive, with 2022 as a standout —
unlike the concentrated "one good year, flat after" shape sometimes seen in
this kind of baseline. A result this strong from a single liquidity
threshold and no other change is worth treating with real caution rather
than taken at face value (see §4).

### 3.4 Neutrality check on the equal-weight construction — fails it

`docs/FIAM.md` §14 is explicit that **neutrality is checked before anything
else**. The equal-weight screened portfolio's beta is **−0.37 (t = −1.36)**
— not statistically significant at conventional levels, but the point
estimate is still sizeable and negative, in the wrong direction for a
strategy the deck would want to call "neutral." Equal-weighting the top/
bottom-100 has no mechanism to prevent this: if the short leg happens to
have a higher average beta than the long leg (as it does here), the
portfolio inherits a negative net beta purely as a byproduct of *which*
stocks got picked, with nothing to correct for it.

### 3.5 Beta-neutral construction — the fix

Replacing equal weighting with the LP construction from §2.5, on the same
$10M-dolvol-screened universe:

| Metric | Equal-weight (§3.3) | Beta-neutral (LP) |
|---|---:|---:|
| Avg monthly return | 3.07% | 2.48% |
| Annualized return (CAGR) | 35.3% | 27.6% |
| Cumulative return (68 months) | 355% | 298% |
| Hit rate (active return > 0) | 63.2% | 61.8% |
| Information Ratio | 0.98 | 0.85 |
| Sharpe ratio | 1.10 | 0.98 |
| Alpha, annualized (t-stat) | 40.7% (t=2.85) | 31.5% (t=2.41) |
| **Beta at formation** (mean, max\|·\|, every month) | not controlled | **~0 (≤3.4×10⁻¹⁵)** |
| **Realized beta vs S&P 500** (t-stat) | −0.37 (t=−1.36) | **−0.17 (t=−0.67)** |
| Correlation with S&P 500 | −0.17 | −0.08 |
| Avg positions / month | 200 | 201 |
| Gross / net exposure | 200% / 0% (exact) | 200% / 0% (exact, to float precision) |

Two different "beta" numbers are reported and they answer different
questions:
- **Beta at formation** is exact by construction — the LP forces
  `Σ w_i · beta_60m_i = 0` every single month (verified: mean 5.6×10⁻¹⁷,
  max absolute value 3.4×10⁻¹⁵ across all 68 months, i.e. zero to floating-
  point precision).
- **Realized beta** comes from regressing the portfolio's actual monthly
  returns against the S&P 500 over the whole OOS window — this is what
  `docs/FIAM.md` §14 actually checks. It fell from −0.37 to −0.17 and is no
  longer statistically distinguishable from zero (t=−0.67 vs. t=−1.36
  before), a real improvement, but it is not itself exactly zero.

The gap between the two exists because `beta_60m` is a **trailing 5-year
estimate**, not the stock's true beta over next month — constraining against
a lagging, noisy proxy reduces realized beta risk but can't cancel it
exactly. This is expected, not a bug: forcing formation-time exposure to
zero is the correct and only actionable thing to do at trade time.

The cost of the fix is a lower return (CAGR 35.3% → 27.6%, IR 0.98 → 0.85):
some of the equal-weight version's return was compensation for carrying
negative beta (a short-biased-in-high-beta-names tilt), which is exactly the
kind of return `docs/FIAM.md` says shouldn't be credited as strategy skill.

### 3.6 Remaining required-reporting fields

`docs/FIAM.md`'s deck-reporting list (lines 269–286) also asks for the long
and short legs' returns separately, turnover, position concentration, and
short-book characteristics. None of these change the model or the
portfolio construction — they're computed directly from the existing
holdings and return series.

| | Equal-weight (§3.3) | Beta-neutral (§3.5) |
|---|---:|---:|
| Long leg CAGR | 9.0% | 11.3% |
| Short leg CAGR | 19.1% | 10.2% |
| Avg monthly turnover (one-way, % of gross) | 39.9% | 39.7% |
| Turnover range | 23.5% – 60.0% | 19.9% – 57.1% |
| Avg position weight | 1.0% | 1.0% |
| Max position weight | 1.0% | 1.0% |
| Avg share of book in top 10 names | 5.0% | 5.0% |
| Short book avg / median market cap | $9,895M / $1,114M | $15,949M / $1,764M |
| Short book avg daily dollar volume | $120M | $151M |
| Short book share with market cap < $1,000M | 48.1% | 38.3% |

Notes on what these numbers actually show:
- **The equal-weight portfolio's short leg (19.1% CAGR) drove much more of
  its return than the long leg (9.0%)** — consistent with §3.4's finding
  that the short leg carried higher average beta: shorting higher-beta names
  paid off especially in 2022's selloff. The beta-neutral version's legs are
  far more balanced (11.3% vs. 10.2%), which is what removing that beta tilt
  should do.
- **Turnover is high either way (~40% one-way per month)** and is not
  reduced by the beta-neutral construction — re-solving the LP independently
  each month has no penalty for trading away from last month's book. This
  was not previously measured or discussed, and would materially affect any
  net-of-cost figures (not computed here — see §4).
- **Position weights and top-10 concentration are trivial by construction**
  (~1% cap, ~200 names ⇒ ~5% in the top 10) for both variants — this reflects
  the deliberately flat weighting scheme, not something discovered from the
  data.
- **No hard-to-borrow data exists in this dataset.** Small market cap is
  used here only as the stated proxy `docs/FIAM.md` itself suggests
  ("small-cap or plausibly hard-to-borrow"); it is not a real borrow-cost or
  short-availability signal.

---

## 4. Known limitations / next steps

1. **Beta neutrality is now enforced at formation but not perfectly realized.**
   The LP construction (§3.5) zeroes out exposure to `beta_60m` exactly every
   month, and realized beta fell from −0.37 (t=−1.36) to −0.17 (t=−0.67) — no
   longer statistically significant, but not exactly zero either, because
   `beta_60m` is a lagging proxy for the stock's actual forward beta.
   Constraining against a rolling shorter-window beta, or re-estimating beta
   more frequently than annually, could close more of that gap.
2. **The LP construction has its own untuned choices.** The 1% per-name cap
   and the $10M dolvol screen were picked to reproduce a similar scale/
   position-count to the earlier equal-weight version, not optimized.
   Stocks missing `beta_60m` (~11% of the liquid universe, mostly younger
   listings) are excluded from this construction entirely, which is a
   further, undocumented-elsewhere universe restriction beyond the dolvol
   screen. `beta_60m`'s 5-year lookback also means very recently listed
   companies can never be included here even once they're otherwise liquid.
3. **No feature deduplication.** All 147 factors are fed to OLS as-is. Some
   are constructed similarly (e.g. several liquidity measures, or `_gr1`/
   `_gr1a`/`_gr3` variants of the same growth concept) and plain OLS handles
   correlated inputs badly, assigning large offsetting weights to near-twins.
   Checking the actual pairwise correlations and dropping or combining
   redundant columns (or using PCA/ridge instead of plain OLS) is a natural
   next step — not yet done here.
4. **Turnover is measured but not costed.** Both variants turn over ~40% of
   the book one-way per month (§3.6) — real, and not trivial, but no
   transaction-cost model converts that into a net-of-cost return. All
   returns in this document are gross of trading costs.
5. **RF convention is stated but unverified** (§2.6) — if precision matters
   later, the pipeline's actual embedded risk-free series should be recovered
   or bounded rather than assumed equal to TB3MS.

This baseline's job is to be a floor to build on (§14 of `docs/FIAM.md`), not
a submission candidate — LASSO/Ridge/ElasticNet, the 8-K text signal, and
explicit neutrality constraints are the logical next layers.
