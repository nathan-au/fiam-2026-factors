# McGill-FIAM Asset Management Hackathon 2026 — First Challenge (Condensed)

Source: *McGill-FIAM Asset Management Hackathon First Challenge 2026.pdf* (Russ Goyenko, McGill; Chengyu Zhang, SJTU). This is a de-fluffed restatement of all actionable requirements.

---

## Contents

- [1. The Task](#1-the-task)
- [2. Hard Constraints (Trading Criteria)](#2-hard-constraints-trading-criteria)
  - [Definitions of neutrality](#definitions-of-neutrality--state-explicitly-which-you-enforce)
- [3. Benchmark and Evaluation Metrics](#3-benchmark-and-evaluation-metrics)
- [4. Data](#4-data)
  - [`chars_final_with_names.parquet` — main numerical panel](#chars_final_with_namesparquet--main-numerical-panel)
  - [Target-variable alignment (critical)](#target-variable-alignment-critical)
  - [`8k_...identified.parquet` — text data](#8k_20150101_20260831_identifiedparquet--text-data-provided-by-quantillium)
  - [Filing-timing and information cutoff](#filing-timing-and-information-cutoff)
  - [Provided code templates (legacy)](#provided-code-templates-legacy--must-be-adapted)
  - [External data (optional)](#external-data-optional)
- [5. Sample Periods and Training Schedule](#5-sample-periods-and-training-schedule)
- [6. Portfolio Construction Baseline](#6-portfolio-construction-baseline)
- [7. Alternative Prediction Targets](#7-alternative-prediction-targets)
- [8. Using LLMs and Text](#8-using-llms-and-text)
- [9. Agentic AI (optional, this year's frontier)](#9-agentic-ai-optional-this-years-frontier)
- [10. Look-Ahead Bias — Zero Tolerance](#10-look-ahead-bias--zero-tolerance)
- [11. Daily Risk Measurement (encouraged, not required)](#11-daily-risk-measurement-encouraged-not-required)
- [12. Deck Requirements (PowerPoint → PDF)](#12-deck-requirements-powerpoint--pdf)
  - [Naming convention (enforced throughout the deck)](#naming-convention-enforced-throughout-the-deck)
- [13. Final Submission Package](#13-final-submission-package)
- [14. Evaluation Criteria](#14-evaluation-criteria)
- [15. Compute Resources](#15-compute-resources)
- [16. Looking Ahead: Carbon Arc (Finals Only)](#16-looking-ahead-carbon-arc-finals-only)
- [References](#references)

---

## 1. The Task

Design and backtest a **market-neutral U.S. equity strategy** using ML, LLMs, agentic workflows, or any combination. You must:

1. Identify stock-level factors predictive of future returns.
2. Construct a market-neutral U.S. equity portfolio (long and short legs balanced).
3. Backtest rigorously.
4. Demonstrate predictive power **strictly out of sample**.
5. Document the workflow — including any agent plan, tools, and prompts — so a third party can reproduce it.

Primary prediction target: **next month's stock return**. Alternatives are allowed (see §7).

**Format:** Stage One (this challenge) → top 10 teams advance to Stage Two finals (2-day mentorship, expanded text data, Carbon Arc alternative data).

---

## 2. Hard Constraints (Trading Criteria)

| Constraint | Limit |
|---|---|
| Positions | **100–500 names**, long + short counted together |
| Gross exposure | **≤ 200%** of capital (max 2x leverage) |
| Net exposure | **−50% to +50%** of capital, **at every rebalance date** |
| Rebalance | Monthly recommended; at minimum, rebalance *some* holdings every 6 months |
| Reporting | Average & max gross exposure, average & extreme net exposure, realised S&P 500 beta over OOS |

**Worked example (valid):** $110 long across 150 names, $90 short across 130 names → gross $200 (200%), net +$20 (+20%), 280 positions.

**Notes:**
- 60 long + 60 short = 120 positions ✓. 40 + 40 = 80 ✗ (below floor).
- Dollar neutrality ($100 vs $100) is the natural centre of the band.
- The ±50% band is for expressing a genuine modest directional view — not for running a long book with a hedge bolted on. **A strategy sitting at the band edge month after month, or with beta materially different from zero over the OOS period, will be evaluated as a directional strategy regardless of how the deck describes it.** Organizers estimate realised beta themselves.

### Definitions of neutrality — state explicitly which you enforce
- **Dollar neutral:** long $ = short $ at each rebalance. Simple, easily verified.
- **Beta neutral:** beta-weighted long exposure = beta-weighted short exposure, portfolio beta ≈ 0. (A dollar-neutral book long high-beta / short low-beta is a levered long in disguise.)
- **Sector / factor neutral** (optional refinement): neutralize within sectors or against size, value, momentum.

**Shorting is not free.** Borrow costs and recall risk are real; a short book concentrated in small, illiquid, hard-to-borrow names may be untradeable. Explicit borrow-cost modeling is not required, but the short book will be read with this in mind.

---

## 3. Benchmark and Evaluation Metrics

**Benchmark = 3-month U.S. T-bill rate + 4% per annum** — a *time series*, computed month by month, not a constant. (Aug 2026 bill ≈ 3.7% → hurdle ≈ 7.7% annualized; in 2021 the hurdle was ≈ 4%.)

- Download **TB3MS** from FRED: https://fred.stlouisfed.org/series/TB3MS
- FRED reports TB3MS as an **annualized percentage**. Monthly decimal benchmark = `TB3MS / 100 / 12 + 0.04 / 12`. Supplied stock returns are in decimal units.
- ⚠ The RF used to build `ret_exc` / `ret_exc_lead1m` is **not necessarily** TB3MS. Do not subtract the risk-free rate twice or assume the series cancel.

**Active return and IR (the headline number):**
```
active_t = R_portfolio,total,t − (TB3MS_t / 100 / 12 + 0.04 / 12)
Information Ratio = sqrt(12) × mean(active_t) / stdev(active_t)
```
Apply the IR formula to **active returns** — *not* to a regression intercept divided by residual volatility (the template's printed "Information Ratio" is the latter and is wrong for this competition).

A weighted long–short stock-return spread and a total return on portfolio capital are different quantities. **State your signed position weights, capital denominator, and cash/collateral convention.** If portfolio total return = the benchmark cash return + a portfolio excess return, active return simplifies to that excess return − 0.04/12; otherwise retain the difference between the cash-rate series.

**Sharpe ratio:** computed from returns in excess of the stated risk-free rate — it must not give credit for merely earning the risk-free rate.

**Alpha / beta regression** (also required; the S&P 500 is shown for context only, not as a target to beat):
```
R_p,t − r_f,t = α + β (R_S&P500,t − r_f,t) + ε_t
```
- Annualize monthly α by ×12. Report **standard errors on both α and β**.
- Beta over the full OOS period should be ≈ 0 — this is your evidence of neutrality. **A beta far from zero is read as a failure of the mandate.**
- Alpha should account for nearly all of the return, since no market exposure remains to explain it.
- Get monthly S&P 500 returns and risk-free rates from FRED.

**Out-of-sample R² (must be reported in the deck):**
```
R²_oos = 1 − Σ(r_{i,t+1} − r̂_{i,t+1})² / Σ(r_{i,t+1})²
```
Note the denominator is **not** demeaned — the benchmark is **zero** (no predictability), not the historical mean. Any positive value captures some predictability. Typical values are 1–2% even for neural nets. Code appears near the end of `penalized_linear_hackathon.py`.

---

## 4. Data

### `chars_final_with_names.parquet` — main numerical panel
- Monthly U.S. stock panel, **01/2015 – 08/2026**, **529,082 stock-month observations**, **6,561 distinct PERMNOs**.
- **198 columns** = **147 characteristics** (selected by the `variable` column in `factor_char_list.csv`) + **51 identifiers/auxiliary columns** (identifiers, dates, returns, prices, market cap, industry classification, label provenance). The 51 support merging/evaluation/implementation — they are *not* a second characteristic list.
- Characteristics span fundamentals, past returns, liquidity, trading costs, and risk. Appendix A of the PDF describes them with research references; `readme.md` documents files and joining conventions.
- **Values and missingness are preserved from the source panel — preprocessing is your responsibility and must be documented.**

### Target-variable alignment (critical)
- For a row labelled `eom` (end-of-month) in month *t*: characteristics belong to month *t*, and **`ret_exc_lead1m` is the excess return in month *t+1***.
- **Use `ret_exc_lead1m` as the target without shifting it again.**
- `ret` and `ret_exc` describe month *t*, **not** the next month.
- **Do not include `ret_exc_lead1m` among the predictors.**
- For legacy templates: map `ret_exc_lead1m` → `stock_exret`, and build the template's `date`/`year`/`month` from the **target month t+1**; preserve the original `eom` and source date separately.
- Because characteristics start Jan 2015, the **first available target month is February 2015** — do not invent Dec 2014 predictors.

### `8k_20150101_20260831_identified.parquet` — text data (provided by Quantillium)
- Cleaned, linked U.S. **8-K current reports**, 01/2015–08/2026: **373,139 filing observations**, **3,687 distinct PERMNOs** (all present in the characteristics panel).
- Single Parquet file (not yearly files) — read selected columns or filter by `filing_date` to work with smaller subsets.
- Each filing carries `permno`, `gvkey`, `iid`, `cusip`, `cik`, `ticker`, `company_name`. **Join to the characteristics panel on `permno` + a dated month key, cross-checking `gvkey` and `iid`.** No `cik_gvkey_linktable.csv` needed. See `readme.md` for the many-to-one join example and identifier-quality flags.
- Some verified security links have no characteristics observation in the exact filing month — retain or exclude these **deliberately** and document it.
- Text is not available for every stock/month. **Keep stock-months without a filing** unless your strategy explicitly restricts its universe (document any restriction).

**Useful 8-K item codes:**
| Item | Content |
|---|---|
| 2.02 | Results of operations / financial condition (earnings announcements — release may be an exhibit, not in the supplied main text) |
| 1.01 / 1.02 | Material agreements entered or terminated: contracts, credit facilities, partnerships, supply deals |
| 2.01 / 2.05 / 2.06 | Transactions and write-downs: completed M&A and disposals, restructuring charges, impairments |
| 5.02 | Officer and director changes (CEO/CFO/board turnover, including abrupt) |
| 4.01 / 4.02 | Auditor changes and non-reliance — **two of the strongest distress signals in the corpus** |
| 7.01 / 8.01 | Reg FD and other events: guidance updates, buybacks, litigation, product announcements |

A single filing may cover several items and may refer to excluded exhibits. **Treat extracted items and dates as annotations to verify, not guarantees.** 8-Ks generally must be filed within four business days of the reportable event, subject to item-specific exceptions.

### Filing-timing and information cutoff
- ⚠ A disclosure dated the 3rd **cannot** be used to form a portfolio on the 1st.
- In this release: `filing_date` is the **provider date**; `filing_timestamp_utc` is a **date/midnight placeholder**; `sec_accepted_at_utc` is **null**; `content_report_date` describes the **reported event**, not a release date.
- **State a conservative availability convention.** Obtain independent acceptance-time evidence before making any intraday or same-day execution claim.
- At monthly frequency: accumulate news within month *t* and extract sentiment to forecast month *t+1*.

### Provided code templates (legacy — must be adapted)
- `penalized_linear_hackathon.py` — linear models (LASSO, Elastic Net, Ridge) and hyperparameter search. Replace `sample_data.csv` with a Parquet/model table, map `stock_exret`, and **replace its 2000–2024 / eight-year training settings with the 2015-based schedule below**. Its dates and line-number references are *not* the 2026 spec.
- `portfolio_analysis_hackathon.py` — decile ranking and performance stats. Its `stock_exret` must refer to the **target holding month**; its printed Information Ratio is an alpha-to-residual-volatility ratio (**not** this competition's IR); its drawdown and turnover examples must be reconciled with your stated definitions. **It does not enforce the 100–500 position or exposure constraints.**
- Fit learned preprocessing on admissible **training** information only. **Do not let target availability define the prediction universe.**

### External data (optional)
Allowed (e.g., WRDS — van Binsbergen, Han & Lopez-Lira (2023) use 67 WRDS financial ratios to predict EPS). If you merge external data, include:
- Data-cleaning code, and a clear description of the source.
- A **security identifier + a date**: preferably `permno` + month, or the (`gvkey`, `iid`) pair + month. CIK is an issuer key and CUSIP can change — verify effective periods; **never join on an undated company name or ticker.**

---

## 5. Sample Periods and Training Schedule

- **Full panel:** 01/2015 – 08/2026.
- **Out-of-sample evaluation:** **01/2021 – 08/2026** (covers post-COVID rebound, 2022 drawdown, 2023–2025 AI concentration).

**Expanding training window + rolling 2-year validation. Assign all splits by the *target return month*.**

| Forecast year | Train | Validate | Test |
|---|---|---|---|
| 1st | 2015–2018 (targets available) | 01/2019 – 12/2020 | 01/2021 – 12/2021 |
| 2nd | through 12/2019 | 01/2020 – 12/2021 | 01/2022 – 12/2022 |
| … | expand annually | roll annually | continue through 08/2026 |

- **Refit annually, forecast monthly.** A January 2021 forecast uses **December 2020** characteristics.
- **Keep any label whose realization falls in the test period out of training and validation**, even when its characteristic month precedes the test boundary.
- Tune hyperparameters on validation, estimate coefficients on training, reserve test purely for OOS evaluation.

---

## 6. Portfolio Construction Baseline

Rank stocks by predicted return at the start of month *t*, split into deciles, long the top decile and short the bottom decile, equal weight. Alternatively long the top 100 / short the bottom 100. Trades execute at the beginning of the month, results evaluated at month-end.

The decile spread is dollar-neutral by construction and zero-cost (shorts finance longs) — **but its beta still needs to be checked**, and it does not by itself satisfy the position-count and exposure constraints.

Within the mandate you are free to choose: position sizing (equal weight, conviction weight, volatility scaling), which neutrality definition to enforce, rebalance frequency, and whether to vary gross exposure over time. **If you vary gross or net exposure, explain what drives the switch and how you predict it in advance** — and gross must stay ≤200% and net inside ±50% throughout.

---

## 7. Alternative Prediction Targets

Direct return prediction is hard. You may instead forecast **fundamentals** that drive returns (earnings/revenue surprises often drive multi-month price drift). Accounting ratios in the panel — e.g. `ebit_sale`, `niq_at`, `niq_be` — support this.

If you do: choose a future fundamental outcome, explain why it matters for returns, and document its construction and availability. **Predictions must use only information available at the forecast date.** For a quarterly fundamental target, explicitly identify the future reporting period and release date — *a shift of three monthly rows in a carried-forward accounting panel does not identify the next fiscal-quarter report.*

---

## 8. Using LLMs and Text

Basic workflow: encode 8-K bodies with a pretrained model (BERT, FinBERT) or have a general-purpose model summarize each into a structured record → use embeddings or extracted fields as features in an ML model (XGBoost, LightGBM) → predict next-month returns, next-quarter EPS, or the sign/size of post-event drift.

Reference: *Can AI Read the Minds of Corporate Executives?* (SSRN).

⚠ Text modeling escalates in complexity quickly — **start simple, then scale.**

---

## 9. Agentic AI (optional, this year's frontier)

An LLM encoder produces a *number*; an agent produces a *process* — one that can be inspected, replayed, criticised and improved. Tools available to an agent: the characteristics dataframe, the filing archive, a Python interpreter, a backtester.

**Use cases the organizers want to see:**
- **Analyst agent** — given a ticker, that month's 8-Ks, and recent characteristic history, return a structured verdict (what happened, was it expected, signed conviction score). Run cross-sectionally, treat output as a factor.
- **Event-triage agent** — distinguish a routine dividend declaration from an auditor resignation; yields a cleaner event set to trade.
- **Signal-discovery agent** — propose, code and backtest combinations of the 147 characteristics. Judged on how the top-ranked idea holds up in a period it never saw. **Multiple-testing discipline is your responsibility, not the model's.**
- **Devil's-advocate agent** — a second agent whose only job is to attack the first's thesis (disconfirming filings, crowding, factor exposures that explain the alpha away).
- **Portfolio-construction agent** — turn forecasts into positions subject to the constraints (100–500 names, 200% gross, ±50% net, turnover budget, sector limits) and write down why each trade was made. **Explainability is a deliverable.**
- **Reproducibility agent** — re-run your pipeline hunting for look-ahead bias, survivorship, and leakage before you submit.

**Two warnings:**
1. **Agents are fluent whether or not they are right.** A well-written thesis is not evidence; the backtest is. Every agent-generated claim in the deck must be traceable back to the provided data.
2. **Agents crowd.** If your edge is "ask a frontier model what it likes," assume hundreds of other teams have the same edge. The value is in *what you feed the agent, how you constrain it, and how you verify it.*

**Agents are not required.** A well-executed classical ML strategy beats a poorly conceived agentic one. If you build one, show the prompts and tool definitions and be honest about failures — *a candid account of an agent that did not work is worth more than a polished account of one that supposedly did.*

---

## 10. Look-Ahead Bias — Zero Tolerance

- You cannot use month *t+1* events to make month *t* decisions. **Python code is submitted and will be checked for forward-looking information.**
- **Model-side look-ahead:** a frontier model trained through 2026 already *knows* what happened to a stock in 2021 — asking it for a view on that period is recall, not forecasting. **Constrain the agent to documents and characteristics available at time *t*, and state explicitly in the deck how you enforced this. Teams that cannot explain how they ruled out model-side look-ahead will be treated as having used it.**

---

## 11. Daily Risk Measurement (encouraged, not required)

Selection is monthly and the required backtest is monthly. But once the month's weights are set, you can hold them fixed and mark the book daily.

Why it matters: a month that drops 6% mid-month and finishes flat shows *zero* volatility and *zero* drawdown in monthly data, but a 6% drawdown and far higher vol/VaR in daily data. Both are arithmetically correct; only one reflects what holding it felt like, and only one tells you whether a prime broker would have issued a margin call (margin is computed on the daily path).

**Sources for daily prices:** WRDS/CRSP daily stock files (cleanest, if your university subscribes), or the Alpha Vantage API (free key covers `TIME_SERIES_DAILY`; `TIME_SERIES_DAILY_ADJUSTED` is paid) — check rate/history limits and cache your pulls.

**Weighting:** monthly and daily analysis are weighted **equally**. A careful monthly-only submission is not penalised, and daily analysis will not rescue a weak strategy. If you report daily-based figures, **label clearly which is which**.

---

## 12. Deck Requirements (PowerPoint → PDF)

**8 pages + Appendix of at most 10 additional pages.** Label every table and chart with the period it covers and state whether returns are gross or net of trading costs.

**Page 1 — Executive Summary.** Strategy, ML algorithm(s) and/or agentic workflow, and performance vs the market-neutral benchmark (T-bill + 4%), with the S&P 500 for context. Use market and benchmark data covering the complete 01/2021–08/2026 window.

**Page 2 — Strategy.** How the long and short legs are constructed; **which definition of neutrality you enforce**; predictive signals used. Top 10 long and top 10 short holdings on average over 01/2021–08/2026. Cumulative performance chart vs benchmark and vs S&P 500.

**Page 3 — Data and Methodology.** Justify the ML/LLM choice. Training structure. Any new architecture or training approach. Any supplementary external data (optional) and whether it proved valuable. **Present OOS R² for the overall sample.** If any part of the pipeline is agentic: architecture, what the agent could see, which tools it could call, step limit, model used, and how model-side look-ahead was prevented — **prompts and tool definitions go in the Appendix.**

**Pages 4–7 — Performance pack** for 01/2021–08/2026, strategy vs benchmark vs S&P 500. Treat it as what you'd hand an investment committee. Put the headline stats table and 3–4 charts here; move remaining tables/exhibits to the Appendix.

*Return statistics:*
- Average monthly return; annualized return, both arithmetic and geometric (CAGR)
- Cumulative return over the full OOS period
- Calendar-year returns, year by year, for strategy / benchmark / S&P 500
- Best and worst month, with dates
- **Hit rate**: share of months with positive *active* return
- **Long-leg and short-leg returns reported separately**

*Risk-adjusted performance:*
- **Information ratio, annualized — the headline number**
- Sharpe ratio, annualized
- Annualized alpha vs S&P 500, **with t-statistic**
- **Beta vs S&P 500, with standard error — evidence of neutrality**
- Correlation of monthly returns with the S&P 500

*Exposure, concentration, implementation:*
- Average number of holdings, split long vs short
- Average gross and average net exposure, each with its range
- Average and maximum single-position weight; share of book in the top 10 names
- Average monthly turnover with range
- **Short-book characteristics:** average market cap, average dollar volume, and share of shorts in small-cap or plausibly hard-to-borrow names

*Charts (minimum):*
- Cumulative return: strategy vs benchmark vs S&P 500
- Underwater (drawdown) chart with S&P 500 overlaid; rolling 12-month active return and rolling 12-month IR
- **Rolling 12-month beta to the S&P 500** — the clearest evidence of whether you stayed neutral throughout or only on average
- Histogram of monthly returns with the benchmark's monthly hurdle marked
- 10 largest positive and 10 largest negative P&L contributors, by ticker and company name

**Page 8 — Discussion.** Did it perform as trained / meet expectations? Main fundamental signals driving performance. Most profitable positions and why. Macroeconomic events that contributed. Potential improvements.

### Naming convention (enforced throughout the deck)
Identify **every** holding, chart label, and table row by **ticker and full company name** — e.g. "NVDA, NVIDIA Corporation". Merge using `permno` and dated `gvkey`–`iid` relationships. Period-supported ticker/name labels are included, but some characteristics observations are unlabeled: **verify historical display labels from an authoritative source and record that source — do not substitute a later company name without verification.**

---

## 13. Final Submission Package

1. **Deck + Appendix as one PDF** (built in PowerPoint, converted to PDF).
2. **CSV of monthly stock holdings** for every month of 01/2021–08/2026. Columns: `Date` (beginning of month, e.g. Sept 01 2026), `PERMNO`, `TICKER`, `COMPANY NAME`, `WEIGHT` (% of portfolio NAV, **positive for long, negative for short**).
   **Plus a separate CSV** with the time series of total portfolio monthly returns (or daily, if available) for the test period.
3. **Python code in one file named `MAIN.py`** clearly describing your process. **Do NOT submit the data** — the only data submitted is the holdings CSV in #2.
4. **Licence:** submission is public domain / free (e.g., Apache). No new technology is being created; the idea plus backtest performance is what is evaluated.
5. **CVs for each team member** — part of the submission and the final team registration; used for recruiting by industry participants, so keep them current.

---

## 14. Evaluation Criteria

Judged by a committee of industry experts and finance academics on:
1. **The investment idea** — which financial factors you focus on and predict. Originality is rewarded; you may predict returns, earnings surprises, P/E, P/S, or anything else that drives future returns.
2. **The technology** — feed-forward nets, gradient boosting, transformer text models, agentic pipelines are all fair game. **What is rewarded is not the sophistication of the tool but the quality of reasoning around it: why this technology, for this signal, on this data.**
3. **Performance vs the T-bill + 4% benchmark** — but **neutrality is checked first, before anything else is looked at.** Beating the S&P 500 is not the objective, and a high beta is not read as a strength.

Overall emphasis is on **originality, choice of technology, and execution of the backtest** — not the raw return number. Purely replicating the provided templates is **not sufficient**: use them to establish a linear baseline, then build on it.

Market frictions (especially turnover costs) will erode alpha — that is expected and normal. What matters is implementing an idea, discussing it, and articulating what you learned and what could be improved.

---

## 15. Compute Resources

- **LightningAI** — free account, ~35h free GPU per account per month.
- Google Cloud trial / AWS student free tier for deep learning workloads.

---

## 16. Looking Ahead: Carbon Arc (Finals Only)

The 147 characteristics derive from prices and accounting statements — the same raw material every quant manager has had since the 1990s. The frontier is **alternative data**: credit-card/receipt panels, web traffic, app downloads, job postings, healthcare claims, point-of-sale and supply-chain records, shipping manifests, foot traffic, weather. High-frequency, *observed* rather than reported, and arriving before the fundamentals they predict.

Finalists are expected to get access to Carbon Arc's Insights Exchange (https://www.carbonarc.co/). **Extra prizes** are under consideration (not guaranteed) for the most creative use of their data, judged separately from the main competition on originality — a team can win one without placing overall.

**You cannot use this data in Stage One and should not design around it.** Two habits that pay off if you advance:
1. Build the pipeline so a new block of features merges on a **firm-month key** without rewriting everything downstream — most teams lose their first finals day to a join.
2. Decide in advance which of your signals is weakest and where knowing what a company is *really* doing would help most. Arriving with a specific question beats arriving hoping to browse.

---

## References

- Chopados, Fan, Goyenko, Laradji, Liu & Zhang (2023), *Can AI Read the Minds of Corporate Executives?*, McGill WP
- Goyenko & Zhang (2022), *The Joint Cross Section of Options and Stock Returns Predictability with Big Data and Machine Learning*, McGill WP
- Gu, Kelly & Xiu (2020), *Empirical Asset Pricing via Machine Learning*, RFS 33(5):2223–2273
- Jensen, Kelly & Pedersen (2022), *Is There a Replication Crisis in Finance?*, Journal of Finance
- Hoerl & Kennard (1970), *Ridge Regression*, Technometrics 12(1):55–67
- Tibshirani (1996), *Regression Shrinkage and Selection via the Lasso*, JRSS-B 58(1):267–288
- Zou & Hastie (2005), *Regularization and Variable Selection via the Elastic Net*, JRSS-B 67(2):301–320
- van Binsbergen, Han & Lopez-Lira (2023), *Man versus Machine Learning*, RFS 36(6):2361–2396
- WSJ (Sept 2026), *The AI Shift Turning Everyday Investors Into Mini Quant Funds*
- FRED, TB3MS: https://fred.stlouisfed.org/series/TB3MS

*Appendix A of the source PDF (pages 25–27) is an image-only table of the 147 characteristics with their research references — consult the PDF directly for it.*
