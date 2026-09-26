# What Nathan has been up to (research run, 2026-09-24)

**TL;DR:**
- I spent a day trying hard to make our system better, with strict anti-overfitting rules. **Nothing I tried beat our current system in a way that holds up statistically.** ("Current system" = the default desk system committed in "Test desks" (6e145be): frozen 18-factor composite + days-to-cover, `lc_t10` LP, 10% short-interest cap, text advisory only; `DEFAULT` in `fiam_desks/system.py`. Called **B0** below. **B1** = B0 + the look-ahead bug fix.)
- I did **fix a look-ahead bug**, get **new short-interest data**, find out **why the strategy lost money in 2015–2020**, and settle **whether text and LLMs help** (text: a little, conditionally; LLMs: no).
- The system we should submit is **B1**: the old default plus the bug fix.
- Two variants look better but are not proven.

Full details: `docs/RESEARCH_RUN_2026-09-24.md` (main report) and `docs/AGENTIC_WORKFLOW.md` (LLM / agent section for the deck). One README per experiment in `experiments/desk_13` … `desk_33`.

---

## 1. How I worked (why the results can be trusted)
- **DEV = 2015–2020** for all decisions. **TEST = 2021–2026** (the FIAM scoring window) is off-limits except for pre-registered checks.
- TEST was looked at only **twice**. The rules were written down *before* each look (`experiments/desk_27_confirmation_test/PREREGISTRATION.md`, `experiments/desk_32_text_confirmation/PREREGISTRATION.md`).
- Every variant I tried is logged in `experiments/desk_research_ledger.csv` (~100 rows). Every TEST look is in `experiments/desk_test_ledger.csv`.
- I corrected for trying many things (false-discovery control, deflated Sharpe, probability of backtest overfitting).
- New code lives in `fiam_research/`. **`fiam_desks/` (our baseline) was not modified** and its regression check still passes.

## 2. The system to submit: B1
B1 = the committed default system (B0) + the look-ahead fix + short-interest data from 2018. One sentence: **each month, rank the liquid large caps on 18 fixed value/quality/etc. characteristics plus days-to-cover, then let a linear program pick ~230 names that maximise that score subject to the FIAM constraints.** Nothing in it is fitted.

### 2a. Pipeline, step by step
Everything is deterministic, and every decision at month *t* uses only data known at *t*. Code: `fiam_desks/` (system, factors_desk, si, pm, lp) plus `fiam_research/core.py` (`b1_score`, `b1_kw`). Text is computed but **advisory only** (it does not change positions).

1. **Universe** (`config.py`, frozen): price ≥ $5, market cap ≥ $2B, 126-day dollar volume ≥ $10M, and both betas (`beta_60m`, `betabab_1260d`) observed.
2. **Factor scores.** 18 characteristics in 7 economic groups, each with a sign fixed in advance from the literature (`FACTOR_GROUPS` in `config.py`):

   | group | factors (direction) |
   |---|---|
   | value | book/market, earnings/price, FCF/price (high = long) |
   | profitability | gross profit/assets, NI/book equity, EBIT/sales (high = long) |
   | investment, issuance, accruals | asset growth, 12m share issuance, accruals/assets (**low** = long) |
   | quality | QMJ, Piotroski F-score (high = long) |
   | surprise | earnings surprise, sales surprise (high = long) |
   | volatility / beta | idiosyncratic vol, max daily return (5 days), BAB beta (**low** = long) |
   | liquidity | Amihud illiquidity (high = long), turnover (**low** = long) |

   Each factor is ranked within the month across universe stocks to [−1, 1], flipped by its sign, and averaged inside its group. Missing values get the monthly median, so they neither help nor hurt.
3. **Days-to-cover as an 8th group.** FINRA days-to-cover (mid-month settlement file, published before month-end, so causal), ranked within the month and flipped (**fewer** days = long). No FINRA value means the group scores 0 (no view).
4. **Composite score** = (sum of the 7 group scores + the dtc score) / 8. Equal weights, no fitting. So dtc is 1/8 of the score, and it is the piece that did *not* replicate (see 3.2), but it is kept because it was part of the committed default.
5. **Portfolio construction: one LP per month** (`lp.py`, scipy/HiGHS). It maximises Σ score × weight (long minus short) subject to:
   - gross exposure = 200% of NAV (about 100% long, 100% short) and dollar-neutral (net = 0);
   - beta-neutral on **both** `beta_60m` and `betabab_1260d` (exactly 0 at formation; realised beta is not exactly 0);
   - net weight in each GICS sector ≤ 5% of NAV; gross weight in each sector ≤ 70% of NAV (35% of gross);
   - per-name weight ≤ 1% of NAV (so it must hold ≥ 200 names; in practice 204–248: ~116 long, ~114 short);
   - one-way turnover ≤ 10% per month versus last month's drifted book. If infeasible, the cap is loosened stepwise (1.5×, 2×, 3×, 5×, none) and logged.
6. **Short-interest cap.** A stock cannot be shorted if FINRA short interest / shares outstanding > 10%. Unknown short interest counts as "allowed".
7. **Costs (assumed, not measured).** Net returns subtract one-way trading costs of 5 / 10 / 20 bp and annual borrow of 30 / 75 / 200 bp on short notional, by market-cap tier (≥ $10B / $2–10B / below). Realised ≈ 3.5 bp/month trading + 4.9 bp/month borrow ≈ 8.4 bp/month. Real borrow costs are one of the two things we never had.
8. **Outputs** (`system.py`): FIAM holdings CSV (Date, PERMNO, TICKER, COMPANY NAME, WEIGHT % of NAV), returns CSV (gross, net, benchmark = T-bill + 4%, exposures, position counts) and a rationale table with a template-generated one-line reason per position (top factor groups for and against, short interest, days-to-cover, advisory text flags).

### 2b. What is different from B0 (the committed default)
| | B0 | B1 |
|---|---|---|
| Universe formation | rows with no next-month return dropped **before** the universe is formed (uses "this stock survives" = look-ahead) | those rows are kept with return = 0 (`PanelX` in `fiam_research/panel_ext.py`), so the universe is what was knowable at month-end. The last month (2026-08) has no next return by design and is not treated as a delisting |
| FINRA short interest | files from 2020-06 | files from **2018-01** (`fiam_research/si_ext.py`, downloaded 2026-09-24) |
| Everything else | same | same (factors, weights, LP, 10% SI cap, costs, advisory text) |

Two consequences to keep in mind:
- **Delisting return = 0 is an assumption.** It is a neutral fill, not a measured delisting return, so the true P&L on stocks that disappear is not captured either way.
- **Before 2018-01 there is no short-interest data.** The dtc group is 0 and the 10% cap cannot bind, so roughly the first half of DEV (2015–17) is effectively the plain 7-group composite. From 2018 on the full B1 rules apply. This is one more reason DEV and TEST are not like-for-like.

### 2c. What the book looks like (TEST 2021-01..2026-08, 68 months)
- **Positions:** 204–248 (avg 230), turnover 10%/month (at the cap), gross 192–200%, net 0, beta at formation 0.
- **The long leg does all the work.** Gross long leg ≈ +12.7%/yr contribution vs ≈ +0.3%/yr from the short leg (short-leg CAGR −1.5%). The shorts mostly hedge market beta and fund the longs rather than earn much.
- **Style exposure (returns regression, TEST):** value 0.58 (t 4.7), quality 0.55 (t 3.8), momentum 0.29 (t 5.0), profitability 0.32 (t 2.3); low-vol only 0.09 (t 1.0). R² 0.76; residual alpha 2.4%/yr, t 0.8. So most of the return is explained by known style premia.
- **Who it holds (average weight per month over TEST):** the largest longs are AAPL, ZTS, A, MO, GOOGL, ADBE, MSFT (~0.7–0.85% each). The largest shorts include biotech (IONS, RARE, INSM, ITCI, ARWR) and growth software/tech (BL, FIVN, TTWO, AXON) at ~0.65–0.95% each. That long-quality / short-growth tilt is consistent with the growth-vs-value story in 3.3.
- **Calendar-year net return on TEST:** 2021 +31.0%, 2022 +18.3%, 2023 +9.1%, 2024 +14.3%, 2025 −2.7%, 2026 (to Aug) +0.3%. **The Sharpe of 0.99 is front-loaded**: the last ~20 months are flat. Worth saying in the deck rather than letting a reviewer find it.
- **Checks passed:** all 7 FIAM constraint checks every month, and two runs give bit-identical holdings (`experiments/desk_29_final_candidate/output/summary.json`).

### 2d. Headline numbers

| | 2015–2020 (DEV) | 2021–2026 (TEST) |
|---|---:|---:|
| net Sharpe | −0.15 | **0.99** |
| net IR (vs T-bill + 4%) | −0.67 | **0.66** |
| max drawdown | −24% | **−9.6%** |
| beta | ≈ −0.15 | ≈ 0 |
| FIAM constraints | pass | pass |

FIAM-format holdings and returns: `experiments/desk_29_final_candidate/output/holdings_B1.csv`, `returns_B1.csv`.
**Honest caveat for the deck:** B1 works in 2021–26 but not 2015–20. Its returns are mostly style bets (quality, profitability, value, low-vol), not stock-specific alpha.

## 3. Things that are now known for sure
1. **Look-ahead bug fixed.** The old panel dropped stocks with no next-month return *before* choosing the universe, which secretly used "this stock survives". The book was shorting future delistings about 3:1. Fixing it moves TEST IR from 0.637 to 0.660.
2. **New data: FINRA short interest back to 2018** (we only had 2020 onward). Run `.venv/bin/python -m fiam_research.download_data`. On those never-seen months:
   - **days-to-cover did NOT replicate**, so treat it as unproven;
   - **the 10% short-interest cap DID replicate** as a risk control.
3. **Why 2015–2020 loses money:** the growth-vs-value cycle. 2020 had the worst value crash on record plus a junk/high-vol rally. Almost every effect that flips sign between periods is this one cycle: value, the short-interest effects, and text in mid-caps.
4. **The LP is not the bottleneck.** A risk-aware optimizer, volatility-scaled caps and more names were all flat or worse.
5. **These failed:**
   - factor timing (factor momentum, dispersion scaling);
   - interaction signals;
   - ML (ridge; a monotone gradient-boosting model looked good on 2017–20 but was weak on TEST);
   - turnover changes.
6. **Warning story for the deck:** the *only* result that survived multiple-testing correction on 2015–20 (tightening the short-interest cap to 5%) **reversed on TEST** (t −2.0, IR 0.66 → 0.30). This is exactly why we don't pick the best-looking backtest.

## 4. Promising but NOT proven (don't claim as results)
| variant | what it is | Sharpe 2015–20 / 2021–26 / all | why not adopted |
|---|---|---|---|
| **A5, JKP 13 themes** | all 146 characteristics grouped into the 13 published Jensen–Kelly–Pedersen themes, using the *published* long/short direction, equal-weighted, nothing fitted | 0.21 / **1.29** / **0.80** (B1: −0.15 / 0.99 / 0.51) | better through lower volatility, not higher returns; the difference is not significant (90% CI −0.14 to +0.73) |
| **Y2, text-conditional** | B1, but the earnings-surprise factor counts double when the company just filed its earnings 8-K, and no view on names with pending mergers | 0.11 / 1.10 / 0.67 | mean return on TEST no better (t 0.12) |

Both are better risk-adjusted in *both* periods. The only clean way to confirm either is a pre-registered test on months after 2026-08. Outputs for A5: `experiments/desk_29_final_candidate/output/*_A5_jkp13z.csv`.

## 5. The text data (handoff folder)
The handoff parquet files are byte-identical to what we'd already been using.
- **No standalone alpha** on stocks we can actually trade.
- **Useful conditional information:**
  - the earnings-8-K timing tells you when the surprise factor works;
  - merger-agreement language tells you where the factor view is wrong.
  - Both are strong in 2015–20 and the same direction but weaker in 2021–26.
- **Text predicts risk:** novel + negative 8-Ks predict next-month volatility, equally in both periods (t 4.7). I haven't found a way to turn that into better returns.

## 6. LLMs / agents (FIAM §9)
Tested with a local model (qwen3.5 9B via ollama; no API key available):
- **Reading 1,775 masked 8-Ks (analyst agent): no signal.**
  - It barely matched the market's reaction to the same news.
  - It predicted next month with the wrong sign.
  - It labelled 54% of filings "good news".
  - Masking wasn't enough: it still guessed the company 22% of the time, which means look-ahead risk. (I also fixed a masker bug that leaked company names.)
- **Explaining positions:** fluent, 0 invented numbers, but **10–20% wrong in meaning**. Examples: 0.5% short interest called "high"; COKE rendered as "Coke (KO)", a different company. **Use the template rationale** we already generate.
- **Devil's advocate:** wrote attacks that cited numbers contradicting its own claims. Our deterministic attack scripts found the real problems.
- **What to say in the deck:** the agentic part that worked is the **research loop itself**, a coding agent running signal discovery with ledgers, pre-registration and multiple-testing control. It found the bug, killed days-to-cover, and caught its own best idea failing out of sample. Plus candid negative results for the LLM reader and explainer. FIAM explicitly rewards honest accounts of agents that didn't work. Prompts and details: `docs/AGENTIC_WORKFLOW.md`.

## 7. What I'd do next
1. Build MAIN.py and the deck on **B1**. Mention A5 / Y2 as "promising, pre-registered for future data".
2. Use `docs/RESEARCH_RUN_2026-09-24.md` for the limitations and discussion pages (the regime story, the reversal story).
3. Don't run more variants on 2021–2026: it has had its two logged looks.
4. Two things we never had and that limit us: **daily returns** (FIAM's daily-risk section) and **real borrow costs**.

**Setup for anyone rerunning:** `.venv/bin/python -m pip install -r requirements-research.txt` (cvxpy, openpyxl), then `.venv/bin/python -m fiam_research.download_data`. The LLM experiments need ollama with `qwen3.5:9b-q4_K_M`. Nothing from this run is committed yet.
