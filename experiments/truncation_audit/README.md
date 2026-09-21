# Truncation-invariance audit of return-derived characteristics, label alignment, and a suspicious-IC scan (TRUNCATION_AUDIT)

Implementation: `truncation_audit.py` (run: `.venv/bin/python experiments/truncation_audit/truncation_audit.py`, about 1.5 minutes).

## Objective
Verify that no return-derived characteristic in the panel uses information from after its characteristic month, that the truncation test is capable of failing, that the label `ret_exc_lead1m` is the plain next-month excess return, and that no characteristic has an implausibly large IC (a leakage red flag) - the FIAM brief's leakage scrutiny.

## Hypothesis
(i) Every return characteristic (`ret_1_0`, `ret_3_1`, `ret_6_1`, `ret_9_1`, `ret_12_1`, `ret_12_7`, `ret_60_12`) can be rebuilt exactly from the monthly `ret` column using rows <= t only. (ii) A deliberately leaky feature is flagged by the same test. (iii) `ret_exc_lead1m(t) == ret_exc(t+1)`. (iv) No characteristic has |IC| > 0.10 in the tradeable universe. Any failure is a finding.

## Research origin
docs/REDDIT_RESEARCH.md sec 2.9 (AQuA, arXiv 2608.12841 and r/LocalLLaMA 1vxajio: an LLM wrote a feature dividing "volume so far" by the day's FINAL total, a second LLM approved it as causal, and only a clean re-split caught it; the fix is to make leakage inexpressible and to test that a feature built from rows <= t equals the full-panel value). The "typed causal-operator registry" half of the idea is for AI-written feature-mining loops and is **not implemented** (no such loop is in scope; NOT_ACTIONABLE); instead every new feature builder in this round (`feat_*`) carries its own truncation test.

## Implementation
`truncation_test(builder, panel, dates)` (embedded): for 3 random test dates t it rebuilds every feature from `panel[eom <= t]` and requires the value at t to equal the full-panel value for every stock. `ret_char_builder` rebuilds each characteristic as the compounded monthly return over the lags each name implies (lags 0, 1-2, 1-5, 1-8, 1-11, 7-11, 12-59). `leaky_builder` is the negative control (next month's return as a "feature"). The label check compares `ret_exc_lead1m` with the calendar-shifted `ret_exc`. The IC scan computes, for all 147 characteristics, the mean monthly rank IC with next-month excess return over all stocks and over the >= $2B universe.

## Experimental setup
Common harness (a verbatim copy of the data / rank-transform / LP / cost / performance code of `experiments/largecap/largecap.py`, embedded so the script runs on its own; the LP functions `_solve_lp` / `build_portfolio` are redefined in the script with optional extra behaviours and reproduce the frozen LP when the options are off).

- **Signal**: the frozen composite (18 factors in 7 economic groups, equal weight, re-ranked in the universe each month; no fitting). It reproduces `experiments/largecap`: universe rank IC 0.0366 (t 1.91), `lc_t10` gross IR 0.615 / net IR 0.541, 2025 net return -12.4%.
- **Universe**: price >= $5, market cap >= $2,000M, 126-day dollar volume >= $10M, both betas observed (about 1,206 stocks per month); 68 test months 2021-01..2026-08.
- **Portfolio**: the frozen LP (dollar-neutral, neutral to `beta_60m` and `betabab_1260d`, net sector <= 5% of NAV, gross sector share <= 35%, gross 200%, per-name cap 1%, 10% one-way turnover cap = `lc_t10`), with tiered *assumed* trading and borrow costs (5/10/20 bp trading and 30/75/200 bp borrow by market-cap tier; not measured).
- **Caveat that applies to every number below**: a single 68-month path; the standard error of a single-path IR is about 0.46, so IR differences under ~0.5 are not distinguishable from noise on their own. Paired monthly net-return t-stats are reported next to IR; with 20+ variants per file, a few |t| > 1 are expected by chance.

## Baseline
None (an audit). The negative control is the comparison.

## Commands
```
.venv/bin/python experiments/truncation_audit/truncation_audit.py
```

## Results
Truncation test of the rebuilt characteristics: passed = **True**, 77,651 values compared at dates 2024-07-31, 2023-11-30 and 2025-08-31, maximum absolute difference between the full-panel and truncated rebuild = 0.0 for all seven.

Rebuilt characteristic vs the panel's own column (all months):

| characteristic | values compared | max |rebuilt - panel| | share within 1e-9 |
|---|---:|---:|---:|
| ret_1_0 | 526,287 | 8.9e-16 | 1.0000 |
| ret_3_1 | 513,021 | 3.6e-15 | 1.0000 |
| ret_6_1 | 493,272 | 7.1e-15 | 1.0000 |
| ret_9_1 | 473,746 | 1.4e-14 | 1.0000 |
| ret_12_1 | 454,576 | 2.1e-14 | 1.0000 |
| ret_12_7 | 454,755 | 7.1e-15 | 1.0000 |
| ret_60_12 | 214,516 | 8.5e-14 | 1.0000 |

**Negative control**: the deliberately leaky feature (next month's return) is flagged: `passed = False` (its truncated build differs from the full build, recorded as null/infinite in `results.json`), while the honest feature `ok_ret_1_0` differs by 0.0. The test is therefore capable of failing.

**Label alignment**: `ret_exc_lead1m(t) == ret_exc(t+1)` for all 522,431 stock-months where both exist (100.0%, max absolute difference 0.0; 0 mismatches).

**Suspicious-IC scan**: 147 characteristics; 11 have |IC| > 0.10 over ALL stocks and **none in the tradeable universe** (largest universe |IC| = 0.034):

| characteristic | ic_all_stocks | t_all | ic_universe | t_universe |
|---|---:|---:|---:|---:|
| ivol_capm_252d | -0.122 | -8.9 | -0.022 | -1.62 |
| bidaskhl_21d | -0.119 | -9.3 | -0.019 | -1.43 |
| ivol_capm_21d | -0.119 | -9.2 | -0.022 | -1.80 |
| ivol_ff3_21d | -0.119 | -9.5 | -0.026 | -2.44 |
| ivol_hxz4_21d | -0.118 | -9.5 | -0.023 | -2.15 |
| rvol_21d | -0.116 | -8.4 | -0.016 | -1.04 |
| rmax5_21d | -0.110 | -8.8 | -0.016 | -1.07 |
| prc | +0.104 | +9.5 | +0.012 | +1.12 |
| rmax1_21d | -0.104 | -8.6 | -0.011 | -0.85 |
| ni_me | +0.101 | +8.7 | +0.014 | +1.22 |
| fcf_me | +0.101 | +9.7 | +0.026 | +2.54 |

## Interpretation
All seven return characteristics are exactly reproducible from returns up to the characteristic month (differences <= 1e-13), so none looks ahead; the label is the plain next-month return; and the truncation test can fail. The 11 high-IC characteristics are the known small-cap lottery / idiosyncratic-volatility / illiquidity family (negative IC for volatility measures, positive for price and earnings yield) documented in `docs/NEGATIVE_RESULT.md`: their IC over all stocks is 0.10-0.12 but only 0.01-0.03 in the tradeable universe, which is the project's central finding, not leakage. **Nothing failed.** Status: IMPLEMENTED_AND_TESTED (audit passed).

## Limitations
- Only the 7 return-derived characteristics can be rebuilt from the panel; the remaining 140 need raw daily or accounting data that is not in the panel, so they are screened (IC scan) rather than proven point-in-time. A large IC is only a necessary-not-sufficient leakage flag; the accounting characteristics' report-date lags are those of the data provider and cannot be verified here.
- The test covers time leakage, not survivorship or universe-membership leakage (docs/NEW.md sec 3.9 item 9).
- 3 random dates: an exhaustive test over all dates was not run.

## Follow-up
Run the same truncation test on any future feature builder before its IC is read (already done for every `feat_*` builder, all passed; the short-interest, TNIC and insider builders rely on external files whose timing rules are stated in their headers).
