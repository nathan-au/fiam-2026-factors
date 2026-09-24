# The assembled default system, end to end (DESK_10_FINAL_SYSTEM)

Implementation: `desk_10_final_system.py` (~30 s), which calls `fiam_desks/system.py`. Outputs (`output/`): `holdings.csv` (FIAM format: Date = first day of the holding month, PERMNO, TICKER, COMPANY NAME, WEIGHT in % of NAV, + long / - short; 15,626 rows, 68 months), `returns.csv` (Date, portfolio_return net of the assumed costs, portfolio_return_gross, benchmark_return = TB3MS/12 + 4%/12, sp500_return, active_return, gross/net exposure, n_positions), `rationale.csv` (one row per position-month: every desk's output and a templated reason string), `monthly_constraints.csv`, `summary.json`.

## What it is
Factors desk = frozen 7-group composite + days-to-cover 8th group. Text desk = advisory (flags and scores are computed, audited and written to the rationale; they do not move positions). Deterministic PM = `lc_t10` LP (dollar-neutral, neutral to beta_60m and betabab_1260d, net sector <= 5%, gross sector <= 35%, gross 200%, per-name cap 1%, 10% one-way turnover budget) + short-interest ratio cap 10% on shorts. Costs: the project's tiered assumed trading and borrow costs (not measured).

## Results
Test (2021-01..2026-08): gross IR +0.721, **net IR +0.637** (matches the published F1+cap row), realised beta +0.023 (t +0.23), max drawdown -9.8%, 203-248 positions, one-way turnover 10.1%, all 7 FIAM constraint checks pass, and month by month: gross 1.95-2.00, |net| <= 3e-16, |beta exposure| <= 8e-16.
Audits: determinism (two runs, identical holdings hash), truncation invariance of the factors output (incl. dtc), the short-interest ratio and the text score at 3 random dates (10,725 values, max diff 0.0).
**Dev disclosure (same system, 2015-02..2020-12; SI data absent so the cap never binds and dtc = 0): net IR -0.757, max DD -30.0%, 2020 -22.2%.**
Sample rationale: "LONG 1.00% NAV; factors +0.34 (for: quality +0.85, profitability +0.78; against: value -0.29); short interest 0.5% of shares, days-to-cover 1.6; 8-K month: novelty 0.08".

## Interpretation
**Demonstrated:** the system is complete, deterministic, constraint-compliant, and traceable (every position has a rationale row derived from desk outputs). **Not demonstrated:** the returns are regime-robust (desk_05/06/11). **Carry forward:** this is the settled implementation; MAIN.py for the submission can be generated from `fiam_desks/` (not done here: the competition wants one file).

**Status: IMPLEMENTED_AND_TESTED**

## Limitations
`gross` dips to 1.95 in some months because the LP may hold a name on both variables (net weight below the long+short sum) - within the <= 200% limit; the frozen LP behaves the same.
