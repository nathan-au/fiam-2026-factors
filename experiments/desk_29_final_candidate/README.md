# desk_29 — Final candidate outputs (FIAM format) and audits

Both books are arms already evaluated in desk_27. No new TEST arm. Outputs in `output/`: `holdings_{B1,A5_jkp13z}.csv` (Date, PERMNO, TICKER, COMPANY NAME, WEIGHT % NAV), `returns_{...}.csv`, `summary.json`.

| | B1 (validated candidate) | A5 JKP13z (promising, not validated) |
|---|---:|---:|
| TEST IC (t) | +0.045 (2.39) | +0.025 (1.83) |
| net / gross IR | +0.660 / +0.743 | +0.856 / +0.955 |
| net Sharpe | 0.99 | 1.29 |
| max DD | −9.6% | −7.7% |
| beta (VW panel market) | −0.03 | −0.04 |
| positions | 204–248 | 201–253 |
| turnover / cost | 10.0% / 8.4 bp/month | 10.0% / 7.7 bp/month |
| all 7 FIAM constraint checks | pass | pass |
| determinism (two runs) | identical | identical |

Look-ahead audit of the new code: theme z-scores rebuilt from a panel truncated at 2017-10, 2018-09 and 2024-08 equal the full-panel values exactly (max diff 0.0). The desk_00 regression gate still passes, and `fiam_desks/` is unmodified.
