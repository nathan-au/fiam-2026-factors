# desk_27 — The single pre-registered TEST confirmation of this research run

`PREREGISTRATION.md` was written (file time 04:59:57) before the script (05:00:41) and before any arm ran on TEST. It lists the arms, the tests, the rules and the prior contamination. Run: `desk_27_confirmation_test.py --confirm` (94 s). Every arm is in `experiments/desk_test_ledger.csv`. Deviations from the pre-registration: none.

## Arms (TEST 2021-01..2026-08, 68 months; combined = DEV+TEST for A0–A2/A5, 2017-01..2026-08 for walk-forward arms)
| arm | TEST IC (t) | TEST net IR | TEST Sharpe | max DD | paired net t vs A0 | IR diff 90% CI | combined Sharpe (A0) | DSR (all trials) | rule a/b/c/d | decision |
|---|---:|---:|---:|---:|---:|---|---:|---:|---|---|
| A0 B1 | +0.0454 (2.39) | +0.660 | +0.99 | −9.6% | – | – | +0.511 | 0.26 | – | reference |
| A1 SI cap 5% | +0.0454 | +0.303 | +0.74 | −8.8% | **−2.00** | [−0.72, −0.04] | +0.613 (0.511) | 0.08 | n/Y/Y/Y | **fail (reversed)** |
| A2 no view on SI>10% both sides | +0.0454 | +0.643 | +0.98 | −10.2% | −0.49 | [−0.10, +0.07] | +0.489 (0.511) | 0.25 | n/Y/Y/n | fail |
| A3 monotone GBM (5 seeds) | +0.0174 (1.12) | −0.233 | +0.24 | −14.2% | −1.79 | [−1.81, −0.03] | +0.295 (0.427) | 0.01 | n/n/Y/n | fail |
| A4 ridge 146 | +0.0170 (0.89) | −0.644 | −0.30 | −24.9% | −2.62 | [−2.17, −0.50] | −0.051 (0.427) | 0.00 | n/n/n/n | fail |
| **A5 JKP13 z (fit-free)** | +0.0251 (1.83) | **+0.856** | **+1.29** | **−7.7%** | −0.03 | [−0.32, +0.69] | **+0.804** (0.511) | 0.40 | n/n/Y/Y | fail (rule a/b) |
| A6 A3 + cap 5% | +0.0174 | −0.724 | −0.26 | −21.9% | −2.23 | [−2.48, −0.40] | +0.020 (0.427) | 0.00 | – | not interpretable (A1, A3 failed) |

**Candidate: A0 = B1 (rule 3: no arm passed).**

## Standalone replications
| id | result | replicated? |
|---|---|---|
| S1 novneg_max IC on the unseen $0.5–2B band | +0.038 (t 2.89), ~138 filers/month | **yes by the rule**, but DEV had −0.022 (t −1.5) in $1–2B (desk_02): sign flip |
| S2 novneg_max in the large tercile | +0.009 (t 1.59) | no |
| S3 novneg_max → next-month idio risk beyond ivol | +0.029 (t 4.71); DEV +0.030 (t 4.65) | **yes (robust)** |
| S4 B1 IC high-SI minus low-SI < 0 | **+0.050 (t +3.03)** (high-SI IC +0.064) | no, **reversed** |
| S5 monotone GBM IC | +0.017 (t 1.12); by year +0.002, +0.014, +0.019, +0.050, −0.008, +0.034 | no |

## Interpretation
- **The only multiple-testing survivor of the DEV search (SI cap 5%) reversed on TEST.** This is exactly the failure mode the protocol was built to catch. If the "best DEV result" had been promoted without a confirmation, it would have cut TEST IR in half.
- **The GBM's DEV edge (IC positive in every 2017–20 year) did not carry over**: IC 0.017 vs the composite's 0.045. More expressive models do not help here.
- **A5 (fit-free JKP themes) has the better risk-adjusted record in both periods:** Sharpe +0.21 vs −0.15 on DEV and +1.29 vs +0.99 on TEST. That comes from lower volatility, not higher mean return, so it fails the pre-registered mean-return rule. desk_28 tests the Sharpe difference post hoc.
- **Text:** its risk information replicates; its return information in the $0.5–2B band flips sign across regimes.
