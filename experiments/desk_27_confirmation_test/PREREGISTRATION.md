# desk_27 — Pre-registration of the ONE test-window confirmation of this research run

Written 2026-09-24 after desk_13–26 (all design decisions on DEV 2015-02..2020-12) and **before any of the arms below was evaluated on TEST (2021-01..2026-08)**.

## Prior TEST contamination (disclosed)
- The composite, dtc and the 10% SI cap were adopted on TEST by earlier experiments (desk_09 ledger).
- desk_13 and desk_15 recomputed B0 / B1 on TEST: a descriptive baseline and an audit correction, not a selection.
- desk_17 (E) looked at the TEST IC-vs-regime slopes (dispersion, market, VIX, value spread). **No regime rule is confirmed here for that reason.**
- None of the arms A1–A6 and none of the standalone tests S1–S5 has been evaluated on TEST before.

## Arms (target months 2021-01..2026-08, universe and panel = research PanelX, costs = frozen tiers)
| arm | definition | origin |
|---|---|---|
| A0 | **B1** = frozen composite + dtc (8th group) + lc_t10 LP + SI cap 10% | reference (desk_15) |
| A1 | A0 with SI cap **5%** | desk_20 C5; only BH-FDR survivor (desk_26) |
| A2 | A0 with names SI > 10% barred on **both** sides (no view on heavily shorted names) | desk_24 I1 (H-SI2) |
| A3 | **monotone GBM**: xgboost depth 2, 300 rounds, eta 0.03, subsample 0.5, colsample 0.5, monotone +1 on the 146 JKP-signed ranks, target = within-month return rank; expanding walk-forward, annual refit, training target months ≤ Nov(Y−1); **average of seeds 0–4** (fixed now); PM = A0's | desk_25 M4 (H-M4) |
| A4 | ridge on the 146 signed ranks, α = 1e4 × training months, same walk-forward; PM = A0's | desk_25 M2 |
| A5 | JKP 13 themes, z-scores, equal weight (fit-free); PM = A0's | desk_19 V3 |
| A6 | A3 + SI cap 5% (only interpreted if A1 AND A3 each pass rule 1) | combination |

## Standalone replications
| id | test | replicated iff |
|---|---|---|
| S1 | novneg_max (sign −) universe-style rank IC on the **$0.5–2B band** (price ≥ $5, 126d dollar volume ≥ $10M, both betas observed, 500 ≤ mcap < 2000), TEST | IC > 0 and t ≥ 2 |
| S2 | novneg_max IC in the largest-cap tercile of the universe, TEST (post-hoc from desk_23 C2) | IC > 0 and t ≥ 2 |
| S3 | novneg_max predicts next-month abs residual return beyond the ivol rank (desk_23 C3), TEST | coefficient > 0 and t ≥ 2 |
| S4 | B1 IC among top-SI-tercile names minus bottom-SI tercile < 0 (desk_24 I1), TEST | diff < 0 and t ≤ −2 |
| S5 | A3's universe IC on TEST | IC > 0 and t ≥ 2 |

## Decision rules (fixed now)
1. **An arm becomes the new candidate architecture iff ALL of the following hold:**
   - (a) TEST paired monthly net-return t ≥ 1.0 vs A0.
   - (b) For signal arms (A3–A5): TEST universe IC ≥ A0's TEST IC − 0.005.
   - (c) TEST max DD not worse than A0's by more than 5 pp, and all FIAM constraint checks pass.
   - (d) Over the **combined** window, its net Sharpe ≥ A0's. The combined window is DEV + TEST for A1/A2/A5 and 2017-01..2026-08 for the walk-forward arms A3/A4/A6.
2. If several arms pass, the candidate is the one with the highest combined-window net Sharpe; A6 is considered only under its condition.
3. If none passes, the candidate remains B1, and the report says so.
4. Report for every arm: bootstrap 90% CI of the TEST IR difference vs A0 (block 4, 5000 draws), and the deflated Sharpe ratio with trials = every row of the research ledger. **No arm is added, dropped or re-parameterised after seeing TEST.** Every arm and test is appended to `experiments/desk_test_ledger.csv`.
