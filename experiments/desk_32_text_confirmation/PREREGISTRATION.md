# desk_32 — Pre-registration of the SECOND test-window look of this research run (text-conditional factors, optional LLM arm)

Written 2026-09-24 after desk_30 (DEV) and while desk_31 (the LLM read on DEV) was still running. **Before any arm below was evaluated on TEST (2021-01..2026-08).**

## Why a second look, and its price
- desk_27 was the run's single pre-registered look. desk_30 found conditional structure in the handoff text tables that was never on TEST in any form. The X1-style item-2.02 timing was tested on TEST only as *earnings-timing flags* (feat_8k_meta, a different construction).
- Because this is a second look, **every p-value here is judged at half the nominal level** (Bonferroni over 2 looks). Concretely, "t ≥ 1" becomes "t ≥ 1.3" and "t ≥ 2" becomes "t ≥ 2.3" (one-sided normal).
- Contamination: B1 is already known on TEST (desk_27 A0). Y1 and Y2 are new.

## Arms (target months 2021-01..2026-08; PM = B1's lc_t10 LP + 10% SI cap; panel = PanelX)
| arm | definition |
|---|---|
| A0 | B1 (reference) |
| Y1 | B1 with the surprise group weighted 2 where the stock filed an item-2.02 8-K in month t or t−1 (`txt_item_2_02`), else 1 |
| Y2 | Y1 + composite score set to 0 (no view) where the stock filed merger-agreement language (`txt_v2_n_101_merger` > 0) in months t−5..t |

## Standalone replications
| id | test | replicated iff |
|---|---|---|
| R1 | X1: surprise-group IC, fresh-2.02 names minus stale names | diff > 0, t ≥ 2.3 |
| R2 | X2: B1 IC, non-merger names minus merger-flagged names | diff > 0, t ≥ 2.3 |
| R3 | X4: A0 book's weighted novneg rank vs next-month abs(book return) | rank corr > 0, t ≥ 2.3 |
| R4 | **only if desk_31 H3 passes on DEV**: the same LLM read on a TEST sample drawn identically (25 filings/month), H3 regression | LLM coef > 0, t ≥ 2.3. If H3 fails on DEV, R4 is not run and no TEST filing is read by the LLM |

## Decision rule (same as desk_27, at the second-look thresholds)
Y2 (or Y1) becomes the candidate iff all of the following hold:
- (a) TEST paired monthly net t ≥ 1.3 vs A0;
- (b) TEST IC ≥ A0 − 0.005;
- (c) TEST max DD not worse by > 5 pp, and all FIAM constraints pass;
- (d) combined DEV+TEST net Sharpe ≥ A0's.

Otherwise B1 stays. No arm is added, dropped or re-parameterised after seeing TEST.
