# The single confirmation run on the test window (DESK_09_CONFIRMATION_TEST)

Implementation: `desk_09_confirmation_test.py --confirm` (~1 min). Read `PREREGISTRATION.md` (written before this ran; committed in the same directory). Outputs: `output/confirmation_test.csv`, `output/results.json`. Every arm is in `experiments/desk_test_ledger.csv`.

## Objective
Check, once, on 2021-01..2026-08 (68 months): (1) the harness reproduces the published rows; (2) which text PM mode / factors option, if any, satisfies the adoption rule; (3) whether the text lane's standalone signals replicate. 12 books + 4 standalone signals declared.

## Results
**Reproduction (published gross / net IR): all four match** - F0 composite 0.615/0.541; F0 + SI cap 0.590/0.508; F1 composite+dtc 0.659/0.584; **F1+cap 0.721/0.637 (default base)**.

Text modes and the rev_1m option layered on F1+cap (paired vs F1+cap; adoption rule 2 = dev a AND test b AND risk c):
| arm | PM-score IC | net IR | paired net t | paired IC t | max DD | 2025 | rule a/b/c | decision |
|---|---:|---:|---:|---:|---:|---:|---|---|
| F1_cap (default base) | +0.0455 | +0.637 | | | -9.8% | -1.0% | | |
| + blend | +0.0460 | +0.668 | +0.09 | +0.54 | -10.5% | -2.3% | n / n / Y | advisory |
| + veto_long | +0.0455 | +0.728 | +0.36 | | -9.5% | +0.2% | n / n / Y | advisory |
| + tilt | +0.0426 | +0.630 | **-0.67** | **-2.54** | -9.9% | -3.9% | **Y** / n / Y | advisory (fails b) |
| + dial | +0.0455 | +0.483 | -1.67 | | -8.6% | -5.3% | n / n / Y | advisory |
| + judge | +0.0455 | +0.658 | -0.27 | | -8.4% | +2.0% | n / n / Y | advisory |
| + rev_1m | +0.0393 | +0.657 | +0.04 | -1.19 | -9.3% | -1.2% | | not adopted |
| (robustness, base F0_cap) + tilt / + judge | 0.0342 / 0.0366 | 0.598 / 0.652 | +0.21 / +0.86 | -2.23 / | -10.8% / -7.3% | | | not adopted |

Standalone text signals on test (rule 3: replicated iff sign as pre-registered and t >= 2): **novneg_max +0.0103 (t +2.87) replicated**; novelty_max +0.0053 (t +1.23) not; abrupt_exit -0.0001 (t -0.01) not; T_A (3-signal desk) +0.0070 (t +1.81) not.

## Interpretation / what worked / what failed / carry forward
**Worked:** the assembled harness is faithful to the published rows; the pre-registered rule produced an unambiguous verdict; `novneg_max` (novelty x negative tone) is a real, small, same-sign signal in the test period. **Failed:** no text PM mode is adopted. `tilt`, the only one that passed the dev criterion, *lowers* the PM-score IC (paired t -2.5) on test - a dev-to-test reversal, consistent with the dev evidence having been noise. `dial` is the worst (net IR -0.15, positions up to 398). `rev_1m` adds nothing. **Uncertain:** veto_long (+0.09 net IR, paired t +0.36) and judge (+0.02) are positive but far from significant (desk_11 bootstrap CIs include 0). **Carry forward:** default = F1+cap with the text desk ADVISORY; `novneg_max` goes to desk_12 as a labelled post-hoc follow-up.

**Status: CONFIRMED (harness) / text modes NOT ADOPTED**

## Deviations from the pre-registration
None in arms or rules. One note: the standalone `abrupt_exit` etc. use the two-sided rank/flag scores of `fiam_desks/text_desk.py`, as in desk_01.

## Limitations
One path of 68 months; the frozen composite, dtc and SI cap had already been evaluated on this window by earlier experiments (they enter as adopted components, their reproduction is a check, not a fresh test).
