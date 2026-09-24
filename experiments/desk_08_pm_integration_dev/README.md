# Deterministic PM integration modes with the text desk, dev period (DESK_08_PM_INTEGRATION_DEV)

Implementation: `desk_08_pm_integration_dev.py --placebos 20` (~8.5 min). Outputs: `output/integration_dev.csv`, `output/results.json`. Dev only. Re-run after a dead-code cleanup: results identical (checked, see bottom).

## Objective / what changed / why
Build the PM layer (`fiam_desks/pm.py`: pure functions of desk outputs -> LP inputs) and compare the ways the text desk can act on positions. Text desk `T_A` = mean of signed [novelty_max, novneg_max, abrupt_exit]; adverse flag `T_B` = any of novel_distress, hard_abrupt, litigation (both fixed before running). Parameters are round numbers, not tuned.

## Arms and results (dev, 71 months, base = frozen composite, lc_t10; `paired t` = paired net-return t vs A0; `placebo` = share of 20 within-(month, sector) shuffles of the text output with net IR >= real)
| arm | rule | PM-score IC | net IR | paired t | max DD | 2020 | placebo |
|---|---|---:|---:|---:|---:|---:|---:|
| A0 factors_only | composite | +0.0106 | -0.87 | | -36.9% | -29.9% | |
| A1 blend w=1 | text 8th group | +0.0112 | -0.88 | +0.05 | -34.9% | -28.2% | 0.00 |
| A2 veto_long | longs barred where flag | +0.0106 | -0.90 | +0.11 | -35.3% | -27.8% | 0.20 |
| A3 tilt 0.25 | pred - 0.25*flag | +0.0107 | -0.83 | +0.77 | -32.5% | -27.1% | 0.00 |
| A4 dial 0.5 | cap x (1+0.5*sign agreement) | +0.0106 | -1.08 | -0.18 | -32.6% | -22.2% | 0.50 |
| A5 judge | shorts need >= 5/7 groups OR flag | +0.0106 | -0.86 | +0.26 | -34.4% | -27.9% | 0.00 |
| A6 judge, factors only | ablation | +0.0106 | -0.97 | -0.71 | -36.4% | -28.7% | |
| A7 agree_dial | cap x (0.5 + group agreement) | +0.0106 | -0.88 | -0.74 | -39.8% | -31.1% | |

All books pass the FIAM constraint checks; the veto never made the LP infeasible.

## Interpretation
**Demonstrated:** the PM layer works end to end (all modes, constraints hold). **Not demonstrated:** any benefit: every paired t is below 1 in absolute value. The placebo needs care: shuffled text books are *worse* than real (e.g. blend placebo -0.995 +- 0.066 vs real -0.878) because random noise dilutes a weak composite - the correct null for "does text add value" is the un-augmented A0, and against it nothing is distinguishable. LP outcomes move by +-0.1 IR with any perturbation (placebo sd 0.04-0.11), the noise floor. **Partly works:** `tilt` (t +0.77, placebo 0.00) and `judge` (the text flag adds +0.11 IR over A6) are the only weakly supportive arms -> pre-registered as candidates. **Failed:** dial (fewer/more concentrated positions, worse IR), agree_dial, judge without text. **Carry forward:** PREREGISTRATION.md rule 2 admits only `tilt` to the confirmation as a potential default; `judge` runs but cannot be adopted (dev t +0.26 < 0.5).

**Status: IMPLEMENTED_AND_TESTED (null on dev; tilt and judge carried)**

## Limitations
Single path; shuffles n=20; the base itself loses money on dev so "improvement" is measured around a negative IR.
