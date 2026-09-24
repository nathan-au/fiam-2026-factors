# Deterministic PM hardening: the pre-existing neutralisation dial, on the dev period (DESK_06_PM_NEUTRAL_DIAL_DEV)

Implementation: `desk_06_pm_neutral_dial_dev.py` (~20 s). Outputs: `output/dial_dev.csv`, `output/results.json`. Dev only.

## Objective / what changed / why
desk_05 showed the composite book loses on dev. Is that the signal or what the LP lets it hold? Use `experiments/neutral_dial` (defined for the test period) unchanged: D1 beta+sector (current), D2 +log size, D3 +12-1 momentum, D4 +idio vol +quality (exact-neutral constraints in the same LP), plus per-name cap 0.5% (`w05`).

## Hypothesis
If the dev loss comes from unintended style exposure (high-vol shorts), deeper neutralisation improves it; if it comes from the signal, it does not.

## Results (dev, net IR / paired net t vs D1 / max DD / 2019 / 2020)
| variant | net IR | paired t vs D1 | max DD | 2019 | 2020 |
|---|---:|---:|---:|---:|---:|
| D1 beta+sector (current) | -0.87 | | -36.9% | -8.2% | -29.9% |
| D2 +size | -0.96 | -1.61 | -40.0% | -9.6% | -32.1% |
| D3 +size+mom | -0.98 | -1.39 | -42.7% | -11.1% | -33.7% |
| D4 +size+mom+vol+quality | -1.66 | -1.34 | -42.6% | -9.5% | -31.3% |
| D1 w05 | -1.07 | +0.47 | -30.5% | -4.9% | -26.3% |
| D4 w05 | -1.76 | -0.54 | -34.0% | -6.7% | -25.0% |

## Interpretation
**Failed:** none of the neutralisation steps helps; D4 (which fixes vol and quality exposure exactly) is the worst. The dev loss is a property of the **signal** in that regime (value/accrual groups had negative IC), not of the LP's freedom. **Worked:** the PM layer reproduces the frozen constraint behaviour on a new period (all constraint checks pass, relaxed months 0-1). **Carry forward:** keep the frozen D1 PM as the default (D4 stays a documented option, as in `neutral_dial`); do not expect PM changes to cure a regime-dependent alpha.

**Status: IMPLEMENTED_AND_TESTED (negative)**

## Limitations
Single 71-month path, IR s.e. ~0.46; D4 relaxes the turnover cap in 1 month.
