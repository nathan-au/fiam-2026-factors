# Does a momentum group help the factors desk? (DESK_07_FACTORS_MOMENTUM_DEV)

Implementation: `desk_07_factors_momentum_dev.py` (~20 s). Dev only. Single pre-declared test (Bonferroni 1): momentum = mean(rank ret_12_1, rank resff3_12_1), sign +, as an 8th group. Adoption rule: PASS on dev AND non-negative at one logged test confirmation.

## Results (dev)
Momentum group IC -0.0051 (t -0.27) [by year 2015 +0.065, 2016 -0.075, 2017 +0.007, 2018 +0.015, 2019 -0.048, 2020 +0.010]; composite+momentum paired gain -0.0036 (t -0.59), perm p 0.63 -> **kill**. Books (dev net IR): composite -0.87; composite+momentum -1.04 (paired t -0.50); momentum only -0.34 (paired t +0.83).

## Interpretation
**Failed:** momentum does not repair the dev-period loss (its own IC is unstable in sign year to year); it is not carried to the test window. **Carry forward:** stop mining the factors desk on dev - after desk_05/06/07 (5 candidate ideas, all killed or inconclusive) further dev searches would be data-dredging. The factors desk stays the frozen composite (+ dtc, + SI cap on test).

**Status: IMPLEMENTED_BUT_FAILED**
