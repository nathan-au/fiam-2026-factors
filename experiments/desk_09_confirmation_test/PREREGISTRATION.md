# Pre-registration for the ONE confirmation run on the test window (written before any desk was run on 2021-2026)

Written after desk_00..desk_08 (all on the DEV period 2015-02..2020-12). Nothing below was chosen by looking at 2021-2026 returns *for these desks*. (The frozen composite, the days-to-cover block and the
SI cap were already evaluated on 2021-2026 by earlier experiments in this repo; they enter as previously-adopted components, reproduced here as a check.)

## What is run (all on test target months 2021-01..2026-08, universe = frozen large-cap universe, LP = lc_t10)
Factors-desk bases (reproduction of earlier published rows, not new tests): F0 composite; F0+SI cap; F1 composite + days-to-cover (8th group); **F1+cap = the default factors desk + PM**.
Text-desk options, layered on F1+cap (5 modes) and, for the two candidates, also on F0 (robustness): blend (w=1), veto_long, tilt (0.25), dial (0.5), judge; text desk = T_A / T_B as in desk_08 (frozen).
Factors option carried from dev: rev_1m as an 8th group on F1+cap (dev: inconclusive, paired gain t +1.15, permutation p 0.005).
Standalone text signals (novelty_max, novneg_max, abrupt_exit and T_A): IC on test, to confirm or contradict dev.
Every test-period arm is appended to `experiments/desk_test_ledger.csv` (experiment, arm, config hash, timestamp). Total test-period arms declared: 12 books + 4 standalone signals.

## Decision rules (fixed now)
1. **Default factors desk** = F1+cap unless its reproduction disagrees with the published rows (net IR 0.637 +-0.005), in which case the harness is wrong and nothing else is reported until fixed.
2. **A text PM mode (or rev_1m) enters the default system iff ALL of:**
   a. dev evidence: paired net-return t vs its non-text base >= 0.5 AND placebo share <= 0.05 in desk_08. Applying it to the desk_08 numbers: only `tilt` qualifies (paired t +0.77, placebo share 0.00);
      `judge` (t +0.26) fails (a) even though its placebo share is 0.00 - it is still run and reported, but cannot enter the default whatever the test says; `blend` (t +0.05), `veto_long` (t +0.11),
      `dial` (t -0.18) fail (a) and are reported as inconclusive-null;
      for rev_1m: dev paired IC gain t >= 1 and permutation p <= 0.05 (met, inconclusive).
   b. test confirmation: paired net-return t vs the same base (non-text) >= 1.0 AND the PM-score universe IC not lower than the base's.
   c. it does not violate any FIAM constraint and does not raise max drawdown by more than 2 percentage points.
   Otherwise it stays implemented but **advisory** (its output appears in the rationale table, it does not change positions).
3. A standalone text signal is reported as "replicated" only if its test IC has the pre-registered sign and t >= 2; else "not replicated".
4. No arm is added, dropped, re-parameterised or re-run after seeing the test numbers. Any deviation is listed in the README as a deviation.
5. Single-path IR standard error is about 0.46 (docs/EXPERIMENTS.md); IR differences below that are not evidence. Paired monthly net-return t and IC t are the primary statistics.
