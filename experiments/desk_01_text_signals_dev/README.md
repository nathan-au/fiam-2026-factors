# Text desk: raw signal study on the development period (DESK_01_TEXT_SIGNALS_DEV)

Implementation: `desk_01_text_signals_dev.py` (run: `.venv/bin/python experiments/desk_01_text_signals_dev/desk_01_text_signals_dev.py`, ~1 min). Outputs: `output/signals_dev.csv`, `output/results.json`. Test window not read.

## Objective / what changed / why
First evaluation of the teammate's `txt_v1` / `txt_v2.1` tables inside this project's frame: tradeable universe (>= $2B, price >= $5, $10M dollar volume, both betas), frozen composite as the base, the repo's kill/pass rule. The handoff's evidence was pre-2021, univariate and (as it turns out) not restricted to the tradeable universe.

## Hypothesis
Pre-registered signs (all adverse: more of it -> lower next-month return). Primary family (3): `novelty_max`, `novneg_max`, `abrupt_exit`. Secondary: 10 more signals + one-sided variants of the rank signals (18 tests in the file). Each added as an 8th equal-weight group to the composite. Extra evidence: residual IC after the composite, 6-month-lag placebo, size terciles, within (month, sector) permutation null (200).

## Research origin
`docs/AGENTIC.md` (text desk), `handoff_text_lane_2026-09-21` (claims: novelty IC -0.015 t -2.9; abrupt exit -0.007; 71-112% survives the 147 characteristics), `experiments/feat_8k_meta` + `feat_8k_502_followup` (metadata blocks killed on 2021-26).

## Results (dev, 71 target months 2015-02..2020-12, universe ~1,005 stocks/month; composite dev IC +0.0106, t 0.68)
Signed IC (+ = predicted direction) and paired gain of adding the signal to the composite. Full 18-row table in `output/signals_dev.csv`.

| signal (two-sided) | IC universe | t | IC all stocks | residual IC | paired gain | paired t | Bonferroni p | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| novelty_max | +0.0014 | +0.34 | +0.0011 | -0.0015 | +0.0003 | +0.26 | 0.82 | kill |
| novneg_max | +0.0071 | +1.63 | +0.0047 | +0.0016 | +0.0006 | +0.53 | 1.00 | kill |
| abrupt_exit | +0.0000 | +0.01 | -0.0070 | +0.0070 | +0.0001 | +0.28 | 1.00 | kill |
| hard_abrupt / distress_801 / litigation / financing / novel_distress / neg_mean / unc_mean / v1 controls | -0.0026..+0.0044 | |t| < 1 | | | |t| < 0.8 | 1.00 | kill (all) |

All 18 tests: kill. No signal has a 6-month-lag IC that is systematically smaller than its own IC (lag IC is as noisy as the signal).

Follow-up diagnostic (ad hoc, in the session, not a scripted arm; reproduced in desk_02): the handoff's IC **does replicate on ALL stocks conditional on a filing** (novelty_max -0.0122, t -2.47; novneg_max -0.0121, t -2.83, raw sign) and vs the SAME-month return is far stronger (novelty -0.025, t -5.1; novneg -0.031, t -6.6). In the tradeable universe: -0.0005 (t -0.08) and -0.0064 (t -1.02).

## Interpretation
**Works:** the tables are clean and the handoff's numbers reproduce on the population they were measured on. **Does not work:** nothing survives the tradeable universe as a ranking group; the effect is (i) a same-month reaction, (ii) present only in names that fail the price/liquidity screens. **Carry forward:** desk_02 (where does it live), desk_03 (other roles). Sign convention note: in this project `+` = long, so an adverse indicator has sign -1.

**Status: IMPLEMENTED_BUT_FAILED (all 18 killed)**

## Limitations
The 18-test family is counted in Bonferroni only within the file; the cumulative dev family of desk_01..08 is ~45 looks. `txt_v2` rule regexes are the teammate's (see desk_00).
