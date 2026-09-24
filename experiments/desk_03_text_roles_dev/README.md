# Other roles for the text desk: persistence, news-gated reversal/drift, tail overlay (DESK_03_TEXT_ROLES_DEV)

Implementation: `desk_03_text_roles_dev.py` (~30 s). Outputs: `output/roles_ic.csv`, `output/tail_overlay.csv`, `output/results.json`. Dev only.

## Objective / what changed / why
After desk_01/02, change the ROLE of text rather than the signal. F1 persistence (decayed intensity, half-life 6, fixed by `feat_8k_502_followup`): 5.02 count, abrupt, hard-abrupt, novel-distress; sign -1. F2 news-gated: `rev_nonovel` = -rank(ret_1_0) where no novel (>= 0.5) filing, sign +; `drift_novel` = +rank(ret_1_0) where a novel filing, sign +. F3 tail overlay: flagged-minus-unflagged next-month tail rate (y <= -15%). Bonferroni 6 (F1+F2); F3 descriptive.

## Results (dev; composite dev IC +0.0106)
| block | IC | t | paired gain | paired t | perm p | verdict |
|---|---:|---:|---:|---:|---:|---|
| m502_hl6 (**the user's own 2021-26 finding**, on 2015-20) | **-0.0112** | **-2.42** | -0.0043 | -1.87 | 1.00 | kill |
| abrupt_hl6 / hard_abrupt_hl6 / novel_distress_hl6 | -0.0041 / -0.0054 / +0.0006 | -0.8 / -1.2 / +0.1 | <= -0.0003 | <= -0.14 | >= 0.45 | kill |
| rev_nonovel | +0.0204 | +1.39 | +0.0072 | +1.27 | 0.005 | inconclusive |
| drift_novel | -0.0056 | -0.62 | -0.0017 | -0.88 | 0.73 | kill |
| control: plain reversal -rank(ret_1_0) | +0.0222 | +1.27 | +0.0087 | +1.15 | 0.005 | (control) |

F3: only `novel_distress` shows a fatter tail (5.24% vs 4.42%, t +1.94); other flags |t| < 1.4; mean-return differences all |t| < 1.1.

## Interpretation
**Failed:** (1) the 5.02 decayed-intensity signal, the only positive 8-K result in this repo (2021-26 IC +0.0125, t 2.87), has the **opposite sign on 2015-2020 (t -2.4)**: it is period-specific, not a stable characteristic. (2) News-gating does not help: gated reversal (0.0204) is *below* plain reversal (0.0222). (3) Tail overlay: one flag borderline out of eight. **Worked (by accident, non-text):** plain 1-month reversal adds to the composite on dev (perm p 0.005) -> tested properly in desk_05. **Carry forward:** flags -> `T_B` for the PM overlays in desk_08; reversal -> desk_05.

**Status: IMPLEMENTED_BUT_FAILED for text (a non-text lead found)**

## Limitations
Reversal result came from a control, so it enters the multiple-testing count of desk_05.
