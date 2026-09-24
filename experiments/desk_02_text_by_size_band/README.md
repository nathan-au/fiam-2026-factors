# Where does the text edge live? Size bands and universe floors (DESK_02_TEXT_BY_SIZE_BAND)

Implementation: `desk_02_text_by_size_band.py` (~1 min). Outputs: `output/band_ic.csv`, `output/floor_sweep.csv`, `output/results.json`. Dev only; test not read. Re-run after a code cleanup: results.json identical.

## Objective / what changed / why
desk_01 changed the question from "does text predict" to "where". The handoff's IC lives in all stocks; the frozen strategy trades a >= $2B universe. If the text edge sits in $0.5-2B names that PM screens still admit, the system's universe floor (not the text desk) should change.

## Hypothesis
Text IC (signed) is positive and significant in some market-cap band above the price/dollar-volume screens. Floors fixed in advance: $500M (pm_ablation), $1B (largecap sensitivity), $2B (frozen).

## Implementation
(a) IC of the signed raw signal among stocks WITH a filing, by band (0-500, 500-1000, 1000-2000, 2000-5000, 5000+), screens on. (b) For each floor: composite dev IC and paired gain of 5 text signals as an 8th group (permutation null 200; Bonferroni 15).

## Results (dev)
Signed IC among filers, screens on (positive = predicted direction):

| signal | 0-500 | 500-1000 | 1000-2000 | 2000-5000 | 5000+ |
|---|---:|---:|---:|---:|---:|
| novelty_max | -0.093 (t -1.2, ~6 names/mo) | -0.051 (t -1.77) | -0.025 (t -1.75) | -0.007 (t -0.77) | +0.005 (t +0.60) |
| novneg_max | +0.158 (t +2.0, ~6 names/mo) | -0.008 | -0.022 (t -1.49) | +0.003 | +0.009 (t +1.21) |

Floor sweep: composite dev IC +0.0127 / +0.0091 / +0.0106 at $500M / $1B / $2B; all 15 (signal, floor) additive tests killed (paired gain |t| < 0.6; unadjusted permutation p >= 0.15, i.e. 1.0 after Bonferroni 15).

## Interpretation
**Worked:** localised the finding. In the screened universe the novelty signal has the **opposite** sign at $0.5-2B (raw IC +0.05, t 1.8) - so the all-stock negative IC in desk_01 comes from names failing the price >= $5 / dollar-volume / beta screens (i.e. untradeable), the same pattern as `pm_ablation`. **Failed:** no floor rescues the text desk. **Carry forward:** keep the $2B floor; text is not a universe-widening argument. The 0-500 band has ~6 names/month: not interpretable.

**Status: IMPLEMENTED_BUT_FAILED**

## Limitations
Bands are cut on characteristic-month market cap; the 0-500 band is tiny after screens.
