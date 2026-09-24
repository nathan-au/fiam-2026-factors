# desk_19 — Signal construction: literature-signed JKP themes vs the frozen composite

**Hypothesis (desk_16 L1/L5/L10/L14):** a composite of all 13 JKP themes, with the *published* long directions (no fitting), is more robust than the project's 18-factor, 7-group composite.
**Implementation:** `fiam_research/themes.py` (cluster map and directions from `cache/jkp/`, from github.com/bkelly-lab/ReplicationCrisis; 146 of 147 panel characteristics map; `intrinsic_value` = JKP `ival_me`), `desk_19_signal_construction.py` (35 s).
**Data:** DEV only. Pre-registered family of 5 vs V0 = B1. The rule is in the script docstring.

| variant | IC (t) | IC 2015–17 / 2018–20 | paired IC gain (t) | style-resid IC (t) | net IR | Sharpe | max DD | paired net t | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| V0 B1 (7 groups + dtc) | +0.0131 (0.87) | +0.045 / −0.018 | – | +0.005 (0.61) | −0.668 | −0.15 | −24.4% | – | reference |
| V1 frozen 7 | +0.0108 (0.70) | +0.045 / −0.022 | −0.002 (−0.91) | +0.003 | −0.599 | −0.11 | −24.4% | +0.40 | no evidence |
| V2 JKP13 ranks | +0.0066 (0.48) | +0.020 / −0.006 | −0.006 (−0.64) | +0.005 | −0.502 | +0.07 | −17.6% | +0.64 | no evidence |
| **V3 JKP13 z-scores** | +0.0062 (0.43) | +0.018 / −0.005 | −0.007 (−0.69) | +0.007 (0.91) | **−0.386** | **+0.21** | **−13.9%** | +0.85 | no evidence |
| V4 JKP13 + dtc | +0.0070 | +0.020 / −0.005 | −0.006 | +0.006 | −0.692 | −0.09 | −22.5% | +0.25 | no evidence |
| V5 all 146 equal | +0.0036 | +0.022 / −0.014 | −0.009 (−1.24) | +0.001 | −0.579 | −0.10 | −26.5% | +0.19 | rejected |

Theme IC on DEV (`theme_ic_by_year_dev.csv`): Quality +0.026 (t 1.6), Low Leverage +0.026 (1.8), Profitability +0.014, Short-Term Reversal +0.013. Wrong-signed: Seasonality −0.030 (t −2.9), Value −0.020, Investment −0.018.

**Interpretation:** no fit-free construction has a DEV IC distinguishable from zero. The JKP composites give a steadier book: a lower value tilt (−0.29 holdings exposure vs −0.09), more momentum (+0.36), and max DD 14% vs 24%. But the gain is not statistically detectable, and it comes precisely from being less exposed to value, which lost on DEV. It is not selected. **Carry forward:** V3 as a robustness signal in desk_20/21, and as a fit-free confirmation arm in desk_27.
