# desk_30 — The handoff text tables: uses no earlier experiment tested

**Data:** `handoff_text_lane_2026-09-21/data/processed/text_features_v{1,2}.parquet`. These are byte-identical (SHA-256) to `cache/text_lane_2026-09-21/`, which desk_01–12 and desk_23 already used.

**Already tested earlier (not repeated here):**
- the text signals as ranking groups and at other universe floors (desk_01/02);
- persistence, news-gated reversal and tail overlay (desk_03), and the rare-event veto (desk_04);
- the PM modes blend, veto, tilt, dial and judge (desk_08/09);
- item-code metadata and earnings-timing flags (feat_8k_meta, on TEST);
- change-based, filing-type, SI-conditioned and risk roles (desk_23).

**This experiment, DEV only.** Signs were fixed before running; BH-FDR q = 0.10 over 4 one-sided tests.

| test (economic reason) | IC A | IC B | diff (t) | BH |
|---|---:|---:|---:|---|
| X1 surprise group IC: fresh 2.02 (t or t−1) vs stale (PEAD is concentrated right after the announcement) | +0.026 | +0.002 | **+0.025 (2.63)** | **discovery** |
| X2 B1 IC: other names vs names with merger-agreement language in t−5..t (deal price pins the stock) | +0.015 | **−0.039** | **+0.054 (2.89)** | **discovery** |
| X3 B1 IC: no fresh news vs novelty ≥ 0.5 (characteristics stale after news) | +0.015 | +0.005 | +0.010 (1.09) | no |
| X4 book's weighted novneg rank vs next-month abs(book return) | rank corr +0.19 | | (1.61) | discovery at q = 0.10 (p 0.053), weak |

Coverage in the universe on DEV: fresh 2.02 is 50%, merger flag 5.3%, novelty ≥ 0.5 is 24%. Merger-flagged names' next-month return equals the rest (no arb-spread effect visible monthly).

**Pre-registered book rules (all failed):**

| arm | IC | net IR | Sharpe | max DD | paired net t | decision |
|---|---:|---:|---:|---:|---:|---|
| B1 | 0.0131 | −0.668 | −0.15 | −24.4% | – | reference |
| X1b surprise zeroed where stale | 0.0120 | −0.867 | −0.35 | −25.7% | −1.68 | rejected |
| X2b merger names barred | 0.0131 | −0.757 | −0.24 | −25.2% | −1.05 | rejected |
| X3b news names barred | 0.0131 | −0.906 | −0.05 | **−14.7%** | +0.52 | no evidence |

**Second stage (`followup_y1.py`).** Designed *after* X1/X2 passed and X1b/X2b failed, so it is 2 extra trials in the ledger:

| arm | IC (paired t) | net IR | Sharpe | max DD | paired net t | decision |
|---|---:|---:|---:|---:|---:|---|
| Y1 surprise ×2 where fresh | 0.0152 (1.27) | −0.560 | −0.02 | −21.5% | +1.15 | no evidence |
| **Y2 = Y1 + merger names set to no view** | **0.0167 (1.92)** | **−0.433** | **+0.11** | −20.9% | **+2.30** | **promising** (second-stage) |

**Methodological note found here.** IR is measured against T-bill+4%. When returns sit below that hurdle, a *lower-volatility* variant gets a *more negative* IR (X3b: Sharpe improved −0.15 → −0.05 while IR fell). The DEV rules of desk_19–26 used IR, which biased them against risk-reducing variants (e.g. JKP13z). Paired monthly net-return t and Sharpe do not have this problem and are reported alongside from here on.

**Interpretation:**
- The text tables carry real **conditional** information on the tradeable universe. The item-2.02 timing tells when the surprise factor is informative, and merger language tells where the factor view is meaningless.
- That is different from the "text as an alpha signal" framing that failed before.
- The information helps only when used as a *weight on the factor signal*, not as a veto. Y2 goes to the second pre-registered TEST look (desk_32) together with the X1/X2 conditional replications.
