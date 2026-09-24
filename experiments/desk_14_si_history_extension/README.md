# desk_14 — FINRA short-interest history extension (data) + fresh OOS test of days-to-cover and the SI cap

**Hypotheses (pre-registered in the script docstring before running):**
- H1: −rank(dtc) has positive IC on months no one has seen.
- H2: dtc as an 8th group raises composite IC.
- H3: the 10% SI cap does not hurt (paired t > −1).

**Data:** FINRA's CDN serves mid-month short-interest files from 2018-01; earlier files return 403. The 29 files for 2018-01..2020-05 were downloaded on 2026-09-24 into `cache/finra_short_interest/`. The logic is the same as `fiam_desks/si.py` (`fiam_research/si_ext.py`) and is identical on the overlap. Universe coverage is 100% of rows from 2018.
Both components were adopted on 2020-06..2026 data, so target months 2018-02..2020-06 (29 months) are genuinely out of sample for them. This is a backward-in-time replication. TEST was not touched.

## Results (`output/results.json`)
| test | result | verdict |
|---|---|---|
| H1: dtc block IC, fresh 29 months | −0.0009 (t −0.08); residual IC −0.0010 | **not replicated** |
| H2: paired IC gain of the 8th group | +0.0042 (t +0.76) | inconclusive |
| H3: SI cap vs no cap (composite+dtc book), paired monthly net | **+0.53%/month (t +2.12, NW t +1.98)** | **holds, and helps** |
| SI > 10% minus rest, next-month return | +0.72%/month (t 1.2) | high-SI names *outperformed* here (squeeze risk) |

DEV books (71 months, net IR):

| book | net IR |
|---|---:|
| composite, no cap | −0.872 |
| composite + extended cap | **−0.648** |
| B0 frozen | −0.757 |
| composite+dtc, extended cap | −0.711 |
| composite+dtc, no cap | −1.003 |

## Interpretation
The **SI cap is confirmed as a risk control on unseen data**. Its mechanism is avoiding squeezes of heavily shorted names, not alpha. **Days-to-cover is not confirmed:** its TEST-period pass (feat_short_interest) did not replicate in 2018–2020. With 29 months the power is low (minimum detectable IC around 0.035), so this is "no evidence" rather than "refuted". It should not be counted as a validated component.
**Carry forward:** use the extended SI in all research. Keep the cap. Treat dtc as unconfirmed (tested again in the signal-construction experiments).
