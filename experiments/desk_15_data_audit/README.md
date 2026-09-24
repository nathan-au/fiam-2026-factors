# desk_15 — Data-quality and implementation audit (+ correction C1)

| item | finding | action |
|---|---|---|
| target alignment | exact (desk_00) | none |
| **target availability defines the universe** | the frozen `Panel` drops rows with missing next-month return *before* forming the universe. 445 universe rows (~3/month), **all** the permno's last observation (delisting / acquisition), i.e. the future fact "survives month t+1" | **C1 correction** (below) |
| S&P 500 series | FRED `SP500` starts 2016-09-12, so harness alpha/beta on DEV use 2016-10.. only | research harness adds beta vs a panel value-weighted market |
| missing inputs | value inputs `be_me`/`ni_me`/`fcf_me` 13–17% missing in 2026 (≤4% before), median-filled → rank 0; `f_score` ~10% missing throughout | disclosed (weakens 2026 value signal) |
| universe size | 938–1,304 names/month, grows over time | disclosed |
| SI join | ticker join, 100% universe coverage from 2018 | none |
| sector | GICS missing for 0.2% of universe rows (sector "NA") | none |
| costs | tiered assumptions (not measured); borrow adversely selected for high-SI names, which the SI cap mitigates | disclosed |

## C1: keep delisting rows (y = 0) — decision rule fixed before running: switch panels if |ΔIR| ≥ 0.05 in either period
| | DEV net IR | TEST net IR | held delisting positions (long/short) | stressed IR (long −30%, short +20%) |
|---|---:|---:|---:|---:|
| frozen panel | −0.757 | +0.637 | – | – |
| with delisting rows | **−0.855** | **+0.660** | 43 (13/30) dev, 62 (13/49) test | −1.005 dev / +0.467 test |

**Decision: material → all research from here uses `fiam_research.panel_ext.PanelX` (y = 0).** The book shorts future delisters about 3:1. They are mostly acquisition targets, whose true final-month returns are unknown without CRSP delisting returns, so the stressed row is the relevant risk. Also note that a ~0.3% change of the input rows moves IR by 0.1. desk_18 quantifies this path noise.
**B1 (research baseline)** = composite + dtc (extended SI) + 10% SI cap (extended) on PanelX. DEV net IR −0.711 before C1; see desk_18 onward for B1 numbers.
