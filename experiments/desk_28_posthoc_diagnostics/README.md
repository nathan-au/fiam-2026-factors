# desk_28 — POST-HOC diagnostics of the desk_27 confirmation (no selection, no new TEST arm)

## P1. A5 (JKP13 z) vs A0 (B1) over all 139 months
| | A5 | A0 |
|---|---:|---:|
| net Sharpe (excess of T-bill), 2015–2026 | **0.80** | 0.51 |
| DEV / TEST Sharpe | +0.21 / +1.29 | −0.15 / +0.99 |
| annual mean excess | 6.5% | 5.2% |
| annual vol | **8.1%** | 10.2% |
| correlation of monthly returns | 0.64 | |

Sharpe difference +0.29, **90% block-bootstrap CI [−0.14, +0.73]**, P(≤0) = 0.13. **Not statistically distinguishable**, but same-signed in both regimes.

## P2. Where A5's lower risk comes from
| | holdings exposure: value / momentum / low-vol / quality / profitability | style-regression alpha t (R²) |
|---|---|---|
| A0 DEV | −0.09 / +0.11 / **+0.35** / +0.92 / +0.47 | 1.26 (0.66) |
| A0 TEST | 0.00 / +0.16 / **+0.32** / +0.94 / +0.53 | 0.80 (0.76) |
| A5 DEV | −0.29 / **+0.36** / +0.10 / +0.73 / +0.63 | **1.75** (0.62) |
| A5 TEST | −0.15 / **+0.36** / +0.05 / +0.71 / +0.66 | **1.89** (0.47) |

A5 drops the low-vol tilt, carries momentum (JKP Momentum includes residual momentum), and has less quality concentration. Its returns are *less* explained by the style proxies in both periods (R² 0.47 vs 0.76 on TEST), and its style-alpha t is larger in both.

## P3. Why the 5% SI cap reversed
Shorts with SI in (5%, 10%] (the names A1 bars) are **~50–58% of the A0 short book's gross**.
- DEV: they cost the book −21% (2019) and −22% (2020). The worst were RNG, SNAP, XLRN, NTRA, FIVN: growth/tech winners of the 2019–20 boom.
- TEST: they earned +16% in 2022. The best included RNG, COUP, KOD, BE, FIVN: the same kind of names in the 2022 growth crash.

The SI-cap flip, the SI-conditional IC flip (desk_24 I1 / S4) and the value-group flip are **one phenomenon, the growth-vs-value cycle**. The cap level was really a growth/value bet in disguise.

## P4. Regime-flip table (`output/regime_flip_table.csv`)
| finding | DEV | TEST | verdict |
|---|---|---|---|
| composite IC | +0.013 | +0.045 | same sign, 3.5× larger on TEST |
| value group IC | −0.016 | +0.05 | **flip** |
| dtc IC | −0.001 (fresh) | +0.035 | **flip / not replicated** |
| B1 IC high-SI minus low-SI | −0.050 (t −2.5) | +0.050 (t +3.0) | **flip, significant both ways** |
| SI cap 5% vs 10% | t +3.19 | t −2.00 | **flip** |
| SI cap 10% vs none | t +2.12 (fresh) | DD −18.9% → −9.8% | same (risk control) |
| novneg IC, $0.5–2B band | −0.022 | +0.038 | **flip** |
| novneg → idiosyncratic risk | t 4.65 | t 4.71 | **same (robust)** |
| monotone GBM IC | +0.023 | +0.017 | same sign, weak |
| JKP13z vs B1 Sharpe | +0.21 vs −0.15 | +1.29 vs +0.99 | **same (A5 better)** |
| dispersion → IC | t −5.4 | t −3.1 | same |
