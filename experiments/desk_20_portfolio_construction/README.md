# desk_20 — Portfolio construction with the signal fixed

**Question (the brief's item 7):** with prediction held constant, does another construction extract more?
**Implementation:** `fiam_research/construct.py` (cvxpy/Clarabel QP with the same constraints, screens, SI cap, relax ladder, drift accounting and cost model as the LP, plus a causal factor risk model: 7 style exposures × a 36-month covariance of their FM returns, and idiosyncratic vol from `ivol_capm_252d`). Script `desk_20_portfolio_construction.py` (37 s, parallel). DEV only. Six pre-registered alternatives, each run on two signals (V0 = B1, V3 = JKP13 z). Promotion requires both signals to pass.

| arm | V0 net IR | V0 max DD | V0 paired t | V3 net IR | V3 paired t | transfer coef (V0) | notes |
|---|---:|---:|---:|---:|---:|---:|---|
| C0 frozen LP, SI cap 10% | −0.668 | −24.4% | – | −0.386 | – | 0.57 | reference |
| C1 caps ∝ 1/idio-vol | −0.848 | −22.1% | −0.38 | −0.456 | −0.06 | 0.53 | |
| C2 name cap 0.5% (400+ names) | −0.968 | −19.5% | −0.06 | −0.627 | −0.71 | 0.66 | higher TC, lower IR |
| C3 risk-aware QP (λ=20) | −1.250 | −26.6% | −1.12 | −0.776 | −0.96 | 0.52 | turnover budget binds; 19–68 relaxed months; fails constraint report |
| C4 score-tracking QP | −0.838 | −19.6% | +0.23 | −0.551 | −0.51 | 0.69 | >500 names in some months |
| **C5 SI cap 5%** | **−0.132** | **−8.1%** | **+3.19** | −0.430 | −0.24 | 0.51 | only effect that survives BH-FDR (desk_26) |
| C6 SI cap 15% | −0.984 | −35.8% | −3.12 | −0.621 | −1.91 | 0.60 | rejected |

**Interpretation:**
- The LP is not the bottleneck. Higher transfer coefficients (C2, C4) do not raise IR, because the extra names carry a weak signal (IC ≈ 0.01).
- Risk-aware QP reduces style tilts (quality 0.92 → 0.54, low-vol 0.35 → 0.12) but loses IR and fights the turnover budget. On DEV the style tilts were not the only loss driver.
- **The SI-cap level matters, and monotonically for V0** (15% → −0.98, 10% → −0.67, 5% → −0.13). For V3 it is flat. The dtc group tilts V0's shorts toward high-SI names, and those squeezed in 2018–20. A tighter cap undoes that.
- Formally C5 is "no evidence": it fails on V3, and the rule requires both signals. It is the only DEV result that survives multiple-testing correction, so it goes to the pre-registered confirmation as a hypothesis.
- **Rejected:** the QP family; wider SI cap. **No evidence:** vol-scaled caps, 0.5% cap.
