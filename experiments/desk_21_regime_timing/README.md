# desk_21 — Causal regime rules (factor momentum, inverse-vol weights, dispersion gross scaling)

**Motivation:** desk_17 (the DEV loss is multi-year style drawdowns; dispersion predicts IC), and desk_16 L2/L3/L6/L7. All parameters were fixed a priori (12-month factor-momentum lookback, 24-month vol window, 80th-percentile dispersion threshold on an expanding window). Everything is causal (`fiam_research/timing.py`). DEV only; 8 variants.

| variant | net IR | max DD | paired t | Δ 2016–17 | Δ 2018–20 | decision |
|---|---:|---:|---:|---:|---:|---|
| base a (B1 groups) | −0.668 | −24.4% | – | | | reference |
| factor momentum, drop losers | −0.785 | −21.7% | −0.32 | − | + | no evidence |
| factor momentum, half weight | −0.798 | −23.7% | −1.02 | − | + | rejected |
| inverse-vol weights | −0.602 | −19.3% | +0.95 | 0 | + | no evidence |
| dispersion gross ×0.5 | −0.600 | −16.7% | +1.27 | − | + | no evidence |
| base b (JKP13 z) | −0.386 | −13.9% | – | | | reference |
| factor momentum, drop | −0.199 | −10.6% | +0.40 | − | + | no evidence |
| factor momentum, half | −0.304 | −12.8% | +0.25 | − | + | no evidence |
| inverse-vol | −0.307 | −14.3% | +0.32 | + | + | no evidence |
| dispersion gross ×0.5 | −0.490 | −11.8% | −0.30 | + | − | no evidence |

**Interpretation:**
- Nothing passes. Every timing rule helps a little in 2018–20 (the value crash) and hurts or does nothing in 2016–17, which is the classic in-regime pattern of timing.
- The dispersion rule fired in 22 of 71 DEV months, mostly in 2019–20, because the expanding threshold keeps firing once dispersion trends up.
- These results match the literature's cautions (Asness et al. 2017; Cederburg et al. 2020): a predictable IC does not translate into a better book after the costs of changing exposure.
- **The 2015–2020 degradation cannot be mitigated by causal, low-parameter timing rules in a way this sample can detect.**
