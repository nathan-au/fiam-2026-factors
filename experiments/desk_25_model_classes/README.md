# desk_25 — More expressive model classes (walk-forward inside DEV)

Expanding window with annual refits and a 1-month purge. Forecast years 2017–2020 (48 OOS months). Inputs: 146 JKP-signed universe ranks. Target: within-month return rank. **Hyperparameters fixed a priori** (no search → no validation multiple testing). The script's second run only added per-year columns; the variants were ledgered once.

| model | OOS IC (t) | 2017 | 2018 | 2019 | 2020 | paired IC vs J13z (t) | net IR 2017–20 | max DD | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| M0 J13z (fit-free) | +0.002 (0.14) | +0.023 | +0.052 | −0.040 | −0.027 | – | −0.569 | −14.1% | reference |
| M1 B1 (fit-free) | +0.001 (0.03) | +0.056 | +0.033 | −0.007 | −0.080 | −0.002 | −1.513 | −24.9% | reference |
| M2 ridge on 146 (α = 1e4·months) | +0.015 (0.69) | +0.017 | +0.049 | −0.001 | −0.003 | +0.013 (0.87) | −0.038 | −15.0% | no evidence |
| M3 theme ridge, equal-weight prior | +0.002 | = M0 | | | | ≈0 | −0.497 | −13.9% | uninformative (the penalty reproduced the prior) |
| **M4 monotone GBM** (xgboost depth 2, 300 × 0.03, monotone in JKP directions) | **+0.023 (1.41)** | +0.024 | +0.049 | **+0.013** | **+0.005** | +0.021 (1.34) | −0.165 | −10.3% | no evidence (2017–18 gain −0.001) |

**Interpretation:**
- The monotone GBM is the only model with positive IC in every OOS year, including 2019–20 when every fit-free composite was negative.
- It is seed-robust (desk_26: 5 seeds give IC 0.020–0.027, all years positive).
- It misses the pre-registered promotion rule by a hair on the first-half condition, and its book still earns less than the T-bill+4% hurdle (DSR 0.05).
- The monotonicity constraint keeps it auditable: it can learn nonlinear shapes and interactions of depth 2, but it cannot reverse a published sign.

**Carried to desk_27** as the primary model-class hypothesis (H-M4), using a 5-seed average fixed now.
