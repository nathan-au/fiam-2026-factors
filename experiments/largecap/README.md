# Large-Cap-Only Strategy, Pre-Registered (LARGECAP)

Implementation: `largecap.py` (run with `.venv/bin/python largecap.py --seeds 5`, about 12 minutes). New file; imports the frozen harness in `et.py` (data, rank transform, folds, LP, costs, performance stats) and edits nothing. Outputs in `output/`: `lc_results.json`, `lc_summary.csv`, `lc_monthly_ic.csv`, `oos_predictions_lc_<arm>.csv`, `portfolio_{holdings,returns}_<variant>_lc_<arm>.csv`, `lc_feature_importance_<arm>.csv`, `lc_run.log`.

**Why.** `docs/PM_ABLATION.md` showed the tree models' edge lives in small-cap shorts, and that a $2B market-cap floor was the one non-negative tradeable row (post-hoc, trained on all stocks). This tests that idea with the design fixed in the script header before any result was seen, and with training/validation restricted to the large-cap universe.

## Pre-registered design (fixed before running; one run, no re-tuning)

| Element | Choice |
|---|---|
| Universe | price ≥ $5, market cap ≥ $2,000M, 126d dollar volume ≥ $10M, both betas observed. ~1,206 stocks/month; 153,392 of 523,125 stock-months. |
| Factors | 18 characteristics picked **by economic group** from `docs/FACTORS.md`, not by IC (`docs/FACTOR_FILTER.md` §2), no momentum: value (`be_me`, `ni_me`, `fcf_me`), profitability (`gp_at`, `ni_be`, `ebit_sale`), investment/issuance/accruals (`at_gr1`, `chcsho_12m`, `oaccruals_at`), quality (`qmj`, `f_score`), surprise (`niq_su`, `saleq_su`), volatility/beta (`ivol_capm_21d`, `rmax5_21d`, `betabab_1260d`), liquidity (`ami_126d`, `turnover_126d`). |
| Arms | `et` Extra-Trees trained + validated on universe rows (headline model); `comp` equal-weight composite of the same 18 factors with pre-specified signs, re-ranked within the universe each month, groups weighted equally, **no fitting**; `et_allrows` same Extra-Trees trained on all stocks (diagnostic). |
| Portfolio | Same LP as `docs/RF_3.md` `pm_*`: dollar-neutral, neutral to both `beta_60m` and `betabab_1260d`, net sector ≤ 5% NAV, gross sector share ≤ 35%, gross 200%. Headline `lc_t10` (10% one-way turnover budget, from the tips). Sensitivities: `lc_free`, `lc_t20`, `lc_t10_w05` (0.5% per-name cap, ≥ 400 names). |
| Costs | Tiered assumptions of `docs/PM_ABLATION.md` §1 (assumed, not measured). |

## Results (2021-01 – 2026-08, 68 months, one path)

| Arm | Portfolio | IR gross | IR net | Sharpe net | CAGR net | β (t) | Rolling-12m β min / max | Max DD net | Short-book median mcap |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| et | lc_t10 (headline) | -0.22 | -0.30 | 0.02 | -0.5% | -0.09 (-0.85) | -0.59 / +0.60 | -35% | $4.7B |
| et | lc_free / t20 / t10_w05 | -0.18 / -0.16 / -0.38 | -0.36 / -0.27 / -0.49 | | | | | | |
| **comp** | **lc_t10 (headline)** | **0.61** | **0.54** | **0.84** | **11.0%** | **-0.06 (-0.55)** | **-0.60 / +0.32** | **-19%** | **$7.3B** |
| comp | lc_free / t20 / t10_w05 | 0.57 / 0.66 / 0.59 | 0.43 / 0.56 / 0.50 | | | | | | |
| et_allrows | lc_t10 | -0.04 | -0.12 | 0.18 | 1.6% | -0.14 (-1.34) | -0.49 / +0.39 | -36% | $5.3B |

Model-level, on universe rows only:

| Arm | Test rank IC (t) | % months > 0 | D10 − D1 (%/month) | OOS R² (universe) | Pred. rank autocorr |
|---|---:|---:|---:|---:|---:|
| et | 0.016 (1.1) | 49% | -0.17 | +0.21% | 0.79 |
| comp | 0.037 (1.9) | 60% | +0.48 | n/a | 0.92 |
| et_allrows | 0.020 (1.5) | 50% | -0.04 | +0.22% | 0.79 |

- Paired monthly IC: `et` − `comp` = −0.020 (t −1.06); `et` − `et_allrows` = −0.004 (t −1.08). Neither is significant, but the direction is against the model in both.
- **Seed noise, `et` headline** (5 seeds): gross IR −0.22, −0.24, −0.39, −0.39, −0.29; universe IC 0.016–0.017. Every seed is negative.
- Composite calendar-year net returns (`lc_t10`): 2021 +35.2%, 2022 +23.0%, 2023 +13.0%, 2024 +5.6%, **2025 −12.4%**, 2026 (8 mo) +3.8%, against T-bill+4% of 4.1 / 6.2 / 9.5 / 9.3 / 8.4 / 5.2%.
- Composite universe IC by year: 2021 +0.077, 2022 +0.077, 2023 +0.006, 2024 +0.062, **2025 −0.026**, 2026 +0.016.
- Composite `lc_t10` legs (gross CAGR): long +13.2%, short −3.9% (shorted names still rose on average). Net alpha t (vs S&P) 2.05; hit rate vs benchmark 56%; positions ~113 long / ~115 short; top-10 share of gross 5%.
- Post-hoc (not pre-registered) per-group universe IC: value +0.047 (t 2.2), investment/issuance/accruals +0.022 (t 2.1), profitability +0.020 (1.6), quality +0.019 (1.5), volatility/beta +0.018 (0.7), liquidity +0.007 (0.6), surprise +0.005 (0.6).

## Findings

1. **The model does not work on large caps; the simple rule does.** Extra-Trees trained on the universe has IC 0.016 and negative IR in every variant and every seed. The equal-weight composite of the same factors has IC 0.037 and IR 0.5–0.66 gross in every variant, with stable beta and a $7B median short book. The composite gets its edge without fitting, so it carries no fitting or tuning selection.
2. **Restricting training to large caps did not rescue the tree model** (`et` vs `et_allrows`: both about zero). Consistent with `docs/ANALOG.md` / `docs/PM_ABLATION.md`: the ceiling is the information in these factors, not the model class.
3. **The composite is not a statistically clean win.** IR standard error over 68 months is about 0.46, so 0.61 is about 1.3 standard errors from zero; universe IC t = 1.9. It is one pre-registered path, and the effect is concentrated in 2021–2022 and 2024, with 2025 negative (−12.4%). Read it as "consistent with a modest value/investment-style tilt", not as proven alpha.
4. **The edge is value and investment/issuance/accruals** (post-hoc group ICs above), not volatility or liquidity, which was the untradeable family in earlier docs. The short leg still loses money; the return comes from the long leg.

## Sensitivity: $1B floor (`--floor 1000`, run after the $2B result; $2B stays the headline, nothing was chosen from this)

Universe ~1,391 stocks/month. Files are tagged `lc1000` (`output/lc1000_results.json`, `lc1000_summary.csv`, `lc1000_run.log`). Arms `et` and `comp` only.

| Arm | Portfolio | IR gross | IR net | Sharpe net | CAGR net | β (t) | Rolling-12m β min / max | Max DD net | Short-book median mcap |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| et | lc_t10 | -0.47 | -0.58 | -0.31 | -5.5% | -0.05 (-0.45) | -0.84 / +0.96 | -38% | $3.3B |
| et | lc_free / t20 / t10_w05 (gross) | -0.43 / -0.31 / -0.51 | | | | | | | |
| comp | lc_t10 | 0.64 | 0.55 | 0.82 | 11.9% | -0.09 (-0.77) | -0.65 / +0.31 | -28% | $4.8B |
| comp | lc_free / t20 / t10_w05 (gross) | 0.65 / 0.58 / 0.54 | | | | | | | |

Universe rank IC: composite 0.044 (t 2.2), Extra-Trees 0.022 (t 1.5); decile spread D10 − D1: composite +0.29, Extra-Trees −0.43 %/month. Composite IC by year: 2021 +0.088, 2022 +0.086, 2023 +0.011, 2024 +0.072, 2025 −0.017, 2026 +0.012 (same shape as at $2B). Paired IC, Extra-Trees − composite: −0.022 (t −1.15).

Reading: the composite result is **not fragile to the floor** (gross IR 0.54–0.65 across all four variants at both $1B and $2B), unlike the all-stock tree models in `docs/PM_ABLATION.md` where $1B was far weaker than $2B. The Extra-Trees model is negative at both floors and worse at $1B. The same caveats apply: one path, IR standard error about 0.46, weak 2025.

## Limitations

- One prediction path, 68 months; no confidence intervals other than the seed spread (ET only; the composite is deterministic).
- Factor set and signs were chosen by me from `docs/FACTORS.md` and the literature before running, but I had read the earlier docs (which report which families were load-bearing), so the choice is not blind. The per-group IC table is post-hoc.
- Costs and borrow are assumed tiers; short-book borrow availability is not modelled (median short is a $7B company, which mitigates but does not remove this).
- Stocks without 5-year beta history are excluded, as everywhere in this project.
- Only two floors were run ($2B pre-registered headline, $1B sensitivity, above). Not run: $500M or a finer sweep.

## Reproduce

```
.venv/bin/python largecap.py --seeds 5                 # arms et, comp, et_allrows at the $2B floor
.venv/bin/python largecap.py --floor 1000 --arms et,comp   # sensitivity (files tagged lc1000)
```
