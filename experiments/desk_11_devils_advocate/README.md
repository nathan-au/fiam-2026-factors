# Deterministic devil's advocate on the assembled system (DESK_11_DEVILS_ADVOCATE)

Implementation: `desk_11_devils_advocate.py --confirm` (~2.5 min). Output: `output/results.json`. Descriptive: nothing here selected or changed a configuration (ledgered as such).

## Attacks (each with its failure criterion fixed in the script) and results on the default system, test window
| # | attack | result | verdict |
|---|---|---|---|
| 1 | time stability | calendar-year net returns 2021 +30.3%, 2022 +17.0%, 2023 +8.1%, 2024 +13.9%, **2025 -1.0%, 2026 (8m) -0.2%**; net IR 0.98 in 2021-01..2023-06 vs **0.21** in 2023-07..2026-08; leave-one-year-out 0.41-0.80 | **FAIL** (2 negative years; fading) |
| 2 | time concentration | without the 3 best months (2021-03, 2022-05, 2021-07) net IR +0.29 | pass (barely) |
| 3 | factor attribution | regress net excess return on market + 6 universe quintile-spread proxies (size, value, momentum, low-vol, quality, profitability): **alpha +3.0%/yr, t +1.07; R2 0.75**; loadings value +0.69 (t 7.4), quality +0.46 (t 3.6), profitability +0.37 (t 3.0), momentum +0.29 (t 4.9), mkt +0.24 (t 3.9) | **FAIL** (return explained by known style exposures) |
| 4 | random-score placebo | 30 within-(month, sector) shuffles of the factor score through the same LP/costs: -1.28 +- 0.54 net IR (max -0.42); share >= real 0.00 | pass (a low bar: random books churn against the 10% cap and pay costs) |
| 5 | bootstrap (block 4, 5000 draws) | default net IR 0.64, **90% CI [0.14, 1.16]**; every text mode's IR difference vs default has a CI containing 0: blend +0.03 [-0.12, +0.18], veto_long +0.09 [-0.05, +0.26], tilt -0.01, dial -0.15 [-0.49, +0.13], judge +0.02 | text modes NOT distinguishable |
| 6 | regime disclosure | same system on 2015-2020: net IR -0.76 (desk_10) | see below |
| 7 | text-lane claim, pooled | novneg_max IC dev +0.0071 (t 1.63), test +0.0103 (t 2.87), pooled +0.0087 (t 3.06); novelty_max pooled +0.0033 (t 1.12); abrupt_exit 0.0000 | descriptive |

## Interpretation
The headline result is **mostly a style bet**: value/quality/profitability exposure explains three quarters of the variance and the residual alpha is not distinguishable from zero (t 1.07), the return has faded since mid-2023, and the same construction lost money in 2015-2020. None of that is hidden by the PM or the text desk. These belong in the deck's limitations page. The devil's advocate did not move any setting.

**Status: IMPLEMENTED_AND_TESTED (two of four fixed criteria FAIL)**

## Limitations
Style proxies are our own quintile spreads on the universe, not Fama-French/Barra factors; 68 observations with 7 regressors; the random-score placebo is easy to beat because of turnover-cap churn.
