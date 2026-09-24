# Factors desk candidates and the dev-period baseline (DESK_05_FACTORS_DESK_DEV)

Implementation: `desk_05_factors_desk_dev.py` (~20 s). Outputs: `output/candidates_dev.csv`, `output/lp_dev.csv`, `output/results.json`. Dev only.

## Objective / what changed / why
The frozen composite anchors the system, so it was run on the development months for the first time. Candidates (declared before running, Bonferroni 3): `rev_group` (8th group = mean(-rank ret_1_0, -rank ret_60_12)), `rev_1m` (ret_1_0 only), `sector_neutral` (composite ranked within GICS sector; replacement). Adoption rule: PASS on dev to enter the default; inconclusive -> carried as an option to the single confirmation; kill -> dropped.

## Results (dev, 71 months)
Descriptive dev IC by group: value -0.016 (t -0.9), profitability +0.020 (t +1.7), investment/issuance/accruals -0.012 (t -1.2), quality +0.007, surprise +0.012, vol/beta +0.010, liquidity +0.009; composite +0.0106 (t +0.68) vs +0.0366 on 2021-26.

| candidate | IC alone (t) | paired gain (t) | perm p | verdict |
|---|---:|---:|---:|---|
| rev_group | -0.0080 (-0.48) | -0.0021 (-0.40) | 0.97 | kill |
| rev_1m | +0.0222 (+1.27) | +0.0087 (+1.15) | 0.005 (Bonf 0.015) | inconclusive |
| sector_neutral | +0.0084 (+0.60) | -0.0023 (-0.41) | - | kill |

**LP books on dev (lc_t10, fresh start): composite gross IR -0.77 / net -0.87, max DD -36.9%, calendar years 2015 +2.6%, 2016 +13.8%, 2017 +0.7%, 2018 -0.7%, 2019 -8.2%, 2020 -29.9%** (short leg -49.7% in 2020: the shorts are high-idiosyncratic-vol names, 67th percentile on average). rev_1m book -0.97 net (paired net t -1.04); sector_neutral -1.04 (t -1.12).

## Interpretation
**Key finding (uncomfortable):** the system's anchor loses money on the 2015-2020 development window although its IC is slightly positive. The 2021-26 result (net IR 0.54) is therefore regime-dependent evidence, not a stable property. **Worked:** the reversal decomposition (only the 1-month piece is informative). **Failed:** long-term reversal, sector-neutral ranking. **Carry forward:** `rev_1m` as an option to the confirmation (it later adds nothing: desk_09); the dev loss -> desk_06, desk_07 (is it the PM or the signal?).

**Status: IMPLEMENTED_AND_TESTED (baseline finding); candidates: 2 killed, 1 inconclusive**

## Limitations
Reversal evidence came from a desk_03 control (exploratory); LP single-path IR s.e. ~0.46.
