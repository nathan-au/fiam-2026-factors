# Total Portfolio Approach, Factors-Only Adaptation (TPA)

Implementation: `tpa.py` (run with `.venv/bin/python tpa.py` for the $2B book and `--floor 1000` for the $1B sensitivity; about 2 minutes each). New file; imports the frozen harness in `et.py` and the pre-registered definitions in `largecap.py`; edits nothing. Outputs in `output/`: `tpa_results_<tag>.json`, `tpa_summary_<tag>.csv`, `tpa_weights_<tag>.csv` (sleeve weights and risk contributions per test year), `tpa_monthly_ic_<tag>.csv`, `tpa_run_<tag>.log`, `oos_predictions_<tag>_tpa_<arm>.csv`, `portfolio_{holdings,returns}_<variant>_<tag>_tpa_<arm>.csv`. No 8-K sleeve yet; that would plug in as an eighth sleeve.

**Idea.** TPA manages one total portfolio against one objective and gives each opportunity capital according to its marginal contribution to total risk and return, instead of equal or fixed buckets. We hold a single long/short equity book, so the "sleeves" are the seven pre-registered factor groups of `docs/LARGECAP.md`. Question: does allocating sleeve capital by contribution to total risk or risk-adjusted return beat the equal-group composite?

## Design (pre-registered in the script header before any result was seen)

| Element | Choice |
|---|---|
| Universe, factors, LP | Exactly `largecap.py`'s: $2B floor (sensitivity $1B), the same 18 factors and 7 groups, the same four LP variants; headline `lc_t10`. |
| Sleeve signal | Group score = mean of the group's signed within-universe monthly ranks, centred within month and scaled to mean \|score\| = 1 (a unit of sleeve capital is a unit of gross exposure). |
| Sleeve return | Return of a unit-gross, score-weighted long/short sleeve in each target month (not beta-neutral; used only for allocation). |
| Estimation | Each test year uses **only** sleeve returns with target month before that year (47 months for 2021 up to 107 for 2026). Weights are fixed within the year. |
| Arms | `tpa_eq` equal capital 1/7 (control). `tpa_erc` equal **risk** contribution, Ledoit-Wolf covariance, no return forecasts. `tpa_ms` long-only max-Sharpe on sleeve returns, means shrunk 50% toward the cross-sleeve mean, Ledoit-Wolf covariance, each sleeve ≤ 35%. |
| Verdict rule | A TPA arm is "useful" only if it beats `tpa_eq` on net IR in **both** `lc_t10` and `lc_free` **and** the paired monthly universe-IC difference has t ≥ 2; beating on IR with t < 2 is "suggestive only". Written expectation before running: 1/N is hard to beat with 47–107 months for 7 sleeves. |

The control checks out: `tpa_eq` reproduces `largecap.py`'s composite (universe IC 0.0368 vs 0.0366; `lc_t10` IR 0.59 gross / 0.52 net vs 0.61 / 0.54).

## Why equal capital is not equal risk

Sleeve share of total risk under `tpa_eq` at $2B (Ledoit-Wolf covariance): volatility/beta 33% (2021) and 30% (2026), profitability 21% / 17%, quality 21% / 16%, while investment/issuance/accruals contributes only 2% / 7% and surprise 9% / 5%. The equal-capital composite is therefore mostly a volatility/beta, profitability and quality bet. `tpa_erc` moves capital toward the low-volatility sleeves (2026 weights: investment 21%, surprise 25%, liquidity 15%, value 11%, quality 11%, profitability 11%, volatility/beta 6%) to equalise risk at 14.3% each.

## Results, $2B floor (2021-01 – 2026-08, 68 months; net of assumed costs unless "gross")

| Arm | Portfolio | IR gross | IR net | Sharpe net | CAGR net | β (t) | Rolling-12m β | Max DD net |
|---|---|---:|---:|---:|---:|---:|---|---:|
| tpa_eq | lc_t10 | 0.59 | 0.52 | 0.82 | 10.7% | −0.09 (−0.84) | −0.64 / +0.29 | −22.6% |
| tpa_eq | lc_free | 0.55 | 0.42 | 0.71 | 9.4% | −0.13 (−1.13) | −0.66 / +0.25 | −24.8% |
| tpa_erc | lc_t10 | 0.67 | 0.60 | 0.90 | 12.0% | −0.13 (−1.21) | −0.69 / +0.19 | −24.5% |
| tpa_erc | lc_free | 0.68 | 0.56 | 0.84 | 11.4% | −0.18 (−1.63) | −0.65 / +0.13 | −25.2% |
| tpa_ms | lc_t10 | −0.05 | −0.15 | 0.24 | 2.0% | 0.00 (+0.05) | −0.45 / +0.50 | −24.8% |
| tpa_ms | lc_free | 0.14 | −0.02 | 0.35 | 3.3% | −0.09 (−0.97) | −0.38 / +0.27 | −26.2% |

`lc_t20` and `lc_t10_w05` rows are in `output/tpa_summary_lc.csv` and tell the same story (`tpa_erc` net IR 0.60 / 0.52 vs `tpa_eq` 0.48 / 0.47; `tpa_ms` −0.03 / −0.28).

| Arm | Universe rank IC (t) | D10 − D1 (%/month) | IC by year 2021 / 22 / 23 / 24 / 25 / 26 | Paired IC vs `tpa_eq` |
|---|---:|---:|---|---|
| tpa_eq | 0.0368 (2.0) | +0.57 | .079 / .075 / .011 / .056 / −.026 / .022 | — |
| tpa_erc | 0.0379 (2.4) | +0.76 | .089 / .077 / .011 / .041 / −.025 / .033 | +0.001 (t 0.21) |
| tpa_ms | 0.0191 (1.8) | +0.41 | .053 / .032 / .004 / .027 / −.020 / .018 | −0.018 (t −1.49) |

Sleeve weights (`tpa_ms`, $2B): capital piles onto profitability, surprise and (later) investment, hitting the 35% cap on surprise in every year, and value gets 0 in every year, because value's pre-2021 sleeve Sharpe was −0.64 and investment's −0.59.

## Sensitivity, $1B floor

| Arm | lc_t10 IR gross / net | lc_free IR gross / net | Universe rank IC (t) | Paired IC vs `tpa_eq` |
|---|---:|---:|---:|---|
| tpa_eq | 0.56 / 0.47 | 0.52 / 0.36 | 0.0440 (2.3) | — |
| tpa_erc | 0.76 / 0.67 | 0.69 / 0.54 | 0.0440 (2.7) | +0.000 (t 0.00) |
| tpa_ms | 0.74 / 0.65 | 0.67 / 0.51 | 0.0302 (2.3) | −0.014 (t −1.22) |

At $1B `tpa_ms` fell back to equal weights for 2021 (no sleeve had a positive shrunk mean) and afterwards put 35% each on investment and surprise.

## Verdict against the pre-registered rule

| Arm | Beats `tpa_eq` on net IR in lc_t10 and lc_free? | Paired IC t ≥ 2? | Verdict |
|---|---|---|---|
| tpa_erc | yes at both floors ($2B: 0.60 vs 0.52 and 0.56 vs 0.42; $1B: 0.67 vs 0.47 and 0.54 vs 0.36) | no (0.21 at $2B, 0.00 at $1B) | **suggestive only** |
| tpa_ms | no at $2B (−0.15, −0.02); yes at $1B (0.65, 0.51) | no (negative at both) | **not useful** (inconsistent across floors) |

## Findings

1. **Equal-risk sleeve budgeting (`tpa_erc`) improves the portfolio numbers, but not the signal.** Its universe rank IC is statistically identical to equal capital (paired t 0.0–0.2), while its net IR is higher in every LP variant at both floors, so the gain is portfolio-level (a more balanced, lower-noise composite) rather than better stock selection. That is exactly what the rule calls suggestive; with one path and 68 months, IR differences of 0.1–0.2 are inside the noise (IR standard error about 0.46).
2. **Return-based allocation (`tpa_ms`) fails, and it fails for an informative reason.** Sleeve returns before 2021 pointed at the wrong sleeves: value and investment had negative pre-2021 Sharpe (−0.64, −0.59) yet were the best-performing groups afterwards (value IC +0.047, investment +0.022 in `docs/LARGECAP.md`). Estimating expected returns from a short history of factor sleeves chases what worked before and misses regime shifts; this is the sleeve-level version of `docs/FACTOR_FILTER.md` §2 (past factor IC does not persist).
3. **Risk-only allocation is the defensible half of TPA here.** It uses no return forecast, is stable year to year (weights move by a few points), and its framing is coherent for the deck: "equal capital across factor groups was 30–33% volatility/beta risk; we budget risk, not capital".
4. **Neutrality is a little looser with `tpa_erc`.** Realised beta is −0.13 to −0.18 (t −1.2 to −1.6 at $2B; −1.8 at $1B, `lc_free`), not statistically different from zero but larger in magnitude than the equal-capital composite's (t −0.8 to −1.1); rolling-12m beta stays within about −0.7 / +0.2 with no window above 1.
5. **What it does not change.** The composite's weaknesses stay: strong 2021–22 and 2024, negative 2025 for every arm, no significant IC. TPA does not add information.

## Limitations

- One path, 68 months; sleeve weights come from 47–107 monthly observations of 7 sleeves, so allocation error is large by construction.
- Sleeve returns are not beta-neutral and are computed on the whole universe, so they measure the raw group premium, not what the constrained LP book earns from that group.
- Shrinkage constants (50% on means, 35% cap) and Ledoit-Wolf were set in advance and not tuned; `tpa_ms` might behave differently with stronger shrinkage, which would move it toward `tpa_eq`.
- The IR comparison across LP variants is not independent (same signals, same months); "beats in every variant" is one result, not eight.
- No 8-K sleeve; TPA's stronger case (allocating between genuinely different information sources) cannot be tested until that block exists.
