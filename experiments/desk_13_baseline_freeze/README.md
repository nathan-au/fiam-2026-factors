# desk_13 — Baseline freeze (B0)

**Hypothesis:** none. This freezes the reference system every later experiment is compared with.
**Implementation:** `desk_13_baseline_freeze.py` (~12 s) calls the new research harness `fiam_research/core.py`, which imports `fiam_desks/` and never modifies it. `baseline_config.json` records the git commit (6e145be), the full configuration and the SHA-256 of every `fiam_desks/*.py`.
**Data:** DEV 2015-02..2020-12 (71 target months), TEST 2021-01..2026-08 (68). TEST numbers were already public in desk_09/10. Re-computing them selects nothing; they are in the ledger as a descriptive freeze.

## B0 = the committed default system
Factors desk = frozen 7-group / 18-factor composite + days-to-cover 8th group (FINRA SI from 2020-06 only). PM = `lc_t10` LP: dollar-neutral, neutral to `beta_60m` and `betabab_1260d`, sector net ≤5%, sector gross ≤35%, 1% name cap, 10% one-way turnover budget, gross 200%. Shorts are allowed only where SI ratio ≤10%. The text desk is advisory. Costs follow the tiered assumption (5/10/20 bp trade, 30/75/200 bp/yr borrow by market cap).

## Metrics (`output/baseline_metrics.json`)
| | DEV | TEST |
|---|---:|---:|
| universe IC (t) | +0.0111 (0.72) | +0.0455 (2.39) |
| net IR / gross IR | −0.757 / −0.642 | +0.637 / +0.721 |
| net Sharpe | −0.30 | +0.97 |
| bootstrap 90% CI net IR (block 4) | [−1.50, −0.04] | [+0.14, +1.16] |
| max DD (net) | −30.0% | −9.8% |
| beta vs panel VW market (t) | −0.17 (−2.7) | −0.03 (−0.3) |
| one-way turnover / positions | 10.1% / 202–243 | 10.1% / 203–248 |
| trade + borrow cost | 3.6 + 4.8 bp/month | 3.5 + 4.9 bp/month |
| long / short leg contribution (ann.) | +11.9% / −13.5% | +12.7% / −0.1% |
| calendar years (net) | 2015 +2.6, 2016 +13.8, 2017 +0.7, 2018 −0.7, 2019 −8.2, 2020 −22.2 | 2021 +30.3, 2022 +17.0, 2023 +8.1, 2024 +13.9, 2025 −1.0, 2026 −0.2 |
| style regression alpha (t), R² | +1.8%/yr (0.77), 0.71 | +2.0%/yr (0.67), 0.75 |
| holdings exposure (weighted rank) | quality +0.95, profitability +0.51, low-vol +0.38, investment +0.33, value −0.09 | quality +0.94, profitability +0.54, low-vol +0.32, value −0.00 |

By regime (DEV): the book **gains in down-market months (+0.80%/month) and loses in up months (−0.60%)**. It is defensive, with realised beta −0.17 on DEV even though it is beta-neutral at formation. On TEST the gains are spread across regimes.

## Weaknesses identified
1. Returns are explained by style exposures: the style alpha has t < 1 in both periods.
2. DEV loses money, and the bootstrap interval excludes 0 on the negative side.
3. Realised DEV beta is negative: "beta-neutral at formation" does not mean neutral ex post for a low-vol/quality book.
4. dtc and the SI cap were adopted on TEST data (checked in desk_14).
5. The panel drops delisting rows before forming the universe (fixed in desk_15).
6. The FRED S&P 500 series starts 2016-09, so the harness's DEV beta and alpha silently skip 2015-02..2016-09. The research harness therefore also reports beta against a value-weighted market built from the panel.

**Carry forward:** B0 is the frozen reference. B1 (desk_15) is the corrected research baseline.
