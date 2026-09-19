# Random Forest, Round 2 — Hardening the Modern Arm

Implementation: `rf_2.py` (run with `.venv/bin/python rf_2.py`; `--stages a,b` for a
subset, `--smoke` for a fast plumbing test). Follows up `docs/RF.md`, where the
Modern arm (15 characteristics) was the best result in the project (IR 1.05).
`rf.py` and its outputs are untouched; everything here is `rf2_*` /
`*_rf2_modern`. Full numbers: `output/rf2_results.json` and one CSV per stage.

**Bottom line.** Modern is a real, positive result — every one of 15 grid
configurations, 5 seeds, and 6 beta/feature variants has a positive IR — but two
things in `docs/RF.md` do not survive stress-testing:

1. **The curated 15-factor set is not special.** 15 *random* characteristics
   give a mean IR of 1.02 (Modern: 1.04–1.20 depending on config/seed). What is
   unusual is the **Graham** set, which is far *worse* than random, not Modern
   being better.
2. **The Graham failure is not a short-squeeze story.** Graham's signal is ~zero
   on *both* sides (details in §5), so the explanation offered in `docs/RF.md`
   is not needed and is not supported.

The honest headline number is **IR ≈ 1.0–1.1 gross** (about 0.9 after a 10 bp
one-way cost), not 1.05–1.20 as a point estimate.

Period for everything: 01/2021–08/2026 (68 monthly OOS returns), **gross of
transaction costs unless a table says otherwise**. Beta-neutral LP on `beta_60m`,
same as every other script.

---

## 1. Wider grid — no gain, and the tuning is noise

Grid widened to `max_depth ∈ {4,6,8,10,12}` × `min_samples_leaf ∈ {50,100,200}`
(200 trees, `max_features=sqrt`, seed 42). The original two configs reproduce
`rf.py` exactly (IR 1.0452, beta t −0.2075 in both), so the harness is unchanged.

| Run | IR | Sharpe | CAGR | α t | Beta (t) | OOS R² | Max DD |
|---|---:|---:|---:|---:|---:|---:|---:|
| rf.py grid (2 configs), tuned per fold | 1.045 | 1.21 | 30.0% | 2.84 | −0.041 (−0.21) | 0.123% | −22.6% |
| **Wide grid (15), tuned per fold** | **1.037** | 1.20 | 29.9% | 2.78 | 0.002 (0.01) | 0.067% | −28.2% |
| Best fixed config `d8_l200`, seed 42 | 1.199 | 1.36 | 34.5% | 3.18 | −0.019 (−0.10) | 0.155% | −22.6% |

- Validation MSE is **identical to four decimals (0.0661) for all 15 configs**:
  the validation window cannot tell configurations apart, so per-fold "tuning"
  is close to picking at random (the picks jump between depth 4 and 12).
- Run at fixed hyperparameters, the 15 configs span **IR 0.85–1.21 (mean 1.03)**,
  all positive, all with |beta t| < 0.5 (`rf2_grid_fixed_configs.csv`). Depth/leaf
  moves IR by more than any plausible signal, with no monotone pattern.
- `d8_l200` (IR 1.20) was picked as "best mean validation MSE", a distinction
  that is noise-level. **Do not headline 1.20.**
- Seed noise (fixed config, 5 seeds, 200 trees): IR 1.06–1.20, mean 1.11, sd 0.055;
  1000 trees gives 1.12. The seed-42 figure sits at the top of its own range.
  Treat **±0.11 (2 sd) as the noise band** on any single IR below.

## 2. Is the win broad-based? Yes — with QMJ carrying the most

Impurity importance (tuned wide-grid models, fold-averaged): top factor `qmj`
13.9%, top 3 34.7%, **effective number of factors 12.4 of 15**. By group: QMJ
29%, "other" (betabab, asset growth, gross profitability) 18%, lottery/vol 17%,
momentum 17%, mispricing 10%, surprise 9%. Not concentrated.

Leave-one-out (fixed `d8_l200`, baseline IR 1.199; seed-mean 1.11):

| Dropped | IR | | Dropped group | IR |
|---|---:|---|---|---:|
| `mispricing_perf` | **0.86** | | **QMJ (3)** | **0.77** |
| `betabab_1260d` | 0.99 | | surprise (2) | 0.93 |
| `gp_at` | 0.99 | | "other" (3) | 1.09 |
| `niq_su` | 1.00 | | mispricing (2) | 1.07 |
| `qmj_growth` | 1.01 | | lottery/vol (2) | 1.18 |
| `at_gr1` | 1.04 | | momentum (3) | **1.28** |
| the other 9 factors | 1.06–1.25 | | | |

Read against the ±0.11 noise band: no single factor is load-bearing (worst case
`mispricing_perf`, IR still 0.86), the **QMJ group** and **earnings-surprise
group** matter most, and **momentum is not needed** — dropping it *raises* IR.
Full table: `rf2_leave_one_out.csv`.

## 3. Feature-set sensitivity — the important negative finding

| Test (fixed config, 15 draws each) | IR mean | sd | min | max |
|---|---:|---:|---:|---:|
| Modern (seed mean / seed-42) | 1.11 / 1.20 | 0.055 (seeds) | 1.06 | 1.20 |
| **2-for-2 random swaps** vs the other 132 factors | 1.03 | 0.15 | 0.60 | 1.21 |
| **15 random characteristics** (null) | **1.02** | 0.24 | 0.61 | 1.42 |

- 3 of 15 random sets (20%) matched or beat Modern's 1.20; 7 of 15 are above 1.0.
- Rank IC tells the same story: Modern 0.095, random sets 0.063–0.105. Graham 0.005.
- Swaps can hurt: one swap (`qmj`, `saleq_su` → `rd_me`, `capx_gr1`) drops IR to
  0.60, consistent with QMJ/surprise being the useful groups.

**What this means.** Most reasonable 15-factor sets produce IR near 1 in this
harness and sample; Modern's curation adds at most a modest edge (~+0.1 vs the
null mean, inside the null's spread). The IR of ~1 therefore says more about the
walk-forward RF + LP long/short construction on this 2021–2026 universe than
about the specific factors. The Graham arm (IR −0.23, IC 0.005) is *below every one
of 15 random draws*, so the notable result is that the Graham set is unusually
bad. Caveats: n=15 draws with one seed each (sd 0.055 from seeds alone), and the
random sets overlap Modern's factors by chance.

## 4. Beta control — the regression vs full-147 RF is noise; betabab does not help

Beta SE on all variants is ≈ 0.2, so "t = −0.21 vs t = 0.01" is a difference of
0.04 in beta — a fifth of one standard error. There is nothing to explain.
Variants (fixed config; `rf2_beta_variants.csv`):

| Variant | IR | Realized beta (t) | Max DD |
|---|---:|---:|---:|
| Baseline: LP on `beta_60m` | 1.20 | −0.019 (−0.10) | −22.6% |
| LP on **`betabab_1260d`** (docs/NEXT.md §2) | 0.91 | +0.056 (0.27) | −35.0% |
| LP on both betas | 1.03 | +0.082 (0.42) | −27.5% |
| + `beta_60m` as a feature | 1.00 | −0.128 (−0.68) | −21.2% |
| + beta family (4 factors) | 1.25 | +0.079 (0.40) | −23.2% |
| + beta family, LP on both | 1.13 | +0.135 (0.69) | −24.1% |

- **The `betabab_1260d` swap does not improve realized beta** (0.056 vs −0.019, both
  ≈ 0) and costs IR and drawdown. Keep `beta_60m` for the LP constraint.
- Adding beta-family factors moves realized beta in both directions; nothing
  systematic. Modern already contains `betabab_1260d` as a feature.
- The book is beta_60m-neutral by construction (0.000 at formation) but carries
  |betabab exposure| ≈ 0.23 on average, and realized beta is still ≈ 0.
- **Neutral on average, not throughout.** The rolling 12-month beta in the deck
  chart ranges from **−0.78 to +1.59** (the +1.59 is the first, noisiest window;
  after that roughly −0.4 to +0.6, ending near −0.8 in mid-2026). FIAM.md §3 says
  this chart is the evidence of neutrality "throughout"; a reviewer will see it.

## 5. Graham failure — long-only check kills the squeeze explanation

`rf2_long_only.csv` (both arms on rf.py's 2-config grid; same investable screen as
the LP; top/bottom 100 names equal-weighted, which is what a capped long-only LP
returns):

| | Graham | Modern |
|---|---:|---:|
| Top-100 **long-only** CAGR (excess of T-bill) | **−2.8%** | 10.1% |
| Equal-weight universe CAGR | 5.3% | 5.3% |
| Top-100 minus universe (annual, t) | −6.5% (−1.05) | +4.8% (1.29) |
| Bottom-100 minus universe (annual) | −6.0% | −19.8% |
| Top minus bottom 100 (annual, t) | −0.5% (−0.04) | 24.6% (2.06) |
| Decile spread D10−D1 (annual, t) | −3.0% (−0.31) | 19.3% (1.92) |
| LP book long-leg / short-leg CAGR | −3.8% / −4.7% | 12.4% / 10.4% |
| Rank IC | 0.005 | 0.095 |

`docs/RF.md` proposed that Graham loses because its **short** leg gets squeezed
in a book that the paper's long-only design avoids. The test says otherwise:
Graham's **long-only** book *also* loses to the universe (−6.5%/yr, insignificant)
and to T-bills, the top-minus-bottom spread is zero, and IC is zero. The signal
simply has no power in this sample on either side. Both legs of the LP book lose
(−3.8% long, −4.7% short). The paper's result therefore does not replicate even
long-only here — with the caveats that the paper's window (Mar 2022–Mar 2026)
differs and the Graham/Modern mapping is this project's own (`docs/RF.md`
limitation 1). **`docs/RF.md`'s explanation paragraph should be revised;** I have
not edited that file.

Side finding for Modern: the edge is **short-heavy**. Top-100 long-only beats the
universe by +4.8%/yr (t 1.3) while the bottom-100 trails it by −19.8%/yr; the
spread is significant (t 2.1) mostly because of the bottom side.

## 6. Turnover and transaction costs

The book is almost fully replaced each month: traded notional **2.03× capital per
month** (gross book = 2.0×; 24× per year), i.e. rf.py's "turnover" of 0.47
(range 0.28–0.83) — this is Σ|Δw| with prior weights drifted by realized returns.
Costs are applied per $ traded, including the initial build; borrow is a flat
rate on the 1.0 short notional. Numbers are for the tuned wide-grid book
(gross IR 1.04), `rf2_costs.csv`:

| One-way cost | IR | Sharpe | CAGR | α t | Max DD |
|---:|---:|---:|---:|---:|---:|
| 0 bp (gross) | 1.04 | 1.20 | 29.9% | 2.78 | −28.2% |
| 5 bp | 0.99 | 1.15 | 28.3% | 2.66 | −28.4% |
| 10 bp | 0.94 | 1.10 | 26.8% | 2.55 | −28.6% |
| 20 bp | 0.84 | 1.00 | 23.8% | 2.32 | −28.9% |
| 30 bp | 0.74 | 0.90 | 20.9% | 2.09 | −29.2% |
| 50 bp | 0.54 | 0.70 | 15.2% | 1.63 | −29.9% |

Borrow cost of 50 / 200 bp p.a. subtracts about 0.02 / 0.08 IR at any trading
cost. Break-even one-way cost (net return = the 4% hurdle) is ≈ **105 bp**.

**Where this understates the risk.** A flat 0–200 bp borrow rate is not what this
short book pays. Short-book stats: average market cap $5.6B but **median $1.3B,
62% of shorts below $2B, 43% below $1B**; the ten largest average shorts are
IOVA, NTLA, AMC, MULN, NKLA, CYTK, MDGL, WOLF, GERN, LAZR — small, volatile,
meme/distressed/biotech names where borrow can cost many times 200 bp or be
unavailable. FIAM.md §2 says the short book "will be read with this in mind". Not
modeled here; my judgment (not measured) is that a realistic all-in cost for this
basket is above the 5–10 bp typical of large caps. Sector/liquidity-tiered costs
or a small-cap short exclusion would be the natural next test.

## 7. Where the return comes from (from the deck exhibits)

- Positive in every calendar year (strategy excess return): 2021 +20.2%, 2022
  +48.3%, 2023 +21.8%, 2024 +12.6%, 2025 +30.8%, 2026 (8 mo) +37.6%.
- **2022 alone is ~27% of total log return**, driven by the short leg (+65% that
  year; the short leg *lost* 13.7% in 2021).
- **Max drawdown (−28.2%) is entirely the first two OOS months**: peak
  12/2020, trough 02/2021, recovered 11/2021. Worst month is 01/2021 (−25.4%) —
  consistent with the 2021 meme-stock squeeze hitting the short book (not
  verified name-by-name).
- **Best month 06/2026 (+18.1%).** Dropping the best 3 months takes IR from 1.04
  to **0.78**. Rolling 12-month IR ranges 0.11–2.67 (`rolling_12m.csv`).
- Long-leg and short-leg CAGR are similar (11.1% / 11.4%) — but see §5 for the
  bottom-side dominance in the sort-based test.
- Largest single P&L contributor is **MULN (+13.7%, shorted while collapsing)**;
  largest detractor is **AMC (−5.6%)**; both sit in the top-10 shorts. Other top
  contributors are similar small/distressed names (LAZR, XELA, FCEL, NOVA, FFIE,
  CLVS). Labels come from the panel's ticker/company_name; **one top-10
  contributor has no ticker or name in the panel** (shown as "(no ticker)" in
  `top10_pnl_contributors.csv`), and FIAM.md §12's naming rule still requires
  verifying historical labels against an authoritative source (not done).

## 8. Deck exhibits generated for the Modern book

Written to `output/rf2_deck/` (tuned wide-grid book; strategy total return =
TB3MS + excess return, stated explicitly; S&P 500 is the FRED price index, not
total return):

| FIAM.md §12 item | File |
|---|---|
| Calendar-year table (strategy excess & total / benchmark / S&P / legs) | `calendar_year_returns.csv` |
| Top-10 long / top-10 short by average weight | `top10_long_short_avg_weight.csv` |
| 10 largest positive / negative P&L contributors | `top10_pnl_contributors.csv`, `pnl_contributors.png` |
| Rolling 12m active return, IR, beta | `rolling_12m.csv`, `rolling_active_ir.png`, `rolling_beta.png` |
| Cumulative vs benchmark vs S&P | `cumulative_return.png` |
| Underwater vs S&P | `underwater.png` |
| Histogram of monthly returns with hurdle | `return_histogram.png` |
| Headline stats (IR, Sharpe, α/β with SE, exposures, concentration, turnover range, short-book stats, best/worst month, legs) | `rf2_results.json` → `grid.wide_grid_tuned_full` |

**Pre-existing inconsistency (all scripts):** `calendar_year_returns` in every
results JSON compounds the strategy's *excess* return next to a *total-return*
benchmark and S&P. `calendar_year_returns.csv` here shows both `strategy_excess`
and `strategy_total` so the deck can pick the consistent one.

## 9. Decision point

- **Keep Modern as a candidate; do not present it as "the curated factors beat the
  paper".** Defensible claims: (i) gross IR ≈ 1.0–1.1 with alpha t ≈ 2.8–3.2 and
  near-zero realized beta, positive in every year and every configuration tested;
  (ii) ≈ 0.9 IR after 10 bp; (iii) robust to dropping any single factor.
- **Not defensible:** that this particular 15-set is uniquely good (random-15 null),
  the 1.20 config number, or that a wider grid helps.
- **Deck risks to address head-on:** short-heavy edge in small/volatile names
  (borrow), 2022 and 3 best months carry a lot of the result, max drawdown
  concentrated in Jan–Feb 2021, and rolling beta not stable around zero.
- I did **not** compare against E2E / Poly / other scripts here, so "which is the
  headline model" is still open; the cross-model comparison should use the same
  net-of-cost, seed-averaged basis used above.

## Limitations

1. Random-15 and swap tests use 15 draws each with a single seed — enough to see
   the mean is ≈ 1, not to estimate tails.
2. Fixed-config robustness stages use `d8_l200`, chosen by noise-level validation
   differences; the 15-config table shows the result does not hinge on it.
3. Transaction-cost model is flat per $ traded with drifted weights; no market
   impact, no name-level borrow. Costs applied to the tuned wide-grid book, not
   the fixed-config book.
4. The wide-grid tuned book (used for costs and deck exhibits) has a different
   max drawdown (−28.2%) than the rf.py book (−22.6%); which grid to present is
   a choice, and IR is statistically the same (1.04 vs 1.05).
5. LOO, swaps, and beta variants use a fixed config with 200 trees; differences
   under ≈ 0.11 IR are indistinguishable from seed noise.
