# Asymmetric rank buffer, L1 turnover penalty, partial rebalance and EMA-smoothed signals on the composite (PORTFOLIO_BUFFER)

Implementation: `portfolio_buffer.py` (run: `.venv/bin/python experiments/portfolio_buffer/portfolio_buffer.py`, about 1 minute).

## Objective
Test whether spending the turnover budget through an entry/hold rank buffer ("buy 10 / hold 50"), an L1 trade penalty, partial rebalancing, or an EMA-smoothed / Garleanu-Pedersen-lite aim signal raises the composite's net IR at equal or lower turnover, compared with the hard 10% one-way turnover cap used today.

## Hypothesis
A hard cap keeps stale names by LP feasibility rather than by design; spending the same turnover on the names whose rank change is most informative should raise net IR. Pre-registered rule (docs/REDDIT_RESEARCH.md sec 2.1): adopt a variant if net IR >= control + 0.10 at equal or lower turnover **and** the paired monthly net-return difference vs the control has t >= 1.

## Research origin
docs/REDDIT_RESEARCH.md sec 2.1 (Blitz et al., FAJ 2023: on MSCI World constituents "buy 10 / hold 50" keeps net alpha above 6% while "buy 20 / hold 20" loses two-thirds to costs; r/quant threads 1rh2h0p / 1693f4p on no-trade bands and rank buffering; Garleanu-Pedersen 2013 "aim in front of the target") and docs/NEW.md sec 3.9 items 3-4 (cost-aware objective, signal smoothing).

## Implementation
`lp2.py`-style redefinition of the LP with cfg keys `enter`/`hold` (a name may be long only if its score percentile is in the top `enter`, or it is already held long and still in the top `hold`; mirror for shorts; if the LP is infeasible the buffer is widened stepwise and this is logged as `buffer_months_widened`), `l1_lambda` (objective term lambda * sum|w - w_prev_drifted| replacing / complementing the hard cap), `trade_frac` (w = f * w_LP + (1 - f) * w_prev, gross renormalised to 2; leaves a small formation-beta residual, reported), and `ema_scores` (per-stock EMA of the composite over months with half-life 1/2/3 months, warm-started from 2015). Variants are listed in `V` inside `main()`. The `*_w2` variants use a 2% per-name cap because at the 1% cap a 10% entry zone (about 120 names) is barely feasible (the buffer had to widen in 37-62 of 68 months for the 10/x variants).

## Experimental setup
Common harness (a verbatim copy of the data / rank-transform / LP / cost / performance code of `experiments/largecap/largecap.py`, embedded so the script runs on its own; the LP functions `_solve_lp` / `build_portfolio` are redefined in the script with optional extra behaviours and reproduce the frozen LP when the options are off).

- **Signal**: the frozen composite (18 factors in 7 economic groups, equal weight, re-ranked in the universe each month; no fitting). It reproduces `experiments/largecap`: universe rank IC 0.0366 (t 1.91), `lc_t10` gross IR 0.615 / net IR 0.541, 2025 net return -12.4%.
- **Universe**: price >= $5, market cap >= $2,000M, 126-day dollar volume >= $10M, both betas observed (about 1,206 stocks per month); 68 test months 2021-01..2026-08.
- **Portfolio**: the frozen LP (dollar-neutral, neutral to `beta_60m` and `betabab_1260d`, net sector <= 5% of NAV, gross sector share <= 35%, gross 200%, per-name cap 1%, 10% one-way turnover cap = `lc_t10`), with tiered *assumed* trading and borrow costs (5/10/20 bp trading and 30/75/200 bp borrow by market-cap tier; not measured).
- **Caveat that applies to every number below**: a single 68-month path; the standard error of a single-path IR is about 0.46, so IR differences under ~0.5 are not distinguishable from noise on their own. Paired monthly net-return t-stats are reported next to IR; with 20+ variants per file, a few |t| > 1 are expected by chance.

Signal universe IC (unchanged by construction except through smoothing): raw 0.0366, EMA half-life 1 month 0.0359, 2 months 0.0348, 3 months 0.0346 (smoothing lowers IC slightly, as the r/quant threads say).

## Baseline
`ctrl_t10` (the frozen `lc_t10`: hard 10% cap, 1% per-name cap): net IR 0.541, one-way turnover 10.0%. Because the free / 2%-cap variants have different turnover or concentration, the matching controls `ctrl_free`, `ctrl_t10_w2`, `ctrl_free_w2` are in the same table.

## Commands
```
.venv/bin/python experiments/portfolio_buffer/portfolio_buffer.py
```

## Results (68 months, single path)
| variant | IR gross | IR net | d IR net | paired t | turnover % | cost bp/m | beta | max |beta| formation | max DD % | 2025 net | buffer widened (months) | adopt_rule |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ctrl_t10 | 0.615 | 0.541 | +0.000 |  | 10.0 | 3.6 | -0.061 | 0.000 | -19 | -0.124 |  | False |
| ctrl_free | 0.572 | 0.433 | -0.108 | -0.76 | 34.3 | 11.4 | -0.120 | 0.000 | -25 | -0.164 |  | False |
| buf_10_50_free | 0.480 | 0.374 | -0.167 | -1.29 | 19.7 | 6.7 | -0.044 | 0.000 | -25 | -0.173 | 37 | False |
| buf_10_30_free | 0.559 | 0.448 | -0.093 | -0.71 | 21.9 | 7.4 | -0.061 | 0.000 | -25 | -0.173 | 62 | False |
| buf_20_50_free | 0.457 | 0.327 | -0.214 | -1.81 | 27.9 | 9.4 | -0.085 | 0.000 | -27 | -0.175 | 0 | False |
| buf_10_50_t10 | 0.549 | 0.469 | -0.072 | -1.00 | 10.0 | 3.5 | -0.024 | 0.000 | -17 | -0.121 | 41 | False |
| l1_0.02_free | 0.628 | 0.516 | -0.025 | -0.05 | 24.6 | 8.3 | -0.110 | 0.000 | -23 | -0.148 |  | False |
| l1_0.05_free | 0.568 | 0.477 | -0.064 | -0.47 | 17.0 | 5.8 | -0.096 | 0.000 | -23 | -0.154 |  | False |
| l1_0.10_free | 0.541 | 0.466 | -0.076 | -0.96 | 11.3 | 4.0 | -0.098 | 0.000 | -22 | -0.153 |  | False |
| l1_0.20_free | 0.678 | 0.611 | +0.070 | +0.28 | 6.5 | 2.4 | -0.035 | 0.000 | -16 | -0.109 |  | False |
| partial50_free | 0.550 | 0.449 | -0.092 | -0.85 | 20.7 | 6.9 | -0.092 | 0.132 | -22 | -0.140 |  | False |
| partial50_buf_10_50 | 0.304 | 0.213 | -0.328 | -2.27 | 14.9 | 5.0 | -0.014 | 0.101 | -27 | -0.175 | 2 | False |
| ctrl_t10_w2 | 0.783 | 0.724 | +0.183 | +1.85 | 10.0 | 3.6 | -0.090 | 0.000 | -32 | -0.216 |  | True |
| ctrl_free_w2 | 0.895 | 0.767 | +0.226 | +1.86 | 39.0 | 13.1 | -0.179 | 0.000 | -30 | -0.188 |  | False |
| buf_10_50_free_w2 | 0.661 | 0.546 | +0.005 | +0.39 | 28.4 | 9.7 | -0.120 | 0.000 | -30 | -0.180 | 2 | False |
| buf_10_50_t10_w2 | 0.338 | 0.274 | -0.267 | -1.47 | 10.0 | 3.6 | -0.070 | 0.000 | -35 | -0.214 | 15 | False |
| ema1_t10 | 0.524 | 0.451 | -0.090 | -1.10 | 10.0 | 3.6 | -0.060 | 0.000 | -23 | -0.167 |  | False |
| ema2_t10 | 0.480 | 0.409 | -0.132 | -1.13 | 10.0 | 3.6 | -0.074 | 0.000 | -24 | -0.154 |  | False |
| ema3_t10 | 0.535 | 0.465 | -0.076 | -0.41 | 10.0 | 3.6 | -0.070 | 0.000 | -21 | -0.139 |  | False |
| ema2_free | 0.544 | 0.450 | -0.092 | -0.54 | 19.0 | 6.6 | -0.077 | 0.000 | -22 | -0.104 |  | False |
| ema2_buf_10_50_free | 0.415 | 0.318 | -0.223 | -1.51 | 15.2 | 5.3 | -0.026 | 0.000 | -22 | -0.170 | 45 | False |
| ema2_partial50_free | 0.477 | 0.398 | -0.143 | -1.15 | 13.5 | 4.7 | -0.052 | 0.142 | -22 | -0.113 |  | False |

Net returns in the crowded-unwind months are in `output/stress_windows_net_return.csv`; the composite lost 4.9% (Jun-2025) and 4.3% (Jul-2025) and made +8.4% in Jul-2026 in the control book; no variant changes that picture materially.

## Interpretation
**Nothing in the turnover-design family beats the hard cap.** Every buffer variant has *lower* net IR than the matching control (`buf_10_50_t10` -0.07, `buf_10_50_free` vs `ctrl_free`: 0.374 vs 0.433) while cutting turnover only from 34% to 20% in the uncapped case; L1 penalties change net IR by -0.08 to +0.07 (paired t -1.0 to +0.3); partial rebalancing by -0.09 (t -0.85), and -0.33 (t -2.3) when combined with the buffer; EMA smoothing lowers IR (-0.08 to -0.13 with the 10% cap, t -0.4 to -1.1) because it lowers IC. The best L1 variant (`l1_0.20_free`, +0.07 net IR at 6.5% turnover, paired t 0.28, max drawdown -16% vs -19%) is well inside noise. The hard 10% cap itself is a better regulariser than a free book (net IR 0.541 vs 0.433 for `ctrl_free`, paired t -0.76) - the opposite of the "hard cap is by feasibility, not design" hypothesis - because the composite's signal is slow (rank autocorrelation 0.92) and the free LP trades on rank noise.
The only row that meets the pre-registered rule is `ctrl_t10_w2` (+0.18 net IR, paired t 1.85), but that is a **2% per-name cap** (concentration), not a turnover design, and its 2025 return and max drawdown are worse (-21.6%, -31.6% vs -12.4%, -18.9%); it is evaluated on its own in `experiments/portfolio_concentration`. With 21 variants tested, one t > 1.8 is not evidence.
**Status: IMPLEMENTED_BUT_FAILED for the buffer / L1 / partial / EMA / Garleanu-Pedersen-lite family** (the rule fired only for a concentration variant, which is a different idea).

## Limitations
- One 68-month path and IR s.e. ~0.46; a +0.1 IR threshold is below the noise floor, so this experiment can only reject large effects.
- Blitz's signals turn over ~1,800% a year; the composite turns over far less, so the buffer's cost saving is small by construction (docs/REDDIT_RESEARCH.md sec 2.1 anticipated this).
- The Garleanu-Pedersen-lite arm uses EMA of the *same* composite as its aim, not separate 1/3/6-month horizon models, so it is a proxy.
- Costs are assumed tiers, not measured.

## Follow-up
None on the composite. The buffer / smoothing ideas may matter for a faster signal (e.g. the Blitz short-term composite, which itself failed on IC in `experiments/feat_blitz`).
