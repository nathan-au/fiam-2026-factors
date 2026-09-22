# Conformal selection of names with an FDR bound (CONFORMAL_SELECT)

Implementation: `conformal_select.py` (run: `.venv/bin/python experiments/conformal_select/conformal_select.py`, about 30 seconds).

## Objective
Toy test of conformal selection (Jin-Candes style: conformal p-values against a calibration null + Benjamini-Hochberg) as a theoretically calibrated way to choose the long / short set and abstain, using the frozen composite.

## Hypothesis
Selecting names whose composite score has a small conformal p-value against past "non-winner" names, with BH at level q, yields a set whose false-discovery share (picks not in the winning 30% of next-month returns) is at or below q, and that beats a same-size top-k by score. Kill: realised FDR > q in the test window (exchangeability broken by regime change), or no return advantage over same-size top-k.

## Research origin
docs/RESEARCH.md Part II H5 (own hypothesis: model-agnostic conformal selection as a calibrated "abstain"; needs a calibration set of past months; the note itself calls for a toy test first). Adjacent literature: conformal selection framework; Conformal Predictive Portfolio Selection (arXiv 2410.16333) is portfolio-level, not name-level.

## Implementation
The composite needs no fitting, so all universe months before the test year (72 months before 2021) are calibration data. win = next-month excess return in the top 30% of the universe cross-section (long); lose = bottom 30% (short, score sign flipped). Calibration null = past units that are not winners (resp. losers). p_i = (1 + #{null scores >= s_i}) / (n_null + 1); BH at q in {0.60, 0.65, 0.70} per test month. The base win rate is 30%, so informative q are close to 1 - 0.30 = 0.70. Compared with the top-k of the same size k.

## Experimental setup
Common harness (a verbatim copy of the data / rank-transform / LP / cost / performance code of `experiments/largecap/largecap.py`, embedded so the script runs on its own; the LP functions `_solve_lp` / `build_portfolio` are redefined in the script with optional extra behaviours and reproduce the frozen LP when the options are off).

- **Signal**: the frozen composite (18 factors in 7 economic groups, equal weight, re-ranked in the universe each month; no fitting). It reproduces `experiments/largecap`: universe rank IC 0.0366 (t 1.91), `lc_t10` gross IR 0.615 / net IR 0.541, 2025 net return -12.4%.
- **Universe**: price >= $5, market cap >= $2,000M, 126-day dollar volume >= $10M, both betas observed (about 1,206 stocks per month); 68 test months 2021-01..2026-08.
- **Portfolio**: the frozen LP (dollar-neutral, neutral to `beta_60m` and `betabab_1260d`, net sector <= 5% of NAV, gross sector share <= 35%, gross 200%, per-name cap 1%, 10% one-way turnover cap = `lc_t10`), with tiered *assumed* trading and borrow costs (5/10/20 bp trading and 30/75/200 bp borrow by market-cap tier; not measured).
- **Caveat that applies to every number below**: a single 68-month path; the standard error of a single-path IR is about 0.46, so IR differences under ~0.5 are not distinguishable from noise on their own. Paired monthly net-return t-stats are reported next to IR; with 20+ variants per file, a few |t| > 1 are expected by chance.

## Baseline
Same-size top-k by raw score (no calibration), and the 30% base win rate.

## Commands
```
.venv/bin/python experiments/conformal_select/conformal_select.py
```

## Results (68 test months, single path)
| side | target FDR q | months_total | months with a selection | avg names selected | win rate of selected | realised FDR | base_rate_win | win rate, same-size top-k | mean ret selected %/m | mean ret unselected %/m | mean ret same-size top-k %/m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| long | 0.60 | 68 | 51 | 40.2 | 0.306 | 0.694 | 0.300 | 0.306 | +0.86 | +0.94 | +0.86 |
| long | 0.65 | 68 | 53 | 47.3 | 0.296 | 0.704 | 0.300 | 0.296 | +1.05 | +0.81 | +1.05 |
| long | 0.70 | 68 | 56 | 54.4 | 0.282 | 0.718 | 0.300 | 0.282 | +0.61 | +0.85 | +0.61 |
| short | 0.60 | 68 | 57 | 62.5 | 0.450 | 0.550 | 0.300 | 0.450 | +0.66 | -0.91 | +0.66 |
| short | 0.65 | 68 | 60 | 81.8 | 0.436 | 0.564 | 0.300 | 0.436 | -0.63 | -0.92 | -0.63 |
| short | 0.70 | 68 | 62 | 103.3 | 0.435 | 0.565 | 0.300 | 0.435 | -0.48 | -0.93 | -0.48 |

("mean ret" is the position-signed mean next-month excess return of the set, in %/month: for the short side it is minus the stock return, so the universe average is about -0.9%/month because the universe rose; compare selected with unselected, not with zero.)

## Interpretation
Two findings. (1) **Conformal selection is exactly top-k with a data-adaptive k**: the p-value is monotone in the score, so BH selects the k highest-scoring names, and the "same-size top-k" columns are identical to the selected sets (same win rate, same return). It therefore adds **no ranking value**; what it adds is the choice of k and the option to abstain (selection in 51-62 of the 68 months). (2) **Calibration is honest on the short side and fails on the long side**: on the short side the realised FDR (0.55-0.57) is below every target q (0.60-0.70) and the win rate of selected shorts (0.43-0.45) is well above the 0.30 base rate, i.e. the composite's worst-ranked names really are more likely to be bottom-30% returns; on the long side the win rate of the selected (0.28-0.31) is at the base rate, so the realised FDR (0.69-0.72) exceeds the q = 0.60 and 0.65 targets (0.70 target slightly exceeded too): the composite's *top* ranks carry no win-rate edge and calibration on 2015-2020 does not transfer. The composite's alpha is on the short side in win-rate terms (consistent with the 30-bucket table in `experiments/portfolio_concentration`, where only the bottom two buckets stand out). Kill conditions: FDR control fails for longs; there is no gain over same-size top-k by construction. Status: IMPLEMENTED_BUT_FAILED (long-side FDR not controlled; no ranking gain over top-k by construction; not adopted).

## Limitations
- Exchangeability between calibration months and test months does not hold (regime change), as the long-side failure shows; conformal guarantees are for exchangeable data.
- The "win" definition (top 30% of the cross-section) is one choice; the FDR level is tied to it.
- Toy test: no portfolio was built from the selected sets; a 68-month path.

## Follow-up
Not pursued. The sensible use of the finding is descriptive: the composite's edge is on avoiding / shorting the worst names.
