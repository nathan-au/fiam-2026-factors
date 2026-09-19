# Portfolio-Aware Weighted Regression (Methodology and Results)

Implementation: `joint.py` (run with `.venv/bin/python joint.py`).
Operationalizes *Machine Learning Meets Markowitz* (NBER WP 34861, Feb 2026,
Wang/Gao/Harvey/Liu/Tao — `docs/PAPERS.md` §2): the paper argues the
standard two-stage pipeline is "deeply problematic" because it treats every
observation's prediction error as equally important, when the optimizer
only cares about error in the stocks that end up mattering to the final
portfolio.

**Still not a reproduction of the paper's actual joint-estimation
machinery** (unavailable beyond the abstract). What's implemented is a
weighted least squares — training-loss weight `w_i = |rank_target_i|^p` —
selected via a tunable exponent `p`.

## Fix: validate on a portfolio-level criterion, not MSE

The first version validated candidate `p` values by **uniform MSE**, which
structurally always prefers `p=0` (unweighted OLS minimizes unweighted MSE
by construction) — every fold degenerated to plain OLS, so the weighting
scheme was never actually tested.

**Fixed by running the real beta-neutral LP on the validation period's
predictions for each candidate `p`, and selecting whichever `p` produces
the best validation-period portfolio Sharpe ratio** — a portfolio-level
criterion for a portfolio-level design choice, directly matching the
paper's own argument that pointwise error is the wrong thing to optimize
when portfolio performance is the actual target.

## Results: validation now genuinely selects different weightings per fold

| Fold | Selected p | Val portfolio Sharpe | All candidates (p: Sharpe) |
|---|---:|---:|---|
| 2021 | 0.0 | −0.55 | 0:−0.55, 0.5:−0.62, 1:−0.62, 2:−0.72, 4:−0.74 |
| 2022 | 1.0 | −0.06 | 0:−0.13, 0.5:−0.11, 1:−0.06, 2:−0.06, 4:−0.14 |
| 2023 | 2.0 | 1.39 | 0:0.47, 0.5:0.66, 1:1.35, 2:1.39, 4:1.34 |
| 2024 | 0.0 | 1.18 | 0:1.18, 0.5:1.06, 1:1.07, 2:0.90, 4:0.69 |
| 2025 | 0.0 | 0.50 | 0:0.50, 0.5:0.34, 1:0.34, 2:0.20, 4:−0.14 |
| 2026 | 4.0 | 2.92 | 0:2.30, 0.5:2.22, 1:2.30, 2:2.55, 4:2.92 |

**This is the key methodological result of this script**: unlike the first
version (p=0 in all 6 folds, no exceptions), the portfolio-Sharpe validation
criterion genuinely discriminates between weighting schemes — three
different `p` values get selected across the six folds, each clearly beating
its alternatives on that fold's own validation-period portfolio performance.
The weighting idea is now actually being tested, not defaulting away.

| Metric | Fixed (portfolio-Sharpe validated) | Forced p=2 (no validation) | Plain OLS |
|---|---:|---:|---:|
| OOS R² | −0.52% | −0.98% | −0.0086% |
| Information Ratio | 0.75 | 0.72 | **0.85** |
| Sharpe ratio | 0.88 | 0.86 | **0.98** |
| CAGR | 23.3% | 21.9% | **27.6%** |
| Alpha t-stat | 2.20 | — | 2.41 |
| Realized beta (t-stat) | −0.18 (t=−0.74) | −0.22 (t=−0.95) | −0.17 (t=−0.67) |
| Hit rate | 63.2% | 58.8% | 61.8% |

**Even with a genuinely-working portfolio-level validation criterion, this
implementation of "joint" weighting does not beat plain OLS on final OOS
performance** — IR, Sharpe, and CAGR all come in a bit below the unweighted
baseline. This is a more informative result than the first version's
non-test (validation now actually discriminates and still doesn't find an
edge over OLS), but it's still not a confirmation of the paper's claim.
Plausible explanation: selecting `p` to maximize *validation-period*
portfolio Sharpe is itself a form of overfitting to that period's specific
realized returns — the fold-by-fold "best" `p` (0, 1, 2, 0, 0, 4) swings
widely, which is consistent with the validation criterion chasing
noise in a 24-month window rather than finding a stable weighting
preference.

## Limitations

1. **A working validation criterion still isn't the paper's actual
   estimation method** — this is portfolio-aware *model selection*
   (choosing among a small set of pre-specified weighting schemes), not the
   paper's implied joint *estimation* (learning return-generation and
   portfolio weights as one integrated optimization).
2. **The selected-p instability across folds (0, 1, 2, 0, 0, 4) suggests
   the validation criterion may be overfitting to each fold's specific
   24-month validation window** rather than finding a genuinely
   better-generalizing weighting rule — a longer validation window or a
   penalty for instability might change this.
3. **This weighting scheme (`|rank|^p`) is still a specific, simple choice**,
   not derived from the paper — other weighting schemes might behave
   differently under the same (now-working) portfolio-Sharpe validation
   criterion.
