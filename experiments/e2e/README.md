# End-to-End Weight Learning — SPSA (Methodology and Results)

Implementation: `e2e.py` (run with `.venv/bin/python experiments/e2e/e2e.py`). Operationalizes
*AlphaZeroBeta: Deep Reinforcement Learning for Market-Neutral Portfolios*
(arXiv 2607.18001, Jul 2026 — `docs/RESEARCH.md` Part III §2).

## Fidelity to the source paper

The paper's actual system: a **CNN-GRU policy trained end-to-end via
Recurrent PPO** (a specific policy-gradient RL algorithm), with a
**composite reward balancing risk-adjusted excess return, benchmark
correlation, and transaction costs**, evaluated across seven equity
indices 2014–2024. No CNN, GRU, PPO, or RL of any kind is built here (no
RL/deep-learning infrastructure in this project) — `theta` is a plain
linear vector optimized by SPSA (Simultaneous Perturbation Stochastic
Approximation, Spall 1992), a zeroth-order method needing only objective
evaluations.

**The first version's training objective was Sharpe alone.** This version
implements the paper's actual **reward structure** — the three named
terms — as `composite_objective()`:

```
objective = Sharpe(monthly returns)
          - lambda_tc * avg_turnover                    (transaction costs)
          - lambda_corr * |corr(returns, S&P 500 returns)|  (benchmark correlation)
```

`avg_turnover` is the mean one-way soft-weight turnover between
**consecutive** months, aligned by permno (a name held in one month but not
the next counts as full turnover). The correlation term uses actual
monthly S&P 500 returns (same FRED series used everywhere else in this
project). Training-month sampling was changed from a random *bag* of 12
months to a random **contiguous** 12-month block, since turnover is only
meaningful between adjacent months. `lambda_tc = lambda_corr = 0.5` are
this project's own calibration choices (the paper doesn't specify weights)
— see Limitations.

Still no RL, no neural network, no CNN-GRU — only the reward's *structure*
(three named terms) now matches the paper; the optimizer and function class
remain unrelated to PPO/CNN-GRU.

## Results

| Fold | OLS warm-start val objective | Best SPSA val objective | Improved? |
|---|---:|---:|---|
| 2021 | −3.42 | −3.42 | No |
| 2022 | −3.58 | −3.54 | **Yes** |
| 2023 | −3.67 | −3.26 | **Yes** |
| 2024 | −3.48 | −3.34 | **Yes** |
| 2025 | −3.51 | −3.41 | **Yes** |
| 2026 | −3.24 | −3.19 | **Yes** |

**5 of 6 folds now show improvement over the OLS warm start** (up from 2
of 6 with the Sharpe-only objective) — the composite objective's extra
structure gives SPSA more to work with, and the optimization converges more
often. Objective values are large and negative (≈−3.2 to −3.7) because the
transaction-cost penalty term dominates in absolute magnitude relative to
Sharpe (≈0–3) at this `lambda_tc` — see Limitations.

| Metric | E2E/SPSA (composite reward) | E2E/SPSA (Sharpe-only, first version) | OLS (baseline) |
|---|---:|---:|---:|
| **OOS R²** | **+0.068%** | +0.066% | −0.0086% |
| Information Ratio | 0.79 | 0.82 | 0.85 |
| Sharpe ratio | 0.93 | 0.96 | 0.98 |
| CAGR | 25.3% | 26.0% | 27.6% |
| Alpha t-stat | 2.33 | 2.35 | 2.41 |
| Realized beta (t-stat) | −0.21 (t=−0.88) | −0.16 (t=−0.66) | −0.17 (t=−0.67) |
| Hit rate | 64.7% | 63.2% | 61.8% |
| Avg monthly turnover | 38.2% | — | 39.7% |

**Still the only model besides `poly.py` with positive pooled OOS R²**, and
essentially unchanged from the first version despite far more folds now
converging — 5 of 6 improving over warm-start doesn't move final portfolio
metrics much beyond what 2 of 6 already achieved. Realized turnover (38.2%)
is nearly identical to OLS's (39.7%) despite explicitly penalizing turnover
in training — the smooth training-time turnover proxy apparently doesn't
translate into materially different real turnover once scores are run
through the actual hard LP each month (which re-optimizes fresh regardless
of small `theta` differences).

## Limitations

1. **`lambda_tc = lambda_corr = 0.5` were not tuned** — chosen once,
   unvalidated. Given the transaction-cost term's absolute magnitude
   dominates Sharpe's in the objective (order 3 vs. order 0–3), the
   optimizer may be shaped more by turnover-avoidance than by the
   "balance" the paper's phrase implies. A validation-based sweep over
   `lambda_tc`/`lambda_corr` (not attempted here) would be the natural
   next step.
2. **Composite reward structure matches the paper; optimizer and function
   class still don't** — no RL, no CNN-GRU, no PPO. This closes one real
   gap (reward shape) while leaving the larger ones (RL, deep network,
   multi-index evaluation) untouched.
3. **SPSA hyperparameters (iteration count, gain sequence, perturbation
   scale) were not re-tuned** for the new objective — same values as the
   first version, which used a differently-scaled objective (Sharpe alone,
   range roughly −1 to 3) than the new one (composite, range roughly −4 to
   0). The gain-sequence pilot calibration (`spsa_train`'s pilot step)
   adapts to this automatically, but the fixed iteration count/checkpoint
   schedule was not revisited.
4. **The positive R² remains small in absolute terms** and, as before, is
   concentrated in specific folds rather than a uniform improvement — worth
   treating as a genuinely interesting result to investigate further, not
   as proof the underlying idea reliably beats pointwise fitting.
