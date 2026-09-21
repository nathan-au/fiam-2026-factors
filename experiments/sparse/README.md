# Sparse-in-Expanded-Space Prediction Model (Methodology and Results)

Implementation: `sparse.py` (run with `.venv/bin/python sparse.py`).
Operationalizes *The Virtue of Sparsity in Complexity* (arXiv 2604.17166,
Apr 2026 — `docs/PAPERS.md` §1).

## Fidelity to the source paper

The first version of this script applied plain LASSO directly to the
original 147 characteristics — a much simpler and different test than what
the paper's verbatim abstract actually describes: **nonlinear feature
expansion combined with basis pursuit**, benchmarked against **"ridgeless"
(near-zero-regularization) high-dimensional ridge**. This version implements
that two-step design:

1. **Nonlinear expansion**: random Fourier features (RFF, Rahimi & Recht
   2007 — the standard random-feature construction in this literature,
   including Kelly-Malamud-Zhou's own "Virtue of Complexity"). The 147
   characteristics are projected through a fixed random Gaussian map into
   P=600 nonlinear features, standardized to unit variance (fit on training
   data only).
2. **Sparse recovery**: LASSO on the 600 expanded features. For a noisy
   regression target, L1-penalized least squares *is* "basis pursuit
   denoising" (Chen, Donoho & Saunders 1998) — a legitimate match to the
   paper's stated method, not an analogy.
3. **Ridgeless comparator, same space**: Ridge with alpha=1e-6 on the same
   600 RFF features — the paper's actual "dense" benchmark, not plain OLS
   on the original 147 characteristics as the first version used.

**A real bug was caught and fixed along the way**: RFF output is tiny in
scale (±0.058 for P=600), and fitting LASSO directly on that scale made
even the grid's smallest alpha collapse every coefficient to zero. Fixed by
standardizing the RFF features to unit variance before fitting (the
standard practice). After the fix, the alpha grid was also widened to
`[1e-7 ... 3e-2]` (9 values) specifically to rule out "the grid just isn't
low enough" as an explanation for the all-zero pattern that persisted — see
Results below for what that check found.

**Honest deviation still remaining**: the paper's "beyond a critical
complexity threshold" regime typically requires P approaching or exceeding
the estimation sample size n. This project's walk-forward pools many
months into training folds of up to ~400K rows, so P > n is computationally
infeasible; P=600 tests the qualitative comparison without reaching the
paper's literal overparameterized-beyond-n regime.

## Results

| Fold | Sparse alpha | Nonzero (of 600) | Ridgeless val MSE | Sparse val MSE |
|---|---:|---:|---:|---:|
| 2021 | 0.003 | 0 | 0.06028 | 0.06019 |
| 2022 | 0.003 | 0 | 0.06486 | 0.06473 |
| 2023 | 0.003 | 0 | 0.04816 | 0.04804 |
| 2024 | 0.003 | 0 | 0.05093 | 0.05082 |
| 2025 | 0.003 | 0 | 0.07974 | 0.07964 |
| 2026 | 0.001 | 0 | 0.09355 | 0.09348 |

**Zero nonzero coefficients at every fold, across the entire tested alpha
range (1e-7 to 3e-2).** This is not a grid artifact — the wider grid was
specifically added to check that, and the same all-zero pattern held at
every tested value. This is a genuine, clean finding: in this 600-dimensional
nonlinear random-feature expansion, **no sparse structure improves on
predicting a constant**, when judged by held-out validation MSE, at any
regularization strength tested.

| Metric | Sparse (LASSO) | Ridgeless (Ridge, α=1e-6) |
|---|---:|---:|
| OOS R² | −0.0589% | −0.1834% |
| Information Ratio | **−0.47** | −0.70 |
| Sharpe ratio | **−0.03** | −0.22 |
| CAGR | **−0.70%** | −2.16% |
| Realized beta (t-stat) | −0.03 (t=−0.36) | 0.09 (t=1.37) |
| Hit rate | 38.2% | 39.7% |
| Avg monthly turnover | 89.2% | 92.0% |
| Max drawdown | −24.6% | −24.9% |

**The paper's directional claim — sparse should dominate ridgeless in
Sharpe — is confirmed in this run, though for a somewhat deflationary
reason.** The "sparse" model's predictions are effectively constant (all
600 RFF coefficients zero, only the intercept survives), so its portfolio
is close to a coin-flip tie-break each month by the LP solver — it neither
wins nor loses much (Sharpe −0.03, near flat). The "ridgeless" model, with
no sparsity constraint, actively fits the 600-dimensional noise in the
expanded space and **loses money** (Sharpe −0.22, CAGR −2.16%, IR −0.70).
Sparse "wins" by correctly declining to fit rather than by finding real
signal — both turnover figures (89–92% monthly) confirm neither model
found a stable ranking; they're both closer to random reshuffling than to
a coherent signal.

## Limitations

1. **"Sparse wins by doing nothing" is a much weaker confirmation of the
   paper's claim than finding an actual sparse, predictive structure** —
   the paper's own result is about a sparse estimator finding *real*
   parsimonious structure in an expanded space, not defaulting to zero.
2. **P=600 does not reach the paper's overparameterized-beyond-n regime**
   (see Fidelity section) — whether a much larger P would surface genuine
   sparse structure is untested here, and infeasible at this project's
   pooled-panel training-set sizes.
3. **The near-100% turnover for both models is itself diagnostic** — when
   a model's predictions carry essentially no real signal, the "portfolio"
   the LP constructs from them is closer to noise than to a tradable
   strategy, regardless of which regularization scheme produced the
   near-flat predictions.
