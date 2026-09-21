# PCA Subspace Stability Diagnostic (Methodology and Results)

Implementation: `pca_diagnostic.py` (run with
`.venv/bin/python experiments/pca_diagnostic/pca_diagnostic.py`). Operationalizes *Principal component
error in high-dimensional factor models* (Bernstein, Goldberg, Gunther,
Kercheval, Lan, Lin & Yao, arXiv 2609.20550, Sep 2026 — `docs/PAPERS.md`
§3).

**This is not a return-prediction or portfolio-construction script**, unlike
the other 7 in this batch (`rf.py`, `sparse.py`, `joint.py`, `e2e.py`,
`btq.py`, `rankglu.py`, `poly.py`). Per `docs/PAPERS.md`, the source paper
itself is a diagnostic tool, not a strategy — it decomposes PCA-based
factor-estimation error into an "out-of-subspace" component (how far an
estimated principal subspace is from the true one) and an "in-subspace"
component (finite-sample noise). It only matters if a PCA/shrinkage-based
prediction model (e.g. Kozak-Nagel-Santosh, `docs/PAPERS.md` §1, still
unbuilt) is pursued — this script answers the prerequisite question: **are
the principal components of this 147-characteristic panel stable enough to
trust a shrinkage prior built on them?**

## Methodology

**Does not reproduce the paper's exact asymptotic error bounds** (unavailable
beyond the abstract). What's implemented: a standard, directly interpretable
stand-in for "out-of-subspace" error — **principal-angle subspace
stability**. For each of the same 6 walk-forward training folds used by
every other script in this batch, the training months are split in half
chronologically; the top-k principal components of the 147 rank-transformed
characteristics' cross-sectional covariance are estimated independently on
each half; the cosine of the principal angles between the two half-period
k-dimensional subspaces (via SVD of the cross-product of the two orthonormal
loading matrices) measures how much the estimated subspace would have
differed had history split differently, for `k = 1..15`. A cosine near 1.0
means the two independent estimates agree closely (stable/trustworthy); near
0 means they diverge (unstable — a shrinkage prior built on this component
would be estimating noise, not signal).

No target variable, no LP, no beta-neutral portfolio, no OOS R² — this
script's only outputs are diagnostic tables.

## Results

Average subspace stability (mean principal-angle cosine) across all 6
folds, by `k` (`experiments/pca_diagnostic/output/pca_stability_by_k.csv`):

| k | Stability (cosine) | k | Stability (cosine) |
|---:|---:|---:|---:|
| 1 | 0.988 | 9 | 0.983 |
| 2 | 0.976 | 10 | 0.978 |
| 3 | 0.963 | 11 | 0.966 |
| 4 | 0.958 | 12 | 0.970 |
| 5 | 0.977 | 13 | 0.958 |
| 6 | **0.861** | 14 | 0.958 |
| 7 | 0.977 | 15 | 0.975 |
| 8 | 0.984 | | |

**The top 15 principal components of this panel are, on the whole,
reliably estimated** — stability stays above 0.95 for all but one value of
k. The one exception, k=6 (0.861), is consistent with two nearby
eigenvalues being close enough in magnitude that their *individual*
eigenvector directions can swap or rotate between the two half-samples even
though the *combined* subspace up to that point is still well-estimated —
a well-known PCA phenomenon (component-level instability near eigenvalue
ties), not evidence the whole top-15 subspace is unreliable.

**Variance explained** (`experiments/pca_diagnostic/output/pca_variance_explained.csv`): 8 components
are needed on average to explain 50% of total cross-sectional variance in
the 147 characteristics; more than 15 are needed to reach 90% (the diagnostic
was capped at k=15). This is a meaningfully diffuse spectrum — no small
handful of components dominates — consistent with the univariate finding
that no individual characteristic carries outsized standalone signal, and
with `experiments/ols/README.md` §4.5's note that the 147 characteristics include many
correlated-but-not-redundant groups rather than a small number of true
underlying factors.

## What this means for a future PCA/shrinkage model

**Green light, with a caveat on dimensionality.** The subspace-stability
result says a Kozak-Nagel-Santosh-style shrinkage prior built on, say, the
top 10–15 principal components of this panel would be resting on a fairly
reliable estimate — out-of-subspace error is low in this regime. The
variance-explained result says those 10–15 components still only capture a
minority of total cross-sectional variation (well under 90%), so a
shrinkage model using just the top components would be discarding real
variation, not just noise, unless combined with enough components to
capture materially more than 50%.

## Limitations

1. **Split-half stability is a proxy for the paper's actual
   out-of-subspace/in-subspace decomposition**, not a reproduction of
   their asymptotic bounds — it answers a related but not identical
   question ("would two independent samples agree" rather than "how far is
   the sample estimate from the true population subspace").
2. **Only the 6 walk-forward training windows were tested**, each split
   into two arbitrary halves by month order — stability under a different
   split (e.g. random month assignment rather than chronological) is
   untested, and could differ if instability is concentrated in a specific
   sub-period (e.g. 2022's volatility regime) rather than spread evenly.
3. **This diagnostic was not used to actually build a shrinkage model** —
   it answers the prerequisite question `docs/PAPERS.md` flagged, but the
   Kozak-Nagel-Santosh model itself remains unbuilt.
