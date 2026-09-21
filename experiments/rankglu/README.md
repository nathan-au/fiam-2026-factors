# Gated Score-Formation Model — RankGLU (Methodology and Results)

Implementation: `rankglu.py` (run with `.venv/bin/python experiments/rankglu/rankglu.py`).
Operationalizes *RankGLU: Residual Gated Score Formation for Cross-Sectional
Stock Prediction* (arXiv 2606.08930, Jun 2026 — `docs/PAPERS.md` §1): a
prediction-head architecture with a direct linear scoring pathway plus a
bounded, gated nonlinear branch — validated in the paper only on Chinese
equity indices (CSI300/CSI800), so transfer to this panel is explicitly
untested going in.

Architecture (this is the closest structural match to a source paper of
any of the 8 scripts in this batch — see `docs/PAPERS.md`'s verbatim
abstract confirmation):

```
z      = X @ W1 + b1                    (hidden, dim H)
gate   = sigmoid(X @ Wg + bg)            (hidden, dim H, in [0,1])
nonlin = tanh(z) * gate                  (bounded branch, in [-1,1])
score  = X @ w_lin + b_lin + nonlin @ w_out + b_out
```

A direct linear pathway plus a bounded, gated nonlinear branch, hand-coded
in numpy (manual forward pass, manual backprop, Adam optimizer — no
autodiff library, since no deep-learning framework is used anywhere else in
this project).

## Fix: widened training budget

The first version trained for 8 epochs over a 4-combination grid
(`h_dim ∈ {8, 16}`, `l2 ∈ {1e-4, 1e-3}`) and produced the worst result of
any model in this project's batch, including a realized beta (−0.47,
t=−2.74) that statistically failed `docs/FIAM.md`'s neutrality mandate
outright. Given the architecture itself matches the paper closely, a
narrow/short training run was the more likely explanation than the
architecture being unsuited to this panel — so this version widens the
search: `h_dim ∈ {8, 16, 32}` (was `{8, 16}`), 20 epochs (was 8), batch
size 4096 (was 8192) — 6 combinations × 20 epochs vs. the original 4 × 8,
roughly 2.8× the training compute.

## Results

| Fold | h_dim | l2 | Val MSE |
|---|---:|---:|---:|
| 2021 | 32 | 0.001 | 0.06065 |
| 2022 | 8 | 0.0001 | 0.06512 |
| 2023 | 8 | 0.0001 | 0.04792 |
| 2024 | 16 | 0.0001 | 0.05069 |
| 2025 | 8 | 0.001 | 0.07980 |
| 2026 | 32 | 0.001 | 0.09361 |

| Metric | RankGLU (widened) | RankGLU (first version) | OLS (baseline) |
|---|---:|---:|---:|
| OOS R² | −0.339% | −0.732% | −0.0086% |
| Information Ratio | 0.57 | 0.36 | 0.85 |
| Sharpe ratio | 0.77 | 0.54 | 0.98 |
| CAGR | 14.0% | 10.0% | 27.6% |
| Alpha t-stat | 1.81 | 1.87 | 2.41 |
| **Realized beta (t-stat)** | **−0.019 (t=−0.12)** | **−0.47 (t=−2.74)** | −0.17 (t=−0.67) |
| Hit rate | 61.8% | 58.8% | 61.8% |
| Max drawdown | −21.3% | −48.0% | −42.4% |

**The realized-beta failure is fixed.** The first version's realized beta
(−0.47, t=−2.74) was statistically significant — per `docs/FIAM.md` §14,
that reads as an outright failure of the neutrality mandate. With more
training, realized beta is now −0.019 (t=−0.12), among the best-neutralized
models in this entire project, comparable to Random Forest's best result.
Max drawdown also improved dramatically (−21.3% vs. −48.0%, now better than
the OLS baseline itself). R² and portfolio-return metrics (IR 0.57, Sharpe
0.77) also improved substantially over the first version but remain the
weakest among the models that don't outright fail neutrality.

**Interpretation**: the previous catastrophic realized-beta result looks,
in hindsight, like an undertrained-model artifact rather than a property of
the architecture itself — a genuinely useful finding given `docs/PAPERS.md`
flagged this model's untested cross-market transfer as the main risk going
in. The architecture transfers to this panel *far* better with adequate
training than the first, deliberately-cheap run suggested.

## Limitations

1. **Still the weakest performer (by return/risk-adjusted metrics) among
   the models that pass the neutrality check** — improved training fixed
   the mandate-failure but did not make this architecture competitive with
   OLS, Poly, or E2E on IR/Sharpe/CAGR.
2. **20 epochs and a 3×2 grid is still modest** relative to what a
   production RankGLU training run would likely use — further widening
   (more epochs, learning-rate schedules, deeper architectures) was not
   attempted, and might continue to help based on the trend observed here.
3. **Exact architecture details beyond the abstract's description remain
   unconfirmed** — layer widths, initialization, and training procedure in
   the actual paper are not available, so this is a best-effort structural
   match, not a verified reproduction.
