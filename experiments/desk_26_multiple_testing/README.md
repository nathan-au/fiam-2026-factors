# desk_26 — Multiple-testing audit of this run (DEV)

Rebuilds the monthly DEV net returns of all 27 full-DEV book variants of desk_19–24 (`output/dev_net_return_matrix.csv`) and applies the corrections implemented in `fiam_research/stats.py`.

| diagnostic | result |
|---|---|
| variants | 27 (full DEV). Total ledgered DEV trials in the run: ~60 (`experiments/desk_research_ledger.csv`) |
| DEV net IR, best variant | **−0.132** (B1 + SI cap 5%). 10/50/90% quantiles of all variants: −1.04 / −0.61 / −0.30. **None beats the T-bill + 4% hurdle on DEV** |
| **PBO (CSCV, 8 blocks, 70 splits)** | **0.49**: choosing the best in-sample variant is a coin flip out of sample |
| deflated Sharpe of the best (27 trials) | 0.06 (SR0 = 0.52 annual) |
| BH-FDR over 33 paired net-return tests (q = 0.10 and 0.20) | **1 discovery: SI cap 5% on B1 (t 3.19)** |
| monotone GBM, seeds 0–4 (OOS 2017–20 IC) | 0.023, 0.020, 0.022, 0.023, 0.027. Positive in every year for every seed |
| monotone GBM 2017–20 book | net IR −0.17, 90% CI [−1.52, +0.66], DSR 0.05 |

**Interpretation:**
- Almost everything in this run is indistinguishable from noise on DEV.
- The one survivor of FDR, the SI cap level, is a *risk* control whose mechanism (squeezes of heavily shorted large caps) has independent support from desk_14 (unseen months) and desk_24 I1.
- The GBM is robust in IC but not in P&L.

These are the candidates for the single pre-registered TEST confirmation (desk_27).
