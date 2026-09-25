# desk_32 — Second pre-registered TEST look: text-conditional factor weighting

`PREREGISTRATION.md` was written before the run. **This is the second TEST look of the run, so thresholds are Bonferroni ×2:** paired t ≥ 1.3, replication t ≥ 2.3. Run: `desk_32_text_confirmation.py --confirm` (27 s). Deviations: none. R4 (LLM on TEST) is conditional on desk_31 H3 (see desk_31).

| arm | TEST IC (t) | TEST net IR | TEST Sharpe | max DD | paired net t | IR diff 90% CI | DEV Sharpe | combined Sharpe | decision |
|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| A0 B1 | 0.0454 (2.39) | 0.660 | 0.99 | −9.6% | – | – | −0.15 | 0.51 | reference |
| Y1 fresh-surprise ×2 | 0.0440 (2.45) | 0.657 | 1.01 | −9.5% | −0.52 | [−0.14, +0.14] | −0.02 | 0.57 | fail (a) |
| **Y2 Y1 + merger no-view** | 0.0434 (2.49) | **0.738** | **1.10** | **−8.5%** | +0.12 | [−0.09, +0.27] | **+0.11** | **0.67** | fail (a) |

| replication | DEV | TEST | replicated (t ≥ 2.3)? |
|---|---|---|---|
| R1 surprise IC, fresh 2.02 minus stale | +0.025 (t 2.63) | +0.015 (t 1.77) | no; same sign, weaker |
| R2 B1 IC, non-merger minus merger names | +0.054 (t 2.89) | +0.015 (t 0.80) | no; same sign, weaker |
| R3 book text-risk vs abs(book return) | +0.19 (t 1.61) | −0.13 (t −1.03) | no; reversed |

**Interpretation:**
- The conditional structure found in the handoff tables keeps its **direction** on unseen data but loses about half its strength. That is typical decay for a DEV discovery.
- Y2 has the higher Sharpe in both regimes (DEV −0.15 → +0.11, TEST 0.99 → 1.10) and a smaller drawdown. Its monthly mean-return improvement on TEST is nil (t 0.12), so under the rule it is **not adopted**.
- Classification: **useful but insufficient**. It is a sensible, explainable use of the text tables (earnings-timing and deal-pinning metadata make the factor view conditional), with no demonstrated performance gain.
