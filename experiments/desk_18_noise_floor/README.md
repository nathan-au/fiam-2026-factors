# desk_18 — Construction noise floor of the LP book

**Hypothesis:** the book IR moves for reasons unrelated to signal quality (LP path dependence), which sets a minimum IR difference worth discussing.
**Implementation:** `desk_18_noise_floor.py` (~7.5 min, 90 LP books, DEV). N1: B1 score + N(0, k·sd) jitter, 20 seeds per k. N2: B1 mixed with a within-(month, sector) shuffled copy.

| test | IC | net IR mean | net IR sd |
|---|---:|---:|---:|
| jitter k=0.02 | 0.0131 | −0.703 | 0.032 |
| jitter k=0.05 | 0.0130 | −0.716 | 0.028 |
| jitter k=0.10 | 0.0130 | −0.733 | 0.047 |
| 50% signal / 50% shuffle | 0.0091 | −0.959 | 0.182 |
| 25% / 75% | 0.0066 | −1.218 | 0.195 |
| pure shuffle (random) | 0.0044 | −1.365 | 0.492 |

**Conclusions:**
- Pure path noise is small (sd ≈ 0.03–0.05 IR). Sampling noise is far larger: bootstrap IR s.e. is about 0.45 on 71 months.
- A random signal's book loses about 1.4 IR through turnover churn. "Better than random" is a very low bar.
- Rule used from here on: construction differences under ~0.15 DEV IR (5× the jitter sd) are not evidence. Paired monthly t-statistics are the primary test.
- A caution for every DEV number in this run: IR here is relative to the T-bill + 4% benchmark. A book with zero excess return and ~7% vol has IR ≈ −0.55. **Negative DEV IRs therefore overstate losses; Sharpe (excess of T-bill) is reported alongside.**
