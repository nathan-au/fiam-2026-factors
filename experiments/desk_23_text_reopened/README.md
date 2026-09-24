# desk_23 — Text Desk reopened: conditional, asymmetric, change-based, and risk roles

**Motivation:** desk_16 L15 (Lazy Prices: *changes* and exec/litigation language matter), L16 (tone and risk), L9 (informed shorts); `novneg_max` (desk_12). DEV only, universe rows. Coverage: 51% of universe stock-months have an 8-K. The family of 5 was pre-registered with BH-FDR q = 0.10; the conditional analyses are descriptive.

| signal | IC (t) | IC where it has a view (t) | share with view | p (one-sided) | BH discovery |
|---|---:|---:|---:|---:|---|
| T1 novneg_max | +0.0071 (1.64) | +0.0064 (1.01) | 51% | 0.051 | no |
| T2 novneg one-sided (adverse half only) | +0.0032 (0.70) | −0.0080 | 25% | 0.24 | no |
| T3 tone change vs the firm's own trailing 12m | +0.0017 (0.37) | +0.0040 | 51% | 0.35 | no |
| T4 novneg only with 5.02 / 8.01 / litigation | −0.0010 (−0.27) | −0.0029 | 26% | 0.61 | no |
| T5 novneg only among top-SI tercile | +0.0029 (0.50) | −0.0029 | 9% | 0.31 | no |

Conditional analyses:
- **C1, text as confidence.** B1 IC among names with top-tercile novneg is +0.002, among other filers +0.018, and among non-filers +0.008. The factor view is weakest where there is fresh adverse text. Not significant (desk_24 I2: t 1.23).
- **C2, size.** novneg IC by universe size tercile: small +0.005, mid +0.002, **large +0.015 (t 2.16)**. Descriptive, one of three cuts.
- **C3, text as a risk predictor.** novneg_max rank-correlates with next-month |residual return|: +0.027 (t 4.2). **After controlling for the stock's own ivol rank: coefficient +0.030 (t 4.65).** neg_mean t 3.2, novelty t 3.4. Text carries genuine information about *risk*.
- **C4, risk overlay.** Halving the name cap of top-decile novneg names gives net IR −0.79 vs −0.67 (paired t −0.91): no evidence. The LP replaces the capped names with other names of similar risk, and 200+ name books diversify single-name vol anyway.

**Verdict on text:**
- Independent alpha: **not demonstrated** in the universe.
- Conditional alpha: **no evidence** (C1 and C2 are hypotheses).
- Portfolio-construction value: **none found**.
- Risk-management value: **text predicts idiosyncratic volatility beyond ivol (t 4.7)**. The information is real, but a book-level use for it was not found.

**Carried to desk_27 as pre-registered hypotheses:**
- (H-T1) novneg_max IC > 0 on the unseen **$0.5–2B band** in TEST.
- (H-T2) novneg_max IC > 0 in the **large tercile** in TEST (suggested by C2, so labelled post-hoc).
- (H-T3) the C3 risk relation replicates in TEST.
