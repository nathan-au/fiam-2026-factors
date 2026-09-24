# Research run 2026-09-24 — can the desk system be made materially better?

Experiments `desk_13` … `desk_29` (one README each), research package `fiam_research/` (imports `fiam_desks/`, never modifies it), research ledger `experiments/desk_research_ledger.csv` (every variant, every window), TEST-look ledger `experiments/desk_test_ledger.csv`. Extra packages: `requirements-research.txt`. External files: `.venv/bin/python -m fiam_research.download_data`.

**Short answer.**
- **No change was confirmed as a material improvement.**
- Of ~60 DEV variants, the one that survived multiple-testing correction (a tighter short-interest cap) reversed on TEST.
- One fit-free, literature-signed construction (JKP 13 themes) had a better Sharpe in *both* regimes (0.80 vs 0.51 over 2015–2026), but the difference is not statistically significant (90% CI [−0.14, +0.73]). It stays a promising hypothesis.
- The research did find and fix a look-ahead flaw, extended the short-interest data back to 2018, and showed that dtc does not replicate on unseen data.
- It traced the 2015–2020 degradation to one phenomenon, the growth-vs-value cycle, which also explains most sign flips between regimes.
- Text carries robust **risk** information but no demonstrated return alpha.

---

## 1. Baseline (desk_13, frozen; desk_15, corrected)
**B0, the committed system:** frozen 18-factor / 7-group composite + days-to-cover 8th group → `lc_t10` LP (dollar- and beta-neutral, sector limits, 1% name cap, 10% turnover budget, 200% gross) + shorts only where SI ratio ≤ 10%; text advisory.

| | DEV 2015-02..2020-12 | TEST 2021-01..2026-08 |
|---|---:|---:|
| IC (t) | +0.011 (0.72) | +0.046 (2.39) |
| net / gross IR (vs T-bill+4%) | −0.757 / −0.642 | +0.637 / +0.721 |
| net Sharpe | −0.30 | +0.97 |
| bootstrap 90% CI of net IR | [−1.50, −0.04] | [+0.14, +1.16] |
| max DD | −30.0% | −9.8% |
| beta (panel VW market) | −0.17 (t −2.7) | −0.03 |
| turnover, positions, costs | 10%, 202–243, 8.5 bp/m | 10%, 203–248, 8.4 bp/m |
| style alpha t (R²) | 0.77 (0.71) | 0.67 (0.75) |

**Weaknesses found:**
1. The returns are style-premium returns: the book holds quality +0.95, profitability +0.5, low-vol +0.35.
2. Realised beta on DEV is negative.
3. dtc and the SI cap had been adopted on TEST.
4. **Look-ahead flaw (fixed, desk_15):** rows without a next-month return were dropped before forming the universe, so the universe used the future fact "this stock survives". Keeping them (y = 0) moves net IR to −0.855 DEV / +0.660 TEST. The book shorts future delisters 3:1. **B1** = B0 + this correction + SI data from 2018 is the research baseline.
5. The FRED S&P 500 series starts 2016-09, so the harness's DEV beta silently skips 20 months.

## 2. Literature (desk_16)
18 primary sources with URL, method, relevance and the hypothesis each motivates:
- JKP 13 themes with published signs.
- Ehsani–Linnainmaa and Gupta–Kelly factor momentum.
- Asness et al. on contrarian timing.
- Arnott et al. on value's drawdown.
- Stivers–Sun on dispersion.
- Cederburg et al. / DeMiguel et al. on vol management.
- Hong et al. on DTC.
- Drechsler² and Stambaugh–Yu–Yuan on short-side mispricing.
- Blitz et al. on residual momentum.
- Grinold on alpha scaling; Clarke–de Silva–Thorley on the transfer coefficient.
- Novy-Marx–Velikov on trading costs.
- McLean–Pontiff on post-publication decay.
- Lazy Prices; Loughran–McDonald.
- Bailey–López de Prado (DSR, PBO); Benjamini–Hochberg.
- Hanauer–Kalsbach on ML.

Rejected before testing: value-spread timing, book-level vol targeting, LLM text reading, more tree models on the 18 factors.

## 3. Experiments
| # | hypothesis | result | statistical evidence | decision |
|---|---|---|---|---|
| 14 | dtc and the SI cap replicate on 29 never-seen months (2018-01..2020-06), using newly downloaded FINRA files | dtc IC −0.001; SI cap +0.53%/month | dtc t −0.08; cap paired t +2.12 | dtc **not replicated**; SI cap **confirmed as risk control** |
| 15 | look-ahead audit | delisting-row leak | ΔIR −0.10 DEV | **correction adopted (B1)** |
| 17 | why DEV loses | value crash (FM value −36% in 2020), low-vol and investment losses; book = persistent style tilts | IC gap p 0.18; return gap p 0.03; dispersion → IC t −5.4 / −3.1 | explained (see §6) |
| 18 | construction noise floor | jitter sd 0.03 IR; random books −1.36 IR | – | ΔIR < 0.15 treated as noise |
| 19 | JKP 13-theme literature-signed composite beats the frozen composite | lower IC, steadier book (DEV Sharpe +0.21 vs −0.15) | paired IC t −0.7, net t +0.85 | no evidence (carried) |
| 20 | construction: vol-scaled caps, 0.5% cap, risk QP, score-tracking, SI-cap level | nothing beats the LP except SI cap 5% on B1 | cap 5% t +3.19; QP −1.1 | **LP kept**; QP rejected; cap 5% carried |
| 21 | causal timing: factor momentum, inverse-vol, dispersion gross scaling | all help 2018–20, hurt 2016–17 | best t +1.27 | no evidence |
| 22 | decay; rebalance speed | profitability/quality slow; composite decays within a month; turnover budget irrelevant | all |t| < 0.5 | no evidence |
| 23 | text: asymmetric, change-based, filing-type, ×SI, confidence, risk | no return signal survives FDR; **text predicts idiosyncratic risk** | risk t 4.65 | risk info yes; overlay no |
| 24 | six conditional-IC interactions; two multiplicative signals | none; composite ranks high-SI names backwards on DEV | I1 t −2.46; signals t −1.5 / −3.1 | rejected |
| 25 | ridge / prior-ridge / monotone GBM, walk-forward in DEV | GBM IC +0.023, positive every OOS year | paired t 1.34 | no evidence (carried) |
| 26 | multiple testing over the run | PBO 0.49; best DSR 0.06; one BH survivor (cap 5%) | – | – |
| 27 | **single pre-registered TEST confirmation** | no arm passes; cap 5% **reverses** (t −2.00); GBM IC 0.017; JKP13z Sharpe 1.29 vs 0.99 | see README | **B1 stays** |
| 28 | post-hoc: why | JKP13z Sharpe diff +0.29, CI [−0.14, 0.73]; cap flip = growth-vs-value | – | descriptive |
| 29 | outputs and audits | constraints, determinism, truncation all pass | – | – |

## 4. Improvements
- **Genuine and validated:** (a) the delisting look-ahead correction; (b) FINRA SI history back to 2018 (29 more months), which gave the first out-of-sample test of dtc (fails) and of the SI cap (holds); (c) evidence that the 10% SI cap is a robust risk control in both regimes.
- **Not validated:** every attempt to raise alpha or IR. The strongest candidate (A5, JKP13z) improves Sharpe in both periods through lower volatility, but the difference is not significant and it failed the pre-registered mean-return rule.

## 5. Failures and why
- **Tighter SI cap:** the (5%, 10%] SI shorts are half the short book and are the growth names. They lost in 2019–20 and won in 2022. The cap level was a growth/value bet, and the best DEV result was a regime artefact.
- **Factor timing / dispersion scaling:** the IC is partly predictable, but converting that into exposure changes did not survive (literature-consistent).
- **Risk QP / vol-scaled caps / more names:** the LP is not the bottleneck. A higher transfer coefficient spreads a weak signal (IC ≈ 0.01 on DEV) over more names.
- **ML (ridge, monotone GBM):** a DEV-only edge. On TEST, IC 0.017 vs the composite's 0.045. More expressive models learn the previous regime.
- **Interactions / multiplicative signals:** noise or wrong-signed.

## 6. Regime dependence (2015–2020)
1. **Economically explainable: yes.** The book is a persistent value + quality + low-risk + profitability tilt. 2017–2020 was the growth boom and value's deepest drawdown (−36% FM value return in 2020), plus the 2020 junk rally.
2. **Statistically distinguishable:** book returns yes (p 0.03), IC no (p 0.18).
3. **Predictable ex ante: partly.** Cross-sectional dispersion predicts lower IC in both periods (t −5.4 / −3.1), but no causal rule converted it into better P&L.
4. **Finite-sample:** the sign flips of value, SI and text-band effects are consistent with **one** regime cycle observed once each way. 139 months contain essentially one growth→value transition, which is the binding limitation.

**Mitigation without overfitting:** the only approach that looked better in both regimes is *diversifying* the style tilts (JKP13z has no low-vol tilt and adds momentum), not *timing* them. It is not statistically confirmed.

## 7. Text Desk
| role | verdict | evidence |
|---|---|---|
| independent alpha | **not demonstrated** in the tradeable universe | novneg_max IC 0.007 / 0.010 (small, same sign); nothing survives FDR |
| conditional alpha | **no evidence** | size, SI and filing-type cuts null or sign-flipping ($0.5–2B band −0.022 DEV, +0.038 TEST) |
| portfolio-construction value | **none found** | half-cap overlay IR −0.79 vs −0.67 |
| risk-management value | **information yes, usage not yet** | novneg_max predicts next-month idiosyncratic volatility beyond ivol, t 4.65 DEV / 4.71 TEST |
| regime indicator | not tested beyond dispersion | – |

## 8. Final candidate
- **Properly validated: B1** = committed system + delisting correction + SI data from 2018. TEST net IR 0.66 (Sharpe 0.99, max DD −9.6%, beta −0.03), all FIAM constraints pass, outputs in `experiments/desk_29_final_candidate/output/*_B1.csv`. **It is not regime-robust:** DEV Sharpe −0.15, style alpha t < 1.
- **Promising, requires further validation: A5**, the JKP 13-theme z-score composite (all 146 characteristics, published directions, equal theme weights, no fitting) with the same PM. Sharpe +0.21 / +1.29 (DEV / TEST) vs −0.15 / +0.99, max DD −13.9% / −7.7%, style alpha t 1.75 / 1.89. Outputs in `*_A5_jkp13z.csv`. To validate: pre-register A5 vs B1 on months after 2026-08 (Sharpe difference, one-sided), or on a non-US JKP sample.
- **Confirmed evidence:** the delisting fix; the SI cap 10% as risk control; dtc not replicating; text predicting risk; LP adequacy vs QP; regime flips as documented.
- **Promising hypotheses:** A5 style diversification; text-based risk inputs to a construction that actually uses risk (not caps).
- **Unresolved:** whether any stock-specific alpha exists in the ≥$2B universe beyond style premia (style alpha t < 2 everywhere); true borrow costs; the daily-path risk profile (no daily data pulled); whether 2021–26 value strength persists.

## 9. Test-window hygiene
- This run looked at TEST: once in the pre-registered desk_27 (7 arms + 5 standalone tests); descriptively in desk_13/15/17/28/29 (baseline freeze, audit correction, attribution, post-hoc, outputs).
- All looks are in `experiments/desk_test_ledger.csv` and `experiments/desk_research_ledger.csv`.
- desk_17's TEST column of IC-vs-regime slopes was seen before desk_27, so no regime rule was put to the confirmation.
- DEV decisions: about 60 variants. PBO 0.49 shows that DEV selection alone carried no information here, which is why the TEST confirmation was decisive.
