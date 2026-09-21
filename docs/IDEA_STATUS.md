# IDEA_STATUS.md — what happened to every idea in NEW.md and REDDIT_RESEARCH.md (2026-09-21)

Every actionable idea in `docs/NEW.md` and `docs/REDDIT_RESEARCH.md` was implemented as an experiment under `experiments/<name>/` (each with `README.md`, one standalone script, and `output/`), executed on the full data, inspected, and given a final status below. **Text-8K / LLM work is excluded by instruction** (anything that reads the 8-K text body or uses an LLM or embeddings); 8-K *item-code metadata* (structured filing dates and item codes) is not text and was tested.

Status vocabulary: `IMPLEMENTED_AND_TESTED` (built, run, evidence supports the claim or the experiment is a calibration / diagnostic), `IMPLEMENTED_BUT_FAILED` (hypothesis rejected by its pre-registered rule), `IMPLEMENTED_BUT_INCONCLUSIVE`, `BLOCKED_WITH_REASON`, `DUPLICATE_OF_EXISTING_EXPERIMENT`, `NOT_ACTIONABLE_WITH_CURRENT_PROJECT`.

All experiments share one frozen harness, copied verbatim from `experiments/largecap/largecap.py` into each script: the >= $2B / $5 / $10M-dollar-volume universe (about 1,206 stocks per month), the 18 pre-registered factors, walk-forward folds 2021–2026, the beta- and sector-neutral LP with a 10% one-way turnover cap, assumed tiered costs. **Every block, portfolio and diagnostic experiment re-derived the composite and reproduced the published numbers exactly** (IC 0.0366, `lc_t10` gross IR 0.615 / net 0.541); **every tree-model experiment's control reproduced `largecap`'s `et` arm** (IC 0.016, gross IR −0.22). Two experiments (`feat_path`, `portfolio_concentration`) were re-run from scratch and their `summary.csv` came out byte-identical; the macro run was identical across two runs.

---

## 1. Bottom line

1. **Nothing made a fitted model tradeable.** Targets (rank, Gaussian-rank, decile classification, sector/size-demeaned, factor-residual, learning-to-rank, multi-horizon), loss weights, recency weights and windows, era-stacking, era-splitting, composite-as-prior and Numerai-style ensembles were tested in ExtraTrees / LightGBM / XGBoost on the same 18 factors. Universe IC of every fitted arm is between −0.003 and +0.024 against the composite's 0.0366 (the one exception is ExtraTrees with the yield-curve slope as an extra input, IC 0.0345, whose gain is within the noise of a shuffled-slope placebo and concentrated in a few months); rank-type targets and all-stock training show a consistent but statistically unconfirmed +0.003 to +0.007 IC. The best fitted arm anywhere is LightGBM trained on all stocks unweighted (IC 0.023, `lc_t10` gross IR +0.08, net 0.00). Weighting toward large names *hurts* (paired t −2.7 / −2.9 for dollar-volume weights).
2. **Of 46 candidate signal blocks built from price paths, industry, 8-K item metadata, peers (correlation and TNIC), insiders and short interest, exactly one passed the pre-registered add-on rule: FINRA days-to-cover** (IC gain on the composite +0.0089, paired t 2.67, in-file Bonferroni p 0.015). Short interest is a persistent state variable (a 6-month-old file predicts as well as a fresh one), about half of its effect is low-SI *out*performance, and its LP-level IR gain is not significant. **Project-wide, that result is not robust to multiple testing** (see section 3). Its most defensible use is as a *short-side filter*: barring shorts with SI ratio > 10% halves the max drawdown (−9.8% vs −18.9%) at similar IR.
3. **Portfolio-layer ideas mostly failed or traded IR for risk.** No buffer / L1 / partial-rebalance / EMA design beat the hard 10% turnover cap. Raising the per-name cap to 1.5–3% raises net IR (0.70–0.75 vs 0.54, paired t ≈ 2) but deepens the max drawdown to −23% … −33%. Deep neutralisation (size + momentum + volatility + quality) cuts the 2025 and unwind-month losses roughly in half for about 0.11 net IR. No crowding / IC / volatility gate improved the book, and a gate cutoff tuned on a validation slice inherited that slice's (losing) regime.
4. **Calibration:** 54% of the composite's cross-sectional variance is explained by public styles + sectors, yet 69% of its IC survives orthogonalisation (residual IC 0.0252, t 2.67). It is statistically indistinguishable from a value + quality screen (paired t 0.42); 54% of its IC is the value group. The unconstrained fundamental law overstates the LP book about 3× (transfer coefficient 0.33). With a 40–50% live haircut, expect net IR of roughly 0.1–0.2, and 68 months can only detect an IC ≥ 0.054.
5. **Leakage audit passed** (return-derived characteristics rebuild exactly from returns ≤ t; a deliberately leaky feature is flagged; label alignment 100%; every new feature builder also passed its own truncation test).

---

## 2. Status of every idea

`N` = docs/NEW.md item, `R` = docs/REDDIT_RESEARCH.md item, `H` = the hypotheses H1–H8 in REDDIT_RESEARCH.md §3.2. "Kill rule" = the pre-registered rule of that document (§4 of NEW.md; §2.x of REDDIT_RESEARCH.md).

| Idea | Experiment | Status | Result (measured) |
|---|---|---|---|
| N1 Rank / Gaussian-rank target | `target_rank` | IMPLEMENTED_BUT_FAILED | IC +0.003…+0.006 over control in ET and LGBM (paired t 0.7–1.4; 3 seeds: +0.005, t 1.2–1.4); book still negative; kill rule (t ≥ 2 on two families) not met |
| N2 Decile classification, expected-decile score | `target_decile_class` | IMPLEMENTED_BUT_FAILED | Expected-decile IC 0.024 / 0.021 vs 0.016 (paired t 1.0 / 0.8); argmax and top-minus-bottom worse; books worse than control |
| N3 Size- / industry-demeaned target | `target_neutral` | IMPLEMENTED_BUT_FAILED | Sector-demeaned +0.004…+0.005 IC (t 0.9); **size-demeaned hurts** (IC 0.012 vs 0.016; gross IR −0.45…−0.51) |
| N4 Factor-residual (Numerai-style) target | `target_neutral` | IMPLEMENTED_BUT_FAILED | IC 0.018–0.021 (+0.002…+0.005, t 0.3–0.6); least-bad books (LGBM gross IR −0.04) but none positive |
| N5 Multi-target, multi-horizon ensemble + ridge meta-model | `multi_target_ens` | IMPLEMENTED_BUT_FAILED | Equal-weight ensemble = single target (IC 0.0168 vs 0.0165); ridge meta-model IC −0.0025 (overfits the 24-month validation window); 3- / 6-month targets slower (rank autocorr 0.88 vs 0.80) but IC 0.013 |
| N6 Learning-to-rank (XGBRanker, LambdaMART, rank-IC objective) | `target_ranker` | IMPLEMENTED_BUT_FAILED | Best `rank_pairwise` +0.010 IC over a weak XGB control (t 1.6) but below the LGBM regression control; lambdarank = control; my rank-IC objective reconstruction not better |
| N7 Quantile / distributional targets | — | NOT_ACTIONABLE_WITH_CURRENT_PROJECT | Source is a quantile-NN paper on conditional skewness with no return-prediction recipe; low-priority in NEW.md |
| N8 Value-/size-weighted training loss (+ train on all stocks, weighted) | `train_weighting` | IMPLEMENTED_BUT_FAILED | Weights hurt: dollar-volume weights paired t −2.7 (ET) / −2.9 (LGBM); best fitted arm anywhere = LGBM on all stocks **unweighted** (IC 0.023, gross IR +0.08), +0.007 IC over control (t 0.9) |
| N9 Recency weighting / rolling window | `recency_window` | IMPLEMENTED_BUT_FAILED | All half-lives and windows below the expanding-window control (paired t −0.4…−1.1) |
| N10 Numerai deep incremental (era-stack) | `recency_window` (`era_stack3`) | IMPLEMENTED_BUT_FAILED | IC 0.020 vs 0.016 (+0.003, t 1.4 / 1.5); books negative |
| N11 Missing-value imputation | — | DUPLICATE_OF_EXISTING_EXPERIMENT | NEW.md itself concludes the current median-fill is fine (Chen & McCoy); nothing to test |
| N12–N13 Information discreteness; path slope / smoothness | `feat_path` | IMPLEMENTED_BUT_FAILED | 5 blocks, paired gain −0.0016…+0.0011 IC (t −0.32…+0.17); continuity adds nothing over plain momentum |
| N14 Residual / industry / 52-week-high-neutral momentum | `feat_industry_rel` | IMPLEMENTED_BUT_FAILED | Gains −0.0007…+0.0030 IC (t ≤ 0.6) |
| N15 Industry-relative characteristics | `feat_industry_rel` | IMPLEMENTED_BUT_FAILED | Within-GICS / FF49 composites slightly worse (IC 0.0324 vs 0.0366; t −0.6…−0.7) |
| N16 Characteristic changes | `feat_char_changes` | IMPLEMENTED_BUT_FAILED | 7 blocks, all gains ≤ +0.0000 IC or negative (12-month composite −0.0050, t −1.5) |
| N17 Characteristic × size interactions | `feat_size_interact` | IMPLEMENTED_BUT_FAILED | Within-tercile ranking −0.0017 (t −0.7); IC-weighted tercile weights −0.0126 (t −1.5) |
| N18 Macro / VIX-conditioned interactions; R13 yield-curve slope | `macro_condition` (slope, credit spread, VIX as tree inputs, with shuffled-slope placebos) and `crowding_gate` (S&P-vol gate / vol scaling) | IMPLEMENTED_BUT_INCONCLUSIVE | Slope +0.018 (ET, IC 0.0345, t 2.5) / +0.009 (LGBM) IC but paired t 1.2 and inside the placebo band's reach (shuffled slope: −0.002…+0.010); gain concentrated in a few months (2022); credit spread hurts; gates G4 / G5 did not help |
| N19 Amihud / spread-adjusted "tradeable" features | `composite_overlap` (leave-liquidity-out, Shapley) | IMPLEMENTED_AND_TESTED | Liquidity group Shapley IC 0.0004 (≈ 0); dropping it leaves IC unchanged (0.0365 vs 0.0366) |
| N20 8-K filing-frequency / information-intensity | `feat_8k_meta` | IMPLEMENTED_BUT_FAILED | `k8_count_12m` IC −0.0013, gain −0.0036 (t −1.05); the Management-Science spread is absent in ≥ $2B stocks 2021–26 |
| N21 News / no-news split of reversal | `feat_8k_meta` | IMPLEMENTED_BUT_FAILED | Reversal is absent with and without news (IC −0.005 / −0.008) |
| N22 8-K event flags (4.02 / 4.01 / 2.05 / 2.06 / 5.02) | `feat_8k_meta`, `feat_8k_502_followup` | IMPLEMENTED_BUT_FAILED (flags); 5.02 IMPLEMENTED_BUT_INCONCLUSIVE | Bad-event flags have wrong-signed residual IC; 5.02 intensity: standalone IC +0.012 (t 2.8, robust to half-life and to size/liquidity/age orthogonalisation) but adds nothing to the composite at any weight |
| N23 Earnings-announcement premium from 2.02 timing; R11 earnings-in-window | `feat_8k_meta` | IMPLEMENTED_BUT_FAILED | Premium absent (IC −0.0004 / −0.0071; gain −0.0033, t −1.7), consistent with Heitz et al. |
| H7 Graded (decayed) event intensity | `feat_8k_meta` | IMPLEMENTED_BUT_FAILED | Decayed bad-event and filing intensities gain −0.0006 / −0.0033 (t −0.4 / −0.9) |
| N24 Naive PEAD / SUE in large caps | — | NOT_ACTIONABLE_WITH_CURRENT_PROJECT | On NEW.md's skip list; supporting evidence: the composite's `surprise` group (`niq_su`, `saleq_su`) has Shapley IC 0.0001 (`composite_overlap`) |
| N25–N29 Lazy Prices on 8-K text, FinBERT+PCA, event extraction, LLM-embedding signals, AI / tariff text scores, look-ahead toolkit; R8 numeric-specificity drift | — | NOT_ACTIONABLE_WITH_CURRENT_PROJECT | **Excluded by instruction** (text 8-K / LLM work). The "sector-ghost" residual-IC check from R8 was applied to every block (`resid IC`) |
| N30 Composite as prior (`init_score`, ridge toward composite); H2 orthogonal-residual training | `composite_prior` | IMPLEMENTED_BUT_FAILED | Every ML-on-top arm is below the composite (paired IC −0.011…−0.018); orthogonal-residual IC 0.0005 (< 0.01 kill bar) |
| N31 Diverse-feature pooled ensembles | `feat_*` blocks, `multi_target_ens` | IMPLEMENTED_BUT_FAILED | Equal-weight of diverse members equals the best member (no gain); no diverse-feature member (peer, 8-K, path) improved the composite |
| N32 IC-weighted dynamic model weights | — (evidence in `feat_size_interact`) | DUPLICATE_OF_EXISTING_EXPERIMENT | `factor_filter` (IC does not persist) and `tpa`; NEW.md itself calls it "a warning, not a lead"; the tercile IC-weighted composite here was worse (−0.0126 IC) |
| N33 TabPFN / tabular foundation model | — | BLOCKED_WITH_REASON | The package installs (isolated venv), but the weights require a human to log in, accept the licence and supply `TABPFN_TOKEN`; the licence also restricts use in commercial decisions. I did not work around the gate. To unblock: accept the licence at ux.priorlabs.ai, export `TABPFN_TOKEN`, and a per-month in-context experiment is ~30 lines |
| N34 Sequence models (Mamba, KAN, Transformers) | — | NOT_ACTIONABLE_WITH_CURRENT_PROJECT | On NEW.md's skip list (evidence is CN / single-stock) |
| N35 CNN "image" factor timing | — | DUPLICATE_OF_EXISTING_EXPERIMENT | `tpa` (`tpa_ms`) failed; the source study is paywalled |
| N36 Peer return gap (correlation peers) | `feat_peer_gap` | IMPLEMENTED_BUT_FAILED | Gains +0.0005 / −0.0042 / −0.0005 IC (t ≤ 0.07 / −1.5); correlation peers no better than plain industry returns |
| N37 TNIC text-based peers | `feat_tnic_peers` | IMPLEMENTED_BUT_FAILED | Peer return gain +0.0001; the pre-registered "laggards catch up" sign is *wrong* (gap gain −0.0046, t −2.07) |
| N38 Customer–supplier momentum | — | BLOCKED_WITH_REASON | Needs supplier-link data (Compustat segments / FactSet Revere) not in the project or free |
| N39 Text-embedding networks; N40 13F asset embeddings | — | NOT_ACTIONABLE_WITH_CURRENT_PROJECT (N39, excluded text/LLM) / BLOCKED_WITH_REASON (N40) | N40: multi-GB EDGAR 13F bulk data + CUSIP mapping + word2vec, and NEW.md itself limits it to quarterly, lagged peer / crowding definitions. **Not run by my choice (effort vs prior), not because the data is unreachable** |
| N41 Triangulated / deep-learning stat arb; N42 GNNs | — | NOT_ACTIONABLE_WITH_CURRENT_PROJECT | Daily / intraday frequency; monthly panel; NEW.md says skip |
| N43 Option-implied features | — | BLOCKED_WITH_REASON | OptionMetrics is a paid WRDS dataset; no access |
| N44 Daily-return signals (DRIF, overnight vs intraday) | — | BLOCKED_WITH_REASON | No daily return source for the full permno history including delisted names (Yahoo rate-limits / survivorship); WRDS CRSP daily not available |
| N45 Analyst forecasts / revisions (I/B/E/S); the analyst leg of Blitz (R2) | — | BLOCKED_WITH_REASON | Paid WRDS dataset |
| N46 Insider Form 4 | `feat_insider_form4` | IMPLEMENTED_BUT_FAILED | Free SEC data 2020–2026Q1; 4 blocks, gains ≤ +0.0017 IC (t ≤ 0.5); insider sales the only suggestive signal (IC +0.019, t 1.6) |
| N47 Short interest (FINRA bi-monthly) | `feat_short_interest`, `feat_si_followup` | IMPLEMENTED_AND_TESTED | **Days-to-cover passes**: +0.0089 IC gain (paired t 2.67, in-file p 0.015), persistent state, half from low-SI outperformance; borrow-stress-robust; **SI ≤ 10% short cap halves the max drawdown**. Not robust project-wide (section 3) |
| N47b FINRA daily short-sale volume | — | BLOCKED_WITH_REASON | Only the last ~365 days are available interactively; no backfill for 2021+ |
| N48 AI-researcher / agent loops | — | NOT_ACTIONABLE_WITH_CURRENT_PROJECT | LLM-agent hypothesis generation is out of scope; the pre-registration and multiple-testing discipline it recommends is what this round applies |
| N49 Neutralise beyond beta; R4 / H3 neutralisation depth dial | `neutral_dial` | IMPLEMENTED_BUT_INCONCLUSIVE | Size / size+momentum lose IR with no stress gain; **full core-factor neutrality (size+mom+vol+quality) halves 2025 and unwind losses for −0.11 net IR** (t −0.6); both source documents partly right; N = 4 stress months |
| N50 Sector-neutral ranking instead of sector caps | `neutral_dial` (S1 / S2) | IMPLEMENTED_BUT_FAILED | Net IR 0.41 / 0.28 vs 0.54 |
| N51 Cost-aware objective (L1 penalty); R1 | `portfolio_buffer` | IMPLEMENTED_BUT_FAILED | Best `l1_0.20` +0.07 net IR (t 0.28), others −0.03…−0.08 |
| N52 Signal smoothing / decay-aware blending; G–P-lite aim | `portfolio_buffer` | IMPLEMENTED_BUT_FAILED | EMA lowers IC and net IR (−0.08…−0.13); G–P-lite −0.14 |
| R1 Asymmetric rank buffer ("buy 10 / hold 50"), no-trade band, partial rebalance | `portfolio_buffer` | IMPLEMENTED_BUT_FAILED | Every buffer variant below its matching control; hard 10% cap beats the free book (0.54 vs 0.43) |
| N53 Confidence gate; N54 dispersion / VIX scaling; R6 crowding gates | `crowding_gate` | IMPLEMENTED_BUT_FAILED | All gates lower or leave IR; IC gate misses the 2025 unwinds; validation-tuned cutoff chosen among all-negative options |
| N55 Concentration / conviction | `portfolio_concentration` | IMPLEMENTED_BUT_INCONCLUSIVE | 1.5–3% caps: net IR 0.70–0.75 (paired t 1.8–2.1, Bonferroni p ≈ 0.3) but max DD −23…−33%; 30-bucket table shows the alpha is mostly avoiding the worst ~7% |
| N56 HRP / TPA sizing | — | DUPLICATE_OF_EXISTING_EXPERIMENT | `hrp`, `tpa` |
| N57 Survivorship / point-in-time discipline | `truncation_audit` (time leakage only) | NOT_ACTIONABLE_WITH_CURRENT_PROJECT | The panel is a retrospective CRSP/Compustat extract; the audit covers time leakage only, not survivorship (stated in the README) |
| N58 Multiple-testing discipline | this file (section 3) | IMPLEMENTED_AND_TESTED | Variant counts and family-wise adjustment below; permutation nulls in every block experiment |
| R2 Blitz short-term composite | `feat_blitz` | IMPLEMENTED_BUT_FAILED | 5 blocks, gains −0.0052…+0.0004 (t ≤ 0.06); industry-relative reversal IC −0.0105; analyst-revision leg blocked (I/B/E/S) |
| R3 Era Splitting; H1 size-environment eras | `era_split` | IMPLEMENTED_BUT_FAILED | My numpy re-implementation (the fork was not built): era criteria worse than the pooled criterion (t −0.9…−1.9); with composite prior `era_dir_init` 0.0212 vs `orig_init` 0.0161 (t +0.18); all below the composite |
| R5 Shared-core overlap test | `composite_overlap` | IMPLEMENTED_AND_TESTED | R² of composite on styles 0.54; **residual IC 0.0252 (t 2.67)** = 69% of raw; no better than a value+quality screen (t 0.42); value = 54% of IC; Shapley values sum exactly to the IC |
| R7 Live-IC haircut | `expectation_calibration` | IMPLEMENTED_AND_TESTED | Transfer coefficient 0.33; 40–50% haircut → net IR 0.09–0.18; 68 months detect only IC ≥ 0.054; bootstrap net-IR 5–95% band (−0.17, 1.22) |
| R9 Truncation-invariance test; typed causal-operator registry | `truncation_audit` (+ a builder test in every `feat_*`) | IMPLEMENTED_AND_TESTED (test); registry NOT_ACTIONABLE | All 7 return characteristics rebuild exactly; negative control flagged; label alignment 100%; the registry is for AI-written feature loops (none in scope) |
| R10 State-dependent borrow stress | `borrow_stress` | IMPLEMENTED_AND_TESTED | Net IR 0.54 → 0.37 (×5) → 0.16 (×10) → −0.26 (×20); a 4%-of-shorts-a-month forced cover at +15% drives IR to ≈ 0 |
| R12 Reinforcements (Attention-Factors replication failure, LdP AUC 0.50, TabPFN reality check, ...) | — | NOT_ACTIONABLE_WITH_CURRENT_PROJECT | Support NEW.md's skip list; nothing to build; the round's negative results are consistent with them |
| R13 Shapley over blocks; IC definitions; power analysis; Toraniko-style exposures; yield-curve macro | `composite_overlap`, `expectation_calibration`, `macro_condition` | IMPLEMENTED_AND_TESTED (yield-curve leg INCONCLUSIVE) | See those READMEs |
| H4 Feature-block permutation null / knockoff FDR | every `feat_*` | IMPLEMENTED_AND_TESTED | Within-(month, sector) permutation null, 200 shuffles, Bonferroni within file; a *random block dilutes the composite*, so the null tests information content, not improvement |
| H5 Conformal selection of names | `conformal_select` | IMPLEMENTED_BUT_FAILED | Conformal selection = top-k with adaptive k (p-values monotone in the score); FDR controlled on the short side (0.55–0.57 ≤ q) but **not** on the long side (0.69–0.72 > q) |
| H6 Pooled international large-cap training | — | BLOCKED_WITH_REASON | The panel is US-only (`excntry` = USA for all 529,082 rows); non-US 148-characteristic panels are a WRDS-hosted dataset |
| H8 Two-stage buffered composite + era-split residual | — | BLOCKED_WITH_REASON | Its pre-registered precondition ("only after 2.1 and 2.3 individually pass") failed: `portfolio_buffer` and `era_split` both failed |

---

## 3. Multiple-testing accounting (NEW.md §4: "count every variant")

| Family | Variants tested against a control |
|---|---|
| Model / target / loss / window / ensemble arms (vs control) | 84 (`target_rank` 4, `target_neutral` 8, `target_decile_class` 6, `target_ranker` 4, `composite_prior` 7, `era_split` 8, `train_weighting` 12, `recency_window` 14, `multi_target_ens` 7, `macro_condition` 14 incl. 6 placebos) |
| Candidate signal blocks added to the composite | 46 blocks + 6 replacement composites (excluding the 5.02 follow-up, whose 7 variants + 3 weights re-test one signal) |
| Portfolio-construction variants | ≈ 51 (`portfolio_buffer` 21, `portfolio_concentration` 9, `neutral_dial` 9, `crowding_gate` 7, `feat_si_followup` 5) |
| **Total** | **≈ 185 tested variants** |

- The pre-registered rules used a **within-file** Bonferroni factor (3 blocks for short interest). Days-to-cover: permutation p 0.005 (the floor of 200 shuffles) × 3 = 0.015 → passes; × 46 blocks project-wide = **0.23**; paired t 2.67 (two-sided p ≈ 0.008) × 46 ≈ 0.35. Short interest was a *prior-motivated* hypothesis (named in NEW.md §3.7 with a mechanism), which supports it but does not remove the discount.
- Expected number of blocks with |paired t| ≥ 2 by chance among 46 independent tests: about 2. One passed (in the right direction).
- Portfolio IR differences below about 0.5 are within one standard error (≈ 0.46) of a single 68-month path; LP corner solutions also move IR by ±0.3 for changes that leave IC unchanged (visible in several block experiments). IR was therefore never used for a decision; paired IC and paired monthly net-return t were.

---

## 4. Problems found and fixed during the round (all documented in the relevant README)

- `composite_prior` and `era_split`: a **negative fitted composite scale flipped the prior's sign** in the 2023 fold (validation IC −0.077 instead of +0.077). Fixed by flooring the scale at +0.005; both experiments re-run. The invalid first-run summaries are kept in `output/first_run_invalid_negative_scale/`; conclusions did not change.
- `feat_tnic_peers`: two implementation slips (wrong zip member, wrong column name) — fixed and re-run.
- `feat_short_interest` / `feat_tnic_peers`: FINRA's CDN rejects urllib's default user agent (HTTP 403), which had silently produced 0 matched months; a **truncation test that compared zero values had "passed"** — the test now fails if it compares nothing.
- `neutral_dial`: a mislabelled summary column (max vs mean exposure) — fixed and re-run.
- `macro_condition`: the S&P series in `cache/` is only ~10 years long, so VIX (FRED `VIXCLS`) replaced S&P realised volatility; and because the yield-curve slope produced a surprisingly large gain, shuffled-slope **placebo arms** were added and the experiment re-run (first run kept in `output/first_run_without_placebo/`).
- `.gitignore` was briefly corrupted by my edit (missing newline) and repaired; the three external-data cache folders (`cache/tnic`, `cache/finra_short_interest`, `cache/sec_form345`, about 0.55 GB) are now ignored. Small FRED files (`cache/T10Y2Y.csv`, `cache/BAA10Y.csv`, `cache/VIXCLS.csv`) follow the existing `cache/TB3MS.csv` convention.

## 5. Things that need you

- **TabPFN (N33)** needs you to accept its licence and supply a token (see table). The licence excludes use "in commercial decision-making"; whether a hackathon counts is your call.
- **Blocked external data** (options, I/B/E/S, daily returns, supplier links, non-US panels) is WRDS-hosted / paid.
- **Output size:** the new `output/` folders total about 0.95 GB (per-arm prediction and holdings CSVs, the same convention as the existing experiments). Nothing was committed; you may want to trim or ignore the largest `oos_predictions_universe_*` and `portfolio_holdings_*` files before committing.
- External-data experiments (`feat_tnic_peers`, `feat_short_interest`, `feat_si_followup`, `feat_insider_form4`, `macro_condition`) need network on their first run to fill `cache/`.
