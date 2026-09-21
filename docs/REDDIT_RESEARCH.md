# REDDIT_RESEARCH.md — Reddit-led deep research, extending NEW.md (2026-09-21)

Project context (from `NEW.md` / `REDDIT.md` / `docs/LARGECAP.md`): monthly US equity long/short from 147 characteristics; the tradeable large-cap universe (price ≥ $5, mcap ≥ $2B, ~1,200 names/month) has rank IC ≈ 0.035; the only honest positive is a no-fit, equal-group composite of 18 factors (net IR ≈ 0.54, IC 0.037, 2025 negative); every fitted model (ET IC 0.016) lost to it.

---

## 0. Read this first — how the research was done, and what Reddit can and cannot tell us

**Access, honestly.** The Reddit MCP was degraded: `browse_subreddit` worked (RSS fallback, no scores/comments), but `search_reddit` and `get_post_details` were refused ("cannot access r/quant…", "Access forbidden"). `WebFetch` on reddit.com is blocked. To still do real Reddit research I used a **public Reddit archive (Arctic Shift API)**, which returned posts *and* comment trees.

**What was actually covered**

| Source | Coverage |
|---|---|
| r/quant | **all 22,196 posts, Jun 2023 → Sep 2026**, grepped locally |
| r/algotrading | **all 13,875 posts, Jun 2024 → Sep 2026** |
| r/SecurityAnalysis | 1,294 posts, Jan 2024 → |
| r/quantfinance | 2,659 posts, Jun 2024 → Apr 2025 (partial; career-dominated) |
| r/Numerai | only 48 posts exist in the archive (the community lives on the Numerai forum, not Reddit) |
| r/MachineLearning, r/datascience, r/LocalLLaMA, r/options, r/econometrics | ~26 targeted title searches, 2023 → 2026 |
| Threads read with comments | ≈ 55 |

**Where the archive is weaker.** Some subreddits' stored `score` values are ingestion-time snapshots (r/MachineLearning posts show 0–1); I ranked by comment count too. Title-search intermittently timed out under load. Anything a commenter says is **unverifiable practitioner testimony**; I graded it as such.

**What Reddit is good for here — and not.** r/quant and r/algotrading are ~70% careers/retail-bot content. On the *specific* problem (monthly cross-sectional ML on large caps) there is **no technical thread that outperforms the literature already in NEW.md**. Reddit's real value was (1) **calibration** (how much IC survives production, how crowding actually behaves, what pod-shop practitioners now care about), (2) **failure stories** and pitfalls, (3) **pointers to primary sources** that NEW.md had missed (Blitz et al., Gârleanu–Pedersen, Era Splitting, AQuA, Toraniko), and (4) one **direct contradiction** of a NEW.md recommendation. The report is organised around that, not around volume.

**Evidence grades** (as requested): **Demonstrated** (concrete experiment/result I could see) · **Reported** (someone claims results, limited evidence) · **Plausible** (technically compelling, thin evidence) · **Speculative** (hypothesis). Statements labelled *Own* are my synthesis, not from Reddit or a paper.

---

## 1. Summary versus NEW.md

| Finding | Status vs NEW.md | Grade | Section |
|---|---|---|---|
| **Asymmetric buy/hold rank buffer ("buy 10 / hold 50") + no-trade band + partial rebalance**, backed by a large-cap monthly study | **New** (NEW.md has only EMA smoothing) | Demonstrated | 2.1 |
| **Blitz et al. short-term composite in a liquid large-cap universe** (industry-relative reversal, 1m industry momentum, seasonality, idio-vol, analyst revisions) | Partly new (components scattered in NEW.md; composite, cost rule and large-cap evidence are not) | Demonstrated (not on our sample) | 2.2 |
| **Era Splitting** tree criterion (Numerai lineage) + *my* extension: size-buckets as environments | **New** | Reported / Speculative | 2.3 |
| **Neutralisation depth is contested**: a 2026 pod-shop practitioner says tightly factor-neutral books did *worst*; NEW.md #6 says neutralise harder | **Contradiction** | Reported | 2.4 |
| **"Shared-core" overlap test** — how much of our alpha is the commoditised composite | **New** | Plausible / Own | 2.5 |
| **Crowding is a variance-shift, not an IC decay** → gate on crowding state, and beware gates calibrated in a single validation regime | Extends NEW.md §3.9 #5 | Reported | 2.6 |
| **Live IC haircut 20–50%** (independent testimony + peer-reviewed 57% cumulative reduction for ML strategies) | **New calibration** | Reported + Demonstrated | 2.7 |
| **Text-signal "sector ghost" check**, and numeric-specificity drift in guidance language | New detail on §3.4 | Reported | 2.8 |
| **Typed causal-operator registry + truncation-invariance test** for any AI-written features | **New** (NEW.md §3.8 has guardrails, not this) | Demonstrated (as a documented failure) / Plausible (as a fix) | 2.9 |
| **Borrow-cost adverse selection**: short-thesis strength correlates with borrow cost/unavailability | New detail | Reported | 2.10 |
| Expected-earnings-in-window as a feature | Conflicts mildly with NEW.md §3.4 #4 (low prior) | Reported | 2.11 |
| Knockoffs / conformal selection / international pooling from adjacent fields | **New (own hypotheses)** | Plausible / Speculative | 3.2 |
| Replication failure of "Attention Factors" stat-arb; LdP pipeline AUC 0.50; TabPFN reality check | Reinforces NEW.md §6 skip list | Reported | 2.12 |

---

## 2. Ideas (detailed)

### 2.1 Asymmetric rank buffer ("buy 10 / hold 50"), no-trade band, partial rebalance, aim-in-front

**Core concept.** Enter a name only when its predicted rank is in the extreme (e.g. top decile), but keep holding it until it falls well outside the entry zone (e.g. out of the top half). Separately, trade only a fraction of the gap to target, and only if the gap exceeds a band. In Gârleanu–Pedersen terms: *aim in front of the target and trade partially toward the aim*.

**Why interesting.** NEW.md §3.9 #4 proposes EMA smoothing and blended horizons; the repo's LP applies a **hard 10% one-way turnover budget** (`docs/LARGECAP.md`). A hard cap makes the LP keep stale names by *feasibility*, not by *design*. An explicit hysteresis rule spends the same turnover on the names where the rank change is most informative.

**Evidence.**
- *Demonstrated (peer-reviewed).* Blitz, Hanauer, Honarvar, Huisman, van Vliet, *Beyond Fama-French Factors: Alpha from Short-Term Signals* (FAJ 2023): MSCI World constituents only (no small/off-benchmark stocks), Dec 1985–Dec 2021, monthly rebalance. Individual signals have 1,300–2,000% annual turnover; with a naive "buy 20 / hold 20" at 25 bp costs the composite's net alpha loses **more than two-thirds** to costs, but with **"buy 10 / hold 50"** the composite's net alpha stays **above 6%**. (Robeco summary; abstract on SSRN 4115411.)
- *Reported.* r/quant, "Minimizing costs for cross sectional strategies" (1rh2h0p): a practitioner with 2.5–2.9 gross Sharpe cut to ~1 by costs is told to use no-trade bands, partial rebalance (25–50% of the gap), and **rank buffering** ("enter top-3, hold until below top-6… one of the biggest cost leaks in top/bottom books"). r/quant "portfolio hysteresis?" (1693f4p): "add a transaction cost term… gives hysteresis for free" or a separate no-trade layer.
- *Demonstrated (theory).* Gârleanu & Pedersen, JF 2013: closed-form optimum = linear combination of current portfolio and an "aim portfolio" (weighted average of today's and expected future Markowitz portfolios); **slower-decaying predictors get more weight**. r/quant threads on combining horizons (1ox4ol7, 1munnm6) converge on "multi-period optimisation" / Boyd's *Multi-Period Trading via Convex Optimization* as the practical route; practitioners say the good version "is guarded IP".

**Source.** https://reddit.com/r/quant/comments/1rh2h0p · https://reddit.com/r/quant/comments/1693f4p · https://reddit.com/r/quant/comments/1ox4ol7 · https://reddit.com/r/quant/comments/1munnm6 · https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4115411 · https://www.robeco.com/en-int/insights/2022/05/beyond-fama-french-alpha-from-short-term-signals · https://www.nber.org/papers/w15205

**Evidence strength.** Demonstrated for the rule's value on a large-cap, monthly universe; not demonstrated for *our* signals or the 2021–26 sample.

**What is new.** NEW.md #10 and §3.9 #4 = EMA of predictions / blended horizons. **Not in NEW.md:** asymmetric entry/exit thresholds, no-trade band, partial rebalancing, or the G–P aim-portfolio formulation.

**Implementation details.**
1. Replace the LP objective's implicit "stay put" with a rank-state machine on the composite/model score: long-enter at score-rank ≤ 10%, long-exit only at rank > 50% (mirror for shorts); LP then only chooses weights over the *allowed* set.
2. Add a cost term to the LP objective (L1 turnover penalty λ·|w − w_prev|) and sweep λ instead of a hard 10% cap; report the implied turnover.
3. G–P-lite: score_aim = Σ_h decay-weight_h · score_h where horizons h ∈ {1m, 3m, 6m} are separate models/composites; trade a fraction (start 0.5) of (aim − current).

**Limitations.** Blitz's signals are *short-horizon, high-turnover*; their headline (1,800% turnover) is **infeasible under the repo's 10% one-way monthly budget**. The buffer rule therefore matters most if the budget is relaxed or if applied to slower composites; on a slow composite the turnover saving is smaller. The 1985–2021 sample is mostly pre-2015; large-cap short-term reversal was reported weaker recently. Buffers concentrate holdings in past winners of the *score* (staleness risk after regime shifts).

**Potential experiments.** On the frozen `largecap.py` composite arm: {hard cap (control), rank buffer 10/30/50, L1 penalty sweep, buffer + partial rebalance}. Pre-registered rule: adopt if net IR ≥ control + 0.1 at equal or lower turnover **and** paired monthly-return t ≥ 1 (IR s.e. is ≈ 0.46, so do not use IR alone).

**Related rabbit holes.** 2.2 (the signals the buffer was designed for), 2.7 (costs are where live decay lives). A commenter with an HFT market-making background (r/quant 1rw4oip) says "I've seen many make money purely on good monetization and bad/no alphas. I've not often seen people make money on bad monetization and good alphas since like 2009" — single testimony from a different horizon, but it agrees with putting the cheap portfolio-layer experiments ahead of new signal work.

---

### 2.2 Blitz short-term composite as a large-cap-native signal block

**Core concept.** Five *cheap* short-horizon signals that survive in liquid large caps and are largely uncorrelated with Fama–French factors: (1) **industry-relative 1-month reversal**, (2) **1-month industry momentum**, (3) **analyst revisions (past 30 days)**, (4) **same-calendar-month return seasonality**, (5) **1-month idiosyncratic volatility**.

**Why interesting.** It is a published composite defined *only* on a large-cap universe, monthly — the exact regime we lose in. The repo's composite deliberately has **no momentum/short-term** content (`LARGECAP.md`: "no momentum"), so this block would be **orthogonal to what we have**.

**Evidence.** *Demonstrated (peer-reviewed).* Robeco summary and the paper's abstract: individual gross returns 5–8%/yr; composite average return > 12%, six-factor alpha > 12% (significant); break-even cost < 25 bp individually, > 30 bp composite; alpha persists out-of-sample and post-publication, across regions, with several-day implementation lags, and uncorrelated with traditional FF factors. Source above.

**What is new.** NEW.md lists industry-relative features (§3.3 #4), residual momentum (#3) and analyst data (§3.7) separately and grades most as [C]. Here is a single **out-of-the-box, benchmark-constituent-only** composite with a cost rule (2.1). The repo already holds `ivol_capm_21d`, `seas_1_1na`, `seas_1_1an`, `seas_2_5na`, `ret_1_0`; **missing** are industry-relative reversal and industry momentum (computable from `ret` + `ff49`) and analyst revisions (needs I/B/E/S).

**Implementation.** Signal 1 = `ret_1_0 − mean(ret_1_0 | ff49, month)` (or value-weighted); Signal 2 = `mean(ret_1_0 | ff49, month)` for the stock's industry; signals 4–5 exist; 3 optional (Round D). Z-score within month, equal-weight, apply 2.1's buffer, *then* test as an additive block on the composite.

**Limitations.** Sample ends 2021 (the paper) and includes the pre-decimalisation-cost era for the older part; our test period 2021–26 contains the July-2025 and Jan-2026 short-term-factor unwinds (Reddit crowding threads, 2.6), which hit exactly short-horizon books. High turnover (see 2.1). Analyst revisions are a large share of the composite's power and are unavailable in the panel.

**Potential experiments.** Paired-IC test of {composite, composite + industry-reversal + industry-momentum} on the ≥ $2B universe; separately with 2.1's buffer. Kill: paired IC t < 1, or gain only in the bottom size tercile.

**Related.** 2.5 (overlap with the commoditised core), 2.6.

---

### 2.3 Era Splitting — invariance-seeking tree splits (and a size-environment variant)

**Core concept.** Standard GBDT chooses the split that maximises impurity reduction over *all rows pooled*. **Era splitting** evaluates the gain **within each era separately** and combines with a smooth min/mean (or a *directional* criterion requiring the split to move the target the same way in every era), so it favours splits that help in *all* periods rather than one dominant regime.

**Why interesting.** Our diagnosed failure is trees that learn small-cap / ivol / lottery structure that does not persist (`FACTOR_FILTER`: universe IC rank corr 0.10; `PM_ABLATION`). That is *precisely* a "split gain concentrated in some environments" problem.

**Evidence.**
- *Reported (authors' abstract).* Era Splitting: Invariant Learning for Decision Trees (arXiv 2309.14496): two new criteria, integrated into GBDTs, "superior performance on the Numerai financial dataset compared to state-of-the-art GBDT". I read only the abstract, not the tables.
- *Implementation exists.* `jefferythewind/scikit-learn-erasplit`: `EraHistGradientBoostingRegressor(gamma=1)`, `fit(X, y, era_column)`; `gain = gamma·era_split_gain + blama·directional_era_split_gain + vanna·original_gain`; `boltzmann_alpha` controls the smooth min/max across eras. The repo publishes **no benchmark numbers**.
- *Related, Reported.* Online learning with **dynamic feature projection** on Numerai (arXiv 2301.00790): LightGBM-dart Sharpe 1.18 → 1.46, Calmar 0.19 → 0.68, GBDTs beat neural nets, and "simple feature engineering paradoxically hurt" (evaluated with Spearman on Numerai data, not US large caps).

**Source.** https://arxiv.org/abs/2309.14496 · https://github.com/jefferythewind/scikit-learn-erasplit · https://arxiv.org/html/2301.00790v4. (Reddit itself has almost no Numerai content — 48 posts — so this lead came from the Numerai literature.)

**What is new.** NEW.md §3.1 e / §3.2 cite Numerai multi-target ensembles and "deep incremental learning", but not tree-level invariance.

**Implementation.** eras = calendar months (≥ 150 eras in-sample). Start with `gamma=1, blama=0, vanna=0` vs the vanilla control on the same features/target; then `blama` directional.

***Own hypothesis (Speculative): size-environment splitting.*** Define eras as **(month × size-tercile)** instead of month. A split only survives if it helps in *large* caps as well as small — an invariance constraint (IRM-style) that directly prices in "edge lives in small caps". Nobody I found has tried this for US equities.

**Limitations.** One published benchmark (Numerai's obfuscated data, ~weekly eras, thousands of names per era); nothing on monthly US large-cap data. Era-min criteria may under-fit and just recreate the composite (which would be a *fine* outcome: it means the invariant signal is the simple one). Tiny custom fork — maintenance risk; check licence and sklearn version.

**Potential experiments.** `et.py`-harness style: {HistGBR vanilla, era-split (month), era-split (month×size)} on the ≥ $2B universe with the composite as `init_score`/baseline. Kill: paired-IC t < 1 vs vanilla, or IC < composite's 0.037.

**Related.** 2.13 (composite-as-prior, reinforced by Reddit), 3.2 (invariance across markets).

---

### 2.4 The neutralisation contradiction: how much factor-neutral is too much?

**Core concept / conflict.** NEW.md §2 #6 and §3.9 #1 recommend neutralising *beyond* beta (size, momentum, vol, quality; Numerai/Barra style). Reddit gives **both** sides:

- *For:* "Barra-neutralise… residual Sharpe and residual IC matter more" (r/quant 1rvb4sm, 12 & 8 upvotes); one desk saw a 2.0 Sharpe fall to ~1.2 *just by removing 1-day reversal*; Toraniko (open-source MIT Barra-style risk model, r/quant 1ekpin6, 179 upvotes) exists to do exactly this.
- *Against (2026, practitioner at a multi-manager HF):* in r/quant 1v30qx3 ("Is anyone in equity stat arb making any money?"), the answer was that the **pain was concentrated in "larger, very short-horizon, tightly factor-neutral" pods**; "**strategies with richer fundamental or alt datasets or those operating under less restrictive risk frameworks held up much better**." A separate top comment (1v02nb2, 91 upvotes) says factor-neutral is "model-dependent" and that a portfolio hard-constrained like other tier-1 pods **converges to the same positions** ("shared core"), so neutralisation to a common risk model *raises* crowding overlap.

**Why it matters.** The repo's LP already neutralises beta + sector. Pushing to size/momentum/vol/quality neutrality (NEW.md #6) could reduce measured factor exposure while *moving the book toward the same constrained feasible set every other neutral book occupies*.

**Evidence strength.** Reported (anonymous practitioner testimony, 2 independent threads; no numbers). The *Sharpe-compression* numbers people quote (30–50% loss going to full-Barra-neutral) are generic and possibly LLM-written — low reliability.

**Implementation / experiment.** Treat neutralisation depth as a **dial with an empirical curve**, not a default: {beta-only, +sector (current), +size, +size+momentum, +size+mom+vol+quality} on the frozen composite and on any candidate signal. For each: net IR, rolling-12m beta, and P&L in the crowded-unwind windows (Jul 2025, Jan 2026, Jul 2026 — see NEW.md §3.9). Use Toraniko or a simple cross-sectional regression for the exposures. The decision rule should weigh *stress-window drawdown* at least as much as full-sample IR.

**Limitations.** With 68 months we cannot tell a real difference from noise; stress windows are 1–3 months each (N≈3). Expect the answer to be "small differences" — the useful output is the *shape*, and the drawdown in 3 windows.

**What is new.** A direct **contradiction** of NEW.md #6 to preserve, not a new technique.

---

### 2.5 "Shared-core" overlap test — is our alpha the commoditised risk premium?

**Core concept.** In the 2026 pod-shop discussion (1v02nb2), capital allocators now ask new research: "**How much of this is just proxying the platform's existing shared core?**" A commenter in 1v30qx3 says the commoditised alphas (vendor-sold, everyone knows them) are really "collecting a *quant risk premium*". Pods with signals "outside the standard factor continuum" (structural anomalies, niche datasets, fundamental idiosyncrasy) get capital.

**Why interesting for us (Own).** The repo's *best and only* result is an equal-weight composite of 18 textbook groups — value, profitability, investment, quality, surprise, ivol/beta, liquidity. That is the **definition of the shared core**. Its 2025 negative year and the Jan/Jul unwinds are consistent with a crowded-core hypothesis. Yet we never measured *how much of any candidate signal is explained by the composite*.

**Supporting practitioner rules.** r/quant 1vvt4m9 ("signal to be neutralised or loaded?", 19 and 8 upvotes on the top answers) gives the decision rule used above: orthogonalise the candidate to the risk model; if it vanishes, it was repackaged beta (neutralise it, unless you actually have a factor-timing edge); if the residual still predicts out of sample, treat the residual as the alpha. One commenter adds a warning relevant to any blending step: "a somewhat predictive signal plus a little of some other predictive signal (or tbh could be beta) = a 'better' predictive signal — chasing Sharpe only can lead to washing out uniqueness of signals" (1rw4oip).

**Evidence.** Reported (crowding testimony) + Own inference. Numerical crowding evidence is in NEW.md §3.9 (MSCI, Hedgeweek).

**Implementation (cheap, no new data).** For any candidate signal s: (i) per-month cross-sectional rank corr and R² of s on the composite; (ii) **orthogonalise s to the composite** each month and re-run the IC/IR test on the *residual*; (iii) report IC of the residual per size tercile. Also (iv) check how the composite's own long/short book correlates with the top-decile of a simple "quality-minus-junk + value" screen.

**Limitations.** A public-factor basket is a crude proxy for the *actual* crowded core (which is defined by managers' positions, not published factors). 13F/portfolio-overlap data is quarterly and lagged (NEW.md §3.6).

**Potential experiments.** For each of the NEW.md rounds' candidate blocks (path shape, 8-K, peer gap, target variants), the *first* diagnostic = residual IC after projecting out the composite. A block that scores IC 0.02 raw but 0.02 residual is worth more than one at 0.03 raw / 0.005 residual.

**Related.** 2.4, 2.6, 2.8 (ghost check is the same pattern).

---

### 2.6 Crowding is a variance-shift: gate on crowding state, and don't calibrate the gate in one regime

**Core concept.** r/quant "Is crowded alpha basically beta now, or is this just cope?" (1u3n4x0, 76 upvotes; top comment 52): informational decay and crowding are **different phenomena** — "in the former, forecast efficacy deteriorates (IC decay); in the latter, the forecast may remain intact, but *monetisation* becomes impaired by liquidity, implementation costs and synchronised de-risking". Alphas increasingly show "**regime-dependent capacity**": stable, then a sharp repricing, not a smooth fade. A follow-up comment: "informational decay is mean-shift, crowding unwind is variance-shift; your backtest captures the mean but not the variance spike." Also: stat-arb capital is allocated on trailing realised Sharpe, so it "crowds performance as much as signals".

**Why interesting.** NEW.md §3.9 #5's gate is on **trailing IC**. If crowding leaves IC intact while breaking P&L, an IC-based gate would *not* fire before Jul 2025/Jan 2026.

**Evidence.** Reported (extended practitioner discussion; the Jan-2026 details corroborated in 1tn45mx: "first three weeks [of January] were horrible… factor/correlation relationships broke down… fresh DD limits"). Consistent with independently reported industry losses in NEW.md §3.9.

**A relevant failure story (Reported).** r/algotrading 1sjicuf (LightGBM ranker, 4 walk-forward folds): the "regime filter" (SPY MA200) zeroed positions on **49.6% of validation dates in the bad fold but 0% of test dates**, and the threshold optimiser selected `enter_thr=0` because the strategy-selection slice had negative Sharpe in *all* folds. Lesson: **abstain/gate parameters tuned on a validation slice inherit that slice's regime and are not evidence about the test regime.** (Also a reminder that mean IC 0.024 with a −0.009 fold is what "IC ≈ 0.03 large-cap" looks like in practice.)

**Implementation (Own).** Candidate crowding-state variables computable from the panel: (a) rolling 3-month cross-sectional **return dispersion of factor-group sleeves** and their pairwise correlation (a jump = unwind), (b) rolling realised beta of the book to a size/momentum/vol basket, (c) 1-month reversal in the *composite's own top-vs-bottom spread* (short-term P&L momentum). Gate/scale gross on (a)–(c) using only data ≤ t; calibrate on the whole pre-2021 history, not a validation fold.

**Limitations.** N of crowding events in our window ≈ 3; any gate that "works" is likely overfit. Vol-management of factors has mixed real-time results (NEW.md §3.9 #6).

**Potential experiments.** Pre-register (a)–(c) and a fixed gross-scaling rule *before* looking at the 2025–26 windows; report only whether it reduces drawdown in Jul 2025 / Jan 2026 / Jul 2026 and by how much it cuts full-sample IR.

---

### 2.7 Expectation calibration: live IC haircut and cost-driven decay

**Core concept.** r/quant "Decline in IC going into prod" (1q02d6g): estimates of how much IC survives production — "**~50%** at a multi-strat across all researchers", "**~40%** on my own book", "**20–40%**" as a general range, and one person's report that a rigorous shop saw ~40% while another large HF with a different research process saw **80–90%**. Advice: "start analysing your true fee/slippage costs versus your simulations." "It is literally not possible to have a rigorously valid hold-out set in this business" — new data arrives too slowly and every idea reuses it.

**Evidence.** Reported (four independent commenters, ranges agree). *Demonstrated (independent):* Azevedo–Hoegner–Velikov, *The Expected Returns on Machine-Learning Strategies* — cumulative performance reduction of **57%** from transaction costs, post-publication decay and post-decimalisation liquidity; sophisticated ML strategies (LSTM-based) still earn net out-of-sample monthly returns of up to 1.42%, but with turnover above 50% and a tilt to difficult-to-arbitrage stocks (per the abstract and Quantpedia summary) (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4702406, AFA abstract).

**Use.** Apply a 40–50% haircut to any reported IC/IR when setting expectations for the deck: composite IC 0.037 → ~0.02, gross IR 0.6 → ~0.3–0.35 by the fundamental law (NEW.md §5). It also tells us how to *prioritise*: an idea that only clears the kill rule by a small margin is very unlikely to survive.

**Limitations.** Anecdotal; "IC of what — a single feature or the ensemble?" was asked and not answered. Not a substitute for our own OOS.

**What is new.** NEW.md §5 states expectations from IC·√breadth but no decay haircut.

---

### 2.8 Text/8-K signals: run a sector-ghost check and try *numeric-specificity drift* (no LLM)

**Core concept.** r/LocalLLaMA 1sqlij0 (single anonymous practitioner, Gemma-4 26B fine-tuned on ~800 of 2,400 earnings-call transcripts, forward-5-day sector-relative label): two signals. **Signal A** — CFOs shifting from numeric guidance ("expect revenue between X and Y") to vague language ("we feel good about our trajectory") → **−1.8% vs sector over 5 days, IC 0.04**, tested on 600 out-of-sample transcripts, "basically zero correlation with momentum, value or any standard factor". **Signal B** — "management confidence" IC 0.09 but **0.85 correlated with sector returns** ("Tech CEOs sound confident when tech is ripping"); killed.

**Why interesting.** Two transferable lessons for §3.4 of NEW.md: (i) **every text-derived score must be sector-neutralised and residualised on `ret_1_0`, `ret_12_1`, `size` before its IC is read**; (ii) the surviving effect was a **structural feature of the text** (numeric specificity), not sentiment.

**Evidence.** Reported only (anonymous, 600-sample OOS, no code, 5-day horizon, earnings calls not 8-Ks). Weak. But the mechanism (text tone ↔ sector return) is well known.

**What is new.** NEW.md §3.4 #6 proposes 8-K text-change similarity; it does not propose a *specificity* feature nor a mandatory sector-ghost check.

**Implementation (Own, cheap).** From the 8-K `text`: digit-token share, count of currency/percentage tokens per sentence, count of forward-looking verbs ("expects", "anticipates") followed by a number vs. not; feature = change vs the same firm's trailing-4 8-K average of the same item. r/algotrading 1ujy65z shows a pure-keyword guidance extractor (median exhibit 99.1 text — our main-text file excludes exhibits, per NEW.md §3.4, so this may be uninformative for 2.02).

**Limitations.** Our 8-K main text is short (median 3.9k chars) and excludes earnings press releases, so guidance language is mostly absent; horizon here is 1 month, not 5 days; large-cap text alpha "dissipates quickly" (NEW.md §3.4 #9).

**Potential experiments.** Round B block: `spec_drift`, residual IC after 2.5's orthogonalisation; kill if residual paired-IC t < 1 on the ≥ $2B universe.

---

### 2.9 If AI writes the features: typed causal operators + truncation-invariance test

**Core concept.** *AQuA: Recursively Self-Improving Quantitative Trading Research Agents* (arXiv 2608.12841) and the r/LocalLLaMA post 1vxajio describe a concrete failure: one LLM wrote an intraday feature dividing "volume so far" by the **day's final total volume**; a second LLM reviewed it "for causality" and **approved the causal-sounding explanation**. It only died on a clean re-split; a manual audit found the leak. The fix in AQuA v2: seal the splits/labels/evaluator outside the agent, and replace arbitrary feature code with a **fixed registry of causal operators** (a full-day normaliser is simply not expressible).

**Why interesting.** NEW.md §3.8 recommends guardrails ("verbatim definition cards, dual reviews, independent re-run"), i.e. more *review*. The failure above shows **review by a similar model fails**; the structural fix is to make leakage inexpressible. A top comment: "a harness where the feature function physically cannot see rows timestamped after the prediction point… then leakage stops being something anyone has to spot."

**Evidence.** Demonstrated as a *failure case* (paper appendix; I saw the abstract and the Reddit summary, not the appendix; no IC published for the failed feature). The proposed fix's effectiveness: Plausible. Note the AQuA abstract also reports IC +0.0843 / Sharpe up to 2.5 on US equities, 2021–25 — I could not verify how (or whether) this handles model-side look-ahead; treat as unverified.

**Implementation (Own, standard, cheap).** (1) **Truncation-invariance test** for every feature: recompute the feature using only rows with `date ≤ t` and assert equality with the value computed from the full panel, for a random sample of (permno, t). This catches leakage regardless of who or what wrote the code. (2) For any feature-mining loop, expose only lag-aware primitives (`lag(x,k)`, `ts_mean(x,k)` with `k≥1`, cross-sectional `rank`, `industry_mean`) — no free-form code. (3) Keep the evaluator frozen (this repo's `largecap.py` harness already is).

**Limitations.** A registry narrows the search space (AQuA acknowledges the tradeoff). Truncation tests catch time leakage, not *survivorship* or *universe* leakage (NEW.md §3.9 #9).

**Potential experiment.** Retrofit the truncation test into the existing feature build and count how many current columns fail — a cheap audit before Round B.

---

### 2.10 Short-side realism: borrow cost is adversely selected

**Core concept.** r/algotrading 1w9yp8g and r/quant 1lpzdwr: the rate you pay is not the average GC rate — "**the reason you want to short is correlated with the borrow becoming expensive or unavailable**"; for HTB names it "can jump an order of magnitude exactly when the short thesis is strongest". A clean backtest that uses a median borrow rate "systematically understates cost in the exact states where the edge is largest". Suggested tests: stress borrow at **2×/5×/10×** and add a **locate-reject / forced-cover flag** as a separate state (a discontinuity in the equity path, not a smooth cost). Availability, not price, is "the bigger problem". Also anecdotes: a PM paper-shorting the same 2k-share inventory every day; the Medallion story that they short "something equivalent" when a locate fails; settlement-calendar arithmetic (a Thu-close/Fri-open short pays 3 days of borrow).

**Evidence.** Reported; mechanism is standard and directly matches our own finding that the ML edge lived in illiquid/biotech shorts (`PM_ABLATION`). Heuristic from practitioners: "price, ADV and industry" predict borrow difficulty; "closed-end funds are expensive".

**What is new.** NEW.md §3.7 lists FINRA short interest as a filter; it does not propose *state-dependent* borrow stress.

**Implementation.** Post-process any candidate book: cost multiplier grid on the short leg; drop any short with price < $10 or ADV below a threshold; forced-cover shock (remove 3–5% of the short book at random each month at a bad price) to see the drawdown tail; report the Jan-2021 month explicitly.

**Limitations.** We have no borrow data; this stresses assumptions rather than measures them. Our current ≥ $2B / ≥ $5 universe already screens the worst offenders.

---

### 2.11 "Earnings in the holding window" as a model input

**Core concept.** r/algotrading 1u19hnl (two live XGBoost momentum models, ~7,600 US stocks, 2015→, monthly rebalance, 10 longs): a single binary `has_earnings_in_window` (earnings date in next 21 days) became the **#1 feature by gain in one model and #5 in the other**, with SHAP treating upcoming earnings as *positive*; a hard "exclude earnings" filter reduced gap risk but lowered CAGR (baselines: 25.3% / 20.2%), and the earnings-feature variant improved drawdown in the growth model to ~−50%. The author still shelved it. Caveat from comments: earnings dates must be point-in-time.

**Evidence.** Reported (a single author; long-only all-cap; no significance test; the feature is an EAP proxy). It **agrees** with the Frazzini–Lamont earnings-announcement premium and **conflicts** with NEW.md's citation that the US premium has disappeared (Heitz et al. 2020) and with the "low prior" grading.

**Implementation.** The 8-K file gives 2.02 filing dates back to 2015: expected-announcement-month = same month last year (and ±3 months). One feature, one test.

**Limitations.** Feature importance ≠ alpha; a binary that identifies *volatile* names can win importance by proxying idiosyncratic vol. Test residual IC after `ivol_capm_21d`.

---

### 2.12 Reinforcements and failure reports (kept short — these support NEW.md's skip list)

| Report | Finding | Grade |
|---|---|---|
| **"Tried to replicate the Attention Factors stat-arb paper…" (r/quant 1w8otws, Sep 2026)** | Top-500 Russell-1000 point-in-time, survivorship-free, 2016–2026, 38 rank-normalised chars, 5 bp + 1 bp shorts: **K=30 mean OOS Sharpe −0.70** (paper: net 2.30 on 24 yrs). Deterministic PCA-residual reversion negative every year: gross −0.59%, costs 5.86%, **turnover 9,190%**. Weekly 1/N same window = **+1.36**. Top comments: "I don't bother with papers that report miraculous SRs and no repo". Post reads as Claude-assisted; single split; different sample length; author found and fixed look-ahead and sign-symmetry bugs. | Reported (not a refutation; the paper's abstract claims OOS Sharpe > 4 gross / 2.3 net on the largest US equities) |
| "Built a full Lopez de Prado pipeline in Rust… AUC = 0.50 OOS" (algotrading 1s77s5k) | 442 tests, CPCV, HMM, meta-labelling, permutation test p = 0.60 → no edge; individual feature IC is real but does not predict *which events win*. Top answer: you built a prediction pipeline before proving there was something to predict. Also 1pm20qy: "No one uses López de Prado's methods, because they don't work" (15 upvotes) vs a 419-upvote meta-labelling explainer (1lnm48w). **Disagreement preserved.** | Reported |
| "Cross sectional IC ranked signal ideas" (algotrading 1vowexr) | A practitioner who tried "thousands of features, ML rankers, z-scores": "*you can basically slightly leverage QQQ by cross-sectional selection*… for every bit of alpha you buy more risk… no signal worth trading". Also: fixing split-adjustment errors **removed** vol/fundamental alpha; top-5% momentum picks worsen *sector concentration*; use rank IC and **sector-neutralise before evaluating**; rare-firing signals cannot work at a monthly rebalance — use always-on graded scores. | Reported |
| r/MachineLearning "What do you think about Tabular Foundation Models" (1thofoe) | TabPFN-3 "basically comparable with a good gradient boost", "somewhat worse than LightGBM with light tuning", heavy inference; **licence: "Non-Commercial Purpose… provided the results are not used in commercial decision-making"**. | Reported |
| r/datascience 1tip0d8, 6-model swap | NN best CAGR; RF/LASSO didn't beat S&P. **Discounted**: default hyper-parameters, unspecified universe, non-walk-forward, target = 10-day forward return, monthly rebalance; top comment (158 upvotes) says exactly this. | Reported, near-worthless |
| r/quant 1jju8jj, 1l59in9, r/algotrading 1progoo, 1iliivd | Consensus: linearise features, start regularised-linear; "ML enhances an existing edge, never found one"; **top-voted fix (60 upvotes): "encode your heuristic as a feature"** — an independent statement of NEW.md's composite-as-prior. | Reported |

---

### 2.13 Additional small findings worth carrying

- **Alpha combination: "close the loop."** r/quant 1w99f1v: for large alpha libraries, evaluate combination methods with the *actual portfolio backtest* (top comment suggests a **Shapley decomposition over the alpha set** across a grid of combination methods and targets); "quite a lot of big wins come from knowing when *not* to listen to a signal". Consistent with the repo's `pm_ablation.py`; new element = Shapley/marginal-contribution over blocks. Reported.
- **Signal prep tradeoff.** 1j7sl2j: EMA-smoothing lowers IC but raises autocorrelation; fitting to forward returns and then MVO "compounds estimation error" — keep to rank + inverse-vol weights. Reported.
- **Definition of IC.** 1rmy6w3: unqualified "IC" = `rank_corr(s, r)`; for monthly books, report raw IC, sector-neutral IC and risk-model-residual IC — our `PM_ABLATION` style. Reported.
- **Power analysis before backtesting** (1r6rc2d): decide the smallest detectable effect first; with 68 months and IR s.e. 0.46 (NEW.md §5) this is the paired-IC discipline already used.
- **Open-source resource:** Toraniko (MIT, numpy+polars, Barra-style market/sector/style factors, "used in production on > $10B AUM", no covariance shrinkage yet, US-only) for 2.4/2.5. https://reddit.com/r/quant/comments/1ekpin6
- **Macro conditioning, weak evidence:** the only feature that moved a 1,400-stock GBT+MLP long-horizon ensemble after 45 rounds of price/volume feature engineering was **yield-curve slope** (algotrading 1rwp4z8); a commenter also flags that 45 rounds of feature-engineer→WFO→repeat is itself an overfitting mechanism. Consistent with NEW.md §3.3 #7 (low priority, high overfit risk).
- **Pod-shop process observation (1v30qx3, Reported):** a centrally netted book pays less spread/impact/financing than several pods trading the same names, and "keeps running lower-Sharpe durable signals through bad stretches because nobody is cut at their drawdown limit". Not actionable for a hackathon; supports the view that costs, not alpha, separate outcomes.
- **Index-event alpha migration (1ppnw7t):** index inclusion/deletion effect is "a stupidly crowded trade", "progressively noisier"; one Vanguard manager's comment that the complex rebalancing is "handed off to our arbitrage desk". Dead end for us; noted so nobody spends time on it.

---

## 3. Novel Ideas

### 3.1 Evidence-backed novel ideas (relative to NEW.md)

1. **Asymmetric rank buffer + partial rebalance / G–P aim portfolio** (2.1) — the only idea here that combines a *large-cap, monthly, peer-reviewed* result with independent practitioner advice. *Demonstrated (rule), Reported (implementation folklore).*
2. **Blitz short-term composite** as a new, orthogonal, large-cap-native block (2.2). *Demonstrated (peer-reviewed), not on our sample.*
3. **Live-IC haircut for expectations** (2.7). *Reported + Demonstrated (57% cumulative reduction).*
4. **Truncation-invariance test and a typed causal-operator registry for AI-generated features** (2.9). *Demonstrated failure mode; fix Plausible.*
5. **Era Splitting** as the tree-level answer to "splits that only work in some regimes" (2.3). *Reported (authors' claim on Numerai).*
6. **Contradiction to preserve:** neutralisation depth (2.4). *Reported.*
7. **Sector-ghost check for text signals** and **numeric-specificity drift** (2.8). *Reported.*
8. **State-dependent borrow stress** (2.10). *Reported.*

### 3.2 New hypotheses (my own — not attributable to Reddit or a paper)

| # | Hypothesis | Grade | Why I think it could work | First test |
|---|---|---|---|---|
| H1 | **Size-environment invariant trees**: era-split with eras = month × size tercile, so a split must help large caps too (2.3) | Speculative | Directly encodes "our edge is small-cap" as a constraint the learner must satisfy; converts the failure into a regulariser | `EraHistGradientBoostingRegressor` vs vanilla on ≥ $2B universe |
| H2 | **Orthogonal-residual training**: train the ML on `y − β·composite_score` (or with composite as `init_score`) *and* require zero rank corr with the composite at prediction time, so ML can only produce what the shared core does not (2.5 + NEW.md #2) | Plausible | Two independent practitioner streams ("encode the heuristic", "shared core") point at the same design | Residual IC by size tercile; kill if < 0.01 |
| H3 | **Neutralisation dial + crowding-stress objective**: pick neutralisation depth by drawdown in the 3 unwind windows, not full-sample IR (2.4/2.6) | Plausible | Resolves the contradiction empirically | 5-step dial, report stress DD |
| H4 | **Feature-block knockoffs**: use a knockoff / permutation null to control the false-discovery rate across the ~25 candidate blocks NEW.md lists. Literature exists for knockoffs in the factor zoo (robust knockoffs 2206.06026; "Controlling FDR under cross-sectional correlations" 2102.07826; NBER "Taming the Factor Zoo" w25481); I found no evidence of use in a *monthly large-cap ML ranking* pipeline | Plausible | NEW.md already demands a multiple-testing count; knockoffs would give an actual FDR-controlled *selection*, not just a Bonferroni discount | Build within-(month, industry) shuffled null for each block's paired-IC gain; adopt only blocks above the 95th percentile of the null-max |
| H5 | **Conformal selection of names**: use model-agnostic conformal selection (finite-sample FDR-controlled subset selection) to choose the long/short set so that the expected share of "wrong" picks is bounded | Speculative | Directly targets the concentration-on-noise problem; a natural "abstain" that is theoretically calibrated, unlike gating on volatile validation slices (2.6). Adjacent literature: conformal selection framework; *Conformal Predictive Portfolio Selection* (2410.16333) is portfolio-level, not name-level | Needs calibration set of past months; likely too little data (68 months) — a toy test on the in-sample folds first |
| H6 | **Pooled international large-cap training**: 148-characteristic panels exist for 46 markets (Cakici–Fieberg–Metko–Zaremba, JEDC 2023: predictability depends on firm size and recent information); pooling developed-market large caps could multiply the effective training sample | Speculative | Sample size (68 test months / 1,200 names) is the binding constraint; more independent cross-sections is the classical fix. *Unverified*: whether FIAM's rules allow external data of this form, and whether characteristic definitions align with our 147 | Feasibility check first (rules + data); if allowed, train on US+Europe/Japan large caps, test on US large caps |
| H7 | **Graded event intensity instead of flags**: convert sparse 8-K event flags (4.02, 4.01, 5.02, 2.05) into exponentially decayed *intensity* scores (half-life 3–6 months) so they exist every month (algotrading 1vowexr: rare-firing signals cannot work at a monthly rebalance) | Plausible | Solves a real mismatch (event alphas are sparse; cross-sectional ranking needs every-month coverage) | Add decayed counts as features; paired-IC on the ≥ $2B universe |
| H8 | **Buffered composite + era-split residual model as a two-stage system** (2.1 → 2.3): stage 1 = buffered composite (low turnover); stage 2 = invariant-tree adjustment allowed only to move a name across the buffer boundary | Speculative | Uses ML where it is cheapest (boundary decisions) and the composite where it is best (core ordering) | Only after 2.1 and 2.3 individually pass |

---

## 4. Best Experimental Candidates

I am not scoring; these are the factors that make each one worth running, in the repo's pre-registration style. Note that the repo's existing NEW.md "Round A–D" plan is unchanged; these slot in.

| Candidate | Evidence | Novelty vs NEW.md | Cost | Impact if it works | Main uncertainty |
|---|---|---|---|---|---|
| **2.1 Rank buffer / L1 penalty / partial rebalance** on the frozen composite | Peer-reviewed large-cap monthly result + independent practitioner advice | High | **Very low** (no new data, one LP change) | Direct: the repo already saw turnover as the recurring drag | Do gains survive the 10% one-way budget? |
| **2.5 Orthogonalise-to-composite diagnostic** | Practitioner testimony + own logic | High | **Very low** (regression per month) | Changes how *every* other candidate is judged | Composite is only a proxy for the real crowded core |
| **2.9 Truncation-invariance audit** | A documented real failure | Med | **Very low** | Protects all later results (and the brief's leakage scrutiny) | None — cheap insurance |
| **2.2 Industry-relative reversal + industry momentum** | Peer-reviewed | Med | Low (`ret` + `ff49`) | New orthogonal large-cap block | Needs 2.1 to be affordable; 2021–26 hostile to short-term factors |
| **2.3 / H1 Era-split trees** | Authors' claim on one dataset; implementation exists | High | Medium (compile fork, CPU-only) | Could let a fitted model finally add to the composite | May underfit and reproduce the composite (an acceptable outcome) |
| **2.4 / H3 Neutralisation dial with stress windows** | Contradictory testimony | Medium (a contradiction) | Medium | Resolves the biggest open decision in NEW.md #6 | N = 3 stress months |
| **H7 Decayed event intensity** | Plausible, own | Med | Low | Makes 8-K block usable at monthly cadence | 8-K value in large caps is unproven |
| **H4 Block knockoffs/permutation null** | Literature exists, no direct evidence for this setting | Med | Medium | Formal discipline for ~25 candidate blocks | Time-series correlation breaks exchangeability; needs care |

---

## 5. What We Still Don't Know

1. **Do our signals survive a rank-buffer under a 10% turnover budget?** Blitz's evidence is for a much higher-turnover regime (2.1).
2. **Neutralisation depth: does harder neutralisation reduce or increase crowded-unwind losses?** Two opposing anecdotes; no quantitative source found (2.4).
3. **Does Era Splitting help on monthly US large caps at all?** Only Numerai (weekly, obfuscated) evidence; the fork ships no benchmarks (2.3). Also unknown whether month is the right era length for ~150 in-sample eras.
4. **How much of our composite is a "shared core"?** No public measure exists; 13F overlap is lagged and quarterly (2.5).
5. **Is the 40–50% IC haircut applicable to a long-only-in-factor-composite book, or mostly to fitted models?** The Reddit figures did not say what type of signal they referred to (2.7).
6. **Was the Attention-Factors negative replication a bug or a real non-transfer?** No code was released by the authors; the replication is single-split, 10 years, and the poster fixed several bugs during the process (2.12). Needs an independent replication before the paper can be used as *evidence* against.
7. **Do tabular foundation models (TabPFN-3 / 3.5) do anything on monthly cross-sectional returns?** Nobody reports; licence excludes commercial decision-making (2.12).
8. **Is a knockoff/conformal approach feasible with ~68 test months and serially correlated cross-sections?** Nothing found (H4/H5).
9. **International pooling:** does FIAM's rules permit it, and does it help a US large-cap target? Completely unexplored here (H6).
10. **Where the research is sparse.** Reddit contains almost nothing on: rank/gauss-rank targets (the topic NEW.md ranks #1), target regularisation, multi-target ensembles, or 8-K/text-change signals in large caps. A grep of the full 40k-post local corpus for "gauss / rank target / rank transform / lambdarank / learning to rank / neutraliz" returned 87 matches, of which only about six are on-topic (the rest are "Gaussian" distributions, copulas and HMMs): 1c6t9b0 (rank vs regression), 1jzt8h9 and 1vvt4m9 (neutralisation), 17m27fe (normalisation), 1rw4oip (raw signal), 1s77s5k. The literature in NEW.md is therefore the *only* basis for the target-design ideas; Reddit neither supports nor contradicts them beyond one thread (1c6t9b0) arguing that quantile targets lose information needed by a downstream optimiser but are a good first hypothesis test (top comment: "beta test with percentiles, then trade the actual return predictions").

---

## 6. Search log, dead ends, and duplicates

**Branches that produced substantive material:** trading-cost/turnover control → Blitz + G–P + Boyd; crowding → 4 pod-shop threads; alpha decay in production; neutralisation; text/LLM signals → ghost check + AQuA; ML success/failure reports; borrow-cost realism; Numerai tree methods (outside Reddit).

**Dead ends (no new information):** r/Numerai (48 posts); careers threads; retail "my bot made X%" posts (r/algotrading 1m5keqj "~1% return/day stat-arb" (1,180 upvotes), 1u6w3ra "gold mine" (354), 1progoo …) — discounted per REDDIT.md §4; index-event alpha; satellite/alt-data folklore (197mah6); crypto cross-sectional threads; options-skew hobbyist threads in r/options (no cross-sectional return-prediction content); r/econometrics ML threads; the r/datascience 6-model swap; r/SecurityAnalysis (human-analyst content, nothing on 8-K/4.02/insider factors beyond retail "cluster buy" claims — one commenter: insider-purchase screens are "one of the worst performing screens"; another: "3+ insiders in a 2-week window" is the strongest signal from a 10-year backtest by the tool's author — contradictory, unquantified).

**Duplicates of NEW.md** (kept out of the main text unless there was new evidence): residual/industry momentum, PEAD dead in large caps, 8-K item flags, Numerai neutralisation & multi-target ensembles, LLM look-ahead, momentum crash & crowding narratives, sequence models and foundation models on the skip list.

**Adversarial final round (searched for what NEW.md/REDDIT.md would not have surfaced).** Turnover-control theory (found), production IC decay (found), tree-level invariance (found), AI-feature-leak infrastructure (found), borrow adverse selection (found), multiple-testing/knockoffs (literature found, no practice), conformal selection (literature found, no practice), international pooling (literature found, no direct evidence), synthetic-data / data-augmentation / self-supervised pre-training for cross-sectional returns (Reddit: nothing substantive — threads are retail synthetic-OOS generators), hierarchical/Bayesian shrinkage and Fama–MacBeth style panel methods (nothing on Reddit), regime detection (retail-dominated, HMMs; nothing with cross-sectional evidence), earnings-window features (found, weak).

---

## 7. Source index

**Reddit (permalinks are `https://reddit.com/r/<sub>/comments/<id>`)**
- Turnover / costs: quant 1rh2h0p, 1693f4p, 1ox4ol7, 1munnm6, 1j7sl2j
- Crowding / pod shops: quant 1v02nb2, 1u3n4x0, 1tn45mx, 1v30qx3, 1ppnw7t, 1r5r2u7
- IC decay & evaluation: quant 1q02d6g, 1rmy6w3, 1rvb4sm, 1rsok6e, 17qmqle, 1r6rc2d, 1w99f1v, 1krbu6a
- Target/rank/model choice: quant 1c6t9b0, 1jju8jj, 1l59in9, 1pm20qy; algotrading 1kqla8u, 1iliivd, 1progoo; datascience 1tip0d8; MachineLearning 1thofoe
- Failures / replications: quant 1w8otws; algotrading 1s77s5k, 1sjicuf, 1vowexr, 1rwp4z8, 1byamsp
- Text / LLM / leakage: LocalLLaMA 1sqlij0, 1vxajio; algotrading 1ujy65z, 1u19hnl
- Short side: algotrading 1w9yp8g; quant 1lpzdwr
- Tools: quant 1ekpin6 (Toraniko)

**External (verified by fetch/search in this session; all abstract- or summary-level unless stated)**
- Blitz et al., *Beyond Fama-French Factors: Alpha from Short-Term Signals* (FAJ 2023) — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4115411 · https://www.tandfonline.com/doi/abs/10.1080/0015198X.2023.2173492 · numbers from https://www.robeco.com/en-int/insights/2022/05/beyond-fama-french-alpha-from-short-term-signals
- Gârleanu & Pedersen, *Dynamic Trading with Predictable Returns and Transaction Costs* (JF 2013) — https://www.nber.org/papers/w15205
- Azevedo, Hoegner, Velikov, *The Expected Returns on Machine-Learning Strategies* — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4702406 · https://quantpedia.com/the-expected-returns-of-machine-learning-strategies/
- Era Splitting — https://arxiv.org/abs/2309.14496 · https://github.com/jefferythewind/scikit-learn-erasplit
- Online learning / dynamic feature projection on Numerai — https://arxiv.org/html/2301.00790v4
- AQuA — https://arxiv.org/abs/2608.12841
- Attention Factors for Statistical Arbitrage — https://arxiv.org/abs/2510.11616 (abstract claims OOS Sharpe > 4 gross, 2.3 net, largest US equities, 24 years)
- Multi-Horizon Equity Returns Predictability via ML (Nechvátalová) — https://www.econstor.eu/bitstream/10419/247369/1/wp2021-02.pdf (summary-level only; PDF saved locally)
- Machine Learning Goes Global — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4141663
- Knockoffs / FDR — https://arxiv.org/html/2206.06026 · https://arxiv.org/pdf/2102.07826 · https://www.nber.org/system/files/working_papers/w25481/w25481.pdf · https://arxiv.org/pdf/1404.5609
- Conformal — https://arxiv.org/abs/2410.16333

**Caveats on this document.** I did not read full paper bodies for Blitz, Era Splitting, AQuA, Attention Factors, Nechvátalová or Azevedo et al.; numbers come from abstracts and reputable summaries and should be re-checked in the papers before appearing in a deck. All Reddit content is unverifiable testimony; where a claim came from a single commenter it is labelled as such.
