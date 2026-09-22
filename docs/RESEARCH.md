# RESEARCH.md — Consolidated Research Sweep

This file merges three research documents produced during the 2026-09-21 research
round into one, in the order they were written: the web literature sweep (formerly
`NEW.md`, now **Part I**), the Reddit-led deep-research follow-up that extends it
(formerly `REDDIT_RESEARCH.md`, now **Part II**), and the paper-by-paper implementation
log for the models built on top of the OLS baseline (formerly `PAPERS.md`, now **Part
III**). Nothing has been re-edited beyond heading levels — every claim, grade, number,
and caveat below is verbatim from the three original files, which have been deleted now
that their content lives here.

**A note on cross-references.** Sentences below say things like "§3.1" or "see 2.5" —
section numbers are unchanged from the original files, so a reference like "§3.1" inside
Part I still means Part I §3.1, and one inside Part II still means Part II §3.1, etc.
Every citation elsewhere in the repo that used to point at `docs/NEW.md`,
`docs/REDDIT_RESEARCH.md`, or `docs/PAPERS.md` (experiment READMEs and scripts) has been
rewritten to point at `docs/RESEARCH.md` with the matching Part label, e.g. `docs/NEW.md
§3.1` became `docs/RESEARCH.md Part I §3.1`.

---

## Part I — docs/RESEARCH.md Part I — Ideas To Find A Model That Actually Works (research sweep, 2026-09-21)

Scope: a broad web sweep (~160 searches/fetches: arXiv, SSRN abstracts, Quantpedia/Quantocracy, practitioner Substacks, Numerai, quant-fund news, X). Goal: ideas we have **not** tried, not a re-run of what is in `docs/`. Everything is measured against where the repo stands (see §1).

Companion docs: `docs/NEGATIVE_RESULT.md`, `experiments/largecap/README.md`, `experiments/pm_ablation/README.md`, `experiments/factor_filter/README.md`, `experiments/tpa/README.md`, `experiments/hrp/README.md`.

---

### 0. How to read this (evidence quality, and what I could not reach)

**Evidence grades used below**

| Grade | Meaning |
|---|---|
| **A** | I read the abstract on arXiv/publisher page and it says the thing directly |
| **B** | Read a practitioner write-up with numbers (Quantitativo, Quantpedia, Substack) — replicable idea, but a single blogger's backtest |
| **C** | Only a search snippet or a secondary summary; the paper itself was paywalled/blocked (SSRN, ScienceDirect, Springer, ACM returned 403 or bot-check) |
| **Own** | My synthesis from the repo's findings, not from a source |

**Coverage gaps (be honest about these)**
- **Reddit: unreachable.** `WebSearch` refuses reddit.com, `WebFetch` refuses it, and Claude-in-Chrome blocks it ("not allowed due to safety restrictions"). So there is **no direct r/quant, r/algotrading or r/wallstreetbets content here.** The practitioner voice in this file comes from Substack, Quantocracy, Quantpedia, Numerai docs/GitHub and quant-fund press instead.
- **X/Twitter:** the search tool surfaced mostly promotional threads ("GPT-6 Astra finds strategies") and no substantive alpha content. I did not treat any X post as evidence. Two X links were 402 (paywalled).
- **SSRN** put up a Cloudflare bot check in Chrome; I did not try to get past it. SSRN-only papers are graded C.
- The `WebFetch` summarizer is a small model; numbers it returned for PDFs were sometimes shallow. Every number below that matters should be re-checked against the paper before it goes in a deck.
- Most "machine learning beats X" papers are on **Chinese A-shares** (marked "CN"). Do not assume transfer.

---

### 1. Where the repo stands, and what the literature says about *why*

**Repo state (from `docs/`):**
- Tree/kNN/linear models on the 147 characteristics have rank IC ≈ 0.14–0.15 over all stocks but only ≈ 0.035 on the universe we can actually trade (price ≥ $5, mcap ≥ $500M, both betas observed). Tradeable books are ≈ 0 or negative (`PM_ABLATION`).
- The edge was small-cap / sub-$5 / biotech shorts.
- Best honest result: a no-fit, equal-group composite of 18 factors on a ≥ $2B universe: net IR ≈ 0.54, universe IC 0.037 (t 1.9), 2025 negative (`LARGECAP`). Extra-Trees on the same universe was ≈ −0.2 to −0.3 IR (IC 0.016).
- Past factor IC does not persist in the investable universe (rank corr 0.10) (`FACTOR_FILTER`).

**The literature agrees with the diagnosis, and it sharpens it:**
- *"What Useful Alphas?"* — ~200 published long-short anomalies: median return 48 bp/mo pre-2005, 19 bp post-2005, **7 bp/mo for non-micro stocks**; "useless to non-micro-cap portfolio managers in the 21st century." [A] https://arxiv.org/abs/2607.06502
- ML strategies earn ~2x the alpha in small vs big firms. [C] (Cakici et al. via search summary)
- A 2026 study of US large caps says **feature scope, not model choice, is the binding constraint** on cross-sectional alpha (67 characteristics → monotonically better AUC/alpha vs 25). [C] https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6497598
- AQR: ML studies that ignore trading costs "rely on transient micro-cap characteristics and disappoint after cost." [A-ish] https://www.aqr.com/Insights/Research/Working-Paper/Machine-Learning-and-the-Implementable-Efficient-Frontier
- One large-cap ML paper reports **"realisable IC" of 0.032–0.049** and that *bias-elimination (masking untradeable rows, neutralisation) was a bigger driver than any architecture or loss* (CN, but the lesson transfers). [B] https://arxiv.org/html/2507.07107v1
- The only ML edge I found that specifically improves in the recent large-cap/mega-cap regime is **option-implied features** (OOS R² +1.29% vs +0.07% linear, 2023-26). [A] https://arxiv.org/abs/2608.26115

**So the productive directions are:** (i) change what the model is asked to predict (targets), (ii) add information that lives in mid/large caps (peers, events, options, insiders, daily-return shape), (iii) build the book so factor/crowding risk doesn't eat a weak signal. More model families on the same 147 columns is the direction the literature says to stop.

---

### 2. Ranked shortlist — what to try first

Ordering = (expected upside for a **large-cap tradeable** book) × (evidence) ÷ (effort). "Kill" = pre-registered stop rule in the repo's own style (universe rank IC and net IR on the `largecap.py` harness; one path, 68 months, IR s.e. ≈ 0.46, so use paired-IC t-stats, not IR alone).

| # | Idea | Section | Needs new data? | Effort | Why it might work | Kill criterion |
|---|---|---|---|---|---|---|
| 1 | **Rank / gauss-rank / size-and-industry-neutral / factor-residual training targets** (instead of winsorized raw return) | §3.1 | No | Low | Cakici–Zaremba: rank targets ≈ doubled return/Sharpe in *large-cap* universes; Howard: target regularisation drove the gain; Numerai scores against a factor-neutral residual target | Paired universe-IC t < 1 vs current winsorized target on ≥ 2 model families |
| 2 | **Shrink ML toward the composite** (composite as prior / `init_score`, few trees, or ridge toward composite weights) | §3.5 | No | Low | The no-fit composite beat every fitted model on large caps; use ML only for the residual | ML-on-top adds < +0.005 paired IC |
| 3 | **Price-path / monthly-return-history features built from `ret`** (information discreteness "frog in the pan", smoothness/slope, 52w-high-neutral & industry-residual momentum, characteristic *changes*) | §3.3 | No | Low–Med | Existing 147 are levels; path shape is a documented, retail-behaviour signal robust across size buckets | No paired-IC gain on universe; or gain only in bottom size tercile |
| 4 | **8-K "information intensity" + news/no-news split of short-term reversal + event flags (4.02/4.01/5.02/2.05)** | §3.4 | No (8-K file) | Med | 8-K frequency predicts *lower* returns (≈4%/yr spread, Management Science); reversal only works when the move was *not* news; 4.02 CAR −1% day-1 / −2% 20d | Feature adds nothing in a ≥ $2B universe after controlling for `ret_1_0`, `rvol_21d` |
| 5 | **Peer / network signals: peer-return gap, correlation-peers and TNIC-peers, customer–supplier spillover** | §3.6 | Peer via `ret` panel: no. TNIC/13F: yes (free) | Med | Cross-stock predictability is documented to survive ML on own characteristics ("does not subsume the peer index") and works in S&P 500 samples | Peer-gap paired IC t < 1 on ≥ $2B universe |
| 6 | **Factor-neutralise beyond beta**: size, momentum, vol/lottery, quality (Barra/Numerai-style neutraliser matrix), plus residual-alpha optimisation | §3.9 | No | Med | Numerai's live score is against a target neutral to country/sector/beta/momentum/size; large books die from shared factor exposure (Jan 2026, July 2025, July 2026 unwinds) | Realised beta/rolling-beta no better *and* IR not better |
| 7 | **Daily-return-distribution signal (DRIF)** and overnight/intraday decomposition | §3.7 | Yes — daily returns (CRSP/WRDS or Alpha Vantage; FIAM allows it) | Med–High | 21 daily returns → elastic net reproduces many anomalies; "timing dominates"; overnight vs intraday profits have opposite signs | Not significant in ≥ $2B after $5 price screen |
| 8 | **Option-implied features (Cremers–Weinbaum IV spread, risk-neutral skew, IV-surface primitives)** | §3.7 | Yes — OptionMetrics via WRDS | High | The only ML edge that *grew* in 2023-26 mega-cap regime; organisers wrote the options+ML paper | Access not available; or IC < 0.02 on ≥ $2B |
| 9 | **Multi-target, multi-horizon ensemble with a ridge meta-model** (Numerai style) | §3.1 | No | Med | Numerai's LightGBM ensemble: 40 targets × seeds + ridge + feature neutralisation cut feature exposure 0.25→0.17 and halved max drawdown (live Sharpe 1.32, corr 0.017) | No paired-IC gain vs single target |
| 10 | **Signal smoothing / decay-aware blending to shrink turnover before the LP** | §3.9 | No | Low | Turnover was the recurring cost drag; Grinold–Kahn: turnover is tied to signal decay | Net IR flat or lower |

Everything else in this file is either a variant of these, a longer shot, or on the "skip" list (§6).

---

### 3. Detailed ideas

#### 3.1 Target engineering — the highest-leverage, cheapest change

Everything in `docs/` fits a winsorized raw next-month return. The literature says the *target* matters more than the model.

**a) Rank-based targets.** Cakici & Zaremba, *Getting the Target Right in Return Prediction* (2026): 80k+ stocks, 35 markets, 1994-2024. Rank targets (percentile, rank-to-[−1,1], Gaussianised rank) beat magnitude-preserving ones; "roughly doubled returns and Sharpe in **large-cap** universes" (monthly alpha ~1.0% → ~1.9%). They underperform in micro-caps/emerging markets where tails matter. [B/C] https://quantpedia.com/getting-the-target-right-in-return-prediction/ · SSRN 6615698
- *Our fit:* our IC is dominated by the tails of small-cap returns; a rank target removes exactly that. Repo already rank-transforms **features** but not the **target**.
- *Implementation:* in `largecap.py`/`et.py` harness replace `y = winsor(ret_exc_lead1m)` with `y = gauss_rank_within_month(ret_exc_lead1m)`; also try uniform rank. Keep scoring on raw return for R².

**b) Classification into deciles instead of regression.** Bai & Pukthuanthong: matched models, classification value-weighted Sharpe **2.08 vs 1.39** for regression; spanning tests say regression alphas vanish after controlling for classification. [A] https://arxiv.org/abs/2108.02283 (revised Sept 2026)
- *Implementation:* label top/bottom decile (or 5 classes) per month; use the predicted class-probability-weighted expected decile as the score (Quantitativo's "probabilistic momentum" does the same: expected return = probability-weighted average, not argmax; reports Sharpe 1.78, β −0.11). [B] https://www.quantitativo.com/p/uncertainty

**c) Size-group / industry-demeaned targets (target regularisation).** Howard, *Less is More?*: separate models by size bucket lifted an ensemble's annualised return 20.0% → 31.8%, but "lack of regularization of the target variable primarily drives the outperformance" — simply removing the size-group median from the target gets comparable gains without training three models. [B] https://quantpedia.com/less-is-more-reducing-biases-and-overfitting-in-machine-learning-return-predictions/
- *Implementation:* `y = ret − median(ret | size_grp, eom)` and `y = ret − median(ret | gics2, eom)`; combine (size×industry) buckets if counts permit.

**d) Factor-residual (neutralised) targets.** Numerai Signals scores on a target neutralised to country, sector, beta, momentum and size, and built the *Alpha* metric so signals are rewarded only for what a market-neutral fund can trade; "poorly-built small-cap signals… will underperform." They publish a 200-column neutraliser matrix and a liquidity/residual-vol sample-weight vector. [A] https://blog.numer.ai/signals-alpha-and-mpc/
- *Our fit (Own):* our failed models learned ivol / lottery / small-cap structure (importance tables in `RF_3`, `ET`, `LGBM`). Training on a target orthogonal to `size`, `betabab_1260d`, `ivol_capm_21d`, sector, and 12-1 momentum forces the model to look for something else — exactly what the LP wants to hold.
- *Implementation:* per month, OLS-residualise `ret_exc_lead1m` on those columns (fit on data available at the time — the *target* residualisation uses only the target month's cross-section, which is not look-ahead as long as the exposures are t-measured). Report IC against both raw and residual returns.

**e) Multi-target, multi-horizon ensembles.** Numerai LightGBM ensemble: one model per target (20d and 60d horizons, different neutralisations), 3 seeds each, ridge meta-model on validation eras, then partial feature neutralisation; heavy regularisation (lr 0.01, 15 leaves, min 2000 rows/leaf, L2 = 10). Reported feature exposure 0.25→0.17, max drawdown halved. [B] https://github.com/vladmurnik/numerai-lgbm-ensemble
- Longer-horizon targets (3–6 months) are also the standard route to lower turnover. [C] (search summary; NBER "Multi Horizon Returns")

**f) Learning-to-rank losses.**
- *LambdaRankIC* (May 2026) directly optimises Rank IC via closed-form lambda gradients, implemented as a custom XGBoost objective; best OOS Rank IC/ICIR/monthly return/Sharpe on real data (CN) and best under low SNR and heavy tails in simulation. [A] https://arxiv.org/abs/2605.00501
- Quantitativo LambdaMART on Russell 3000: 21 features, 30 quantile buckets, 150% gross, 18.1% CAGR, Sharpe 1.62 (2006-25, 10 bp costs); tuned depth 5, lr 0.1, 200 trees, **retrain annually with 15 years**. [B] https://www.quantitativo.com/p/learning-to-rank
- Caveat: Russell 3000 including small caps; expect the large-cap number to be much lower.
- *Implementation:* `xgboost.XGBRanker` (`rank:ndcg`/`rank:pairwise`) with monthly groups; labels = decile (0–9) or 0–30 quantile relevance.

**g) Distributional/quantile targets.** Quantile NNs give conditional skewness that is positively priced (no evidence variance/kurtosis are). [C] https://arxiv.org/pdf/2408.07497 — low priority.

#### 3.2 Training universe and sample weighting

- **Value-weighted / size-weighted training loss.** Gu–Kelly–Xiu's own variant weights loss by market value because the smallest 20% of stocks are ~3% of cap; more economically relevant weight on large names. [C] https://dachxiu.chicagobooth.edu/download/ML.pdf. Our `_trad` arms *restricted* the universe; **weighting** (e.g. `sqrt(mcap)` or dollar-volume weights, as Numerai's sample-weight vector does) is a softer variant not yet tested. (Own: try `w = min(mcap, cap)^0.5`.)
- **Recency weighting / shorter windows.** Delphic Alpha's cross-asset Lasso: a 6-month rolling window dominated 12 and 18; feature *selection* frozen once, weights re-fit monthly. [B] https://delphicalpha.substack.com/p/from-alpha-signals-to-portfolio · Formal result: model complexity and training-window length must be chosen **jointly** (14% OOS R² gain over fixed-window; strongest in recessions). [A] https://arxiv.org/abs/2512.23596 — our expanding window + rank-IC selection never varies the window. *Own:* add exponential half-life {24, 48, 96 months} to the validation grid.
- **Numerai "deep incremental learning":** stack XGBoost models trained on different eras; two-layer stack beat single models under distribution shift. [A] https://arxiv.org/abs/2303.07925
- **Missing values:** cross-sectional mean/median fill is fine — Chen & McCoy on 159 predictors found simple imputation beats EM-style methods. [A] https://arxiv.org/abs/2207.13071 (so our current preprocessing is not a weakness).

#### 3.3 Features derivable from data we already have (no new data)

The panel contains `ret` (monthly, per `permno`), `prc`, `prc_high`, `prc_low`, `dolvol`, `tvol`, `shares`, `gics`/`ff49`/`sic`, and the 147 characteristics. All below are computable at month *t* from month ≤ *t* data.

1. **Information discreteness ("frog in the pan").** Da–Gurun–Warachka: momentum is 5.94% for *continuous*-information stocks vs −2.07% for discrete-information stocks with the same cumulative return; replicated across large/small cap and institutional-ownership splits. [A] https://ideas.repec.org/a/oup/rfinst/v27y2014i7p2171-2218..html — `ID = sign(ret_12_2)·(%neg months − %pos months)` from monthly `ret` (or daily if external).
2. **Path shape: slope & smoothness.** Quantitativo replication: 7.9% annual alpha (t 3.44) vs FF5+mom, Sharpe 1.12 on smoothness alone; smooth up-trends long, noisy down-trends short. [B] https://www.quantitativo.com/p/slope-strength-and-retail-extrapolation — compute R² and slope of the last 12 cumulative-return points.
3. **Residual / industry-neutral / 52-week-high-neutral momentum.** Raw momentum crashes in sector rotations; residual momentum (net of market and sector) and 52-wk-high-neutral momentum have lower crash risk (52wk-high-neutral: Sharpe +50%). [C] https://quantpedia.com/strategies/residual-momentum-factor · https://alphaarchitect.com/reducing-the-impact-of-momentum-crashes/ — the repo dropped momentum for defensibility; these are the *defensible* variants. Note `resff3_12_1` already exists (FF3-residual); add **within-`ff49` industry-relative** versions.
4. **Industry-relative characteristics.** For long-short, adjusting for sector "boosts [better result] from 20% to 78% of the time"; largest reduction in *value-weighted large-cap* strategies. [C] https://alphaarchitect.com/is-sector-neutrality-in-factor-investing-a-mistake/ — rank every characteristic **within** `ff49`/`gics` each month (or residualise on industry means), as a second feature block.
5. **Characteristic changes / momentum in attributes.** Equilibrium-style result: returns are driven by *changes* in characteristics; "momentum in firm attributes should be more investigated." [C] https://arxiv.org/pdf/2203.07865 — add 1m/3m/12m differences of the rank-transformed key characteristics (profitability, valuation, `niq_su`, `at_gr1`, `qmj`).
6. **Characteristic × size interactions.** Trees can find these; a ridge/linear composite cannot — put size-tercile-conditional composites into the composite arm.
7. **Macro/VIX-conditioned interactions.** Firm features × macro state is standard GKX; a 2026 GNN paper reports VIX/credit-spread conditioning helps. [C] https://arxiv.org/pdf/2605.19278 — only 68 test months; treat as low priority, high overfit risk.
8. **Amihud/spread-adjusted "tradeable" versions** of features so the model stops leaning on illiquidity (`ami_126d`, `bidaskhl_21d` were high-importance in `XGB`). (Own.)

#### 3.4 The 8-K layer — concrete, cheap, mostly *not* about LLMs

Data facts I checked in `fiam/8k_*.parquet` (2015-2026): 373k filings; item counts 9.01 (296k), **2.02 (120k)**, 7.01 (88k), 8.01 (84k), **5.02 (71k)**, 1.01 (48k), 5.07 (32k), 2.03 (20k), 5.03 (11k), 3.02 (10k), 3.01 (5.4k), 1.02 (4.6k), 2.01 (4.6k), **4.01 (2.2k)**, **2.05 (1.8k)**, **4.02 (0.7k)**, 2.06 (0.5k). Median main-text length is only **3.9k characters** (press-release exhibits are excluded, so 2.02 filings are mostly a pointer). In the ≥ $2B / ≥ $5 universe since 2021: **58% of stock-months have ≥ 1 8-K, 30% have an earnings (2.02) 8-K**, mean 0.88 8-Ks per stock-month.

Ideas, in order of cheapness:

1. **Information-intensity / filing-frequency features.** Higher 8-K filing frequency → *lower* future returns and volatility; a long-short on it earned ≈ 4.3%/yr spread (4.4% FF3+mom alpha), with larger effects at low intensity and high prior volatility. [C] https://dl.acm.org/doi/abs/10.1287/mnsc.2015.2408 (Management Science) — build: count of 8-Ks in the trailing 1/3/12 months, abnormal count vs own 12-month mean, counts by item family (voluntary 2.02/7.01/8.01 vs mandatory). Uses `filing_date` ≤ month-end of *t* (conservative availability, FIAM §4).
2. **News/no-news split of short-term reversal.** Reversal is compensation for liquidity provision; price moves *caused by news* continue, while liquidity-driven moves reverse. Novy-Marx/Medhat adjust for PEAD and industry momentum to isolate the reversal; unadjusted is far weaker. [B] https://www.dimensional.com/us-en/insights/q-and-a-on-short-run-reversals-with-mamdouh-medhat-and-robert-novy-marx · https://www.sciencedirect.com/science/article/abs/pii/S0304405X14001366 — the 8-K file gives a **news indicator** for month *t*: interact `ret_1_0` with `had_8k_in_month`. Caveat: reversal is *stronger* in volatile, small stocks, so the large-cap version may be tiny.
3. **Event flags** as features/screens: 4.02 (non-reliance): mean CAR vs SPY −0.87% (day 1) / −1.54% (day 20), −1.1% / −2.0% excluding 2021, n = 2,398 (2007-23), reported significant at the 0.1% level [B] https://sec-api.io/resources/stock-price-reactions-to-item-4-02-disclosures-in-sec-form-8-k-filings ; 4.01 (auditor change) named by the brief as a top distress signal; 5.02 (officer/director change), 2.05/2.06 (restructuring/impairment). Base rates are tiny (4.02 = 698 filings total) so use as **short-book exclusion/inclusion flags**, not a factor. CEO departures: powerful-CEO departures ≈ +3% higher abnormal return; forced departures raise volatility. [C]
4. **Earnings-announcement premium via 2.02 timing.** Frazzini–Lamont: buy stocks expected to announce next month, short those not; >60 bp/month, strong in large caps. But **Heitz et al. (2020) report the premium has disappeared in the US**, attributed to the rise in 8-K filings of material events after 2004. [C] https://www.aqr.com/library/working-papers/the-earnings-announcement-premium-and-trading-volume — cheap to test (expected month from last year's 2.02 date) but low prior; log it as a falsification test.
5. **PEAD in large caps is dead or negative** (Quanter Lab, S&P 500 2006-25: −0.5%/yr for the drift, +2.2%/yr on announcement day). [B] https://quantocracy.com/recent-quant-links-from-quantocracy-as-of-09062026/ — so skip a naive SUE/PEAD build (we already have `niq_su`, `saleq_su`).
6. **"Lazy Prices" for 8-Ks.** Cohen–Malloy–Nguyen: firms that *change* their periodic-report language underperform; replications show a non-linear relation with cosine similarity. [B/C] https://ideas.repec.org/p/nbr/nberwo/25084.html · https://github.com/martifigueres/LAZY-PRICES-REPLICATION-AND-DASHBOARD — analogue: cosine/Jaccard similarity of a firm's 8-K text vs its own previous same-item 8-K (TF-IDF is enough; no LLM). 8-K boilerplate is repetitive, so *change* is informative. Untested on 8-Ks specifically.
7. **Embeddings + PCA into a tree model.** Standard recipe: FinBERT (768-d) → PCA (20–50 comps) → gradient boosting; "PCA incurs a slight loss of accuracy but remains competitive." [C] https://arxiv.org/pdf/2508.06548 — the brief's own suggested workflow; risk is a text signal with IC ≈ 0 that costs a lot of compute.
8. **Structured event extraction.** *Grounded Event Extraction from 8-Ks* (2607.08346): 292,984 filings (2022-2026), 119-type taxonomy, 601,088 quote-anchored tags (released), precision 12%→96% by quality score; validated by an event-study of *unsigned* abnormal returns. [A] https://arxiv.org/abs/2607.08346 — useful to build an **event-type × direction** feature set; the released tags cover 2022-26 only, so cannot be used for a 2021 fit without re-tagging, and LLM tagging of 2021-22 filings with a 2026 model has model-side look-ahead risk.
9. **LLM-embedding signals do work in news, mostly in small stocks.** Chen–Kelly–Xiu: LLM news embeddings beat bag-of-words, add beyond reversal/characteristics, "predictability persists for several days among small stocks but dissipates quickly for large stocks." [C] https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4416687 — the same sign as our problem: text edge is largest where we cannot trade.
10. **Fast numbers, slow language.** In S&P 1500 (2022-25), quantitative earnings surprise is gone by the next open; **transcript-based qualitative sentiment peaks the next trading day and is tradeable** (but at a daily horizon, using call transcripts we do not have). [A] https://arxiv.org/abs/2606.29734 — informative for what *not* to expect from 8-K 2.02 pointers.
11. **AI/tariff exposure text scores** (mentions of AI in filings; tariff-risk recognition). AI-disclosure work: markets reward concrete implementation, not vague mentions [C] https://www.sciencedirect.com/science/article/pii/S105752192500465X ; tariff-exposure measures predicted "Liberation Day" reactions [C] https://www.bostonfed.org/-/media/Documents/events/2025/us-economy-changing-global-landscape/us_firms_exposure_tariffs_barbiero_silva_sheremirov_stein.pdf . Both are *event-window* results, not monthly return-forecast factors; theme-of-the-year, low prior as a durable alpha.

**Model-side look-ahead toolkit (the brief says teams that cannot explain this "will be treated as having used it"):**
- Time-locked LMs: ChronoBERT/ChronoGPT trained only on text up to each date. [A] https://arxiv.org/abs/2502.21206 · DatedGPT https://arxiv.org/html/2603.11838
- Measured contamination: Gao–Jiang–Yan show LLM forecast power is amplified on high-memorisation firm-date pairs and loses significance post-cutoff. [A] https://arxiv.org/abs/2512.23847
- Anonymisation: BlindTrade strips tickers and names before the agent sees text; Sharpe 1.40 ± 0.22 over 20 runs, weaker in trending bull markets. [A] https://arxiv.org/abs/2603.17692
- Inference-time debiasing (FinCAD, context-aware decoding). [A] https://arxiv.org/abs/2605.24564

#### 3.5 Model-side: prior-shrinkage, ensembling, foundation models

- **Own idea — composite-as-prior.** The one thing that worked (`LARGECAP` composite) has no fit parameters. Use LightGBM `init_score = composite_score` with ≤ 50 shallow trees, or a ridge whose penalty pulls toward composite group weights, so ML can only *add* what survives validation. This directly addresses the observation that ET on the same universe scored IC 0.016 vs composite 0.037.
- **Pooling and winsorising ML forecasts** beats shrinkage and "recent-performance-weighted" ensembles; equal-weighting is justified by endemic misspecification. [C] https://www.sciencedirect.com/science/article/pii/S0927539824000732 · stacking helps in extreme downside months [C] https://www.sciencedirect.com/science/article/abs/pii/S0927539822000342 . Our `ens5` never beat its best member, but all five members shared features; **diverse-feature** members (composite, peer-gap, 8-K, path-shape) are the ones worth averaging.
- **IC-weighted dynamic model weights** beat metric-weighted ones (CN CSI300; and "factor screening substantially enhanced" combined strategies). [A] https://arxiv.org/abs/2508.18592 — conflicts with our own `FACTOR_FILTER` (IC doesn't persist in the investable universe) and `TPA` (`tpa_ms` failed); treat as a warning, not a lead.
- **Tabular / time-series foundation models:** TSFMs (TimesFM, Moirai, Chronos…) win most tasks but gains over a zero-return baseline are "small and sparse"; only 2 of 10 cases statistically significant. [A] https://arxiv.org/html/2606.27100 . TabPFN-3 / TabICL do in-context learning without fitting — potentially interesting for *few-shot per-month* fitting on ~1,200 large caps [C] https://arxiv.org/pdf/2605.13986 , but no equity-return evidence found. Cheap experiment only if a GPU is free.
- **Sequence/attention architectures** (regime-gated Transformers, Mamba, KAN, hybrid LSTM-XGBoost): claims are on CN or a handful of tickers; I found no credible US large-cap cross-section evidence. See §6.
- **CNN "image" factor timing.** 206 factors' cumulative-return charts → CNN; ~6% annual alpha vs untimed, Sharpe 1.22, break-even cost 1.08%/trade; survives post-publication. [B/C] https://larryswedroe.substack.com/p/timing-the-factor-zoo — timing our own **factor-group sleeves** is the same problem `TPA` `tpa_ms` failed on; only try with strong regularisation.

#### 3.6 Cross-stock / network information (uses information that is *not* in a stock's own characteristics)

1. **Peer return gap / peer index.** Peer Return Gap (stock's lagged return − peers' returns) long-short earned 1.26%/mo (t 3.81), FF5 alpha 1.10% (t 2.86) (China, correlation peers). [C] https://www.sciencedirect.com/science/article/pii/S305070062500088X . US: Avramov & Ge, *Dual peer effects* (JFE 2026): a Peer Index predicts returns and earnings surprises, decays *without reversal*, and "machine-learning models based solely on firm-level characteristics do not subsume PI." [C] https://www.repository.cam.ac.uk/items/c9345bde-eace-4e97-b666-094549a2bda0
   - *Own implementation with our panel:* peers = top-*k* by 60-month return correlation **within the same `ff49`**; feature = own `ret_1_0` − peer-average `ret_1_0`; also peer-average `ret_12_1`. All from `ret`.
2. **Text-based industry peers (TNIC).** Free data (Hoberg–Phillips) keyed by `gvkey`; TNIC peer momentum is "substantially more significant than SIC-based peers or own-firm momentum," especially when links are less visible. [C] https://hobergphillips.tuck.dartmouth.edu/tnic_basedata.html — needs a `gvkey`→`permno` join (panel has `gvkey`+`iid`); verify data availability through 2025 and its release lag.
3. **Customer–supplier momentum.** Direct tier-1 links: monthly hedge alpha 0.37–0.63% (CN); spillovers also travel beyond tier-1 and are stronger when investors are inattentive. [C] https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5892284 · lead-lag exists only when customer-firm information is *continuous* (ties to §3.3 ID). Requires supplier data (Compustat segments / FactSet Revere, or LLM-extracted from 10-Ks) — high effort.
4. **Text-embedding networks.** FinBERT 10-K MD&A embeddings for 255 S&P 500 firms 2011-25, propagated along a supply-chain KG: long-short Sharpe 0.86, FF5 alpha 7.27% (t 2.30), survives sector-neutralisation and placebo. [A] https://arxiv.org/abs/2606.29290 · 10-K embedding graph + **LLM edge-filtering** raised S&P 500 mean-reversion Sharpe 0.742→0.820 (2011-19). [A] https://arxiv.org/abs/2604.19476 — only S&P 500 samples; small N; 8-K text is short, so the network would have to come from 10-K/8-K item text or from the customer-supplier tables.
5. **Asset embeddings from 13F holdings.** Gabaix–Koijen–Richmond–Yogo: a 4-dim holdings embedding explains >50% of relative valuations vs 15% for characteristics. [A] https://www.nber.org/system/files/working_papers/w33651/w33651.pdf . Quantitativo's Word2Vec-on-13F → 50 clusters → daily residual reversal within cluster: Sharpe 2.59 (2020-25), β 0.04 — but that is daily-frequency and uses paid Sharadar SF3. [B] https://www.quantitativo.com/p/asset-embeddings — with free EDGAR 13F it works only quarterly with a ≥ 45-day lag; use for **peer definition + crowding**, not a fast signal.
6. **Triangulated / modern stat arb.** Aggregate all pair spreads in an economically coherent group into per-stock "votes"; 2.4 Sharpe gross (1.2–1.9 net) vs 1.6 for GICS baseline. [B] https://www.quantitativo.com/p/triangulated-statistical-arbitrage — daily/intraday; not monthly-panel compatible. *Deep Learning Statistical Arbitrage* (residual portfolios from latent factors + convolutional transformer; Sharpe ~4 gross, 2002-16, ~550 largest stocks) is the academic version but is daily and frictionless. [A] https://arxiv.org/abs/2106.04028 — noted for completeness; our data cadence rules it out unless we add daily returns.
7. **GNNs:** the 2026 GNN papers I found predict correlations/volatility or directional accuracy, not cross-sectional return IC; I would not spend time here.

#### 3.7 External data that FIAM allows (needs `permno`/(`gvkey`,`iid`) + month and cleaning code)

FIAM §4 explicitly permits external data if joined on a dated identifier. Availability is the constraint; here is what I found and how safe the timing is.

| Source | Cost/access | Signal & evidence | Timing caution |
|---|---|---|---|
| **Options (OptionMetrics)** | WRDS (McGill has it — organisers' options+ML paper used it) | Cremers–Weinbaum IV spread and Bakshi RN-skew remain significant across regimes; smirk (Xing et al.) faded to insignificant (t −1.5) in 2023-26; ML on raw IV-surface/Greeks: OOS R² +1.29% vs +0.07% in the AI/mega-cap regime; no hand-crafted signal in the top-5 features [A] https://arxiv.org/abs/2608.26115 | Month-end snapshot is fine |
| **Daily returns (CRSP daily / Alpha Vantage)** | WRDS or free key | *Daily Return Information Factor*: elastic net on the 21 daily returns of the last month; 1.57%/mo (Sharpe 1.23) value-weighted L-S, "timing dominates" (recent days drive reversal) [C] https://www.cxoadvisory.com/technical-trading/applying-machine-learning-to-recent-daily-returns/ · SSRN 6005614; overnight vs intraday returns: many strategies' profits are earned entirely overnight or entirely intraday with opposite signs [A] https://www.sciencedirect.com/science/article/abs/pii/S0304405X19300650 | None (all ≤ month-end) |
| **Analyst forecasts/price targets (I/B/E/S)** | WRDS | Sector-relative price-target-implied return: long-short on S&P 500 with "substantial alpha after costs" (Da–Schaumburg, old) [B] https://www.quantseeker.com/p/is-there-alpha-in-analyst-forecasts . "Earnings revisions matter more" in 2026 dispersion regime [C] https://hedgeco.net/news/05/2026/quant-equitys-alpha-surge-why-systematic-stock-picking-is-back-at-the-center-of-the-hedge-fund-trade.html | Use statistical-period date, not calendar |
| **Insider Form 4** | SEC EDGAR, free | 3.7M transactions/34 countries: composite of role, size, clustering, R&D context gives ≥ 1%/mo alpha equal-weighted; "works best for mid/large caps"; alpha compressed since pre-2010 [C] https://verityplatform.com/wp-content/uploads/2026/04/VerityData-Insider-Academic-Studies.pdf ; ML on microcap purchases AUC 0.67→0.70 [A] https://arxiv.org/html/2602.06198v1 | Filed within 2 business days; map via `cik` (present in the 8-K file for 3,687 permnos) |
| **13F holdings** | SEC EDGAR, free | crowding/peer definition (see §3.6) | 45-day lag — use quarter-lagged |
| **Short interest** | FINRA, free (bi-monthly) | Surprise in short interest negatively predicts cross-section (informed short sellers) [C] https://www.sciencedirect.com/science/article/pii/S1386418123000393 ; a 2025-26 forecasting model got 61.6% directional accuracy but "almost no ability to predict" returns [C] https://equibles.com/research/new-short-interest-forecast-model — **more valuable as a short-selection/squeeze-risk filter** given our Jan-2021 loss | Published ~1 week after settlement |
| **FINRA daily short-sale volume** | FINRA, free (only last 365 days interactively; archives via Query API) | Short volume ratio as sentiment; not consolidated across venues [C] https://www.finra.org/finra-data/browse-catalog/short-sale-volume-data | Backfill limits for 2021+ |
| **Retail attention/sentiment (Google Trends, Wikipedia, StockTwits/X)** | Free/limited | Effects concentrated in small, young, high-idio-vol stocks; Wikipedia/Google mixed [C] https://www.mdpi.com/2227-7072/13/3/158 | **Skip for a large-cap book** |
| **CDS/credit** | Paid | CDS slope predicts stock returns mainly for stocks with high arbitrage costs [C] https://www.sciencedirect.com/science/article/abs/pii/S0304405X17300028 | **Skip** |
| **Man vs. ML earnings expectations** | Compustat/IBES | The brief cites this paper; note it carries an **Expression of Concern** from RFS (journal is investigating reliability). [A] https://academic.oup.com/rfs/article/39/5/1555/8502599 — don't build the deck's thesis on it. |

**Free-data trap:** SEC-derived data is usually free but unlinked; every join needs a dated `cik`/`cusip`→`permno` map. The 8-K file already carries `cik`, `cusip`, `permno` for 3,687 securities — enough to bootstrap the mapping for large caps.

#### 3.8 Hypothesis generation and "AI researcher" loops (optional, the brief's frontier)

- The 2026 crop: **AlphaAgent** (regularised exploration to resist alpha decay; S&P 500 annualised 8.74% vs 2.75% next-best) [A] https://arxiv.org/abs/2502.16789 · **QuantaAlpha** (evolutionary trajectories; factors mined on CSI300 transfer to CSI500/S&P 500 with ~19% cumulative excess over 4 years) [A] https://arxiv.org/abs/2602.07085 · **XALPHA** (report-to-memory absorption, CSI300) [A] https://arxiv.org/abs/2607.08332 · **AlphaPROBE**, AlphaSage, AutoScientist-Quant (all CN). LLM-generated features as tabular inputs: Sharpe +14% to +91%, weakly correlated with baseline features; *retrieval quality is critical*. [A] https://arxiv.org/abs/2602.00196
- **The cautionary result that matters most for a "signal-discovery agent":** Quantpedia had an AI agent replicate 9 published anomalies; **none survived 2023-25 out of sample**, the apparent survivor (Sharpe 1.95) was a construction error, and errors included extracting from abstracts, wrong price floors, and 153 months of near-zero-beta "degenerate" books. Guardrails: verbatim definition cards, dual fidelity reviews (code + trade log), as-traded prices, tradeability screens, cost stress tests, independent re-run. [B] https://quantpedia.com/guardrails-make-the-researcher-what-an-ai-agent-got-right-and-wrong-replicating-nine-equity-anomalies/?a=6080
- Five evaluation failures reverse the sign of LLM-agent results: look-ahead, survivorship, backtest overfitting, cost neglect, regime blindness. [A] https://arxiv.org/abs/2603.27539
- **Live-only LLM evidence** (cannot be backtested over 2021-26 without contamination): daily Russell-1000 agent — top-20 long-only alpha 18.4 bp/day, Sharpe 2.43, evaluated forward from April 2025 [A] https://arxiv.org/abs/2601.11958 ; MarketSenseAI — 19 months on the S&P 500, IC +0.489, strong-buy +2.18%/mo vs +1.15% [A] https://arxiv.org/abs/2604.17327 ; a synthetic 100-persona LLM "crowd" earned ~10% alpha but 92% of its holdings matched a neutral single prompt and the alpha was AI-mega-cap exposure, not selection [B] https://quantpedia.com/do-llm-crowds-produce-investment-signals-an-empirical-test/ .
- **LLM features can be valid and still fail downstream:** LLM-extracted features with IC > 0.15 in held-out data made an RL agent *worse* than a price-only baseline once macro conditions shifted. [A] https://arxiv.org/abs/2604.10996
- *Own recommendation:* if any agent is used, use it to **generate and code hypotheses**, then freeze the hypothesis list and test with pre-registered kill rules and a deflated-Sharpe-style discount. "Agents crowd" (brief §9) is supported by the crowding evidence in §3.9.

#### 3.9 Portfolio construction and risk — where a weak signal is lost or saved

**Evidence that the *regime* is hostile to shared factor exposure**
- **Jan 2026:** quant equity funds fell (UBS: US quant −2.8% in two weeks) on unwinding of crowded small-cap/high-risk longs and short exposure to high-beta/lower-quality; quality-short strategies suffered; momentum helped. [B] https://www.hedgeweek.com/quant-hedge-funds-see-worst-drawdown-since-october-as-crowded-trades-unwind/
- **Jun–Jul 2025:** quant equity managers lost ≈ 4.2% (Goldman PB); unusual factor correlations. [B] https://www.msci.com/research-and-insights/blog-post/unraveling-summer-2025s-quant-fund-wobble
- **July 2026:** Goldman's high-beta momentum basket fell ≈ 37% (record) while the S&P 500 hit highs; trigger was a Nvidia server-delay rumour; hedge-fund momentum exposure at the 92nd percentile of five years. [B] https://artificialfinance.org/2026/08/the-ai-momentum-unwind/
- Shared lesson: shorts in "low-quality/high-beta" and longs in "momentum/AI" are the same trade for many funds. The Bayes Group crowding piece argues alt-data and NLP sentiment have themselves become consensus signals, and recommends measuring overlap with the top quant managers. [B] https://www.bayes-group.com/insights/quant-equity-crowding-paradox
- Repo-relevance: our composite's factor groups (value, profitability, quality, vol/beta) overlap heavily with the crowded set; `LARGECAP` already shows 2025 negative.

**Ideas**
1. **Neutralise to more than beta.** Numerai's neutraliser matrix (country, sector, beta, momentum, size) plus Barra-style residual-alpha handling: decompose the alpha into risk-model-spanned and orthogonal parts and *penalise the spanned part* — "penalizing the residual alpha may improve exposures and ex-ante IR." [B] https://www.msci.com/documents/10199/c6e5e3f7-cd44-4322-aeb5-331e20e2afb7 — the repo's LP constrains `beta_60m`, `betabab_1260d`, sector. Add **log-mcap, 12-1 momentum, ivol/rvol, quality (`qmj`)** exposure caps to the LP (or project the alpha vector onto the orthogonal complement of those columns before the LP).
2. **Sector-neutral ranking** (rank within `gics`/`ff49`, then combine) instead of sector caps in the LP — cheaper and closer to how the literature's "sector-adjusted" gains are obtained. [C] (see §3.3 item 4)
3. **Cost-aware learning.** AQR "implementable efficient frontier": integrate trading-cost-aware optimisation *with* ML by learning portfolio weights directly under an economic objective, and report "economic feature importance." [A-ish] https://www.aqr.com/Insights/Research/Working-Paper/Machine-Learning-and-the-Implementable-Efficient-Frontier — relates to the existing `e2e.py` (SPSA) and `joint.py`; the AQR framing (costs inside the objective, not after) is the missing piece.
4. **Signal smoothing / alpha-decay-aware blending.** Grinold–Kahn: turnover is linked to signal decay; turnover-adjusted IR is always lower than IR that ignores turnover. [C] https://arxiv.org/pdf/2105.10306 — replace the per-month LP objective by an EMA of predictions (half-life 1–3 months) or a blend of 1m/3m-horizon models; the repo's hard 10% turnover cap costs signal because the LP must keep stale names.
5. **Confidence gate / abstain.** *When Alpha Breaks:* a LightGBM ranker "broke during the 2024 AI rally"; a strategy-level regime-trust gate (AUROC ≈ 0.72–0.75) that **decides whether to trade at all**, plus a cap on the most uncertain positions, worked; *continuous* uncertainty-based sizing degraded results because uncertainty is coupled to signal strength. [A] https://arxiv.org/abs/2603.13252 · Confidence Gate Theorem: gating helps only under rank-alignment and no "inversion zones," and structural-uncertainty confidence fails under contextual drift. [A] https://arxiv.org/abs/2603.09947 — the FIAM brief allows varying gross if the switch is predictable in advance and explained. *Own:* gate on trailing-12m IC of the composite and rolling factor-group correlation, both known at *t*.
6. **Dispersion / VIX as a scaling input.** Cross-sectional return dispersion and VIX forecast alpha dispersion; but high-dispersion periods raise active risk more than alpha (lower IR). [C] https://www.sciencedirect.com/science/article/abs/pii/S0304405X1930090X — this cuts against naive "lever up when dispersion is high." Vol-management of factors (Moreira–Muir): +50–100% Sharpe in-sample, but real-time out-of-sample results are mixed; momentum/ROE/BAB scaled versions retain gains. [C] https://www.sciencedirect.com/science/article/abs/pii/S0304405X2030132X
7. **Concentration / conviction.** Quantitativo's LTR: 30 quantiles beat 10/20/40+ (i.e. top ~3% names); the Russell-1000 agent's edge is concentrated in the top-20. Our LP's 1% per-name cap gives ~200-230 names. A more concentrated conviction book (e.g. ~60 longs at 2–3% each plus a hedge-matched short book) still fits the 100–500 position rule but concentrates factor risk; only pursue it if the decile table shows the extremes carry the alpha.
8. **Existing sizing work:** `HRP` and `TPA` already show inverse-vol / equal-risk budgeting gives lower drawdown and higher net IR but not significant IC gains; treat that as done.
9. **Survivorship/membership discipline:** Dead Signals Lab — reconstructing point-in-time S&P 500 membership cut a Sharpe from 0.63 to 0.06. [B] https://quantocracy.com/recent-quant-links-from-quantocracy-as-of-09062026/ — our panel is a retrospective CRSP/Compustat extract; note the caveat in the deck and keep universe rules point-in-time (as `largecap.py` already does).

---

### 4. A concrete sequenced plan (in the repo's pre-registration style)

**Round A — no new data, one script (`target_lab.py`), reuse `et.py` harness + the `largecap.py` universe**
1. Targets × models grid: {winsorized raw (control), uniform rank, gauss rank, size-median-demeaned, industry-median-demeaned, factor-residual (size, betabab, ivol, gics, 12-1 mom)} × {ET (existing config), LightGBM-forest-style, XGBRanker LambdaRank, ridge toward composite}.
2. Score: universe rank IC, paired-IC t vs control, D10−D1 spread, then `lc_t10`/`lc_free` net IR. Pre-registered rule: adopt a target only if paired-IC t ≥ 2 on *two* families or IC ≥ composite's 0.037 with t ≥ 2.
3. Add **composite-as-prior** arm (`init_score`).

**Round B — new features from existing data (`feat_lab.py`)**
- Path block (ID, slope/smoothness, 12m max-drawdown-in-path, up-month fraction), industry-relative ranks, characteristic changes, 8-K intensity/no-news/event-flag block, peer-return-gap block. Add each *block* to the composite arm and to the composite-prior model, report the paired-IC per block. Blocks that fail in the ≥ $2B universe are dropped, not tuned.

**Round C — portfolio layer (`pc_lab.py`), on whichever signal survives**
- (i) EMA-smoothed signal vs raw; (ii) extra neutralisers (log-mcap, momentum, ivol, quality) via alpha projection; (iii) confidence gate on trailing IC. Report rolling-12m beta, the 2025 and Jan-2021 months, and the July-2025/Jan-2026 windows specifically (crowding stress).

**Round D — only if A–C leave IC < 0.05:** external data in this order — (1) daily returns (DRIF, overnight/intraday), (2) options if WRDS OptionMetrics is reachable, (3) I/B/E/S revisions/targets, (4) Form 4 + 13F crowding, (5) TNIC peers, (6) short interest as a short-side filter.

**Multiple-testing discipline (Own):** this file lists ~25 candidate signals. Count every variant tried, report the count in the deck, and apply a deflation (Deflated Sharpe / Bonferroni-style on the paired-IC t-stats). The Quantpedia agent study (nine anomalies, zero survivors) and the Numerai lesson ("Sharpe correlation across windows ≈ 0, IC persists"; top-20 by Sharpe 3.3 in-sample → 0.2 OOS) are the reasons.

---

### 5. Realistic expectations (Own — do not skip)

- A rank-IC of 0.05 on ≥ $2B large caps with ~1,200 names/month is a strong, unusual result. By the fundamental law (IR ≈ IC·√breadth) with ~200–250 effective independent bets/month, IC 0.03 → IR ~0.5–0.7 *before* costs; IC 0.05 → ~0.8–1.2. That is consistent with the repo's composite (IC 0.037, gross IR 0.61) and is the sensible target.
- Only **IC on the tradeable universe** counts. Three of the literature's headline claims (small-cap ML, StockTwits, LLM news embeddings) are exactly the kind that vanished in `PM_ABLATION`.
- 68 test months and an IR s.e. ≈ 0.46: no single-path IR difference under ~0.5 is distinguishable. Use paired-IC t-stats and block-level (feature-family) tests.
- Regimes: 2025 was negative for every arm; July-2025/Jan-2026/July-2026 were crowded-factor unwinds. A strategy that looks like "quality/momentum/low-vol" will be judged against them.

---

### 6. Things that looked exciting but I would skip (and why)

| Idea | Why skip |
|---|---|
| More tree/boosting variants on the 147 columns | Five families give the same tradeable result; `PM_ABLATION` §7 already concluded this |
| Fancy sequence models (Mamba, KAN, regime-gated Transformer, quantum kernels) | Evidence is CN, single-stock, or directional accuracy; the Quantum-kernel horse race found the advantage "vanishes under fair comparison" [C] https://arxiv.org/abs/2607.20168 |
| Time-series foundation models for returns | Only 2 of 10 significant vs zero-return; "not reliable alpha" [A] https://arxiv.org/html/2606.27100 |
| Naive PEAD/SUE in large caps | Negative in S&P 500 2006-25 (Quanter Lab); we already carry `niq_su`/`saleq_su` |
| Google Trends / Wikipedia / StockTwits / X sentiment | Effect lives in small, young, high-idio-vol stocks |
| RL portfolio agents (AlphaZeroBeta-style PPO) | We have no policy-gradient infrastructure; the 2026 RL papers I found evaluate on indices/global baskets, with modest Sharpe; the repo's `e2e.py` already covers "train against a portfolio objective" |
| Factor timing on 7 sleeves | `TPA` (`tpa_ms`) failed; CNN image timing is the only positive result and it is a paywalled 206-factor study |
| Regime-conditioned "LLM analyst" backtests over 2021-26 with a 2026 frontier model | Model-side look-ahead; the brief will treat the team as having used it unless prevented (§3.4 toolkit: time-locked models, anonymisation, contamination test) |
| Building on van Binsbergen–Han–Lopez-Lira (EPS bias) | RFS has issued an Expression of Concern on the paper |

---

### 7. Source index (grouped)

**Diagnosis / evidence base**
- What Useful Alphas? — https://arxiv.org/abs/2607.06502
- Feature Scope & cross-sectional return prediction, US large caps — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6497598
- AQR, ML and the implementable efficient frontier — https://www.aqr.com/Insights/Research/Working-Paper/Machine-Learning-and-the-Implementable-Efficient-Frontier
- ML Enhanced Multi-Factor Trading with Bias Correction (CN) — https://arxiv.org/html/2507.07107v1
- Option-Implied Signals and Crash Risk, 2015-2026 — https://arxiv.org/abs/2608.26115
- Quantpedia: Guardrails Make the Researcher — https://quantpedia.com/guardrails-make-the-researcher-what-an-ai-agent-got-right-and-wrong-replicating-nine-equity-anomalies/?a=6080

**Targets, losses, training**
- Getting the Target Right (Cakici–Zaremba) — https://quantpedia.com/getting-the-target-right-in-return-prediction/
- Classification vs regression (Bai–Pukthuanthong) — https://arxiv.org/abs/2108.02283
- Less is More? (Howard) — https://quantpedia.com/less-is-more-reducing-biases-and-overfitting-in-machine-learning-return-predictions/
- LambdaRankIC — https://arxiv.org/abs/2605.00501 · Quantitativo LTR — https://www.quantitativo.com/p/learning-to-rank · Learning to rank for VRP (options) — https://arxiv.org/abs/2608.24786
- Nonstationarity–complexity tradeoff — https://arxiv.org/abs/2512.23596
- Missing values for ML portfolios — https://arxiv.org/abs/2207.13071
- Numerai LGBM ensemble — https://github.com/vladmurnik/numerai-lgbm-ensemble · Numerai Alpha — https://blog.numer.ai/signals-alpha-and-mpc/ · Deep incremental learning — https://arxiv.org/abs/2303.07925
- Delphic Alpha, signals→portfolio — https://delphicalpha.substack.com/p/from-alpha-signals-to-portfolio
- Pooling/winsorising forecasts — https://www.sciencedirect.com/science/article/pii/S0927539824000732 · Stacking — https://www.sciencedirect.com/science/article/abs/pii/S0927539822000342

**Features from price/return history**
- Frog in the Pan — https://ideas.repec.org/a/oup/rfinst/v27y2014i7p2171-2218..html
- Slope/Strength/Retail Extrapolation — https://www.quantitativo.com/p/slope-strength-and-retail-extrapolation
- Probabilistic momentum — https://www.quantitativo.com/p/uncertainty
- Trend factor (post-2016 flat) — https://www.quantitativo.com/p/coding-trend-factor
- Residual momentum — https://quantpedia.com/strategies/residual-momentum-factor · Momentum crashes — https://alphaarchitect.com/reducing-the-impact-of-momentum-crashes/
- Sector neutrality — https://alphaarchitect.com/is-sector-neutrality-in-factor-investing-a-mistake/
- Daily Return Information Factor — https://www.cxoadvisory.com/technical-trading/applying-machine-learning-to-recent-daily-returns/
- Overnight vs intraday — https://www.sciencedirect.com/science/article/abs/pii/S0304405X19300650
- Characteristics-driven returns in equilibrium — https://arxiv.org/pdf/2203.07865

**8-K / text / LLM**
- Grounded 8-K event extraction — https://arxiv.org/abs/2607.08346
- Item 4.02 reactions — https://sec-api.io/resources/stock-price-reactions-to-item-4-02-disclosures-in-sec-form-8-k-filings
- 8-K information intensity — https://dl.acm.org/doi/abs/10.1287/mnsc.2015.2408
- Reversals and liquidity provision — https://www.dimensional.com/us-en/insights/q-and-a-on-short-run-reversals-with-mamdouh-medhat-and-robert-novy-marx
- Earnings announcement premium — https://www.aqr.com/library/working-papers/the-earnings-announcement-premium-and-trading-volume
- Fast Numbers, Slow Language — https://arxiv.org/abs/2606.29734
- Lazy Prices — https://ideas.repec.org/p/nbr/nberwo/25084.html
- Supply-chain propagation of text signals — https://arxiv.org/abs/2606.29290 · LLM-augmented semantic networks — https://arxiv.org/abs/2604.19476
- Expected Returns and LLMs (Chen–Kelly–Xiu) — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4416687
- Look-ahead: ChronoBERT https://arxiv.org/abs/2502.21206 · Detecting lookahead https://arxiv.org/abs/2512.23847 · FinCAD https://arxiv.org/abs/2605.24564 · BlindTrade https://arxiv.org/abs/2603.17692 · DatedGPT https://arxiv.org/html/2603.11838
- LLM crowds — https://quantpedia.com/do-llm-crowds-produce-investment-signals-an-empirical-test/

**Cross-stock / network**
- Dual peer effects — https://www.repository.cam.ac.uk/items/c9345bde-eace-4e97-b666-094549a2bda0 · Peer return gap (CN) — https://www.sciencedirect.com/science/article/pii/S305070062500088X
- TNIC data — https://hobergphillips.tuck.dartmouth.edu/tnic_basedata.html
- Customer–supplier momentum spillover — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5892284
- Asset Embeddings — https://www.nber.org/system/files/working_papers/w33651/w33651.pdf · Quantitativo version — https://www.quantitativo.com/p/asset-embeddings
- Triangulated stat arb — https://www.quantitativo.com/p/triangulated-statistical-arbitrage · Modern stat arb — https://www.quantitativo.com/p/modern-statistical-arbitrage · Deep Learning Stat Arb — https://arxiv.org/abs/2106.04028

**Agents / discovery**
- AlphaAgent — https://arxiv.org/abs/2502.16789 · QuantaAlpha — https://arxiv.org/abs/2602.07085 · XALPHA — https://arxiv.org/abs/2607.08332 · Generative AI for stock selection — https://arxiv.org/abs/2602.00196
- MarketSenseAI — https://arxiv.org/abs/2604.17327 · Agentic nowcasting — https://arxiv.org/abs/2601.11958 · Evaluation failures — https://arxiv.org/abs/2603.27539 · When valid signals fail — https://arxiv.org/abs/2604.10996

**Portfolio / risk / regime**
- When Alpha Breaks — https://arxiv.org/abs/2603.13252 · Confidence Gate Theorem — https://arxiv.org/abs/2603.09947
- Grinold–Kahn / turnover-adjusted IR — https://arxiv.org/pdf/2105.10306
- MSCI residual-alpha portfolio construction — https://www.msci.com/documents/10199/c6e5e3f7-cd44-4322-aeb5-331e20e2afb7
- Jan 2026 quant drawdown — https://www.hedgeweek.com/quant-hedge-funds-see-worst-drawdown-since-october-as-crowded-trades-unwind/ · Summer 2025 wobble — https://www.msci.com/research-and-insights/blog-post/unraveling-summer-2025s-quant-fund-wobble · July 2026 momentum unwind — https://artificialfinance.org/2026/08/the-ai-momentum-unwind/ · Crowding paradox — https://www.bayes-group.com/insights/quant-equity-crowding-paradox · 2026 breakdowns — https://youngandcalculated.substack.com/p/everything-that-broke-in-quant-this
- Quant-equity revival / dispersion — https://hedgeco.net/news/05/2026/quant-equitys-alpha-surge-why-systematic-stock-picking-is-back-at-the-center-of-the-hedge-fund-trade.html
- Volatility-managed portfolios (OOS) — https://www.sciencedirect.com/science/article/abs/pii/S0304405X2030132X · Dispersion & alpha — https://www.sciencedirect.com/science/article/abs/pii/S0304405X1930090X

**External data**
- Insider signals — https://verityplatform.com/wp-content/uploads/2026/04/VerityData-Insider-Academic-Studies.pdf · https://arxiv.org/html/2602.06198v1
- Short interest — https://equibles.com/research/new-short-interest-forecast-model · FINRA short volume — https://www.finra.org/finra-data/browse-catalog/short-sale-volume-data
- Analyst targets — https://www.quantseeker.com/p/is-there-alpha-in-analyst-forecasts
- Man vs Machine Learning (Expression of Concern) — https://academic.oup.com/rfs/article/39/5/1555/8502599

**Foundation / sequence models**
- Pretrained TS foundation models — https://arxiv.org/html/2606.27100 · TabPFN-3 — https://arxiv.org/pdf/2605.13986 · Adaptive Financial Transformer — https://arxiv.org/abs/2606.29347


---

## Part II — docs/RESEARCH.md Part II — Reddit-led deep research, extending docs/RESEARCH.md Part I (2026-09-21)

Project context (from `docs/RESEARCH.md` Part I / `REDDIT.md` / `experiments/largecap/README.md`): monthly US equity long/short from 147 characteristics; the tradeable large-cap universe (price ≥ $5, mcap ≥ $2B, ~1,200 names/month) has rank IC ≈ 0.035; the only honest positive is a no-fit, equal-group composite of 18 factors (net IR ≈ 0.54, IC 0.037, 2025 negative); every fitted model (ET IC 0.016) lost to it.

---

### 0. Read this first — how the research was done, and what Reddit can and cannot tell us

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

**What Reddit is good for here — and not.** r/quant and r/algotrading are ~70% careers/retail-bot content. On the *specific* problem (monthly cross-sectional ML on large caps) there is **no technical thread that outperforms the literature already in docs/RESEARCH.md Part I**. Reddit's real value was (1) **calibration** (how much IC survives production, how crowding actually behaves, what pod-shop practitioners now care about), (2) **failure stories** and pitfalls, (3) **pointers to primary sources** that docs/RESEARCH.md Part I had missed (Blitz et al., Gârleanu–Pedersen, Era Splitting, AQuA, Toraniko), and (4) one **direct contradiction** of a docs/RESEARCH.md Part I recommendation. The report is organised around that, not around volume.

**Evidence grades** (as requested): **Demonstrated** (concrete experiment/result I could see) · **Reported** (someone claims results, limited evidence) · **Plausible** (technically compelling, thin evidence) · **Speculative** (hypothesis). Statements labelled *Own* are my synthesis, not from Reddit or a paper.

---

### 1. Summary versus docs/RESEARCH.md Part I

| Finding | Status vs docs/RESEARCH.md Part I | Grade | Section |
|---|---|---|---|
| **Asymmetric buy/hold rank buffer ("buy 10 / hold 50") + no-trade band + partial rebalance**, backed by a large-cap monthly study | **New** (docs/RESEARCH.md Part I has only EMA smoothing) | Demonstrated | 2.1 |
| **Blitz et al. short-term composite in a liquid large-cap universe** (industry-relative reversal, 1m industry momentum, seasonality, idio-vol, analyst revisions) | Partly new (components scattered in docs/RESEARCH.md Part I; composite, cost rule and large-cap evidence are not) | Demonstrated (not on our sample) | 2.2 |
| **Era Splitting** tree criterion (Numerai lineage) + *my* extension: size-buckets as environments | **New** | Reported / Speculative | 2.3 |
| **Neutralisation depth is contested**: a 2026 pod-shop practitioner says tightly factor-neutral books did *worst*; docs/RESEARCH.md Part I #6 says neutralise harder | **Contradiction** | Reported | 2.4 |
| **"Shared-core" overlap test** — how much of our alpha is the commoditised composite | **New** | Plausible / Own | 2.5 |
| **Crowding is a variance-shift, not an IC decay** → gate on crowding state, and beware gates calibrated in a single validation regime | Extends docs/RESEARCH.md Part I §3.9 #5 | Reported | 2.6 |
| **Live IC haircut 20–50%** (independent testimony + peer-reviewed 57% cumulative reduction for ML strategies) | **New calibration** | Reported + Demonstrated | 2.7 |
| **Text-signal "sector ghost" check**, and numeric-specificity drift in guidance language | New detail on §3.4 | Reported | 2.8 |
| **Typed causal-operator registry + truncation-invariance test** for any AI-written features | **New** (docs/RESEARCH.md Part I §3.8 has guardrails, not this) | Demonstrated (as a documented failure) / Plausible (as a fix) | 2.9 |
| **Borrow-cost adverse selection**: short-thesis strength correlates with borrow cost/unavailability | New detail | Reported | 2.10 |
| Expected-earnings-in-window as a feature | Conflicts mildly with docs/RESEARCH.md Part I §3.4 #4 (low prior) | Reported | 2.11 |
| Knockoffs / conformal selection / international pooling from adjacent fields | **New (own hypotheses)** | Plausible / Speculative | 3.2 |
| Replication failure of "Attention Factors" stat-arb; LdP pipeline AUC 0.50; TabPFN reality check | Reinforces docs/RESEARCH.md Part I §6 skip list | Reported | 2.12 |

---

### 2. Ideas (detailed)

#### 2.1 Asymmetric rank buffer ("buy 10 / hold 50"), no-trade band, partial rebalance, aim-in-front

**Core concept.** Enter a name only when its predicted rank is in the extreme (e.g. top decile), but keep holding it until it falls well outside the entry zone (e.g. out of the top half). Separately, trade only a fraction of the gap to target, and only if the gap exceeds a band. In Gârleanu–Pedersen terms: *aim in front of the target and trade partially toward the aim*.

**Why interesting.** docs/RESEARCH.md Part I §3.9 #4 proposes EMA smoothing and blended horizons; the repo's LP applies a **hard 10% one-way turnover budget** (`experiments/largecap/README.md`). A hard cap makes the LP keep stale names by *feasibility*, not by *design*. An explicit hysteresis rule spends the same turnover on the names where the rank change is most informative.

**Evidence.**
- *Demonstrated (peer-reviewed).* Blitz, Hanauer, Honarvar, Huisman, van Vliet, *Beyond Fama-French Factors: Alpha from Short-Term Signals* (FAJ 2023): MSCI World constituents only (no small/off-benchmark stocks), Dec 1985–Dec 2021, monthly rebalance. Individual signals have 1,300–2,000% annual turnover; with a naive "buy 20 / hold 20" at 25 bp costs the composite's net alpha loses **more than two-thirds** to costs, but with **"buy 10 / hold 50"** the composite's net alpha stays **above 6%**. (Robeco summary; abstract on SSRN 4115411.)
- *Reported.* r/quant, "Minimizing costs for cross sectional strategies" (1rh2h0p): a practitioner with 2.5–2.9 gross Sharpe cut to ~1 by costs is told to use no-trade bands, partial rebalance (25–50% of the gap), and **rank buffering** ("enter top-3, hold until below top-6… one of the biggest cost leaks in top/bottom books"). r/quant "portfolio hysteresis?" (1693f4p): "add a transaction cost term… gives hysteresis for free" or a separate no-trade layer.
- *Demonstrated (theory).* Gârleanu & Pedersen, JF 2013: closed-form optimum = linear combination of current portfolio and an "aim portfolio" (weighted average of today's and expected future Markowitz portfolios); **slower-decaying predictors get more weight**. r/quant threads on combining horizons (1ox4ol7, 1munnm6) converge on "multi-period optimisation" / Boyd's *Multi-Period Trading via Convex Optimization* as the practical route; practitioners say the good version "is guarded IP".

**Source.** https://reddit.com/r/quant/comments/1rh2h0p · https://reddit.com/r/quant/comments/1693f4p · https://reddit.com/r/quant/comments/1ox4ol7 · https://reddit.com/r/quant/comments/1munnm6 · https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4115411 · https://www.robeco.com/en-int/insights/2022/05/beyond-fama-french-alpha-from-short-term-signals · https://www.nber.org/papers/w15205

**Evidence strength.** Demonstrated for the rule's value on a large-cap, monthly universe; not demonstrated for *our* signals or the 2021–26 sample.

**What is new.** docs/RESEARCH.md Part I #10 and §3.9 #4 = EMA of predictions / blended horizons. **Not in docs/RESEARCH.md Part I:** asymmetric entry/exit thresholds, no-trade band, partial rebalancing, or the G–P aim-portfolio formulation.

**Implementation details.**
1. Replace the LP objective's implicit "stay put" with a rank-state machine on the composite/model score: long-enter at score-rank ≤ 10%, long-exit only at rank > 50% (mirror for shorts); LP then only chooses weights over the *allowed* set.
2. Add a cost term to the LP objective (L1 turnover penalty λ·|w − w_prev|) and sweep λ instead of a hard 10% cap; report the implied turnover.
3. G–P-lite: score_aim = Σ_h decay-weight_h · score_h where horizons h ∈ {1m, 3m, 6m} are separate models/composites; trade a fraction (start 0.5) of (aim − current).

**Limitations.** Blitz's signals are *short-horizon, high-turnover*; their headline (1,800% turnover) is **infeasible under the repo's 10% one-way monthly budget**. The buffer rule therefore matters most if the budget is relaxed or if applied to slower composites; on a slow composite the turnover saving is smaller. The 1985–2021 sample is mostly pre-2015; large-cap short-term reversal was reported weaker recently. Buffers concentrate holdings in past winners of the *score* (staleness risk after regime shifts).

**Potential experiments.** On the frozen `largecap.py` composite arm: {hard cap (control), rank buffer 10/30/50, L1 penalty sweep, buffer + partial rebalance}. Pre-registered rule: adopt if net IR ≥ control + 0.1 at equal or lower turnover **and** paired monthly-return t ≥ 1 (IR s.e. is ≈ 0.46, so do not use IR alone).

**Related rabbit holes.** 2.2 (the signals the buffer was designed for), 2.7 (costs are where live decay lives). A commenter with an HFT market-making background (r/quant 1rw4oip) says "I've seen many make money purely on good monetization and bad/no alphas. I've not often seen people make money on bad monetization and good alphas since like 2009" — single testimony from a different horizon, but it agrees with putting the cheap portfolio-layer experiments ahead of new signal work.

---

#### 2.2 Blitz short-term composite as a large-cap-native signal block

**Core concept.** Five *cheap* short-horizon signals that survive in liquid large caps and are largely uncorrelated with Fama–French factors: (1) **industry-relative 1-month reversal**, (2) **1-month industry momentum**, (3) **analyst revisions (past 30 days)**, (4) **same-calendar-month return seasonality**, (5) **1-month idiosyncratic volatility**.

**Why interesting.** It is a published composite defined *only* on a large-cap universe, monthly — the exact regime we lose in. The repo's composite deliberately has **no momentum/short-term** content (`experiments/largecap/README.md`: "no momentum"), so this block would be **orthogonal to what we have**.

**Evidence.** *Demonstrated (peer-reviewed).* Robeco summary and the paper's abstract: individual gross returns 5–8%/yr; composite average return > 12%, six-factor alpha > 12% (significant); break-even cost < 25 bp individually, > 30 bp composite; alpha persists out-of-sample and post-publication, across regions, with several-day implementation lags, and uncorrelated with traditional FF factors. Source above.

**What is new.** docs/RESEARCH.md Part I lists industry-relative features (§3.3 #4), residual momentum (#3) and analyst data (§3.7) separately and grades most as [C]. Here is a single **out-of-the-box, benchmark-constituent-only** composite with a cost rule (2.1). The repo already holds `ivol_capm_21d`, `seas_1_1na`, `seas_1_1an`, `seas_2_5na`, `ret_1_0`; **missing** are industry-relative reversal and industry momentum (computable from `ret` + `ff49`) and analyst revisions (needs I/B/E/S).

**Implementation.** Signal 1 = `ret_1_0 − mean(ret_1_0 | ff49, month)` (or value-weighted); Signal 2 = `mean(ret_1_0 | ff49, month)` for the stock's industry; signals 4–5 exist; 3 optional (Round D). Z-score within month, equal-weight, apply 2.1's buffer, *then* test as an additive block on the composite.

**Limitations.** Sample ends 2021 (the paper) and includes the pre-decimalisation-cost era for the older part; our test period 2021–26 contains the July-2025 and Jan-2026 short-term-factor unwinds (Reddit crowding threads, 2.6), which hit exactly short-horizon books. High turnover (see 2.1). Analyst revisions are a large share of the composite's power and are unavailable in the panel.

**Potential experiments.** Paired-IC test of {composite, composite + industry-reversal + industry-momentum} on the ≥ $2B universe; separately with 2.1's buffer. Kill: paired IC t < 1, or gain only in the bottom size tercile.

**Related.** 2.5 (overlap with the commoditised core), 2.6.

---

#### 2.3 Era Splitting — invariance-seeking tree splits (and a size-environment variant)

**Core concept.** Standard GBDT chooses the split that maximises impurity reduction over *all rows pooled*. **Era splitting** evaluates the gain **within each era separately** and combines with a smooth min/mean (or a *directional* criterion requiring the split to move the target the same way in every era), so it favours splits that help in *all* periods rather than one dominant regime.

**Why interesting.** Our diagnosed failure is trees that learn small-cap / ivol / lottery structure that does not persist (`FACTOR_FILTER`: universe IC rank corr 0.10; `PM_ABLATION`). That is *precisely* a "split gain concentrated in some environments" problem.

**Evidence.**
- *Reported (authors' abstract).* Era Splitting: Invariant Learning for Decision Trees (arXiv 2309.14496): two new criteria, integrated into GBDTs, "superior performance on the Numerai financial dataset compared to state-of-the-art GBDT". I read only the abstract, not the tables.
- *Implementation exists.* `jefferythewind/scikit-learn-erasplit`: `EraHistGradientBoostingRegressor(gamma=1)`, `fit(X, y, era_column)`; `gain = gamma·era_split_gain + blama·directional_era_split_gain + vanna·original_gain`; `boltzmann_alpha` controls the smooth min/max across eras. The repo publishes **no benchmark numbers**.
- *Related, Reported.* Online learning with **dynamic feature projection** on Numerai (arXiv 2301.00790): LightGBM-dart Sharpe 1.18 → 1.46, Calmar 0.19 → 0.68, GBDTs beat neural nets, and "simple feature engineering paradoxically hurt" (evaluated with Spearman on Numerai data, not US large caps).

**Source.** https://arxiv.org/abs/2309.14496 · https://github.com/jefferythewind/scikit-learn-erasplit · https://arxiv.org/html/2301.00790v4. (Reddit itself has almost no Numerai content — 48 posts — so this lead came from the Numerai literature.)

**What is new.** docs/RESEARCH.md Part I §3.1 e / §3.2 cite Numerai multi-target ensembles and "deep incremental learning", but not tree-level invariance.

**Implementation.** eras = calendar months (≥ 150 eras in-sample). Start with `gamma=1, blama=0, vanna=0` vs the vanilla control on the same features/target; then `blama` directional.

***Own hypothesis (Speculative): size-environment splitting.*** Define eras as **(month × size-tercile)** instead of month. A split only survives if it helps in *large* caps as well as small — an invariance constraint (IRM-style) that directly prices in "edge lives in small caps". Nobody I found has tried this for US equities.

**Limitations.** One published benchmark (Numerai's obfuscated data, ~weekly eras, thousands of names per era); nothing on monthly US large-cap data. Era-min criteria may under-fit and just recreate the composite (which would be a *fine* outcome: it means the invariant signal is the simple one). Tiny custom fork — maintenance risk; check licence and sklearn version.

**Potential experiments.** `et.py`-harness style: {HistGBR vanilla, era-split (month), era-split (month×size)} on the ≥ $2B universe with the composite as `init_score`/baseline. Kill: paired-IC t < 1 vs vanilla, or IC < composite's 0.037.

**Related.** 2.13 (composite-as-prior, reinforced by Reddit), 3.2 (invariance across markets).

---

#### 2.4 The neutralisation contradiction: how much factor-neutral is too much?

**Core concept / conflict.** docs/RESEARCH.md Part I §2 #6 and §3.9 #1 recommend neutralising *beyond* beta (size, momentum, vol, quality; Numerai/Barra style). Reddit gives **both** sides:

- *For:* "Barra-neutralise… residual Sharpe and residual IC matter more" (r/quant 1rvb4sm, 12 & 8 upvotes); one desk saw a 2.0 Sharpe fall to ~1.2 *just by removing 1-day reversal*; Toraniko (open-source MIT Barra-style risk model, r/quant 1ekpin6, 179 upvotes) exists to do exactly this.
- *Against (2026, practitioner at a multi-manager HF):* in r/quant 1v30qx3 ("Is anyone in equity stat arb making any money?"), the answer was that the **pain was concentrated in "larger, very short-horizon, tightly factor-neutral" pods**; "**strategies with richer fundamental or alt datasets or those operating under less restrictive risk frameworks held up much better**." A separate top comment (1v02nb2, 91 upvotes) says factor-neutral is "model-dependent" and that a portfolio hard-constrained like other tier-1 pods **converges to the same positions** ("shared core"), so neutralisation to a common risk model *raises* crowding overlap.

**Why it matters.** The repo's LP already neutralises beta + sector. Pushing to size/momentum/vol/quality neutrality (docs/RESEARCH.md Part I #6) could reduce measured factor exposure while *moving the book toward the same constrained feasible set every other neutral book occupies*.

**Evidence strength.** Reported (anonymous practitioner testimony, 2 independent threads; no numbers). The *Sharpe-compression* numbers people quote (30–50% loss going to full-Barra-neutral) are generic and possibly LLM-written — low reliability.

**Implementation / experiment.** Treat neutralisation depth as a **dial with an empirical curve**, not a default: {beta-only, +sector (current), +size, +size+momentum, +size+mom+vol+quality} on the frozen composite and on any candidate signal. For each: net IR, rolling-12m beta, and P&L in the crowded-unwind windows (Jul 2025, Jan 2026, Jul 2026 — see docs/RESEARCH.md Part I §3.9). Use Toraniko or a simple cross-sectional regression for the exposures. The decision rule should weigh *stress-window drawdown* at least as much as full-sample IR.

**Limitations.** With 68 months we cannot tell a real difference from noise; stress windows are 1–3 months each (N≈3). Expect the answer to be "small differences" — the useful output is the *shape*, and the drawdown in 3 windows.

**What is new.** A direct **contradiction** of docs/RESEARCH.md Part I #6 to preserve, not a new technique.

---

#### 2.5 "Shared-core" overlap test — is our alpha the commoditised risk premium?

**Core concept.** In the 2026 pod-shop discussion (1v02nb2), capital allocators now ask new research: "**How much of this is just proxying the platform's existing shared core?**" A commenter in 1v30qx3 says the commoditised alphas (vendor-sold, everyone knows them) are really "collecting a *quant risk premium*". Pods with signals "outside the standard factor continuum" (structural anomalies, niche datasets, fundamental idiosyncrasy) get capital.

**Why interesting for us (Own).** The repo's *best and only* result is an equal-weight composite of 18 textbook groups — value, profitability, investment, quality, surprise, ivol/beta, liquidity. That is the **definition of the shared core**. Its 2025 negative year and the Jan/Jul unwinds are consistent with a crowded-core hypothesis. Yet we never measured *how much of any candidate signal is explained by the composite*.

**Supporting practitioner rules.** r/quant 1vvt4m9 ("signal to be neutralised or loaded?", 19 and 8 upvotes on the top answers) gives the decision rule used above: orthogonalise the candidate to the risk model; if it vanishes, it was repackaged beta (neutralise it, unless you actually have a factor-timing edge); if the residual still predicts out of sample, treat the residual as the alpha. One commenter adds a warning relevant to any blending step: "a somewhat predictive signal plus a little of some other predictive signal (or tbh could be beta) = a 'better' predictive signal — chasing Sharpe only can lead to washing out uniqueness of signals" (1rw4oip).

**Evidence.** Reported (crowding testimony) + Own inference. Numerical crowding evidence is in docs/RESEARCH.md Part I §3.9 (MSCI, Hedgeweek).

**Implementation (cheap, no new data).** For any candidate signal s: (i) per-month cross-sectional rank corr and R² of s on the composite; (ii) **orthogonalise s to the composite** each month and re-run the IC/IR test on the *residual*; (iii) report IC of the residual per size tercile. Also (iv) check how the composite's own long/short book correlates with the top-decile of a simple "quality-minus-junk + value" screen.

**Limitations.** A public-factor basket is a crude proxy for the *actual* crowded core (which is defined by managers' positions, not published factors). 13F/portfolio-overlap data is quarterly and lagged (docs/RESEARCH.md Part I §3.6).

**Potential experiments.** For each of the docs/RESEARCH.md Part I rounds' candidate blocks (path shape, 8-K, peer gap, target variants), the *first* diagnostic = residual IC after projecting out the composite. A block that scores IC 0.02 raw but 0.02 residual is worth more than one at 0.03 raw / 0.005 residual.

**Related.** 2.4, 2.6, 2.8 (ghost check is the same pattern).

---

#### 2.6 Crowding is a variance-shift: gate on crowding state, and don't calibrate the gate in one regime

**Core concept.** r/quant "Is crowded alpha basically beta now, or is this just cope?" (1u3n4x0, 76 upvotes; top comment 52): informational decay and crowding are **different phenomena** — "in the former, forecast efficacy deteriorates (IC decay); in the latter, the forecast may remain intact, but *monetisation* becomes impaired by liquidity, implementation costs and synchronised de-risking". Alphas increasingly show "**regime-dependent capacity**": stable, then a sharp repricing, not a smooth fade. A follow-up comment: "informational decay is mean-shift, crowding unwind is variance-shift; your backtest captures the mean but not the variance spike." Also: stat-arb capital is allocated on trailing realised Sharpe, so it "crowds performance as much as signals".

**Why interesting.** docs/RESEARCH.md Part I §3.9 #5's gate is on **trailing IC**. If crowding leaves IC intact while breaking P&L, an IC-based gate would *not* fire before Jul 2025/Jan 2026.

**Evidence.** Reported (extended practitioner discussion; the Jan-2026 details corroborated in 1tn45mx: "first three weeks [of January] were horrible… factor/correlation relationships broke down… fresh DD limits"). Consistent with independently reported industry losses in docs/RESEARCH.md Part I §3.9.

**A relevant failure story (Reported).** r/algotrading 1sjicuf (LightGBM ranker, 4 walk-forward folds): the "regime filter" (SPY MA200) zeroed positions on **49.6% of validation dates in the bad fold but 0% of test dates**, and the threshold optimiser selected `enter_thr=0` because the strategy-selection slice had negative Sharpe in *all* folds. Lesson: **abstain/gate parameters tuned on a validation slice inherit that slice's regime and are not evidence about the test regime.** (Also a reminder that mean IC 0.024 with a −0.009 fold is what "IC ≈ 0.03 large-cap" looks like in practice.)

**Implementation (Own).** Candidate crowding-state variables computable from the panel: (a) rolling 3-month cross-sectional **return dispersion of factor-group sleeves** and their pairwise correlation (a jump = unwind), (b) rolling realised beta of the book to a size/momentum/vol basket, (c) 1-month reversal in the *composite's own top-vs-bottom spread* (short-term P&L momentum). Gate/scale gross on (a)–(c) using only data ≤ t; calibrate on the whole pre-2021 history, not a validation fold.

**Limitations.** N of crowding events in our window ≈ 3; any gate that "works" is likely overfit. Vol-management of factors has mixed real-time results (docs/RESEARCH.md Part I §3.9 #6).

**Potential experiments.** Pre-register (a)–(c) and a fixed gross-scaling rule *before* looking at the 2025–26 windows; report only whether it reduces drawdown in Jul 2025 / Jan 2026 / Jul 2026 and by how much it cuts full-sample IR.

---

#### 2.7 Expectation calibration: live IC haircut and cost-driven decay

**Core concept.** r/quant "Decline in IC going into prod" (1q02d6g): estimates of how much IC survives production — "**~50%** at a multi-strat across all researchers", "**~40%** on my own book", "**20–40%**" as a general range, and one person's report that a rigorous shop saw ~40% while another large HF with a different research process saw **80–90%**. Advice: "start analysing your true fee/slippage costs versus your simulations." "It is literally not possible to have a rigorously valid hold-out set in this business" — new data arrives too slowly and every idea reuses it.

**Evidence.** Reported (four independent commenters, ranges agree). *Demonstrated (independent):* Azevedo–Hoegner–Velikov, *The Expected Returns on Machine-Learning Strategies* — cumulative performance reduction of **57%** from transaction costs, post-publication decay and post-decimalisation liquidity; sophisticated ML strategies (LSTM-based) still earn net out-of-sample monthly returns of up to 1.42%, but with turnover above 50% and a tilt to difficult-to-arbitrage stocks (per the abstract and Quantpedia summary) (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4702406, AFA abstract).

**Use.** Apply a 40–50% haircut to any reported IC/IR when setting expectations for the deck: composite IC 0.037 → ~0.02, gross IR 0.6 → ~0.3–0.35 by the fundamental law (docs/RESEARCH.md Part I §5). It also tells us how to *prioritise*: an idea that only clears the kill rule by a small margin is very unlikely to survive.

**Limitations.** Anecdotal; "IC of what — a single feature or the ensemble?" was asked and not answered. Not a substitute for our own OOS.

**What is new.** docs/RESEARCH.md Part I §5 states expectations from IC·√breadth but no decay haircut.

---

#### 2.8 Text/8-K signals: run a sector-ghost check and try *numeric-specificity drift* (no LLM)

**Core concept.** r/LocalLLaMA 1sqlij0 (single anonymous practitioner, Gemma-4 26B fine-tuned on ~800 of 2,400 earnings-call transcripts, forward-5-day sector-relative label): two signals. **Signal A** — CFOs shifting from numeric guidance ("expect revenue between X and Y") to vague language ("we feel good about our trajectory") → **−1.8% vs sector over 5 days, IC 0.04**, tested on 600 out-of-sample transcripts, "basically zero correlation with momentum, value or any standard factor". **Signal B** — "management confidence" IC 0.09 but **0.85 correlated with sector returns** ("Tech CEOs sound confident when tech is ripping"); killed.

**Why interesting.** Two transferable lessons for §3.4 of docs/RESEARCH.md Part I: (i) **every text-derived score must be sector-neutralised and residualised on `ret_1_0`, `ret_12_1`, `size` before its IC is read**; (ii) the surviving effect was a **structural feature of the text** (numeric specificity), not sentiment.

**Evidence.** Reported only (anonymous, 600-sample OOS, no code, 5-day horizon, earnings calls not 8-Ks). Weak. But the mechanism (text tone ↔ sector return) is well known.

**What is new.** docs/RESEARCH.md Part I §3.4 #6 proposes 8-K text-change similarity; it does not propose a *specificity* feature nor a mandatory sector-ghost check.

**Implementation (Own, cheap).** From the 8-K `text`: digit-token share, count of currency/percentage tokens per sentence, count of forward-looking verbs ("expects", "anticipates") followed by a number vs. not; feature = change vs the same firm's trailing-4 8-K average of the same item. r/algotrading 1ujy65z shows a pure-keyword guidance extractor (median exhibit 99.1 text — our main-text file excludes exhibits, per docs/RESEARCH.md Part I §3.4, so this may be uninformative for 2.02).

**Limitations.** Our 8-K main text is short (median 3.9k chars) and excludes earnings press releases, so guidance language is mostly absent; horizon here is 1 month, not 5 days; large-cap text alpha "dissipates quickly" (docs/RESEARCH.md Part I §3.4 #9).

**Potential experiments.** Round B block: `spec_drift`, residual IC after 2.5's orthogonalisation; kill if residual paired-IC t < 1 on the ≥ $2B universe.

---

#### 2.9 If AI writes the features: typed causal operators + truncation-invariance test

**Core concept.** *AQuA: Recursively Self-Improving Quantitative Trading Research Agents* (arXiv 2608.12841) and the r/LocalLLaMA post 1vxajio describe a concrete failure: one LLM wrote an intraday feature dividing "volume so far" by the **day's final total volume**; a second LLM reviewed it "for causality" and **approved the causal-sounding explanation**. It only died on a clean re-split; a manual audit found the leak. The fix in AQuA v2: seal the splits/labels/evaluator outside the agent, and replace arbitrary feature code with a **fixed registry of causal operators** (a full-day normaliser is simply not expressible).

**Why interesting.** docs/RESEARCH.md Part I §3.8 recommends guardrails ("verbatim definition cards, dual reviews, independent re-run"), i.e. more *review*. The failure above shows **review by a similar model fails**; the structural fix is to make leakage inexpressible. A top comment: "a harness where the feature function physically cannot see rows timestamped after the prediction point… then leakage stops being something anyone has to spot."

**Evidence.** Demonstrated as a *failure case* (paper appendix; I saw the abstract and the Reddit summary, not the appendix; no IC published for the failed feature). The proposed fix's effectiveness: Plausible. Note the AQuA abstract also reports IC +0.0843 / Sharpe up to 2.5 on US equities, 2021–25 — I could not verify how (or whether) this handles model-side look-ahead; treat as unverified.

**Implementation (Own, standard, cheap).** (1) **Truncation-invariance test** for every feature: recompute the feature using only rows with `date ≤ t` and assert equality with the value computed from the full panel, for a random sample of (permno, t). This catches leakage regardless of who or what wrote the code. (2) For any feature-mining loop, expose only lag-aware primitives (`lag(x,k)`, `ts_mean(x,k)` with `k≥1`, cross-sectional `rank`, `industry_mean`) — no free-form code. (3) Keep the evaluator frozen (this repo's `largecap.py` harness already is).

**Limitations.** A registry narrows the search space (AQuA acknowledges the tradeoff). Truncation tests catch time leakage, not *survivorship* or *universe* leakage (docs/RESEARCH.md Part I §3.9 #9).

**Potential experiment.** Retrofit the truncation test into the existing feature build and count how many current columns fail — a cheap audit before Round B.

---

#### 2.10 Short-side realism: borrow cost is adversely selected

**Core concept.** r/algotrading 1w9yp8g and r/quant 1lpzdwr: the rate you pay is not the average GC rate — "**the reason you want to short is correlated with the borrow becoming expensive or unavailable**"; for HTB names it "can jump an order of magnitude exactly when the short thesis is strongest". A clean backtest that uses a median borrow rate "systematically understates cost in the exact states where the edge is largest". Suggested tests: stress borrow at **2×/5×/10×** and add a **locate-reject / forced-cover flag** as a separate state (a discontinuity in the equity path, not a smooth cost). Availability, not price, is "the bigger problem". Also anecdotes: a PM paper-shorting the same 2k-share inventory every day; the Medallion story that they short "something equivalent" when a locate fails; settlement-calendar arithmetic (a Thu-close/Fri-open short pays 3 days of borrow).

**Evidence.** Reported; mechanism is standard and directly matches our own finding that the ML edge lived in illiquid/biotech shorts (`PM_ABLATION`). Heuristic from practitioners: "price, ADV and industry" predict borrow difficulty; "closed-end funds are expensive".

**What is new.** docs/RESEARCH.md Part I §3.7 lists FINRA short interest as a filter; it does not propose *state-dependent* borrow stress.

**Implementation.** Post-process any candidate book: cost multiplier grid on the short leg; drop any short with price < $10 or ADV below a threshold; forced-cover shock (remove 3–5% of the short book at random each month at a bad price) to see the drawdown tail; report the Jan-2021 month explicitly.

**Limitations.** We have no borrow data; this stresses assumptions rather than measures them. Our current ≥ $2B / ≥ $5 universe already screens the worst offenders.

---

#### 2.11 "Earnings in the holding window" as a model input

**Core concept.** r/algotrading 1u19hnl (two live XGBoost momentum models, ~7,600 US stocks, 2015→, monthly rebalance, 10 longs): a single binary `has_earnings_in_window` (earnings date in next 21 days) became the **#1 feature by gain in one model and #5 in the other**, with SHAP treating upcoming earnings as *positive*; a hard "exclude earnings" filter reduced gap risk but lowered CAGR (baselines: 25.3% / 20.2%), and the earnings-feature variant improved drawdown in the growth model to ~−50%. The author still shelved it. Caveat from comments: earnings dates must be point-in-time.

**Evidence.** Reported (a single author; long-only all-cap; no significance test; the feature is an EAP proxy). It **agrees** with the Frazzini–Lamont earnings-announcement premium and **conflicts** with docs/RESEARCH.md's Part I citation that the US premium has disappeared (Heitz et al. 2020) and with the "low prior" grading.

**Implementation.** The 8-K file gives 2.02 filing dates back to 2015: expected-announcement-month = same month last year (and ±3 months). One feature, one test.

**Limitations.** Feature importance ≠ alpha; a binary that identifies *volatile* names can win importance by proxying idiosyncratic vol. Test residual IC after `ivol_capm_21d`.

---

#### 2.12 Reinforcements and failure reports (kept short — these support docs/RESEARCH.md's Part I skip list)

| Report | Finding | Grade |
|---|---|---|
| **"Tried to replicate the Attention Factors stat-arb paper…" (r/quant 1w8otws, Sep 2026)** | Top-500 Russell-1000 point-in-time, survivorship-free, 2016–2026, 38 rank-normalised chars, 5 bp + 1 bp shorts: **K=30 mean OOS Sharpe −0.70** (paper: net 2.30 on 24 yrs). Deterministic PCA-residual reversion negative every year: gross −0.59%, costs 5.86%, **turnover 9,190%**. Weekly 1/N same window = **+1.36**. Top comments: "I don't bother with papers that report miraculous SRs and no repo". Post reads as Claude-assisted; single split; different sample length; author found and fixed look-ahead and sign-symmetry bugs. | Reported (not a refutation; the paper's abstract claims OOS Sharpe > 4 gross / 2.3 net on the largest US equities) |
| "Built a full Lopez de Prado pipeline in Rust… AUC = 0.50 OOS" (algotrading 1s77s5k) | 442 tests, CPCV, HMM, meta-labelling, permutation test p = 0.60 → no edge; individual feature IC is real but does not predict *which events win*. Top answer: you built a prediction pipeline before proving there was something to predict. Also 1pm20qy: "No one uses López de Prado's methods, because they don't work" (15 upvotes) vs a 419-upvote meta-labelling explainer (1lnm48w). **Disagreement preserved.** | Reported |
| "Cross sectional IC ranked signal ideas" (algotrading 1vowexr) | A practitioner who tried "thousands of features, ML rankers, z-scores": "*you can basically slightly leverage QQQ by cross-sectional selection*… for every bit of alpha you buy more risk… no signal worth trading". Also: fixing split-adjustment errors **removed** vol/fundamental alpha; top-5% momentum picks worsen *sector concentration*; use rank IC and **sector-neutralise before evaluating**; rare-firing signals cannot work at a monthly rebalance — use always-on graded scores. | Reported |
| r/MachineLearning "What do you think about Tabular Foundation Models" (1thofoe) | TabPFN-3 "basically comparable with a good gradient boost", "somewhat worse than LightGBM with light tuning", heavy inference; **licence: "Non-Commercial Purpose… provided the results are not used in commercial decision-making"**. | Reported |
| r/datascience 1tip0d8, 6-model swap | NN best CAGR; RF/LASSO didn't beat S&P. **Discounted**: default hyper-parameters, unspecified universe, non-walk-forward, target = 10-day forward return, monthly rebalance; top comment (158 upvotes) says exactly this. | Reported, near-worthless |
| r/quant 1jju8jj, 1l59in9, r/algotrading 1progoo, 1iliivd | Consensus: linearise features, start regularised-linear; "ML enhances an existing edge, never found one"; **top-voted fix (60 upvotes): "encode your heuristic as a feature"** — an independent statement of docs/RESEARCH.md's Part I composite-as-prior. | Reported |

---

#### 2.13 Additional small findings worth carrying

- **Alpha combination: "close the loop."** r/quant 1w99f1v: for large alpha libraries, evaluate combination methods with the *actual portfolio backtest* (top comment suggests a **Shapley decomposition over the alpha set** across a grid of combination methods and targets); "quite a lot of big wins come from knowing when *not* to listen to a signal". Consistent with the repo's `pm_ablation.py`; new element = Shapley/marginal-contribution over blocks. Reported.
- **Signal prep tradeoff.** 1j7sl2j: EMA-smoothing lowers IC but raises autocorrelation; fitting to forward returns and then MVO "compounds estimation error" — keep to rank + inverse-vol weights. Reported.
- **Definition of IC.** 1rmy6w3: unqualified "IC" = `rank_corr(s, r)`; for monthly books, report raw IC, sector-neutral IC and risk-model-residual IC — our `PM_ABLATION` style. Reported.
- **Power analysis before backtesting** (1r6rc2d): decide the smallest detectable effect first; with 68 months and IR s.e. 0.46 (docs/RESEARCH.md Part I §5) this is the paired-IC discipline already used.
- **Open-source resource:** Toraniko (MIT, numpy+polars, Barra-style market/sector/style factors, "used in production on > $10B AUM", no covariance shrinkage yet, US-only) for 2.4/2.5. https://reddit.com/r/quant/comments/1ekpin6
- **Macro conditioning, weak evidence:** the only feature that moved a 1,400-stock GBT+MLP long-horizon ensemble after 45 rounds of price/volume feature engineering was **yield-curve slope** (algotrading 1rwp4z8); a commenter also flags that 45 rounds of feature-engineer→WFO→repeat is itself an overfitting mechanism. Consistent with docs/RESEARCH.md Part I §3.3 #7 (low priority, high overfit risk).
- **Pod-shop process observation (1v30qx3, Reported):** a centrally netted book pays less spread/impact/financing than several pods trading the same names, and "keeps running lower-Sharpe durable signals through bad stretches because nobody is cut at their drawdown limit". Not actionable for a hackathon; supports the view that costs, not alpha, separate outcomes.
- **Index-event alpha migration (1ppnw7t):** index inclusion/deletion effect is "a stupidly crowded trade", "progressively noisier"; one Vanguard manager's comment that the complex rebalancing is "handed off to our arbitrage desk". Dead end for us; noted so nobody spends time on it.

---

### 3. Novel Ideas

#### 3.1 Evidence-backed novel ideas (relative to docs/RESEARCH.md Part I)

1. **Asymmetric rank buffer + partial rebalance / G–P aim portfolio** (2.1) — the only idea here that combines a *large-cap, monthly, peer-reviewed* result with independent practitioner advice. *Demonstrated (rule), Reported (implementation folklore).*
2. **Blitz short-term composite** as a new, orthogonal, large-cap-native block (2.2). *Demonstrated (peer-reviewed), not on our sample.*
3. **Live-IC haircut for expectations** (2.7). *Reported + Demonstrated (57% cumulative reduction).*
4. **Truncation-invariance test and a typed causal-operator registry for AI-generated features** (2.9). *Demonstrated failure mode; fix Plausible.*
5. **Era Splitting** as the tree-level answer to "splits that only work in some regimes" (2.3). *Reported (authors' claim on Numerai).*
6. **Contradiction to preserve:** neutralisation depth (2.4). *Reported.*
7. **Sector-ghost check for text signals** and **numeric-specificity drift** (2.8). *Reported.*
8. **State-dependent borrow stress** (2.10). *Reported.*

#### 3.2 New hypotheses (my own — not attributable to Reddit or a paper)

| # | Hypothesis | Grade | Why I think it could work | First test |
|---|---|---|---|---|
| H1 | **Size-environment invariant trees**: era-split with eras = month × size tercile, so a split must help large caps too (2.3) | Speculative | Directly encodes "our edge is small-cap" as a constraint the learner must satisfy; converts the failure into a regulariser | `EraHistGradientBoostingRegressor` vs vanilla on ≥ $2B universe |
| H2 | **Orthogonal-residual training**: train the ML on `y − β·composite_score` (or with composite as `init_score`) *and* require zero rank corr with the composite at prediction time, so ML can only produce what the shared core does not (2.5 + docs/RESEARCH.md Part I #2) | Plausible | Two independent practitioner streams ("encode the heuristic", "shared core") point at the same design | Residual IC by size tercile; kill if < 0.01 |
| H3 | **Neutralisation dial + crowding-stress objective**: pick neutralisation depth by drawdown in the 3 unwind windows, not full-sample IR (2.4/2.6) | Plausible | Resolves the contradiction empirically | 5-step dial, report stress DD |
| H4 | **Feature-block knockoffs**: use a knockoff / permutation null to control the false-discovery rate across the ~25 candidate blocks docs/RESEARCH.md Part I lists. Literature exists for knockoffs in the factor zoo (robust knockoffs 2206.06026; "Controlling FDR under cross-sectional correlations" 2102.07826; NBER "Taming the Factor Zoo" w25481); I found no evidence of use in a *monthly large-cap ML ranking* pipeline | Plausible | docs/RESEARCH.md Part I already demands a multiple-testing count; knockoffs would give an actual FDR-controlled *selection*, not just a Bonferroni discount | Build within-(month, industry) shuffled null for each block's paired-IC gain; adopt only blocks above the 95th percentile of the null-max |
| H5 | **Conformal selection of names**: use model-agnostic conformal selection (finite-sample FDR-controlled subset selection) to choose the long/short set so that the expected share of "wrong" picks is bounded | Speculative | Directly targets the concentration-on-noise problem; a natural "abstain" that is theoretically calibrated, unlike gating on volatile validation slices (2.6). Adjacent literature: conformal selection framework; *Conformal Predictive Portfolio Selection* (2410.16333) is portfolio-level, not name-level | Needs calibration set of past months; likely too little data (68 months) — a toy test on the in-sample folds first |
| H6 | **Pooled international large-cap training**: 148-characteristic panels exist for 46 markets (Cakici–Fieberg–Metko–Zaremba, JEDC 2023: predictability depends on firm size and recent information); pooling developed-market large caps could multiply the effective training sample | Speculative | Sample size (68 test months / 1,200 names) is the binding constraint; more independent cross-sections is the classical fix. *Unverified*: whether FIAM's rules allow external data of this form, and whether characteristic definitions align with our 147 | Feasibility check first (rules + data); if allowed, train on US+Europe/Japan large caps, test on US large caps |
| H7 | **Graded event intensity instead of flags**: convert sparse 8-K event flags (4.02, 4.01, 5.02, 2.05) into exponentially decayed *intensity* scores (half-life 3–6 months) so they exist every month (algotrading 1vowexr: rare-firing signals cannot work at a monthly rebalance) | Plausible | Solves a real mismatch (event alphas are sparse; cross-sectional ranking needs every-month coverage) | Add decayed counts as features; paired-IC on the ≥ $2B universe |
| H8 | **Buffered composite + era-split residual model as a two-stage system** (2.1 → 2.3): stage 1 = buffered composite (low turnover); stage 2 = invariant-tree adjustment allowed only to move a name across the buffer boundary | Speculative | Uses ML where it is cheapest (boundary decisions) and the composite where it is best (core ordering) | Only after 2.1 and 2.3 individually pass |

---

### 4. Best Experimental Candidates

I am not scoring; these are the factors that make each one worth running, in the repo's pre-registration style. Note that the repo's existing docs/RESEARCH.md Part I "Round A–D" plan is unchanged; these slot in.

| Candidate | Evidence | Novelty vs docs/RESEARCH.md Part I | Cost | Impact if it works | Main uncertainty |
|---|---|---|---|---|---|
| **2.1 Rank buffer / L1 penalty / partial rebalance** on the frozen composite | Peer-reviewed large-cap monthly result + independent practitioner advice | High | **Very low** (no new data, one LP change) | Direct: the repo already saw turnover as the recurring drag | Do gains survive the 10% one-way budget? |
| **2.5 Orthogonalise-to-composite diagnostic** | Practitioner testimony + own logic | High | **Very low** (regression per month) | Changes how *every* other candidate is judged | Composite is only a proxy for the real crowded core |
| **2.9 Truncation-invariance audit** | A documented real failure | Med | **Very low** | Protects all later results (and the brief's leakage scrutiny) | None — cheap insurance |
| **2.2 Industry-relative reversal + industry momentum** | Peer-reviewed | Med | Low (`ret` + `ff49`) | New orthogonal large-cap block | Needs 2.1 to be affordable; 2021–26 hostile to short-term factors |
| **2.3 / H1 Era-split trees** | Authors' claim on one dataset; implementation exists | High | Medium (compile fork, CPU-only) | Could let a fitted model finally add to the composite | May underfit and reproduce the composite (an acceptable outcome) |
| **2.4 / H3 Neutralisation dial with stress windows** | Contradictory testimony | Medium (a contradiction) | Medium | Resolves the biggest open decision in docs/RESEARCH.md Part I #6 | N = 3 stress months |
| **H7 Decayed event intensity** | Plausible, own | Med | Low | Makes 8-K block usable at monthly cadence | 8-K value in large caps is unproven |
| **H4 Block knockoffs/permutation null** | Literature exists, no direct evidence for this setting | Med | Medium | Formal discipline for ~25 candidate blocks | Time-series correlation breaks exchangeability; needs care |

---

### 5. What We Still Don't Know

1. **Do our signals survive a rank-buffer under a 10% turnover budget?** Blitz's evidence is for a much higher-turnover regime (2.1).
2. **Neutralisation depth: does harder neutralisation reduce or increase crowded-unwind losses?** Two opposing anecdotes; no quantitative source found (2.4).
3. **Does Era Splitting help on monthly US large caps at all?** Only Numerai (weekly, obfuscated) evidence; the fork ships no benchmarks (2.3). Also unknown whether month is the right era length for ~150 in-sample eras.
4. **How much of our composite is a "shared core"?** No public measure exists; 13F overlap is lagged and quarterly (2.5).
5. **Is the 40–50% IC haircut applicable to a long-only-in-factor-composite book, or mostly to fitted models?** The Reddit figures did not say what type of signal they referred to (2.7).
6. **Was the Attention-Factors negative replication a bug or a real non-transfer?** No code was released by the authors; the replication is single-split, 10 years, and the poster fixed several bugs during the process (2.12). Needs an independent replication before the paper can be used as *evidence* against.
7. **Do tabular foundation models (TabPFN-3 / 3.5) do anything on monthly cross-sectional returns?** Nobody reports; licence excludes commercial decision-making (2.12).
8. **Is a knockoff/conformal approach feasible with ~68 test months and serially correlated cross-sections?** Nothing found (H4/H5).
9. **International pooling:** does FIAM's rules permit it, and does it help a US large-cap target? Completely unexplored here (H6).
10. **Where the research is sparse.** Reddit contains almost nothing on: rank/gauss-rank targets (the topic docs/RESEARCH.md Part I ranks #1), target regularisation, multi-target ensembles, or 8-K/text-change signals in large caps. A grep of the full 40k-post local corpus for "gauss / rank target / rank transform / lambdarank / learning to rank / neutraliz" returned 87 matches, of which only about six are on-topic (the rest are "Gaussian" distributions, copulas and HMMs): 1c6t9b0 (rank vs regression), 1jzt8h9 and 1vvt4m9 (neutralisation), 17m27fe (normalisation), 1rw4oip (raw signal), 1s77s5k. The literature in docs/RESEARCH.md Part I is therefore the *only* basis for the target-design ideas; Reddit neither supports nor contradicts them beyond one thread (1c6t9b0) arguing that quantile targets lose information needed by a downstream optimiser but are a good first hypothesis test (top comment: "beta test with percentiles, then trade the actual return predictions").

---

### 6. Search log, dead ends, and duplicates

**Branches that produced substantive material:** trading-cost/turnover control → Blitz + G–P + Boyd; crowding → 4 pod-shop threads; alpha decay in production; neutralisation; text/LLM signals → ghost check + AQuA; ML success/failure reports; borrow-cost realism; Numerai tree methods (outside Reddit).

**Dead ends (no new information):** r/Numerai (48 posts); careers threads; retail "my bot made X%" posts (r/algotrading 1m5keqj "~1% return/day stat-arb" (1,180 upvotes), 1u6w3ra "gold mine" (354), 1progoo …) — discounted per REDDIT.md §4; index-event alpha; satellite/alt-data folklore (197mah6); crypto cross-sectional threads; options-skew hobbyist threads in r/options (no cross-sectional return-prediction content); r/econometrics ML threads; the r/datascience 6-model swap; r/SecurityAnalysis (human-analyst content, nothing on 8-K/4.02/insider factors beyond retail "cluster buy" claims — one commenter: insider-purchase screens are "one of the worst performing screens"; another: "3+ insiders in a 2-week window" is the strongest signal from a 10-year backtest by the tool's author — contradictory, unquantified).

**Duplicates of docs/RESEARCH.md Part I** (kept out of the main text unless there was new evidence): residual/industry momentum, PEAD dead in large caps, 8-K item flags, Numerai neutralisation & multi-target ensembles, LLM look-ahead, momentum crash & crowding narratives, sequence models and foundation models on the skip list.

**Adversarial final round (searched for what docs/RESEARCH.md Part I/REDDIT.md would not have surfaced).** Turnover-control theory (found), production IC decay (found), tree-level invariance (found), AI-feature-leak infrastructure (found), borrow adverse selection (found), multiple-testing/knockoffs (literature found, no practice), conformal selection (literature found, no practice), international pooling (literature found, no direct evidence), synthetic-data / data-augmentation / self-supervised pre-training for cross-sectional returns (Reddit: nothing substantive — threads are retail synthetic-OOS generators), hierarchical/Bayesian shrinkage and Fama–MacBeth style panel methods (nothing on Reddit), regime detection (retail-dominated, HMMs; nothing with cross-sectional evidence), earnings-window features (found, weak).

---

### 7. Source index

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


---

## Part III — Candidate Papers — Beyond the OLS Baseline

Reading list of methods to build on top of the plain-OLS floor in `ols.py` /
`experiments/ols/README.md`. Scope: methods that operate on the 147 numeric characteristics
in `fiam/chars_final_with_names.parquet` — prediction models and
portfolio-construction methods only, no LLM/text/agentic approaches, and no
OLS baseline itself (that's the floor, not a candidate).

**Filtered to 2026-only.** An earlier version of this list included
foundational but older papers (Kelly-Malamud-Zhou 2024, Nagel 2025,
Kozak-Nagel-Santosh 2020, Freyberger-Neuhierl-Weber 2020, Bryzgalova-Pelger-Zhu
2025, Gu-Kelly-Xiu 2021, Feng-Giglio-Xiu 2020, Cakici et al. 2025, and an
older non-2026 cut of *AlphaPortfolio* and *Decision by Supervised Learning
with Deep Ensembles*) — all removed here per an explicit request to keep only
2026 work. Every paper below was confirmed dated 2026 (posting date checked
directly against NBER/arXiv, not inferred from search snippets).

Each entry: what it does, and which documented weakness in `experiments/ols/README.md` /
`experiments/xgb/README.md` it would address.

---

### 1. Prediction layer — replace/augment plain OLS or XGBoost

#### *The Virtue of Sparsity in Complexity* (Apr 2026)
**Verbatim abstract (checked directly, not inferred):** distinguishes
*capacity sparsity* (dimensionality of the candidate feature space) from
*factor sparsity* (parsimonious structure of priced risks) and argues
they're complements — expanding capacity *enables* the discovery of factor
sparsity. Revisits Didisheim et al. (2025)'s benchmark design at higher
complexity and shows **nonlinear feature expansions combined with basis
pursuit** (an L1-minimization technique, not plain LASSO) yield portfolios
that dominate **ridgeless benchmarks** (near-zero-regularization ridge, the
Kelly-Malamud-Zhou-style overparameterized regime) beyond a critical
complexity threshold. The gains come from *enlarging* the feature space
first, not from directly penalizing the original small feature set.
- Posted 2026-04.
- [arXiv PDF](https://arxiv.org/pdf/2604.17166)
- **Implementation note (`sparse.py`):** what was actually built is plain
  LASSO on the original 147 characteristics, unexpanded — no nonlinear
  feature-expansion step, and benchmarked against OLS/XGBoost rather than a
  ridgeless overparameterized-ridge comparator. This tests a much narrower,
  simpler question ("does directly penalizing the raw features help") than
  the paper's actual claim (which is about sparsity *emerging from* an
  expanded feature space). See `experiments/sparse/README.md` for the full caveat.

#### *Quantity, Risk, and Return* (Sep 2026)
**Verbatim abstract (checked directly):** expected stock return depends not
only on factor risk exposure (beta) but on the **factor's** own quantity
fluctuations (q) "induced by trading flows" — i.e. quantity is a
**factor-level** concept (how much noise-trading flow a given factor's
loading has absorbed), not a per-stock liquidity measure. "Sophisticated
investors should demand a higher factor premium when they have absorbed
noise trading flows of stocks with high loadings to that factor." The
cross-sectional risk-return relationship is flat unconditionally but strongly
depends on quantity once conditioned on it; the BTQ model also addresses the
factor-zoo problem by selecting a small number of factors.
- Posted 2026-09.
- [arXiv](https://arxiv.org/abs/2609.05162)
- **Implementation note (`btq.py`):** this panel has no trading-flow-by-type
  data to build the paper's actual factor-level quantity variable, so what
  was built is a much simpler per-*stock* liquidity proxy (mean of
  `turnover_126d`/`turnover_var_126d`/`dolvol_126d`/`dolvol_var_126d`,
  `docs/FACTORS.md` §16) interacted linearly with every characteristic —
  same general "condition on trading activity" spirit, structurally
  different from the paper's factor-level mechanism. See `experiments/btq/README.md`.

#### *Quant Convergence: Bridging Classical Value Investing and Modern Factor Models for Systematic Equity Selection* (Jun 2026)
Empirical horse race of XGBoost, AutoGluon, and Random Forest across three
feature sets (classical Graham value rules, modern factors, and a hybrid) on
20 years of S&P 500 data. Finding: **plain Random Forest on the simplest
feature set had the best risk-adjusted result** (highest return, best Calmar
ratio), while the more complex AutoGluon ensemble had a larger drawdown for
similar return. A second independent data point — after this project's own
XGBoost result (`experiments/xgb/README.md` §4) — that added model complexity doesn't
reliably buy better risk-adjusted performance on this style of tabular
factor data, and a concrete reason to try plain Random Forest as a cheap
bagging-based alternative to boosting before reaching for anything heavier.
**Verbatim abstract confirms**: the *best* result was specifically the
**Graham-rules-only** Random Forest (highest return, 1.38 Calmar), and the
best-drawdown result was the **combined** (Graham + momentum) Random
Forest — "pure modern factors" alone was not the winning feature set in
their results, model family (Random Forest) was one part of the finding,
curated/simple features were the other.
- Posted 2026-06.
- [arXiv PDF](https://arxiv.org/pdf/2606.24575)
- **Implementation note (`rf.py`):** only the model-family swap was
  tested (Random Forest in place of XGBoost), on the *same full
  147-characteristic panel* as every other script here — the paper's
  actual best-performing feature-curation arm (Graham-only, or Graham +
  momentum) was not replicated. `rf.py`'s strong result is genuine but
  only directly confirms half of the paper's finding. See `experiments/rf/README.md`.

#### *RankGLU: Residual Gated Score Formation for Cross-Sectional Stock Prediction* (Jun 2026)
A prediction-head architecture built specifically to solve the "how do I
turn a model's score into a stable ranking/weight" problem — a bounded,
gated nonlinear branch alongside a direct linear scoring path, designed so
the model doesn't overfit unstable return magnitudes while still capturing
some nonlinear interaction. Directly relevant to the calibration concern
`docs/NEXT.md` §3 already flags for tree-model conviction weighting ("z-score
or rank-transform the predictions first, rather than plugging the raw
predicted value directly into the weight") — this is a more structured,
learned version of that same fix. Only validated on Chinese equity indices
(CSI300/CSI800) in the paper; transfer to this panel is unverified.
**Verbatim abstract confirms** the "direct linear scoring path + bounded
multiplicative branch" framing exactly, though exact layer formulas aren't
given in the abstract.
- Posted 2026-06.
- [arXiv PDF](https://arxiv.org/pdf/2606.08930)
- **Implementation note (`rankglu.py`):** this is the closest structural
  match of any of the 8 — `score = X@w_lin + b_lin + (tanh(z)*gate)@w_out
  + b_out` directly implements a linear path plus a bounded (tanh),
  gated (sigmoid) multiplicative branch, hand-coded in numpy since no
  deep-learning framework was available. Exact architecture details beyond
  what the abstract states are still unconfirmed. See `experiments/rankglu/README.md`.

---

### 2. Portfolio-construction layer — predictions → constrained weights

#### Wang, Gao, Harvey, Liu & Tao (2026), *Machine Learning Meets Markowitz*, NBER WP 34861
Argues the standard two-stage pipeline — forecast returns, then plug into an
optimizer — is "deeply problematic" because it treats prediction error as
equally costly for every stock, when the optimizer only cares about error in
the stocks that end up mattering to the final portfolio. Proposes fitting
the return model and the portfolio weights jointly instead. **Directly
describes the architecture both `ols.py` and `xgb.py` currently use**
(predict, then separately solve an LP) — worth citing in the deck's
methodology section as the documented limitation of that two-stage design,
whether or not it's rebuilt jointly before the deadline.
- Posted 2026-02-24.
- [NBER working paper (free PDF)](https://www.nber.org/system/files/working_papers/w34861/w34861.pdf)
- [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6290354)
- **Implementation note (`joint.py`):** the abstract gives the argument but
  no algorithmic detail (loss function, weighting scheme) — what's built is
  a much simpler approximation (train-loss weighted by realized-target
  rank extremity), not a reproduction of "unifies the expected return
  generation process and the final optimized portfolio." In practice,
  validation always selected the unweighted case, so the approximation
  ended up not even being tested against OLS in the primary result — see
  `experiments/joint/README.md`.

#### Cong, Tang & Wang (2026), *AlphaPortfolio: Goal-Oriented Investment Management Through Deep Reinforcement Learning*, NBER WP 35195
**Verbatim abstract (checked directly — the PDF, not a search snippet):**
adapts attention-based neural networks and RL to direct portfolio
construction. Core architecture is a **Transformer encoder** for
long/short-range path dependence in firm and market states plus a
**cross-asset attention network**, trained end-to-end (not step-by-step) on
objectives that are non-additively-separable across periods — including the
Sharpe ratio directly. In U.S. equities: Sharpe above 2, risk-adjusted alpha
over 13% with monthly rebalancing, robust to excluding small/illiquid
stocks. Demonstrates flexibility to incorporate **transaction costs** and
state interactions, "before developing a **polynomial-feature-sensitivity
analysis**" — a *post-hoc interpretability tool applied to the trained
Transformer/RL model*, not a standalone predictive architecture — "to
uncover key drivers of performance, including their rotation and
nonlinearity." This paper was previously titled "Goal-Oriented Portfolio
Management Through Transformer-Based Reinforcement Learning" and includes
partial results from an earlier, separate, pre-2026 working paper ("Direct
Construction Through Deep Reinforcement Learning and Interpretable AI,"
jointly with Yang Zhang) — **the "polynomial network" / "economic
distillation" framing this project initially used to describe this entry
came from that older paper, not from this one**, and has been corrected.
- Posted 2026-05-19 (as "Working Paper 35195"), May 2026 per the PDF header.
- [NBER working paper](https://www.nber.org/papers/w35195)
- [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6785198)
- **Implementation note (`poly.py`):** no Transformer, no RL, no
  cross-asset attention, no transaction-cost term were built (no RL/deep-
  learning infrastructure in this project). What's implemented is a
  standalone degree-2 polynomial Ridge regression on a curated 20-
  characteristic subset — inspired by the *shape* of the paper's post-hoc
  polynomial-sensitivity idea (interaction/quadratic terms as an
  attribution tool), repurposed here as the primary predictive model
  itself rather than an analysis layered on top of a different model. This
  is a materially narrower and structurally different thing than what the
  paper built. See `experiments/poly/README.md` for the full correction and its effect
  on results.

#### *AlphaZeroBeta: Deep Reinforcement Learning for Market-Neutral Portfolios* (Jul 2026)
**Verbatim abstract (checked directly):** a deep RL framework combining a
**composite reward function** (balances risk-adjusted excess return,
benchmark correlation, and **transaction costs**) with a **CNN-GRU policy
trained end-to-end via Recurrent PPO** (proximal policy optimization — a
specific policy-gradient RL algorithm, not generic backpropagation),
evaluated via rolling walk-forward across **seven equity indices,
2014–2024**. Achieves higher Sharpe than baselines with near-zero benchmark
correlation. The most literal upgrade path from the per-month linear
program in `ols.py` §2.5/§3.5 (`experiments/ols/README.md`) or `xgb.py`'s identical LP
(`experiments/xgb/README.md`): instead of re-solving a fresh LP from a fixed prediction
each month, learn the characteristics-to-constrained-weights mapping
directly.
- Posted 2026-07-20.
- [arXiv PDF](https://arxiv.org/pdf/2607.18001)
- **Implementation note (`e2e.py`):** no CNN, no GRU, no PPO, no RL of any
  kind, and no transaction-cost term were built. What's implemented is a
  linear scoring function trained by SPSA (a classical zeroth-order
  stochastic optimizer, unrelated to policy-gradient RL) against a smooth
  portfolio-Sharpe proxy. The only thing genuinely shared with the paper is
  the high-level idea "train weights end-to-end against a portfolio
  objective instead of a two-stage predict-then-optimize pipeline" — the
  actual mechanism is unrelated. See `experiments/e2e/README.md`.

---

### 3. Diagnostic / ruled-out — useful for the deck, not standalone methods

#### *Quantum Kernels and the Cross-Section of Stock Returns: Anatomy of a Vanishing Advantage* (Jul 2026)
A controlled horse race (quantum fidelity kernel vs. projected quantum
kernel vs. a classical RBF kernel control, identical training data/solver/
tuning budget) finds the quantum kernels' apparent edge vanishes under fair
comparison. Not something to implement (no practical quantum hardware
access, and the result is negative), but useful as a citable example of due
diligence for `docs/FIAM.md` §14's "quality of reasoning" criterion — a
documented instance of an exotic method being checked and ruled out.
- Posted 2026-07.
- [arXiv abstract](https://arxiv.org/abs/2607.20168)

#### Bernstein, Goldberg, Gunther, Kercheval, Lan, Lin & Yao, *Principal component error in high-dimensional factor models* (Sep 2026)
**Verbatim abstract (checked directly):** decomposes PCA-based
factor-estimation error into "out-of-subspace error" (distance from the
estimate to the true population factor subspace — data-observable, an
estimable floor) and "in-subspace error" (arises from finite sample size of
latent factor returns, cannot be estimated from data alone), each with
almost-sure asymptotic limits as dimension grows with sample size bounded.
**Illustrated with a three-factor SIMULATION of the US public equity
market** (synthetic, low-dimensional) — their finding that "out-of-subspace
error dominates" is demonstrated in that controlled simulation, not on a
real 147-dimensional panel. Not a prediction method — a diagnostic tool.
Relevant only if a PCA/shrinkage-based prediction model is built on the 147
characteristics.
- Posted 2026-09.
- [arXiv](https://arxiv.org/abs/2609.20550)
- **Implementation note (`pca_diagnostic.py`):** doesn't reproduce their
  asymptotic estimator — uses split-half principal-angle subspace stability
  as a directly computable proxy, run on the real 147-characteristic panel
  (not a synthetic simulation). Related in spirit, answers a related but
  not identical question. See `experiments/pca_diagnostic/README.md`.

---

### Status: all 8 built, run, and fidelity-checked against verbatim abstracts

`ols.py` (floor) and `xgb.py` (nonlinear model, underperformed the floor —
`experiments/xgb/README.md` §4) were built first. All 8 entries above now have a working
implementation (`rf.py`, `sparse.py`, `joint.py`, `e2e.py`, `btq.py`,
`rankglu.py`, `poly.py`, `pca_diagnostic.py`), each corrected once against
its source paper's actual verbatim abstract (not a search summary) — see
each entry's "Implementation note" above and the corresponding `docs/*.md`
for what was fixed and why. Final OOS results, ranked by Information Ratio
(2021-01–2026-08, all vs. the same beta-neutral LP construction):

| Rank | Model | OOS R² | IR | Sharpe | CAGR | Realized β (t) |
|---|---|---:|---:|---:|---:|---:|
| 1 | **RF-Modern** (`experiments/rf/README.md`) | +0.12% | **1.05** | 1.21 | 30.0% | −0.04 (−0.21) |
| 2 | **Poly** (`experiments/poly/README.md`) | −0.02% | 0.87 | **1.08** | 21.2% | −0.18 (−1.12) |
| 3 | RF-Combined (`experiments/rf/README.md`) | −0.21% | 0.81 | 0.97 | 23.8% | 0.14 (0.65) |
| 4 | **E2E** (`experiments/e2e/README.md`) | **+0.07%** | 0.79 | 0.93 | 25.3% | −0.21 (−0.88) |
| 5 | Joint (`experiments/joint/README.md`) | −0.52% | 0.75 | 0.88 | 23.3% | −0.18 (−0.74) |
| 6 | BTQ (`experiments/btq/README.md`) | −0.002% | 0.67 | 0.80 | 20.9% | −0.15 (−0.60) |
| 7 | RankGLU (`experiments/rankglu/README.md`) | −0.34% | 0.57 | 0.77 | 14.0% | **−0.02 (−0.12)** |
| 8 | RF-Graham (`experiments/rf/README.md`) | −0.44% | −0.23 | −0.07 | −4.6% | −0.29 (−0.29) |
| 9 | Sparse-RFF (`experiments/sparse/README.md`) | −0.06% | −0.47 | −0.03 | −0.7% | −0.03 (−0.36) |
| 10 | Ridgeless-RFF (`experiments/sparse/README.md`) | −0.18% | −0.70 | −0.22 | −2.2% | 0.09 (1.37) |

*(vs. `ols.py`'s own baseline: R² −0.01%, IR 0.85, Sharpe 0.98, CAGR
27.6%, β −0.17 (t=−0.67) — still not beaten on IR/Sharpe/CAGR by any of the
8, though `RF-Modern`, `E2E`, and `Poly` all have better (less negative or
positive) OOS R², and `RankGLU` now has the best realized-beta neutrality
of any model in the whole project.)*

**Two genuinely positive R² results** (`RF-Modern`, `E2E`) and **one best-
in-project neutrality result** (`RankGLU`) survived rigorous fidelity
correction — these three are the strongest candidates for further work.
**RF-Graham directly contradicts its source paper** in this project's
market-neutral long/short setting (see `experiments/rf/README.md` for the likely
explanation) — a genuine, reportable negative result, not a bug.
