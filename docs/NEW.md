# NEW.md — Ideas To Find A Model That Actually Works (research sweep, 2026-09-21)

Scope: a broad web sweep (~160 searches/fetches: arXiv, SSRN abstracts, Quantpedia/Quantocracy, practitioner Substacks, Numerai, quant-fund news, X). Goal: ideas we have **not** tried, not a re-run of what is in `docs/`. Everything is measured against where the repo stands (see §1).

Companion docs: `docs/NEGATIVE_RESULT.md`, `experiments/largecap/README.md`, `experiments/pm_ablation/README.md`, `experiments/factor_filter/README.md`, `experiments/tpa/README.md`, `experiments/hrp/README.md`.

---

## 0. How to read this (evidence quality, and what I could not reach)

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

## 1. Where the repo stands, and what the literature says about *why*

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

## 2. Ranked shortlist — what to try first

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

## 3. Detailed ideas

### 3.1 Target engineering — the highest-leverage, cheapest change

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

### 3.2 Training universe and sample weighting

- **Value-weighted / size-weighted training loss.** Gu–Kelly–Xiu's own variant weights loss by market value because the smallest 20% of stocks are ~3% of cap; more economically relevant weight on large names. [C] https://dachxiu.chicagobooth.edu/download/ML.pdf. Our `_trad` arms *restricted* the universe; **weighting** (e.g. `sqrt(mcap)` or dollar-volume weights, as Numerai's sample-weight vector does) is a softer variant not yet tested. (Own: try `w = min(mcap, cap)^0.5`.)
- **Recency weighting / shorter windows.** Delphic Alpha's cross-asset Lasso: a 6-month rolling window dominated 12 and 18; feature *selection* frozen once, weights re-fit monthly. [B] https://delphicalpha.substack.com/p/from-alpha-signals-to-portfolio · Formal result: model complexity and training-window length must be chosen **jointly** (14% OOS R² gain over fixed-window; strongest in recessions). [A] https://arxiv.org/abs/2512.23596 — our expanding window + rank-IC selection never varies the window. *Own:* add exponential half-life {24, 48, 96 months} to the validation grid.
- **Numerai "deep incremental learning":** stack XGBoost models trained on different eras; two-layer stack beat single models under distribution shift. [A] https://arxiv.org/abs/2303.07925
- **Missing values:** cross-sectional mean/median fill is fine — Chen & McCoy on 159 predictors found simple imputation beats EM-style methods. [A] https://arxiv.org/abs/2207.13071 (so our current preprocessing is not a weakness).

### 3.3 Features derivable from data we already have (no new data)

The panel contains `ret` (monthly, per `permno`), `prc`, `prc_high`, `prc_low`, `dolvol`, `tvol`, `shares`, `gics`/`ff49`/`sic`, and the 147 characteristics. All below are computable at month *t* from month ≤ *t* data.

1. **Information discreteness ("frog in the pan").** Da–Gurun–Warachka: momentum is 5.94% for *continuous*-information stocks vs −2.07% for discrete-information stocks with the same cumulative return; replicated across large/small cap and institutional-ownership splits. [A] https://ideas.repec.org/a/oup/rfinst/v27y2014i7p2171-2218..html — `ID = sign(ret_12_2)·(%neg months − %pos months)` from monthly `ret` (or daily if external).
2. **Path shape: slope & smoothness.** Quantitativo replication: 7.9% annual alpha (t 3.44) vs FF5+mom, Sharpe 1.12 on smoothness alone; smooth up-trends long, noisy down-trends short. [B] https://www.quantitativo.com/p/slope-strength-and-retail-extrapolation — compute R² and slope of the last 12 cumulative-return points.
3. **Residual / industry-neutral / 52-week-high-neutral momentum.** Raw momentum crashes in sector rotations; residual momentum (net of market and sector) and 52-wk-high-neutral momentum have lower crash risk (52wk-high-neutral: Sharpe +50%). [C] https://quantpedia.com/strategies/residual-momentum-factor · https://alphaarchitect.com/reducing-the-impact-of-momentum-crashes/ — the repo dropped momentum for defensibility; these are the *defensible* variants. Note `resff3_12_1` already exists (FF3-residual); add **within-`ff49` industry-relative** versions.
4. **Industry-relative characteristics.** For long-short, adjusting for sector "boosts [better result] from 20% to 78% of the time"; largest reduction in *value-weighted large-cap* strategies. [C] https://alphaarchitect.com/is-sector-neutrality-in-factor-investing-a-mistake/ — rank every characteristic **within** `ff49`/`gics` each month (or residualise on industry means), as a second feature block.
5. **Characteristic changes / momentum in attributes.** Equilibrium-style result: returns are driven by *changes* in characteristics; "momentum in firm attributes should be more investigated." [C] https://arxiv.org/pdf/2203.07865 — add 1m/3m/12m differences of the rank-transformed key characteristics (profitability, valuation, `niq_su`, `at_gr1`, `qmj`).
6. **Characteristic × size interactions.** Trees can find these; a ridge/linear composite cannot — put size-tercile-conditional composites into the composite arm.
7. **Macro/VIX-conditioned interactions.** Firm features × macro state is standard GKX; a 2026 GNN paper reports VIX/credit-spread conditioning helps. [C] https://arxiv.org/pdf/2605.19278 — only 68 test months; treat as low priority, high overfit risk.
8. **Amihud/spread-adjusted "tradeable" versions** of features so the model stops leaning on illiquidity (`ami_126d`, `bidaskhl_21d` were high-importance in `XGB`). (Own.)

### 3.4 The 8-K layer — concrete, cheap, mostly *not* about LLMs

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

### 3.5 Model-side: prior-shrinkage, ensembling, foundation models

- **Own idea — composite-as-prior.** The one thing that worked (`LARGECAP` composite) has no fit parameters. Use LightGBM `init_score = composite_score` with ≤ 50 shallow trees, or a ridge whose penalty pulls toward composite group weights, so ML can only *add* what survives validation. This directly addresses the observation that ET on the same universe scored IC 0.016 vs composite 0.037.
- **Pooling and winsorising ML forecasts** beats shrinkage and "recent-performance-weighted" ensembles; equal-weighting is justified by endemic misspecification. [C] https://www.sciencedirect.com/science/article/pii/S0927539824000732 · stacking helps in extreme downside months [C] https://www.sciencedirect.com/science/article/abs/pii/S0927539822000342 . Our `ens5` never beat its best member, but all five members shared features; **diverse-feature** members (composite, peer-gap, 8-K, path-shape) are the ones worth averaging.
- **IC-weighted dynamic model weights** beat metric-weighted ones (CN CSI300; and "factor screening substantially enhanced" combined strategies). [A] https://arxiv.org/abs/2508.18592 — conflicts with our own `FACTOR_FILTER` (IC doesn't persist in the investable universe) and `TPA` (`tpa_ms` failed); treat as a warning, not a lead.
- **Tabular / time-series foundation models:** TSFMs (TimesFM, Moirai, Chronos…) win most tasks but gains over a zero-return baseline are "small and sparse"; only 2 of 10 cases statistically significant. [A] https://arxiv.org/html/2606.27100 . TabPFN-3 / TabICL do in-context learning without fitting — potentially interesting for *few-shot per-month* fitting on ~1,200 large caps [C] https://arxiv.org/pdf/2605.13986 , but no equity-return evidence found. Cheap experiment only if a GPU is free.
- **Sequence/attention architectures** (regime-gated Transformers, Mamba, KAN, hybrid LSTM-XGBoost): claims are on CN or a handful of tickers; I found no credible US large-cap cross-section evidence. See §6.
- **CNN "image" factor timing.** 206 factors' cumulative-return charts → CNN; ~6% annual alpha vs untimed, Sharpe 1.22, break-even cost 1.08%/trade; survives post-publication. [B/C] https://larryswedroe.substack.com/p/timing-the-factor-zoo — timing our own **factor-group sleeves** is the same problem `TPA` `tpa_ms` failed on; only try with strong regularisation.

### 3.6 Cross-stock / network information (uses information that is *not* in a stock's own characteristics)

1. **Peer return gap / peer index.** Peer Return Gap (stock's lagged return − peers' returns) long-short earned 1.26%/mo (t 3.81), FF5 alpha 1.10% (t 2.86) (China, correlation peers). [C] https://www.sciencedirect.com/science/article/pii/S305070062500088X . US: Avramov & Ge, *Dual peer effects* (JFE 2026): a Peer Index predicts returns and earnings surprises, decays *without reversal*, and "machine-learning models based solely on firm-level characteristics do not subsume PI." [C] https://www.repository.cam.ac.uk/items/c9345bde-eace-4e97-b666-094549a2bda0
   - *Own implementation with our panel:* peers = top-*k* by 60-month return correlation **within the same `ff49`**; feature = own `ret_1_0` − peer-average `ret_1_0`; also peer-average `ret_12_1`. All from `ret`.
2. **Text-based industry peers (TNIC).** Free data (Hoberg–Phillips) keyed by `gvkey`; TNIC peer momentum is "substantially more significant than SIC-based peers or own-firm momentum," especially when links are less visible. [C] https://hobergphillips.tuck.dartmouth.edu/tnic_basedata.html — needs a `gvkey`→`permno` join (panel has `gvkey`+`iid`); verify data availability through 2025 and its release lag.
3. **Customer–supplier momentum.** Direct tier-1 links: monthly hedge alpha 0.37–0.63% (CN); spillovers also travel beyond tier-1 and are stronger when investors are inattentive. [C] https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5892284 · lead-lag exists only when customer-firm information is *continuous* (ties to §3.3 ID). Requires supplier data (Compustat segments / FactSet Revere, or LLM-extracted from 10-Ks) — high effort.
4. **Text-embedding networks.** FinBERT 10-K MD&A embeddings for 255 S&P 500 firms 2011-25, propagated along a supply-chain KG: long-short Sharpe 0.86, FF5 alpha 7.27% (t 2.30), survives sector-neutralisation and placebo. [A] https://arxiv.org/abs/2606.29290 · 10-K embedding graph + **LLM edge-filtering** raised S&P 500 mean-reversion Sharpe 0.742→0.820 (2011-19). [A] https://arxiv.org/abs/2604.19476 — only S&P 500 samples; small N; 8-K text is short, so the network would have to come from 10-K/8-K item text or from the customer-supplier tables.
5. **Asset embeddings from 13F holdings.** Gabaix–Koijen–Richmond–Yogo: a 4-dim holdings embedding explains >50% of relative valuations vs 15% for characteristics. [A] https://www.nber.org/system/files/working_papers/w33651/w33651.pdf . Quantitativo's Word2Vec-on-13F → 50 clusters → daily residual reversal within cluster: Sharpe 2.59 (2020-25), β 0.04 — but that is daily-frequency and uses paid Sharadar SF3. [B] https://www.quantitativo.com/p/asset-embeddings — with free EDGAR 13F it works only quarterly with a ≥ 45-day lag; use for **peer definition + crowding**, not a fast signal.
6. **Triangulated / modern stat arb.** Aggregate all pair spreads in an economically coherent group into per-stock "votes"; 2.4 Sharpe gross (1.2–1.9 net) vs 1.6 for GICS baseline. [B] https://www.quantitativo.com/p/triangulated-statistical-arbitrage — daily/intraday; not monthly-panel compatible. *Deep Learning Statistical Arbitrage* (residual portfolios from latent factors + convolutional transformer; Sharpe ~4 gross, 2002-16, ~550 largest stocks) is the academic version but is daily and frictionless. [A] https://arxiv.org/abs/2106.04028 — noted for completeness; our data cadence rules it out unless we add daily returns.
7. **GNNs:** the 2026 GNN papers I found predict correlations/volatility or directional accuracy, not cross-sectional return IC; I would not spend time here.

### 3.7 External data that FIAM allows (needs `permno`/(`gvkey`,`iid`) + month and cleaning code)

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

### 3.8 Hypothesis generation and "AI researcher" loops (optional, the brief's frontier)

- The 2026 crop: **AlphaAgent** (regularised exploration to resist alpha decay; S&P 500 annualised 8.74% vs 2.75% next-best) [A] https://arxiv.org/abs/2502.16789 · **QuantaAlpha** (evolutionary trajectories; factors mined on CSI300 transfer to CSI500/S&P 500 with ~19% cumulative excess over 4 years) [A] https://arxiv.org/abs/2602.07085 · **XALPHA** (report-to-memory absorption, CSI300) [A] https://arxiv.org/abs/2607.08332 · **AlphaPROBE**, AlphaSage, AutoScientist-Quant (all CN). LLM-generated features as tabular inputs: Sharpe +14% to +91%, weakly correlated with baseline features; *retrieval quality is critical*. [A] https://arxiv.org/abs/2602.00196
- **The cautionary result that matters most for a "signal-discovery agent":** Quantpedia had an AI agent replicate 9 published anomalies; **none survived 2023-25 out of sample**, the apparent survivor (Sharpe 1.95) was a construction error, and errors included extracting from abstracts, wrong price floors, and 153 months of near-zero-beta "degenerate" books. Guardrails: verbatim definition cards, dual fidelity reviews (code + trade log), as-traded prices, tradeability screens, cost stress tests, independent re-run. [B] https://quantpedia.com/guardrails-make-the-researcher-what-an-ai-agent-got-right-and-wrong-replicating-nine-equity-anomalies/?a=6080
- Five evaluation failures reverse the sign of LLM-agent results: look-ahead, survivorship, backtest overfitting, cost neglect, regime blindness. [A] https://arxiv.org/abs/2603.27539
- **Live-only LLM evidence** (cannot be backtested over 2021-26 without contamination): daily Russell-1000 agent — top-20 long-only alpha 18.4 bp/day, Sharpe 2.43, evaluated forward from April 2025 [A] https://arxiv.org/abs/2601.11958 ; MarketSenseAI — 19 months on the S&P 500, IC +0.489, strong-buy +2.18%/mo vs +1.15% [A] https://arxiv.org/abs/2604.17327 ; a synthetic 100-persona LLM "crowd" earned ~10% alpha but 92% of its holdings matched a neutral single prompt and the alpha was AI-mega-cap exposure, not selection [B] https://quantpedia.com/do-llm-crowds-produce-investment-signals-an-empirical-test/ .
- **LLM features can be valid and still fail downstream:** LLM-extracted features with IC > 0.15 in held-out data made an RL agent *worse* than a price-only baseline once macro conditions shifted. [A] https://arxiv.org/abs/2604.10996
- *Own recommendation:* if any agent is used, use it to **generate and code hypotheses**, then freeze the hypothesis list and test with pre-registered kill rules and a deflated-Sharpe-style discount. "Agents crowd" (brief §9) is supported by the crowding evidence in §3.9.

### 3.9 Portfolio construction and risk — where a weak signal is lost or saved

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

## 4. A concrete sequenced plan (in the repo's pre-registration style)

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

## 5. Realistic expectations (Own — do not skip)

- A rank-IC of 0.05 on ≥ $2B large caps with ~1,200 names/month is a strong, unusual result. By the fundamental law (IR ≈ IC·√breadth) with ~200–250 effective independent bets/month, IC 0.03 → IR ~0.5–0.7 *before* costs; IC 0.05 → ~0.8–1.2. That is consistent with the repo's composite (IC 0.037, gross IR 0.61) and is the sensible target.
- Only **IC on the tradeable universe** counts. Three of the literature's headline claims (small-cap ML, StockTwits, LLM news embeddings) are exactly the kind that vanished in `PM_ABLATION`.
- 68 test months and an IR s.e. ≈ 0.46: no single-path IR difference under ~0.5 is distinguishable. Use paired-IC t-stats and block-level (feature-family) tests.
- Regimes: 2025 was negative for every arm; July-2025/Jan-2026/July-2026 were crowded-factor unwinds. A strategy that looks like "quality/momentum/low-vol" will be judged against them.

---

## 6. Things that looked exciting but I would skip (and why)

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

## 7. Source index (grouped)

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
