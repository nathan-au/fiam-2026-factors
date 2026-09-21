# REDDIT.md — What I would have searched on Reddit (if it were reachable)

Reddit was blocked in `WebSearch`, `WebFetch` and Claude-in-Chrome (see `NEW.md` §0). This is the search plan to run by hand (or via any tool that can reach it). Context: monthly US equity long/short, 147 characteristics, ML, **large-cap tradeable universe**, market-neutral, tradeable IC ≈ 0.035, best result is a no-fit composite.

Tip for every search: sort by **Top → Past year** and again by **Top → All time**; then read the top comments, not just posts. Weight replies from people who mention costs, universe, or out-of-sample periods. Discount anything with "my bot made X%" and no universe/cost details.

---

## 1. Subreddits, in priority order

| Priority | Subreddit | Why |
|---|---|---|
| 1 | r/quant | Practitioners; best signal on what works in production vs. academic backtests |
| 2 | r/algotrading | Huge, noisy; useful for implementation traps, data sources, and "why my backtest died" threads |
| 3 | r/quantfinance | Career-skewed, but has methodology Q&A and paper discussion |
| 4 | r/MachineLearning | Cross-sectional/tabular ML, target design, financial time-series validation debates |
| 5 | r/datascience | Leakage, walk-forward CV, "I tried ML on stocks" post-mortems |
| 6 | r/Numerai (and r/numerai_signals if present) | Directly analogous problem: cross-sectional ranking, neutralisation, era CV |
| 7 | r/SecurityAnalysis | Fundamentals/event signals (8-K, insider, restatements) from human analysts |
| 8 | r/thetagang, r/options | Options-implied signals, IV spread/skew as return predictors (organisers like options) |
| 9 | r/wallstreetbets | Not for alpha — for **crowding / squeeze / retail-flow** behaviour (our Jan-2021 loss, meme shorts) |
| 10 | r/StockMarket, r/investing | Only for regime narrative (July 2026 momentum unwind, AI concentration) |
| 11 | r/FinancialCareers, r/quant_hiring | Skip except for "what do quant interviews say works" folklore |
| 12 | r/LocalLLaMA, r/singularity | Only for LLM look-ahead / memorisation and agent-backtest discussions |

---

## 2. Search list (exact queries)

Format: `subreddit | query`. Use Reddit's own search with `restrict to subreddit` on.

### A. Target and loss design (highest priority — matches NEW.md §3.1)
- r/quant | `rank target OR "rank transform" OR "gauss rank" cross-sectional`
- r/quant | `"predict rank" vs "predict return" long short`
- r/quant | `demean target sector size "neutralized target"`
- r/quant | `residual return target beta neutral ML`
- r/quant | `lambdarank OR "learning to rank" stocks`
- r/quant | `classification vs regression stock returns decile`
- r/MachineLearning | `cross-sectional stock ranking loss function IC`
- r/MachineLearning | `financial ML target leakage overlapping returns`
- r/Numerai | `feature neutralization proportion`
- r/Numerai | `multi target ensemble 20d 60d`
- r/Numerai | `era boosting OR "era-wise" overfit`
- r/Numerai | `Signals alpha metric neutralized target`

### B. "Why doesn't ML work on the tradeable universe" (our core problem)
- r/quant | `ML alpha only in small caps`
- r/quant | `"micro cap" backtest illusion shorting borrow`
- r/quant | `IC 0.03 realistic large cap`
- r/quant | `rank IC decays after costs`
- r/quant | `factor zoo large cap only what survives`
- r/algotrading | `backtest great until I add liquidity filter`
- r/algotrading | `short leg untradeable borrow cost backtest`
- r/quant | `capacity turnover cost ML equity`
- r/quant | `how much of a published anomaly survives transaction costs`

### C. Feature engineering from data we already have (NEW.md §3.3)
- r/quant | `information discreteness OR "frog in the pan"`
- r/quant | `residual momentum OR "industry neutral momentum"`
- r/quant | `52 week high momentum neutral`
- r/quant | `smoothness momentum path shape`
- r/quant | `industry relative ranking within sector features`
- r/quant | `characteristic changes OR "delta features" cross-section`
- r/algotrading | `feature importance stock ML what features matter`
- r/quant | `short term reversal residual OR "no news" OR "news-driven"`
- r/quant | `idiosyncratic volatility large cap still works`

### D. 8-K / filings / events (NEW.md §3.4)
- r/SecurityAnalysis | `8-K item 4.02 OR "non-reliance" OR restatement signal`
- r/SecurityAnalysis | `auditor resignation 8-K`
- r/SecurityAnalysis | `CEO departure abrupt 8-K returns`
- r/algotrading | `SEC 8-K NLP trading signal`
- r/algotrading | `EDGAR filings sentiment FinBERT backtest`
- r/quant | `10-K text change similarity "lazy prices"`
- r/quant | `earnings announcement premium still works`
- r/quant | `PEAD large caps dead`
- r/quant | `8-K filing frequency information intensity`
- r/datascience | `SEC filings embeddings return prediction`
- r/MachineLearning | `LLM look-ahead bias backtest memorization`
- r/LocalLLaMA | `LLM knows the future stock prices backtest contamination`

### E. Cross-stock / peer / network signals (NEW.md §3.6)
- r/quant | `peer momentum OR "peer returns" lead lag`
- r/quant | `customer supplier momentum`
- r/quant | `correlation clusters stat arb monthly`
- r/quant | `13F embeddings word2vec crowding`
- r/quant | `TNIC OR "text based industry" peers`
- r/quant | `graph neural network stocks actually works`

### F. External data worth adding (NEW.md §3.7)
- r/quant | `option implied signals cross-section stock returns IV spread skew`
- r/options | `IV spread call put predicts stock returns`
- r/quant | `analyst revisions signal 2025 OR 2026`
- r/quant | `insider buying cluster signal large cap`
- r/algotrading | `Form 4 insider data backtest`
- r/quant | `short interest signal squeeze filter`
- r/algotrading | `FINRA short volume ratio backtest`
- r/quant | `daily returns distribution predict next month`
- r/quant | `overnight vs intraday returns anomaly`
- r/algotrading | `free data sources CRSP alternative daily prices survivorship`

### G. Portfolio construction / risk (NEW.md §3.9)
- r/quant | `factor neutral optimizer alpha residual risk model Barra`
- r/quant | `beyond beta neutral size momentum neutral long short`
- r/quant | `turnover constraint signal smoothing EMA alpha`
- r/quant | `"alpha decay" blend horizons turnover`
- r/quant | `long short equity crowded factor unwind July 2025`
- r/quant | `momentum crash 2026 quant drawdown`
- r/quant | `dispersion scaling gross exposure`
- r/quant | `vol targeting long short factor still works out of sample`
- r/quant | `confidence gate abstain regime ML ranker`
- r/quant | `stress test crowded shorts squeeze January 2021`
- r/wallstreetbets | `short squeeze list high short interest` (for crowding mechanics only)

### H. Validation / discipline
- r/quant | `walk forward purge embargo monthly returns overlapping`
- r/quant | `deflated sharpe multiple testing how many trials`
- r/algotrading | `backtest overfitting how do you know`
- r/datascience | `time series cross validation leakage stock`
- r/quant | `point in time universe survivorship S&P 500 membership`
- r/quant | `IR standard error 5 years monthly not significant`

### I. Model families (to confirm the "skip" list)
- r/quant | `XGBoost vs neural net cross-sectional equity`
- r/MachineLearning | `tabular foundation model TabPFN finance`
- r/quant | `Transformer stock prediction reality`
- r/quant | `KAN OR Mamba OR "state space" stock returns`
- r/quant | `reinforcement learning portfolio real results`
- r/quant | `LLM alpha mining AlphaAgent OR "factor mining" agent overfit`

### J. Trend scanning (what people are excited about right now)
- r/quant | `2026 what is working in quant equity` (sort: new, past month)
- r/quant | `new paper cross-section` (past month)
- r/algotrading | `2026 strategy ideas` (past month)
- r/MachineLearning | `[D] financial ML` (past year)

---

## 3. What to extract from each thread

1. **Universe** they trade (Russell 1000? all-cap?) and whether costs/borrow are included.
2. **Target and loss** used (raw, rank, residual, classification).
3. **Time period** of out-of-sample and whether it includes 2022 and 2025.
4. **Turnover and holding period.**
5. **Failure stories** ("worked until I…") — these are the highest-value comments.
6. **Data source** and whether it is free/point-in-time.
7. Any **paper title** → add to NEW.md source index and grade it.

## 4. Filters for trustworthiness

- Keep: comments citing a paper, a costs assumption, a live/paper-traded period, or a concrete failure.
- Discount: single-backtest Sharpe > 2 with no universe; "guaranteed" language; AI-generated posts; crypto-only results; sub-$5 or micro-cap universes.
- Cross-check anything promising against `experiments/pm_ablation/README.md`: if the idea's edge would live in small/illiquid shorts, it fails our tradeability screen.
