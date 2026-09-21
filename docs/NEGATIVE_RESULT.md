# An Honest Negative Result: Tree Models on the 147 Characteristics

Draft text for the deck's Discussion page / Appendix. Every number below comes from `experiments/rf_3/README.md`, `experiments/et/README.md`, `experiments/lgbm/README.md`, `experiments/cat/README.md`, `experiments/xgb_2/README.md` and `experiments/pm_ablation/README.md` (period 01/2021 – 08/2026, 68 monthly returns, one prediction path per model). The FIAM brief says a candid account of something that did not work is worth more than a polished account of something that supposedly did; this is that account.

---

## 1. Summary

We trained five tree-based models (Random Forest, Extra-Trees, LightGBM, CatBoost, XGBoost) on the provided characteristics to predict next-month excess return, and built market-neutral long/short portfolios from them. Under the loose portfolio used in our earlier work (a $10M dollar-volume screen, dollar- and beta-neutral, 200% gross, 1% position cap) every model looked good: information ratio (IR) 0.9–1.3 against the T-bill + 4% benchmark.

**That performance was not tradeable.** When we required the book to hold only stocks priced at least $5 with market cap of at least $500M, to be neutral to two betas, to respect sector limits and to trade at most 10% of the book per month, every model fell to an IR of about zero or below, and below zero after assumed trading and borrow costs.

We therefore do **not** present the loose-portfolio numbers as a result. We present them as a diagnostic that showed where the models' apparent skill came from.

## 2. What we did

- **Models.** Random Forest, Extra-Trees, LightGBM (forest-style, plus its random-forest mode), CatBoost, and XGBoost re-tuned after our first attempt failed. All fit on a 12-factor momentum-free set (a 15-factor set with momentum as reference); XGBoost also on 139 factors.
- **Discipline.** Hyperparameters were chosen per fold on validation rank IC, never on test results. Splits follow the FIAM schedule (expanding training window, 2-year rolling validation, refit annually, assigned by target month). The label is next-month excess return (`ret_exc_lead1m`), winsorized at the training fold's 1st/99th percentiles for fitting only.
- **Two portfolios per model.** The *loose* portfolio, and a *tradeable* one built from the constraints in the tips from a previous podium participant and from our financial engineer: price ≥ $5, market cap ≥ $500M, neutral to both `beta_60m` and `betabab_1260d`, net sector exposure ≤ 5% of NAV, and a hard turnover budget (10% headline, 20% and none as sensitivity). Both are reported gross and net of assumed costs.
- **Pre-specified.** The constraint set and the 10% cap were fixed from the tips before any result was seen. Later diagnostics (training on tradeable stocks only, a top/bottom-decile baseline, a market-cap-floor sweep) were added after the first negative result, as follow-ups, and are labelled as such.

## 3. What we found

### 3.1 The headline numbers (Random Forest, 12 factors; other models are in the same range)

| Portfolio | Gross IR | Net IR | Short-leg return (gross CAGR) |
|---|---:|---:|---:|
| Loose (old) portfolio | 1.26 | 1.09 | +18.2% |
| Tradeable, no turnover cap (`pm_free`) | 0.07 | −0.13 | −13.8% |
| Tradeable, 20% turnover cap | −0.04 | −0.17 | −13.8% |
| **Tradeable, 10% turnover cap (`pm_t10`, headline)** | **−0.21** | **−0.31** | **−15.0%** |

Across the five models, tradeable-portfolio gross IR (10% cap) ranges from −0.12 to −0.31; the loose-portfolio range is 0.90 to 1.26. A negative short-leg return means the names we shorted rose.

### 3.2 Where the apparent skill came from

1. **A price screen alone removes most of it.** Adding only "price ≥ $5" to the loose portfolio takes the Random Forest short leg from +18.2% to −3.8% per year, while the long leg barely moves (13.4% → 13.0%). Market-cap floors of $250M–$1B push the short leg to −10% to −14%. The edge was in shorting small, low-priced stocks.
2. **Rank skill is concentrated in small stocks.** Mean monthly rank IC between prediction and next-month return is 0.14–0.15 across all stocks, about 0.20 for stocks under $250M market cap, and only 0.034–0.037 on the universe the tradeable portfolio can hold.
3. **It was a sector bet.** The loose Random Forest book averaged 42% of NAV net short Health Care and reached 58% in one month. The largest shorts by average weight were IOVA (Iovance Biotherapeutics), NTLA (Intellia Therapeutics), LAZR (Luminar Technologies), AMC (AMC Entertainment), NKLA (Nikola), ALT (Altimmune), MULN (Mullen Automotive), PLUG (Plug Power), SPCE (Virgin Galactic) and FCEL (FuelCell Energy) — by our own characterization biotechs, EV/hydrogen/space names and meme stocks that are likely hard or expensive to borrow. We did not check borrow availability (no such data in the panel). Adding sector limits alone cut IR by 0.08–0.41.
4. **It is not the portfolio optimizer.** A plain top-decile-long / bottom-decile-short equal-weight book on tradeable stocks (the FIAM §6 baseline) has gross IR −0.04 to −0.21 across the models, and the return spread between the top and bottom prediction decile on that universe is between −0.09 and +0.24 percentage points per month — smaller than the monthly hurdle.

### 3.3 What did *not* explain it

- **Model family.** Random Forest, Extra-Trees, LightGBM, CatBoost and XGBoost all give about zero on the tradeable portfolio. An equal-weight ensemble of the five never beat its best member.
- **Momentum.** Removing the three momentum factors lowered loose-portfolio IR by 0.07–0.27 in all five models, so momentum was not what hurt. (`mispricing_perf` embeds a momentum component and stays in every arm.) We keep momentum out for defensibility, not because it cost performance.
- **The training universe.** Training and validating only on tradeable-universe rows did not create tradeable alpha (loose IR stayed 0.7–1.4; tradeable IR stayed −0.4 to +0.25).
- **Number of factors.** A 139-factor XGBoost gave loose IR 1.25 but tradeable IR −0.54; more factors fit more of the small-cap structure.

### 3.4 Positive by-products

- **Our first XGBoost result (IR 0.12) was a tuning problem, not a model problem.** Selecting tree count on validation rank IC instead of MSE, with forest-style regularization, moved loose-portfolio IR to 0.90–1.25 depending on feature set. (Several changes were bundled; no single fix is credited.)
- **Neutrality mechanics work.** Formation beta is exactly zero on both betas every month by construction. Realized beta over the 68 months is within ±0.13 (|t| < 0.8) for every full-universe tradeable book; a few books trained on tradeable stocks only reach |t| ≈ 2 (β −0.25 to −0.28). Rolling 12-month beta exceeds 1 in exactly one window — the first, ending December 2021, which contains the January 2021 short squeeze — and stays roughly within ±0.6 afterwards. Beta-neutrality itself cost nothing (adding the second beta constraint raised gross IR for four of five models and left XGB_2 essentially unchanged, 0.90 → 0.88).
- **A turnover cap is cheap where a book has an edge.** On the loose book a 10% cap lowered gross IR by 0.05–0.26. The loose book trades about 1.7× capital per month (43% one-way of gross); tradeable books hold to 10% by construction.

## 4. Why this matters for the evaluation

The organizers say neutrality and short-book tradeability are checked before performance, and that a short leg concentrated in small, hard-to-borrow names "will be read with this in mind." Our loose-portfolio result would fail exactly that read. Reporting it as performance would be presenting a number we know does not survive the criteria we were told we would be judged on.

## 5. What we would do next

The limiting factor is the information in the signal, not the model or the constraints. More model variants on these 147 characteristics are unlikely to change that.

1. **Add information that lives in mid and large caps** — for example an event layer from the 8-K filings (a teammate's workstream) or the alternative data available to finalists — and re-run through the same tradeable-portfolio harness, with the same reporting.
2. **Model borrow cost and availability** for small-cap shorts, so any small-cap edge could be sized honestly instead of flattered by our flat assumed costs.
3. **Test the least-bad tradeable result properly.** A $2B market-cap floor gave Random Forest gross IR 0.67 / net 0.51 with the most stable rolling beta of any book (max 0.38). It is one row, chosen after the fact, and unvalidated; it needs a pre-registered re-test and a smoother weighting than the LP's extreme picks.

## 6. Limitations of this result

- One prediction path per model (single seed); no confidence intervals. Earlier work put 2-sd seed noise for Random Forest at about ±0.11 IR.
- Trading costs (5 / 10 / 20 bp one-way) and borrow costs (30 / 75 / 200 bp per year) are assumed tiers by market cap, not measured. Real borrow on the legacy shorts would be far higher, so the net loose-portfolio numbers are flattering.
- Sector labels are the panel's supplied GICS codes; their point-in-time accuracy is unverified. Stocks with no 5-year beta history (about 11% of liquid names) are excluded from every portfolio.
- 68 months, of which the first two contain the January 2021 squeeze; a single out-of-sample path.
- The `_trad` runs, decile baseline, market-cap sweep and sector diagnostics were added after the first negative result.
