# Should We Filter the 147 Factors? (FACTOR_FILTER)

Implementation: `factor_filter.py` (run with `.venv/bin/python factor_filter.py`, about 4 minutes). New file; nothing existing is edited. Outputs in `output/`: `factor_redundancy_clusters.csv`, `factor_redundancy_pairs.csv`, `factor_ic_persistence.csv`, `factor_filter_selected_features.csv`, `factor_filter_summary.csv`, `factor_filter_monthly_ic.csv`, `factor_filter_paired_tests.csv`, `factor_filter_results.json`.

**The question.** Instead of giving a model all 147 characteristics, would criteria, filters or preprocessing for including a factor help — for example dropping ones that are highly correlated (`docs/FACTORS.md` shows many near-variants of the same idea)?

## Bottom line

1. **Correlation pruning is worth doing for hygiene, not for performance.** 24 factor pairs have |ρ| ≥ 0.9 (9 at ≥ 0.95), and 23 clusters of 2–4 factors have average |ρ| > 0.8 (§1). But in a walk-forward test, pruning at |ρ| > 0.8 gave **no gain** in rank IC or portfolio results (paired t = −0.5; §3). Trees are largely indifferent to redundant inputs.
2. **Using fewer factors helped a little; picking them by past IC helped a little more — neither is statistically distinguishable from noise, and no rule fixed the tradeable portfolio.** In the LP-investable universe, mean rank IC went from 0.023 (all 139) to 0.026 (a *random* 30) to 0.034 (30 chosen by past IC) and 0.037 (the hand-picked 12). The best paired t-statistic against "all 139" is 1.5; against a random 30 it is 1.2 (§3). Every rule's portfolio still has negative gross IR (best: −0.21 for `pm_t10`).
3. **The reason IC-based filters are fragile:** in the investable universe a factor's past IC does not predict its future IC — the rank correlation between train-era and test-era IC across the 139 factors is **0.10**, and only 35% of the 20 "best" factors keep their sign (§2). In *all stocks* the same statistics are 0.89 and 100% — factor IC is stable exactly where the edge is untradeable (small caps; `docs/PM_ABLATION.md`).
4. **Recommendation:** dedupe the near-duplicates (list in §1) and move to a small, economically-motivated, *pre-specified* factor set chosen by group, not by IC. Do not expect filtering to create tradeable alpha; that still needs new information (§5).

## Literature context (from web-search summaries, not full paper reads)

- The standard advice for redundant features is to group related ones and keep a representative, and to judge importance on held-out rather than training data; selecting among correlated inputs is unstable ([survey snippet](https://www.luxalgo.com/library/concept/feature-selection/)).
- There is a literature on correlation-robust factor selection in the factor zoo ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S092753982400032X)); tree ensembles (Random Forest, XGBoost, LightGBM) are reported to do best out of sample on factor-zoo features ([Modern Finance](https://mf-journal.com/article/view/606)).
- Gu, Kelly & Xiu ([NBER w25398](https://www.nber.org/papers/w25398)) report that dominant signals across methods are variations of momentum, liquidity and volatility, and that trees/nets gain from nonlinear interactions. Note this includes momentum, which a financial engineer asked us to treat with suspicion (`docs/RF_3.md` §4).

## 1. FACTORS.md review: what is redundant (unsupervised, no return data)

Computed on the pre-2021 LP-investable universe (price ≥ $5, market cap ≥ $500M, $10M dollar volume, both betas observed), all 147 characteristics, using their cross-sectional ranks (so Pearson = Spearman).

- **Overall the panel is not very redundant:** median |ρ| between two factors is 0.07. 24 pairs are ≥ 0.9 in |ρ|, 60 are ≥ 0.8.
- **Effective dimension:** it takes 9 principal components to explain 50% of the variance, 34 for 80%, 54 for 90% and 72 for 95% — consistent with `docs/PCA_DIAGNOSTIC.md` ("a meaningfully diffuse spectrum"). So roughly 54 components carry 90% of the variance — far fewer than 147, but not a handful either.
- **Average-linkage clusters at |ρ| > 0.8 (23 clusters with 2+ members, covering 58 factors; 147 → 112 clusters):**

| Idea (`docs/FACTORS.md` section) | Cluster | Note |
|---|---|---|
| Volatility (§15) | `ivol_capm_21d`, `ivol_ff3_21d`, `ivol_hxz4_21d`, `rvol_21d` | three idiosyncratic-vol variants (ρ up to 0.98) plus total vol |
| Volatility / lottery (§15) | `iskew_ff3_21d`, `iskew_hxz4_21d`; `rmax1_21d`, `rmax5_21d` | |
| Liquidity (§16) | `turnover_126d`, `zero_trades_126d`, `zero_trades_21d`, `zero_trades_252d`; `dolvol_var_126d`, `turnover_var_126d` | **`turnover_126d` vs `zero_trades_126d` has ρ = −1.000** (`zero_trades` is tie-broken by turnover): an exact duplicate |
| Size / liquidity (§1, §16) | `ami_126d`, `dolvol_126d`, `market_equity` | size, dollar volume and illiquidity are one dimension (ρ down to −0.94); also the same variables the portfolio screens use |
| Momentum / seasonality (§4, §6) | `ret_12_1`, `ret_9_1`, `seas_1_1na`; `ret_60_12`, `seas_2_5na` | |
| Profitability: "lagged-assets" twins (§7) | `gp_at`/`gp_atl1`, `op_at`/`op_atl1`/`cop_at`/`cop_atl1`, `ope_be`/`ope_bel1`; `niq_at`/`niq_be`; `ocf_at`/`qmj_prof`; `at_turnover`/`opex_at`/`sale_bev` | the `…l1` versions differ only in the divisor (last year's assets/equity) |
| Value (§2) | `at_me`, `be_me`, `bev_mev`; `debt_me`/`netdebt_me`; `eqnpo_me`/`eqpo_me` | |
| Investment / accruals (§9, §10) | `lnoa_gr1a`, `ncoa_gr1a`, `nncoa_gr1a`, `noa_gr1a`; `inv_gr1`/`inv_gr1a`; `fnl_gr1a`/`nfna_gr1a`; `rd5_at`/`rd_sale` | |
| Growth / issuance (§3, §8) | `niq_at_chg1`/`niq_be_chg1`; `chcsho_12m`/`eqnpo_12m` | |

The full cluster and pair lists are in `factor_redundancy_clusters.csv` and `factor_redundancy_pairs.csv`. A sensible dedup keeps one factor per row above (and one of `turnover_126d` / `zero_trades_126d`).

## 2. Does a factor's past IC predict its future IC?

Per-factor mean monthly rank IC with next-month return, target months before 2021 ("train era", 71 months) vs 2021-01 – 2026-08 ("test era", 68 months), 139 non-momentum factors:

| | Investable universe | All stocks |
|---|---:|---:|
| Rank correlation across the 139 factors: train-era IC vs test-era IC | **0.10** | **0.89** |
| The 20 factors with the largest train-era \|t\|: share whose test-era IC has the same sign | **35%** | **100%** |
| Those 20 factors: mean test-era IC, signed by their train-era sign | -0.0038 | +0.1171 |
| Factors with \|t\| ≥ 2 in the train era (chance alone: ≈ 7) | 10 | 99 |
| Factors with \|t\| ≥ 2 in the test era | 38 | 106 |
| Mean \|IC\| per factor: train era / test era | 0.0114 / 0.0193 | 0.0308 / 0.0577 |

Reading: in the universe we can trade, the factors that looked best in 2015–2020 were no more likely to work in 2021–2026 than any other (rank correlation ≈ 0.1; the top-20 keep their sign 35% of the time, worse than a coin flip). Many more factors are individually significant in the test era (38 vs 10), but they are largely different ones. In all stocks the picture is the opposite (0.89, 100%): the small-cap/illiquidity/volatility pattern is persistent, and it is precisely the part `docs/PM_ABLATION.md` shows cannot be traded.

## 3. Walk-forward filter experiment

Extra-Trees (`docs/ET.md`; grid `min_samples_leaf` {300, 1000, 3000}, 200 trees, selected on validation rank IC) trained and validated **only on LP-investable rows**, with each rule choosing its factors from rows whose target month is before that fold's validation window. Test window 2021-01 – 2026-08 (68 months), scored on every test stock; the headline metric is rank IC on the LP-investable test universe.

| Rule | Avg. factors | Test rank IC, investable (t) | Test rank IC, all stocks | OOS R² | `pm_free` IR gross / net | `pm_t10` IR gross / net |
|---|---:|---:|---:|---:|---:|---:|
| `all139` (baseline: every non-momentum factor) | 139 | 0.0233 (1.5) | 0.1222 | -0.021% | -0.78 / -0.97 | -0.63 / -0.72 |
| `decorr` (cluster-prune at |ρ| > 0.8) | 105 | 0.0219 (1.5) | 0.1201 | -0.021% | -0.55 / -0.73 | -0.68 / -0.78 |
| `ic_top30` (30 largest |train IC|) | 30 | 0.0339 (2.3) | 0.1338 | +0.008% | -0.16 / -0.40 | -0.26 / -0.38 |
| `decorr_ic_top30` (prune, then top 30 by |train IC|) | 30 | 0.0319 (2.0) | 0.1303 | -0.013% | -0.51 / -0.72 | -0.47 / -0.58 |
| `sign_stable` (|t| ≥ 2 and same sign in both halves) | 13 | 0.0301 (2.2) | 0.1056 | -0.038% | -0.31 / -0.55 | -0.57 / -0.69 |
| `curated12` (RF_3's hand-picked Modern, no momentum) | 12 | 0.0366 (2.2) | 0.1265 | -0.014% | -0.11 / -0.31 | -0.21 / -0.29 |
| `random30` ×5 seeds (null; mean ± sd) | 30 | 0.0256 ± 0.0015 | 0.1153 | -0.026% | -0.48 / -0.69 | -0.53 ± 0.11 / -0.63 |

Paired differences in the monthly investable-universe IC series (the same 68 test months for every arm, so differences are paired):

| Arm | vs. baseline | Mean monthly IC difference | Paired t | Months arm is better |
|---|---|---:|---:|---:|
| `all139` | `random30_mean` | -0.0023 | -0.76 | 47% |
| `decorr` | `all139` | -0.0014 | -0.53 | 53% |
| `decorr` | `random30_mean` | -0.0037 | -1.09 | 46% |
| `ic_top30` | `all139` | +0.0106 | +1.47 | 54% |
| `ic_top30` | `random30_mean` | +0.0083 | +1.18 | 62% |
| `decorr_ic_top30` | `all139` | +0.0085 | +1.43 | 53% |
| `decorr_ic_top30` | `random30_mean` | +0.0063 | +1.04 | 54% |
| `sign_stable` | `all139` | +0.0068 | +0.82 | 53% |
| `sign_stable` | `random30_mean` | +0.0045 | +0.56 | 53% |
| `curated12` | `all139` | +0.0133 | +1.28 | 53% |
| `curated12` | `random30_mean` | +0.0110 | +1.12 | 51% |
| `random30_mean` | `all139` | +0.0023 | +0.76 | 53% |

What it says:
- **Pruning correlated factors does nothing** (IC 0.022 vs 0.023; paired t = −0.5). 139 → about 105 factors removed exact/near duplicates without changing what the trees can learn.
- **Fewer factors helps slightly.** A random 30 beats all 139 by +0.002 IC (paired t = 0.8) — consistent with noise features costing a tree ensemble a little.
- **Choosing by past IC helps a bit more, weakly.** `ic_top30` is +0.011 over all 139 (t = 1.5, better in 54% of months) and +0.008 over a random 30 (t = 1.2). The `decorr_ic_top30` (t = 1.4) and `sign_stable` (t = 0.8) variants are similar. **None of these clears t = 2.** With 68 months and one seed, we cannot tell them from noise.
- **The hand-picked 12 (`curated12`) has the best IC (0.037)**, but it was chosen with knowledge of earlier results (a 2026 paper's "Modern" set, then `docs/RF_2.md`'s leave-one-out), so it carries human look-ahead; the walk-forward rules do not.
- **No rule makes the tradeable portfolio positive.** Best `pm_t10` gross IR is −0.21 (`curated12`); the random-30 null is -0.53 ± 0.11 and all-139 is -0.63. Filtering moves the tradeable book from clearly negative to mildly negative — the same "≈ 0 or below" as every other model in this batch.
- **Selection is unstable.** Across the six annual refits `ic_top30` keeps on average 58% of its factors from one fold to the next (Jaccard), `sign_stable` only 39%. The nine factors selected in all six folds are `bidaskhl_21d`, `ivol_capm_252d`, `ivol_ff3_21d`, `niq_at`, `niq_be`, `rmax5_21d`, `turnover_126d`, `zero_trades_126d`, `zero_trades_252d` — seven of the nine are volatility/liquidity-family factors (five of them members of the near-duplicate clusters of §1: `ivol_ff3_21d`, `rmax5_21d`, `turnover_126d`, `zero_trades_126d`, `zero_trades_252d`), and the other two, `niq_at` and `niq_be`, are themselves a duplicate pair. So plain IC screening mostly re-selects the same redundant family, and that family (volatility/illiquidity) is where the untradeable small-cap edge lives.

## 4. Is filtering a good idea? Summary judgment

| Kind of filter | Verdict here |
|---|---|
| Drop near-duplicates (|ρ| > 0.9, or `…l1` twins, exact duplicates) | Yes for interpretability, speed and cleaner importance tables; no measurable effect on accuracy. |
| Prune to one-per-cluster at |ρ| > 0.8 | Harmless, no gain (t = −0.5). |
| Fewer factors, chosen *before* looking at returns, by economic group | Reasonable; the RF_3 set is an example. Best defensibility. Gain over all-139 is small and not significant. |
| Keep factors with the best past IC (walk-forward) | Weak evidence of a small gain (t ≈ 1.5); unstable selections; and IC does not persist in the investable universe (§2). Only useful if combined with a persistence requirement, and even then expect little. |
| Filter to raise tradeable alpha | Did not work: all rules ≤ 0 on the tradeable portfolio. |

## 5. What to do

1. Use a modest, documented dedup (one factor per cluster in §1) and a factor set chosen by group up front (e.g. 2–3 from each of value, profitability, investment, quality/surprise, volatility/lottery, liquidity), so the deck can say why each factor is in.
2. If an IC screen is used at all, keep it walk-forward, require the same sign in both halves of the training window, and report the random-subset null next to it (as here).
3. Do not expect this to change the tradeable result. New information that is not just small-cap microstructure (e.g. the 8-K layer) remains the higher-value work (`docs/PM_ABLATION.md` §7).
4. Keep the feature block modular so new features can merge on the `permno`–month key without rewriting downstream code.

## 6. Limitations

- One model family (Extra-Trees), one seed; five random subsets for the null (sd of IC across them is only 0.0015, but that understates the uncertainty of any single IC mean, which has t ≈ 1.5–2.3).
- Training windows for the IC filters are short (47 months for the 2021 fold, up to 107 for 2026), so per-factor ICs are noisy — the practical reason such filters are weak, not an implementation bug.
- Models train on LP-investable rows only (the universe that matters); results for filtering on the full universe, where IC is persistent, were not tested.
- The cluster cutoff (|ρ| > 0.8) and K = 30 were set before running and not tuned. `curated12` is not a walk-forward rule.
- Stocks without 5-year beta history are excluded from the investable universe, as in every portfolio in this project.
