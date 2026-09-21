# Shared-core overlap test, Shapley decomposition and the three definitions of IC (COMPOSITE_OVERLAP)

Implementation: `composite_overlap.py` (run: `.venv/bin/python experiments/composite_overlap/composite_overlap.py`, about 40 seconds).

## Objective
Measure how much of the composite - the project's only positive result - is the commoditised risk premium ("shared core"): its exact group-level Shapley decomposition, its IC under three definitions (raw, sector-neutral, risk-model-residual), whether it is more than a plain value + quality screen, and what happens to the book when each group is dropped (including the liquidity group, docs/NEW.md item 8).

## Hypothesis
The composite - an equal-weight blend of textbook value / profitability / investment / quality / surprise / volatility / liquidity groups - IS the shared core: (i) most of its variance is explained by a public style set + sectors; (ii) its IC largely disappears once orthogonalised to that set; (iii) it is about as good as a be_me + qmj screen; (iv) its IC is concentrated in one or two groups, and the liquidity group contributes nothing. Pre-registered reading: if the risk-model-residual IC t-stat is < 1 the composite has no alpha beyond the shared core; a group with Shapley value <= 0 should be dropped.

## Research origin
docs/REDDIT_RESEARCH.md sec 2.5 (r/quant 1v02nb2 / 1v30qx3 / 1vvt4m9: "how much of this is just proxying the platform's existing shared core?"; orthogonalise to the risk model, treat the residual as the alpha), sec 2.13 (Shapley decomposition over the alpha set, r/quant 1w99f1v; report raw, sector-neutral and risk-model-residual IC, 1rmy6w3; Toraniko-style Barra-lite exposures, 1ekpin6) and docs/NEW.md sec 3.3 item 8 (tradeable / liquidity-adjusted features).

## Implementation
`shapley_ic`: exact Shapley values over the 7 groups with v(S) = mean monthly universe rank IC of the equal-weight sub-composite of S (128 subsets). `month_ols_resid`: month-by-month cross-sectional OLS of the composite score on {log market cap, `ret_12_1`, `betabab_1260d`, `ivol_capm_21d`, `turnover_126d`, `be_me`} + GICS-sector dummies (universe rows; a public, Barra-flavoured style set), giving the mean R^2 and the residual whose IC is the risk-model-residual IC; the sector-neutral IC uses sector dummies only. Composite vs the `be_me + qmj` percentile-rank screen: rank correlation, top-quintile overlap, paired IC. Leave-one-group-out `lc_t10` portfolios.

## Experimental setup
Common harness (a verbatim copy of the data / rank-transform / LP / cost / performance code of `experiments/largecap/largecap.py`, embedded so the script runs on its own; the LP functions `_solve_lp` / `build_portfolio` are redefined in the script with optional extra behaviours and reproduce the frozen LP when the options are off).

- **Signal**: the frozen composite (18 factors in 7 economic groups, equal weight, re-ranked in the universe each month; no fitting). It reproduces `experiments/largecap`: universe rank IC 0.0366 (t 1.91), `lc_t10` gross IR 0.615 / net IR 0.541, 2025 net return -12.4%.
- **Universe**: price >= $5, market cap >= $2,000M, 126-day dollar volume >= $10M, both betas observed (about 1,206 stocks per month); 68 test months 2021-01..2026-08.
- **Portfolio**: the frozen LP (dollar-neutral, neutral to `beta_60m` and `betabab_1260d`, net sector <= 5% of NAV, gross sector share <= 35%, gross 200%, per-name cap 1%, 10% one-way turnover cap = `lc_t10`), with tiered *assumed* trading and borrow costs (5/10/20 bp trading and 30/75/200 bp borrow by market-cap tier; not measured).
- **Caveat that applies to every number below**: a single 68-month path; the standard error of a single-path IR is about 0.46, so IR differences under ~0.5 are not distinguishable from noise on their own. Paired monthly net-return t-stats are reported next to IR; with 20+ variants per file, a few |t| > 1 are expected by chance.

## Baseline
The composite itself (IC 0.0366, t 1.91; `lc_t10` gross IR 0.615 / net 0.541) and, for the screen question, the be_me + qmj screen.

## Commands
```
.venv/bin/python experiments/composite_overlap/composite_overlap.py
```

## Results (68 months)
Group contributions (Shapley values sum exactly to the composite's IC of 0.0366; the two right-hand columns are the leave-one-group-out LP books, single path, IR s.e. ~0.46):

| group | Shapley IC | share of composite IC % | IC without this group | net IR without this group | 2025 net without |
|---|---:|---:|---:|---:|---:|
| value | 0.0197 | 54 | 0.0276 | 0.42 | -0.129 |
| profitability | 0.0041 | 11 | 0.0377 | 0.56 | -0.129 |
| investment_issuance_accruals | 0.0070 | 19 | 0.0342 | 0.52 | -0.126 |
| quality | 0.0028 | 8 | 0.0388 | 0.57 | -0.117 |
| surprise | 0.0001 | 0 | 0.0360 | 0.58 | -0.113 |
| volatility_beta | 0.0026 | 7 | 0.0380 | 0.68 | -0.143 |
| liquidity | 0.0004 | 1 | 0.0365 | 0.51 | -0.166 |

Three definitions of IC (the composite's mean monthly cross-sectional R^2 on the style set + sectors is **0.54**):

| definition | IC | t |
|---|---:|---:|
| raw rank IC | 0.0366 | 1.91 |
| sector-neutral (composite demeaned within GICS sector) | 0.0331 | 1.96 |
| risk-model residual (size, momentum, beta, ivol, liquidity, book-to-market + sectors) | 0.0252 | 2.67 |

IC by size tercile:

| size tercile | raw IC | risk-model-residual IC |
|---|---:|---:|
| small | 0.0512 | 0.0282 |
| mid | 0.0359 | 0.0310 |
| large | 0.0145 | 0.0167 |

Composite vs a be_me + qmj screen: screen IC **0.0311** (t 2.04) vs composite 0.0366; paired composite minus screen +0.0055 (t 0.42); mean monthly rank correlation 0.54; mean top-quintile overlap 0.41 (random = 0.20).

## Interpretation
(i) **About 54% of the composite's cross-sectional variance is explained by public styles + sectors** (mean R^2 0.54) - the composite is substantially the shared core, as the Reddit thread suspects. (ii) **But its IC survives orthogonalisation**: the risk-model-residual IC is 0.0252 with a *higher* t-stat (2.67) than the raw IC (1.91): removing the style-timing noise leaves a cleaner, if smaller, signal. So the pre-registered "no alpha beyond the core" reading is **not** supported; the residual is about 69% of the raw IC. (iii) **It is statistically indistinguishable from a simple be_me + qmj screen** (screen IC 0.0311, t 2.04; paired difference +0.0055, t 0.42; top-quintile overlap 41% vs 20% random): the 18-factor, 7-group construction adds little over two factors. (iv) **The IC is concentrated in value**: Shapley value of `value` is 0.0197 (54% of the IC), investment/issuance/accruals 0.0070, profitability 0.0041, quality 0.0028, volatility/beta 0.0026, liquidity 0.0004 and surprise 0.0001 - the last two contribute nothing (Shapley ~0) and dropping surprise or liquidity leaves IC unchanged (0.0360 / 0.0365); dropping `value` cuts IC to 0.0276 and net IR to 0.42. Dropping the volatility/beta group *raises* net IR to 0.68 (IC 0.0380), a single-path result within noise but consistent with the group's Shapley value being close to zero. The composite's IC is concentrated in the small and mid terciles (0.051 / 0.036 raw) and is weak in the large tercile (0.0145 raw, 0.0167 residual), which is where the tradeable book actually sits. Status: IMPLEMENTED_AND_TESTED (diagnostic; no adopt rule). Practical consequence: judge every new candidate by its residual IC after the composite (done in every `feat_*` experiment).

## Limitations
- The public style set is a crude proxy for the *actual* crowded core (which is defined by managers' positions); the high R^2 also partly reflects that the composite's own groups are styles.
- IC by size tercile is computed within the universe; the composite's weaker large-cap IC is not tested for significance.
- Leave-one-group-out IR differences are within the single-path noise (s.e. ~0.46).
- `value` is the largest contributor but `be_me` is in both the composite and the residualisation set, which lowers the residual IC.

## Follow-up
A composite restricted to value + investment (Shapley-weighted) is an obvious simplification but choosing it from these results would be a post-hoc selection, so it is not run.
