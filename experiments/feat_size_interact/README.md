# Size-conditional composites (FEAT_SIZE_INTERACT)

Implementation: `feat_size_interact.py` (run: `.venv/bin/python experiments/feat_size_interact/feat_size_interact.py`). Everything it writes goes to `output/`.

## Objective
Test whether ranking every composite factor within size terciles (a size-neutral composite) or weighting the seven factor groups differently by size tercile (weights fitted only on earlier months) improves the composite.

## Hypothesis
The composite's value/investment tilt works differently across size terciles; letting it differ by tercile (interaction a linear composite cannot express but trees can) raises universe IC.

## Research origin
docs/RESEARCH.md Part I sec 3.3 item 6 (characteristic x size interactions: "put size-tercile-conditional composites into the composite arm") and sec 3.2.

## Implementation
`size_tercile_rows` gives the within-month size tercile of the universe. `comp_within_size_tercile`: `grp_scores(key=tercile)`. `comp_tercile_ic_weighted`: for each test year, tercile-specific weights = max(mean monthly rank IC of each group score within the tercile over all universe months before the test year, 0), normalised (the only fitted quantity; expanding window from 2015, so only 72-132 months). Both are replacement composites compared with the frozen composite.

## Experimental setup
Common harness (a verbatim copy of the data / rank-transform / walk-forward / LP / cost / performance code of `experiments/largecap/largecap.py`, embedded so the script runs on its own):

- **Data**: `fiam/chars_final_with_names.parquet` (523,125 stock-months with a next-month return; 147 characteristics; `ret`, `me`, `ff49`, `gvkey`, `ticker`, ... for feature building).
- **Universe** (IC scoring and LP): price >= $5, market cap >= $2,000M, 126-day dollar volume >= $10M, both betas observed; about 1,206 stocks per month; 68 test months 2021-01..2026-08 (82,026 universe test rows).
- **No model is fitted.** The baseline is the frozen composite: equal-weight of 7 factor groups over 18 pre-specified factors, re-ranked within the universe each month. Each candidate block is added as an **8th equal-weight group** with a sign fixed before any result was seen (`sign` column below; the sign is applied to the within-month universe percentile rank of the block, NaN -> neutral 0).
- **Tests per block**: (1) IC of the signed block alone; (2) paired monthly IC gain of composite+block vs the composite (mean difference, t); (3) residual IC after orthogonalising the block to the composite each month (the "shared-core" check, docs/RESEARCH.md Part II sec 2.5); (4) the gain within each size tercile; (5) a within-(month, sector) permutation null, 200 shuffles, for the additive gain (docs/RESEARCH.md Part II H4). **Note**: a random block *dilutes* the composite, so the null gain is negative on average; the permutation p therefore tests "does this block carry information", while the paired t tests "does adding it improve the composite". (6) LP `lc_t10` gross / net IR (**single-path IR s.e. is about 0.46**, and LP corner solutions make IR jump by +-0.3 for changes that leave IC unchanged; IR is reported but never used for decisions). (7) A **truncation-invariance test**: the feature builder is re-run on rows with eom <= t only for random test dates t and must give identical values at t for every stock.
- **Pre-registered rule** (docs/RESEARCH.md Part I sec 2 / 4): KILL if the paired IC gain has t < 1 or the gain is confined to the smallest size tercile. PASS requires paired t >= 2, Bonferroni-adjusted permutation p <= 0.05 (factor = number of blocks in the file) and not small-tercile-only.
- **Baseline sanity**: the composite reproduces `experiments/largecap` (IC 0.0366, t 1.91; `lc_t10` gross IR 0.615 / net 0.541).


## Baseline
The frozen 7-group composite of `experiments/largecap` on the same universe and months (composite IC 0.0366 (t 1.91); lc_t10 gross IR 0.615, net IR 0.541, 2025 net return -12.4%); every block is compared with it (paired), and the LP book with it (gross/net IR, single path).

## Commands
```
.venv/bin/python experiments/feat_size_interact/feat_size_interact.py
```

## Results

Replacement composites (compared directly with the frozen composite):

| name | IC | IC t | composite IC | IC diff vs composite | paired t | rank corr w/ composite | diff small | diff mid | diff large | IR gross | IR net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| comp_within_size_tercile | 0.0350 | 1.94 | 0.0366 | -0.0017 | -0.70 | 0.98 | +0.0009 | -0.0002 | +0.0008 | 0.677 | 0.603 |
| comp_tercile_ic_weighted | 0.0241 | 1.46 | 0.0366 | -0.0126 | -1.55 | 0.85 | -0.0068 | -0.0123 | -0.0149 | 0.250 | 0.172 |

Standalone block IC by calendar year and by size tercile (all blocks, signed as tested):

(no additive blocks in this experiment)

Truncation-invariance test of the feature builder: not applicable (no time-series feature is built; the size tercile is a same-month cross-sectional quantity).


## Interpretation
Ranking within size terciles changes essentially nothing (IC 0.0350 vs 0.0366, paired t -0.70, rank correlation 0.98 with the composite). The IC-weighted tercile version, which actually uses a fit, is clearly worse (IC 0.0241, paired t -1.55; gross IR 0.25 vs 0.61): group-IC weights estimated on 2015-2020 do not carry into 2021-26 (the same non-persistence found in `experiments/factor_filter`). **Killed: both variants have paired t < 1 or negative.**

**Status: IMPLEMENTED_BUT_FAILED (both variants killed)**

## Limitations
- The IC-weight fit is short (72 months at the first fold) and mixes regimes; a longer sample could be better.
- Terciles inside a $2B floor cover a narrow size range.
- Two variants tested.

## Follow-up
None.
