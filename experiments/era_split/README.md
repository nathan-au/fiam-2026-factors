# Era Splitting (invariance-seeking tree splits) and size-environment eras (ERA_SPLIT)

Implementation: `era_split.py` (run: `.venv/bin/python experiments/era_split/era_split.py`, about 6 minutes).

## Objective
Test Era Splitting (per-era split criteria that reward only splits that help in every era) and my extension H1 (eras = month x size tercile, so a split must also help large caps) as a way to make a fitted tree model beat the composite in the >= $2B universe.

## Hypothesis
Fitted trees learn structure whose gain is concentrated in some periods / size buckets (small-cap, ivol, lottery) and does not persist. Era-wise criteria should raise the universe rank IC over the same trees with the standard pooled criterion, and the composite-initialised versions should add to (not lose to) the composite. The criteria may underfit and reproduce the composite: an acceptable, informative outcome.
Pre-registered kill rule (docs/REDDIT_RESEARCH.md sec 2.3): paired monthly IC t < 1 vs the vanilla-criterion control (`orig`), or IC < the composite's 0.037.

## Research origin
docs/REDDIT_RESEARCH.md sec 2.3 (Era Splitting, arXiv 2309.14496: two new tree criteria; Numerai dataset; grade Reported, only the abstract and the fork README were read) and H1 (size-environment invariance, "speculative"); H8 (two-stage buffered composite + era-split) was conditional on this and on the buffer experiment both passing, and neither did (see the final report).

## Implementation
The upstream `scikit-learn-erasplit` fork needs a source build of a patched scikit-learn and was **not used**. `era_split.py` implements a small numpy histogram GBDT (20 bins on the rank-transformed features, depth 3, 250 rounds at lr 0.03, min leaf 300 rows, lambda 100, 50% row / 50% column subsampling, target = month-demeaned winsorised return) with pluggable split criteria computed from era-wise gradient/count histograms: `orig` (pooled second-order gain: the control, same code path), `era_avg` (sum of each era's own gain), `era_softmin` (Boltzmann soft-minimum over eras of the per-row gain, so a split must help in *every* era), `era_dir` (directional: an era's gain counts positively if its left-vs-right direction agrees with the pooled direction and negatively otherwise) and `era_dir_size` (eras = month x size tercile). `*_init` arms boost from `init_score = a*composite` (a = fitted slope floored at +0.005, see the bug note). The number of trees (0 allowed for `*_init`) is chosen on validation IC. These are re-implementations from the paper's description, so a negative result is about the idea as implemented here, not about the fork.

## Experimental setup
All four target experiments use the same frozen harness (a verbatim copy of the data / rank-transform / walk-forward / LP / cost / performance code of `experiments/largecap/largecap.py`, embedded in the script so it runs on its own).

- **Data**: `fiam/chars_final_with_names.parquet`, 523,125 stock-months after dropping rows without a next-month return; 18 factors chosen by economic group in `experiments/largecap` (`FACTOR_GROUPS`), cross-sectionally median-filled and rank-transformed to [-1, 1] over all stocks each month.
- **Universe** (training, validation and scoring rows): price >= $5, market cap >= $2,000M, 126-day dollar volume >= $10M, both betas observed (about 1,206 stocks per month; 153,392 rows).
- **Walk-forward**: for test year Y = 2021..2026, train on target months before Jan Y-2, validate on Y-2..Y-1, refit annually; candidate hyper-parameters are chosen on **validation rank IC against the raw next-month excess return** (for every arm, whatever the fit target), never on test data.
- **Models**: Extra-Trees (largecap grid, 300 trees) and forest-style LightGBM (num_leaves 7, min_child_samples {500, 2000}, extra_trees, lambda 100, checkpoints at 100/200/300 trees). Seed 42 unless stated.
- **Scoring**: universe rank IC of the OOS prediction vs the raw next-month return (68 months, 2021-01..2026-08), decile spread, and the frozen LP portfolios `lc_t10` (10% one-way turnover cap, headline) and `lc_free`, gross and net of the assumed tiered costs.
- **Pre-registered rule** (docs/NEW.md sec 4): adopt an alternative target only if paired monthly universe-IC t >= 2 vs the control on **two** model families (rule A), or IC >= the composite's 0.037 with t >= 2 (rule B).
- **Sanity check that the harness is faithful**: the control arm `et__control` reproduces `experiments/largecap` `et` exactly: IC 0.0164 (published 0.016), lc_t10 gross IR -0.22 (published -0.22).

## Baseline
`orig` (same GBDT, pooled criterion) and the composite (IC 0.0366; gross IR 0.615).

## Commands
```
.venv/bin/python experiments/era_split/era_split.py
```

## Results (68 test months)
| arm | ic | ic_t | IC diff vs comp | t vs comp | IC diff vs orig | t vs orig | resid IC | IC small | IC mid | IC large | IR gross | IR net | max DD % | avg trees |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| comp | 0.0366 | 1.91 | +0.0000 |  | +0.0185 | 0.848 |  | 0.0512 | 0.0359 | 0.0145 | 0.615 | 0.541 | -19 |  |
| orig | 0.0181 | 1.42 | -0.0185 | -0.848 | +0.0000 |  | +0.0048 | 0.0259 | 0.0121 | 0.0067 | -0.411 | -0.493 | -41 | 167 |
| era_avg | 0.0054 | 0.37 | -0.0312 | -1.336 | -0.0127 | -1.910 | -0.0043 | 0.0096 | -0.0025 | -0.0012 | -0.843 | -0.931 | -43 | 217 |
| era_softmin | 0.0042 | 0.27 | -0.0324 | -1.329 | -0.0139 | -1.773 | -0.0043 | 0.0137 | -0.0043 | -0.0083 | -0.729 | -0.810 | -42 | 142 |
| era_dir | 0.0142 | 1.03 | -0.0224 | -1.057 | -0.0039 | -0.933 | +0.0005 | 0.0234 | 0.0073 | 0.0029 | -0.438 | -0.527 | -33 | 183 |
| era_dir_size | 0.0129 | 0.92 | -0.0238 | -1.129 | -0.0052 | -1.127 | -0.0005 | 0.0213 | 0.0074 | -0.0008 | -0.330 | -0.410 | -35 | 158 |
| orig_init | 0.0161 | 1.00 | -0.0206 | -1.773 | -0.0020 | -0.117 | -0.0200 | 0.0348 | 0.0150 | -0.0132 | -0.061 | -0.166 | -25 | 92 |
| era_dir_init | 0.0212 | 1.32 | -0.0154 | -1.437 | +0.0031 | 0.178 | -0.0162 | 0.0390 | 0.0196 | -0.0062 | -0.183 | -0.285 | -24 | 100 |
| era_dir_size_init | 0.0197 | 1.21 | -0.0169 | -1.627 | +0.0016 | 0.090 | -0.0178 | 0.0380 | 0.0179 | -0.0094 | 0.078 | -0.028 | -20 | 92 |

**Implementation bug found and fixed (kept for the record).** Same as in `composite_prior`: the first run used an unfloored composite scale, whose slope was about zero in the 2023 fold, flipping the `*_init` arms' prior. Invalid first-run summary (kept in `output/first_run_invalid_negative_scale/`):

| arm | ic | d_ic_vs_comp | paired_t_vs_comp | paired_t_vs_orig | ir_gross |
|---|---:|---:|---:|---:|---:|
| comp | 0.0366 | +0.0000 |  | 0.848 | 0.615 |
| orig | 0.0181 | -0.0185 | -0.848 |  | -0.411 |
| era_avg | 0.0054 | -0.0312 | -1.336 | -1.910 | -0.843 |
| era_softmin | 0.0042 | -0.0324 | -1.329 | -1.773 | -0.729 |
| era_dir | 0.0142 | -0.0224 | -1.057 | -0.933 | -0.438 |
| era_dir_size | 0.0129 | -0.0238 | -1.129 | -1.127 | -0.330 |
| orig_init | 0.0238 | -0.0129 | -0.693 | 0.574 | -0.231 |
| era_dir_init | 0.0262 | -0.0104 | -0.592 | 0.818 | -0.371 |
| era_dir_size_init | 0.0238 | -0.0128 | -0.747 | 0.551 | -0.187 |

Only the `*_init` arms changed (IC 0.0161-0.0212 after the fix vs 0.0238-0.0262 before); the conclusion is unchanged.

## Interpretation
No era criterion beats the vanilla pooled criterion in the way the hypothesis needs. On raw arms the era criteria are worse than `orig` (IC 0.0181): `era_avg` 0.0054 (t vs orig -1.9), `era_softmin` 0.0042 (-1.8), `era_dir` 0.0142 (-0.9), `era_dir_size` (H1) 0.0129 (-1.1). With the composite prior, the directional criteria are marginally above the pooled one (`era_dir_init` 0.0212 vs `orig_init` 0.0161, `era_dir_size_init` 0.0197; paired t vs `orig_init` +0.18 and +0.09), but every arm is well below the composite (paired t vs composite -0.8 to -1.8). The size-environment idea H1 does not help (`era_dir_size` <= `era_dir`). Directional criteria do behave like regularisers (fewer or smaller splits), which is the theory, but on this data that regularisation buys nothing that beats the composite. **Kill rule fires for all arms.** Status: IMPLEMENTED_BUT_FAILED (my re-implementation; not the upstream fork).

## Limitations
- Re-implementation: the fork's criteria (gamma / blama / vanna mixing, Boltzmann alpha) are not reproduced exactly; `era_softmin` in particular uses an adaptive scale that I chose.
- Only 47-132 monthly eras are available at the folds (from 2015), far fewer than the Numerai setting; per-era histograms of about 1,000 rows over 20 bins are sparse.
- One seed, one depth, one learning-rate setting.

## Follow-up
Not warranted. H8 (two-stage system) is not implemented because its stated precondition (buffer and era-split each pass individually) failed.
