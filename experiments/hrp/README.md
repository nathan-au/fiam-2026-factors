# Hierarchical Risk Parity Sizing on the Large-Cap Composite Book (HRP)

Implementation: `hrp.py` (run with `.venv/bin/python experiments/hrp/hrp.py` for the $2B book and `--floor 1000` for the $1B sensitivity; about 10 seconds each). New file; contains a verbatim copy of the parts of the frozen `et.py` harness it uses (standalone) and reads the holdings saved by `largecap.py` (`output/portfolio_holdings_lc_t10_<lc|lc1000>_comp.csv`); edits nothing. Outputs in `output/`: `hrp_summary_<tag>.csv`, `hrp_results_<tag>.json`, `hrp_run_<tag>.log`, `portfolio_{holdings,returns}_<scheme>_cap<per-mille>_<tag>_comp.csv`.

**Question.** `experiments/largecap/README.md`'s headline book is sized by an LP whose linear objective pushes positions to the per-name cap (roughly equal weight). Does risk-aware sizing (López de Prado's HRP) lower volatility and drawdown without hurting neutrality or return? HRP is a sizing method, not a signal: it adds no information, so the only expected effect is on risk.

## Design (pre-registered in the script header before any result was seen)

| Element | Choice |
|---|---|
| Selection | The names each month are exactly those held by `largecap.py`'s `lc_t10` book on the composite (`comp`) arm. Only the sizes change. |
| Schemes | `ew` equal weight within each leg (**control**: same names, same repair); `ivp` inverse-volatility within each leg (**control**: risk-aware sizing without clustering); `hrp` HRP within each leg. |
| HRP | 60-month trailing correlation of monthly excess returns (`ret_exc`, months up to the characteristic month), distance √((1−ρ)/2), single linkage, quasi-diagonalisation by dendrogram leaf order, recursive bisection with inverse-variance cluster variance. No covariance inversion, no shrinkage, pairwise-complete correlations (≥ 48 obs). |
| Neutrality | Each leg's target sizes (sum 1 per leg, gross 200%) are repaired by the smallest L1 change that restores the selection LP's constraints exactly: dollar-neutral, neutral to `beta_60m` and `betabab_1260d`, net sector ≤ 5% NAV, sector gross ≤ 35%, side of every name unchanged, per-name cap. Headline cap 2%, sensitivity 1% (the LP's own cap). The repair was feasible in every month (0 fallbacks). |
| Turnover | No turnover constraint in the repair; realised turnover and tiered costs (`experiments/pm_ablation/README.md` §1) are in the net figures. |
| Verdict rule | HRP is "useful" only if, versus `ew` at the same cap, **net Sharpe is not lower AND net max drawdown is shallower AND neutrality is not worse** (beta t-stat, rolling-12m beta range). `hrp` vs `ivp` is reported separately. Not tuned; not re-run. |

## Results, $2B floor (2021-01 – 2026-08, 68 months, net of assumed costs unless "gross")

| Scheme | Cap | IR gross | IR net | Sharpe net | CAGR net | Ann. vol net | Max DD net | β (t) | Rolling-12m β | One-way turnover, % of gross |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| lp (`lc_t10`, selection book) | 1% | 0.61 | 0.54 | 0.84 | 11.0% | — | −19.0% | −0.06 (−0.55) | −0.60 / +0.32 | 10.0% |
| ew | 2% | 0.62 | 0.54 | 0.84 | 10.9% | 13.3% | −19.0% | −0.06 (−0.53) | −0.53 / +0.31 | 12.0% |
| ew | 1% | 0.62 | 0.54 | 0.85 | 10.8% | 13.2% | −17.7% | −0.05 (−0.42) | −0.53 / +0.32 | 11.7% |
| ivp | 2% | 0.54 | 0.45 | 0.79 | 9.0% | 11.9% | −13.2% | −0.03 (−0.33) | −0.53 / +0.35 | 13.8% |
| ivp | 1% | 0.64 | 0.55 | 0.87 | 10.6% | 12.5% | −14.7% | −0.05 (−0.46) | −0.51 / +0.29 | 13.6% |
| **hrp** | **2%** | 0.69 | 0.58 | 0.88 | 11.3% | 13.2% | −13.7% | −0.12 (−1.08) | −0.64 / +0.23 | 24.0% |
| hrp | 1% | 0.69 | 0.59 | 0.90 | 11.3% | 12.8% | −13.5% | −0.08 (−0.73) | −0.57 / +0.30 | 16.9% |

The `ew` control at 1% reproduces the LP book (IR 0.54 net, drawdown −17.7% vs −19.0%), so the re-sizing machinery changes nothing by itself.

Paired monthly net-return differences at the 2% cap (mean per month, t): `hrp − ew` +0.03% (t 0.17), `hrp − ivp` +0.19% (t 1.32), `ivp − ew` −0.16% (t −1.24). At the 1% cap: `hrp − ew` +0.03% (t 0.34).

## Sensitivity, $1B floor

| Scheme | Cap | IR gross | IR net | Sharpe net | Ann. vol net | Max DD net | β (t) | One-way turnover |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lp (`lc_t10`) | 1% | 0.64 | 0.55 | 0.82 | — | −27.5% | −0.09 (−0.77) | 10.0% |
| ew | 2% | 0.57 | 0.47 | 0.73 | 15.0% | −27.0% | −0.09 (−0.71) | 12.4% |
| ivp | 2% | 0.54 | 0.43 | 0.73 | 13.3% | −23.1% | −0.06 (−0.54) | 13.1% |
| hrp | 2% | 0.46 | 0.33 | 0.60 | 14.9% | −25.9% | −0.15 (−1.22) | 23.4% |
| hrp | 1% | 0.55 | 0.42 | 0.71 | 14.2% | −27.6% | −0.08 (−0.73) | 17.2% |

Paired `hrp − ew` net: −0.17% per month (t −0.88) at 2%, −0.08% (t −0.84) at 1%.

## Verdict against the pre-registered rule

| Floor, cap | Net Sharpe ≥ `ew`? | Max DD shallower? | Neutrality not worse? | Verdict |
|---|---|---|---|---|
| $2B, 2% | yes (0.88 vs 0.84) | yes (−13.7% vs −19.0%) | **no** (β t −1.08 vs −0.53) | **not useful** by the rule |
| $2B, 1% | yes (0.90 vs 0.85) | yes (−13.5% vs −17.7%) | **no** (β t −0.73 vs −0.42) | **not useful** by the rule |
| $1B, 2% | **no** (0.60 vs 0.73) | yes (−25.9% vs −27.0%) | **no** | **not useful** |
| $1B, 1% | **no** (0.71 vs 0.74) | **no** (−27.6% vs −27.4%) | roughly equal | **not useful** |

## Findings

1. **Risk-aware sizing trims drawdown at $2B but not robustly.** HRP cuts max drawdown from −19% to −13.5% and IVP to −13% to −15%, with unchanged or slightly higher return. The differences in monthly returns are nowhere near significant (paired t ≈ 0.2–0.3 for HRP vs equal weight, |t| ≤ 1.3 for every comparison), and at the $1B floor the drawdown benefit is mostly gone and net Sharpe is lower.
2. **HRP's clustering adds nothing over simple inverse-volatility sizing.** IVP delivers a similar drawdown reduction (and the lowest volatility, 11.9% vs 13.3%) with about half of HRP's turnover. HRP − IVP is +0.19% per month at 2% (t 1.3) at $2B and negative at $1B.
3. **HRP costs turnover.** One-way turnover is 17–24% of gross vs 12% for equal weight, because sizes move with the trailing correlation matrix each month; this is already inside the net figures. It is the practical reason not to prefer it.
4. **Neutrality slips a little.** Realised beta is more negative with HRP (t −1.08 at the 2% cap vs −0.53), still not statistically different from zero; rolling-12m beta stays within about ±0.65 with no window above 1. This is what fails the pre-registered rule.
5. **Bottom line for the deck.** HRP is feasible and works mechanically, but it is not a reliable improvement here; the honest statement is "risk-aware sizing (IVP or HRP) reduced the 2021–2026 drawdown at the $2B floor, HRP no more than IVP, and the effect did not hold at $1B". If a risk-sizing scheme is shown at all, inverse-volatility is the simpler and cheaper choice.

## Limitations

- One path, 68 months; no confidence intervals beyond the paired t-statistics above.
- Only the sizes of an existing selection book are changed; the selection LP still chooses names by the composite and applies its own turnover cap, so this does not test HRP inside the optimisation.
- 60-month correlation from monthly returns on about 115 names per leg is noisy; no shrinkage was applied (pre-specified, not tuned). Single linkage was used as in the original paper and was not compared with other linkages.
- Costs and borrow are assumed tiers; the repair LP has no turnover constraint, which is why turnover rises.
- The rule's "neutrality not worse" test compares two insignificant t-statistics; it is a strict reading of the pre-registered criterion, not evidence that neutrality is broken.
