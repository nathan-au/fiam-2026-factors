# desk_17 — Why does the construction lose on 2015–2020? (attribution)

**Question:** is the DEV degradation explainable, distinguishable from noise, predictable ex ante, or a finite-sample artefact?
**Data:** B1 over DEV and TEST, descriptive; ledgered as "descriptive, no selection". TEST was already seen for this system. **Caveat:** the TEST column of part E is a TEST look at the regime variables, disclosed in desk_27.
**Outputs:** `group_ic_by_year.csv`, `fm_group_returns_by_year.csv`, `book_decomposition_{dev,test}.csv`, `book_group_exposure_*.csv`, `sector_by_year_*.csv`, `regime_table.csv`, `results.json`.

## A. Which factors (universe IC by year)
The value group is the swing factor: IC +0.07 (2016), −0.06 (2018), **−0.11 (2020)**, then +0.09 / +0.07 (2021 / 2022). Investment/issuance: −0.05 in 2020, +0.07 / +0.05 in 2021 / 22. Vol/beta: −0.06 in 2020, then +0.10 in 2022. The composite is +0.08 (2015), +0.06 (2017), **−0.08 (2020)**, +0.09 (2021 / 22), −0.00 (2025).

## B. Book decomposition (monthly exposure to each group × its Fama-MacBeth return, summed per year)
| year | value | profitability | inv/iss/accr | quality | vol/beta | dtc | stock-specific residual | gross total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2019 | −0.9% | +5.5% | −1.6% | −7.1% | −2.3% | +2.0% | −0.5% | −5.0% |
| **2020** | **−18.6%** | +3.4% | −6.2% | +2.0% | −5.8% | −4.4% | +13.0% | **−20.3%** |
| 2021 | +13.3% | +12.1% | +5.6% | −3.6% | −2.3% | +4.9% | −1.9% | +29.9% |
| 2022 | +12.0% | −2.0% | +3.9% | −5.9% | +4.8% | +5.7% | −1.8% | +19.0% |
| 2025 | −1.1% | −8.4% | −2.2% | +1.4% | −2.5% | +6.7% | +5.8% | −1.6% |

The book carries **persistent, nearly constant** group exposures (value ~0.5, profitability ~0.75–0.93, quality ~0.9–1.1, investment ~0.35, vol/beta ~0.3). Its P&L is mostly those groups' factor returns. The 2020 loss is the value crash (FM value return −36%, the largest in the sample), with investment and low-vol also losing. 2021–22 are the value rebound.

## C. Legs and sectors
DEV losses sit in the short leg (2017 −19%, 2019 −32%, 2020 −33% of short-leg P&L) during a bull market, and the realised DEV beta is −0.15. Sector contributions are small (all within ±1.5%/yr on DEV).

## D. Is DEV different beyond noise?
- Monthly IC, DEV vs TEST: 0.013 vs 0.045, Welch t 1.34, p 0.18. **Not distinguishable.**
- Monthly net return: −0.10% vs +1.00%, t 2.20, p 0.03. **Distinguishable.**
- IR difference: 1.33, 90% CI [0.40, 2.31].

So the *prediction quality* difference is within noise, while the *book return* difference is not. The book converts style-factor returns, not stock-specific IC, into P&L, and those factor returns differ between the regimes.

## E. Ex-ante predictability of the composite's monthly IC (NW t, slope per 1 sd)
| predictor at formation | DEV slope (t) | TEST slope (t) |
|---|---:|---:|
| cross-sectional return dispersion | **−0.048 (−5.4)** | **−0.024 (−3.1)** |
| trailing 12m market return | +0.029 (+2.5) | +0.027 (+1.9) |
| trailing 6m market vol | −0.033 (−3.2) | −0.019 (−1.3) |
| value spread | −0.045 (−3.1) | −0.006 (−0.3) |
| VIX | −0.034 (−2.1) | −0.005 (−0.4) |
| lagged IC / trailing-12m IC | +0.016 (1.0) / +0.006 (0.2) | +0.007 / −0.001 |

## Answers
1. **Economically explainable: yes.** The composite is a value + quality + low-risk style book. 2015–2020 contains the deepest value drawdown on record (Arnott et al. 2021) plus the 2020 junk/high-vol rally.
2. **Statistically distinguishable: the book returns yes (p 0.03); the IC no (p 0.18).**
3. **Predictable ex ante: partly.** Dispersion and market state predict the IC in both periods. Value-spread and VIX effects appear only on DEV. The IC is not autocorrelated, so IC-momentum gating has no basis.
4. **Finite-sample artefact: the IC gap may be one.** The return gap is a regime (style-premium) effect.

desk_21 tests whether any of this can be exploited causally.
