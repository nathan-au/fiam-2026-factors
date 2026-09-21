# Portfolio-Constraint Ablation and the Tradeability Finding (PM_ABLATION)

Implementation: `pm_ablation.py` (run with `.venv/bin/python experiments/pm_ablation/pm_ablation.py` **after** `rf_3.py`, `et.py`, `lgbm.py`, `cat.py`, `xgb_2.py`; no refitting — it reads their saved predictions `output/oos_predictions_<tag>_modern_nomom.csv`). New file; nothing existing is edited. Runs in about 10 minutes.

**Why this exists.** The first run of `rf_3.py` showed the project's legacy LP at IR ≈ 1.3 but every "portfolio-manager" variant (the constraints suggested in *Valentino's FIAM Tips* and by the financial engineer) at IR ≈ 0 and net-negative. All five models later agreed. This script isolates *which* constraint does it.

**Bottom line.**
1. **The models' edge is a small-cap, sub-$5, short-leg edge, concentrated in Health Care / biotech.** Adding *only* a price ≥ $5 screen to the legacy LP takes the short leg from about +18%/yr to about −4%/yr (RF); adding a market-cap floor of $250M–$1B makes the short leg −10% to −14%/yr (§2).
2. **Rank skill in the tradeable universe is small and mostly not exploitable.** Test rank IC is 0.14–0.15 over all stocks, 0.20 below $250M market cap, 0.05 on all names passing the price/market-cap/dollar-volume screens and only 0.034–0.037 on the universe the PM LP can actually hold (screens plus both betas observed) (§3). Sorting that universe by prediction gives a decile spread of about zero (§5), and the FIAM §6 baseline book (top decile long / bottom decile short, equal weight) has gross IR −0.04 to −0.21 (§5).
3. **The constraints that are not about universe are cheap or mildly costly:** dual-beta neutrality does not hurt (IR 1.26 → 1.38 for RF), a 10% turnover cap costs about 0.05–0.26 IR gross depending on model, sector limits cost about 0.08–0.41 IR — because the legacy book is on average **42% of NAV net short Health Care** (§4).
4. Practical reading: through FIAM's neutrality/tradeability lens, these models show no demonstrated tradeable edge in this sample. Reporting that honestly (the brief says a candid negative result is worth more than a polished one) is a valid deliverable; a positive tradeable result needs a different signal (§7).

## 1. Setup

**Inputs.** Predictions of the five models on the headline `modern_nomom` arm, plus `ens5` — an equal-weight ensemble (per-month cross-sectional percentile rank of each model's prediction, averaged). All are out-of-sample (2021-01 – 2026-08).

**Portfolios.** All ablation rows are the legacy LP (`experiments/ols/README.md` §2.5) plus the change named in the row; `pm_free` is the combination of the screens + dual beta + sector limits, and `pm_t10` adds the 10% turnover cap. The constraint details: price ≥ $5, market cap ≥ $500M, both `beta_60m` and `betabab_1260d` neutral, |net| ≤ 5% of NAV and gross ≤ 35% of the book per 2-digit GICS sector, one-way drift-adjusted turnover ≤ 10% of gross (same convention as `experiments/rf_3/README.md`). `decile_ew_tradeable` is FIAM §6's baseline: within price ≥ $5, market cap ≥ $500M, $10M dollar-volume stocks with both betas observed, long the top decile / short the bottom decile of the prediction, equal weight (100% long, 100% short), **not** beta-neutral, no sector or turnover control.

**Cost assumptions (net figures).** Not measured — the panel has no borrow or spread by name:

| Market cap | One-way trading cost | Annual borrow on short notional |
|---|---:|---:|
| ≥ $10B | 5 bp | 30 bp |
| $2B – $10B | 10 bp | 75 bp |
| < $2B | 20 bp | 200 bp |

Trading cost is applied to every dollar traded, including the initial build, using drift-adjusted trades; exits are charged at the last-held tier. Real borrow on the small/biotech/meme shorts of the legacy book (§4) would be far above 200 bp, so **net figures for the legacy book are flattering**.

**Universe note.** Every LP in this project excludes stocks with no `beta_60m` (≈ 11% of liquid names: recent listings), since they cannot enter the beta constraint. The FIAM rule is to state universe restrictions; this is one.

## 2. Constraint ablation (headline arm `modern_nomom`, 2021-01 – 2026-08)

Gross IR (net-of-cost IR follows):

| Portfolio | rf3 | et | lgbm | cat | xgb2 | ens5 |
|---|---:|---:|---:|---:|---:|---:|
| legacy | 1.26 | 0.96 | 0.92 | 0.92 | 0.90 | 0.94 |
| +price>=5 | 0.48 | 0.35 | 0.37 | 0.39 | 0.26 | 0.46 |
| +mcap>=250M | 0.27 | 0.12 | 0.14 | 0.15 | 0.03 | 0.17 |
| +mcap>=500M | 0.22 | 0.05 | 0.09 | 0.04 | -0.04 | 0.12 |
| +mcap>=1000M | 0.21 | 0.11 | 0.19 | 0.11 | 0.08 | 0.15 |
| +mcap>=2000M | 0.67 | 0.45 | 0.40 | 0.35 | 0.34 | 0.49 |
| +price5&mcap500 | 0.09 | 0.08 | 0.08 | 0.04 | -0.05 | 0.09 |
| +dual_beta | 1.38 | 1.09 | 1.00 | 1.05 | 0.88 | 0.98 |
| +sector_limits | 0.85 | 0.78 | 0.77 | 0.80 | 0.82 | 0.85 |
| +turnover_10pct | 1.00 | 0.91 | 0.68 | 0.77 | 0.79 | 0.75 |
| pm_free | 0.07 | -0.11 | 0.04 | -0.03 | -0.24 | -0.13 |
| pm_t10 | -0.21 | -0.28 | -0.12 | -0.31 | -0.17 | -0.27 |
| decile_ew_tradeable | -0.04 | -0.18 | -0.21 | -0.15 | -0.10 | -0.10 |

Net IR (tiered costs):

| Portfolio | rf3 | et | lgbm | cat | xgb2 | ens5 |
|---|---:|---:|---:|---:|---:|---:|
| legacy | 1.09 | 0.81 | 0.80 | 0.78 | 0.76 | 0.81 |
| +price>=5 | 0.32 | 0.19 | 0.22 | 0.22 | 0.08 | 0.31 |
| +mcap>=250M | 0.12 | -0.02 | 0.00 | -0.01 | -0.12 | 0.02 |
| +mcap>=500M | 0.06 | -0.10 | -0.06 | -0.12 | -0.21 | -0.03 |
| +mcap>=1000M | 0.04 | -0.05 | 0.03 | -0.06 | -0.09 | -0.01 |
| +mcap>=2000M | 0.51 | 0.31 | 0.26 | 0.20 | 0.19 | 0.34 |
| +price5&mcap500 | -0.07 | -0.07 | -0.08 | -0.13 | -0.23 | -0.08 |
| +dual_beta | 1.20 | 0.93 | 0.86 | 0.90 | 0.73 | 0.84 |
| +sector_limits | 0.71 | 0.64 | 0.63 | 0.66 | 0.67 | 0.71 |
| +turnover_10pct | 0.90 | 0.82 | 0.60 | 0.69 | 0.70 | 0.67 |
| pm_free | -0.13 | -0.30 | -0.16 | -0.23 | -0.44 | -0.34 |
| pm_t10 | -0.31 | -0.37 | -0.22 | -0.41 | -0.27 | -0.36 |
| decile_ew_tradeable | -0.20 | -0.32 | -0.35 | -0.31 | -0.27 | -0.25 |

Short-leg CAGR (gross, excess of risk-free; a negative number means the shorted names rose):

| Portfolio | rf3 | et | lgbm | cat | xgb2 | ens5 |
|---|---:|---:|---:|---:|---:|---:|
| legacy | 18.2 | 16.4 | 11.3 | 9.9 | 12.2 | 13.8 |
| +price>=5 | -3.8 | -5.2 | -7.6 | -6.0 | -6.2 | -2.2 |
| +mcap>=250M | -10.8 | -11.6 | -14.2 | -13.1 | -13.1 | -10.1 |
| +mcap>=500M | -10.9 | -12.1 | -14.0 | -13.4 | -13.5 | -10.6 |
| +mcap>=1000M | -10.2 | -10.7 | -11.9 | -12.4 | -11.3 | -9.7 |
| +mcap>=2000M | -2.0 | -4.2 | -6.6 | -6.2 | -5.1 | -4.2 |
| +price5&mcap500 | -12.2 | -11.2 | -13.7 | -13.2 | -12.7 | -10.7 |
| +dual_beta | 17.1 | 14.2 | 9.8 | 9.1 | 8.0 | 12.1 |
| +sector_limits | 7.7 | 8.9 | 8.1 | 7.4 | 8.7 | 12.2 |
| +turnover_10pct | 15.8 | 15.1 | 6.6 | 7.5 | 10.1 | 10.8 |
| pm_free | -13.8 | -16.0 | -13.4 | -15.4 | -14.8 | -14.1 |
| pm_t10 | -15.0 | -16.8 | -15.3 | -16.1 | -14.2 | -14.1 |
| decile_ew_tradeable | -11.4 | -13.7 | -14.7 | -13.8 | -11.8 | -12.7 |

Long-leg CAGR (gross):

| Portfolio | rf3 | et | lgbm | cat | xgb2 | ens5 |
|---|---:|---:|---:|---:|---:|---:|
| legacy | 13.4 | 9.3 | 14.8 | 13.2 | 10.4 | 11.9 |
| +price>=5 | 13.0 | 11.9 | 15.2 | 13.2 | 9.9 | 11.6 |
| +mcap>=250M | 14.5 | 11.6 | 15.0 | 13.8 | 10.3 | 11.8 |
| +mcap>=500M | 14.2 | 11.3 | 14.6 | 12.7 | 10.5 | 11.9 |
| +mcap>=1000M | 13.6 | 12.2 | 15.5 | 13.8 | 11.9 | 12.3 |
| +mcap>=2000M | 12.4 | 12.1 | 13.5 | 11.9 | 10.4 | 12.2 |
| +price5&mcap500 | 13.2 | 12.0 | 14.9 | 13.3 | 10.4 | 11.7 |
| +dual_beta | 15.4 | 11.9 | 15.6 | 16.5 | 12.8 | 13.1 |
| +sector_limits | 12.8 | 10.8 | 11.4 | 12.1 | 11.1 | 9.9 |
| +turnover_10pct | 6.6 | 7.1 | 9.6 | 10.0 | 8.5 | 7.6 |
| pm_free | 14.3 | 12.8 | 13.0 | 14.0 | 9.1 | 11.6 |
| pm_t10 | 10.5 | 10.5 | 11.9 | 9.1 | 10.7 | 8.5 |
| decile_ew_tradeable | 10.3 | 8.7 | 9.1 | 10.2 | 9.0 | 9.7 |

Short-book median market cap ($M):

| Portfolio | rf3 | et | lgbm | cat | xgb2 | ens5 |
|---|---:|---:|---:|---:|---:|---:|
| legacy | 932 | 916 | 851 | 923 | 882 | 903 |
| +price>=5 | 1707 | 1637 | 1532 | 1710 | 1678 | 1657 |
| +mcap>=250M | 1440 | 1392 | 1312 | 1420 | 1390 | 1403 |
| +mcap>=500M | 1816 | 1737 | 1641 | 1804 | 1799 | 1747 |
| +mcap>=1000M | 2632 | 2520 | 2381 | 2600 | 2629 | 2587 |
| +mcap>=2000M | 4524 | 4407 | 4098 | 4469 | 4451 | 4358 |
| +price5&mcap500 | 2137 | 2019 | 1893 | 2118 | 2110 | 2046 |
| +dual_beta | 975 | 986 | 931 | 988 | 954 | 1033 |
| +sector_limits | 980 | 955 | 886 | 966 | 914 | 959 |
| +turnover_10pct | 1017 | 1037 | 969 | 1036 | 993 | 1058 |
| pm_free | 2363 | 2236 | 2014 | 2261 | 2247 | 2280 |
| pm_t10 | 2591 | 2411 | 2215 | 2430 | 2493 | 2553 |
| decile_ew_tradeable | 2414 | 2216 | 2047 | 2358 | 2350 | 2222 |

Rolling-12m β, maximum (window count above 1 is one — the first window — for every legacy/PM row of the model scripts):

| Portfolio | rf3 | et | lgbm | cat | xgb2 | ens5 |
|---|---:|---:|---:|---:|---:|---:|
| legacy | 1.53 | 1.59 | 2.20 | 2.06 | 2.07 | 2.02 |
| +price>=5 | 1.38 | 1.42 | 1.54 | 1.39 | 1.46 | 1.64 |
| +mcap>=250M | 1.66 | 1.67 | 1.73 | 1.61 | 1.73 | 1.79 |
| +mcap>=500M | 1.28 | 1.26 | 1.26 | 1.22 | 1.19 | 1.41 |
| +mcap>=1000M | 1.02 | 1.07 | 1.09 | 1.16 | 1.04 | 1.26 |
| +mcap>=2000M | 0.38 | 0.35 | 0.39 | 0.51 | 0.49 | 0.53 |
| +price5&mcap500 | 1.16 | 1.13 | 1.13 | 1.13 | 1.06 | 1.31 |
| +dual_beta | 1.53 | 1.56 | 2.25 | 1.99 | 1.98 | 2.24 |
| +sector_limits | 2.11 | 2.03 | 2.18 | 1.99 | 1.88 | 2.10 |
| +turnover_10pct | 1.19 | 1.50 | 2.15 | 1.63 | 1.81 | 1.96 |
| pm_free | 1.14 | 1.32 | 1.22 | 1.23 | 1.06 | 1.45 |
| pm_t10 | 1.26 | 1.36 | 1.26 | 1.15 | 1.12 | 1.40 |
| decile_ew_tradeable | 1.20 | 1.27 | 1.24 | 1.32 | 1.35 | 1.30 |

What the tables say:
- **The price screen and the market-cap floors are the whole story.** `+price>=5` alone: RF IR 1.26 → 0.48, short leg +18.2% → −3.8%, while the long leg is unchanged (13.4% → 13.0%). `+mcap>=500M`: 0.22. The long leg does not care about the universe; the short leg turns from the source of the alpha into a loss.
- **A $2B floor is the exception** (RF 0.67 gross, 0.51 net; short leg −2.0%; max rolling β 0.38 — the most stable-beta book of all): with only large caps the short leg roughly breaks even and the long leg carries a modest edge. This is one row on one model set; not a tuned choice.
- **Dual-beta neutrality is free** (gross IR up for RF, ET, LightGBM, CatBoost and the ensemble; flat for XGB_2: 0.90 → 0.88). It does not, by itself, remove the first-window rolling-β peak.
- **A 10% turnover cap costs less than the other constraints** (RF 1.26 → 1.00, ET 0.96 → 0.91, LightGBM 0.92 → 0.68) but not nothing; with the tiered costs RF's net IR still falls from 1.09 to 0.90 (the cap is applied to a book whose gross IR also falls, so the cost saving does not offset it in these results).
- **Sector limits** cost 0.08–0.41 IR (RF 1.26 → 0.85, XGB_2 0.90 → 0.82) — see §4.
- **The ensemble never beats its best member** (ties RF at `+sector_limits`, 0.85; `ens5` legacy 0.94 vs RF 1.26), unsurprising when the members share the same features and the same source of skill.
- The `pm_free` → `pm_t10` step (adding the 10% turnover cap to a book with no edge) lowers gross IR for every model except XGB_2 (−0.24 → −0.17). My interpretation, not tested: with no signal there is nothing to offset the stale positions the cap forces the LP to keep.

## 3. Where the rank skill lives

Mean monthly Spearman rank IC between prediction and next-month return, 2021-01 – 2026-08, by size bucket (a stock belongs to every bucket it satisfies; buckets overlap):

| Bucket (stocks with characteristics that month) | Avg. stocks/month | rf3 | et | lgbm | cat | xgb2 | ens5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| all_stocks | 3952 | 0.143 | 0.152 | 0.155 | 0.144 | 0.148 | 0.152 |
| liquid_$10M+ | 1739 | 0.082 | 0.084 | 0.088 | 0.079 | 0.083 | 0.086 |
| price<5 | 997 | 0.151 | 0.164 | 0.156 | 0.151 | 0.152 | 0.158 |
| price>=5 | 2955 | 0.069 | 0.074 | 0.078 | 0.070 | 0.074 | 0.075 |
| mcap<250M | 1366 | 0.198 | 0.212 | 0.210 | 0.197 | 0.202 | 0.209 |
| mcap250-500M | 435 | 0.093 | 0.108 | 0.105 | 0.094 | 0.095 | 0.102 |
| mcap500M-2B | 781 | 0.070 | 0.081 | 0.085 | 0.071 | 0.073 | 0.078 |
| mcap>=2B | 1370 | 0.045 | 0.041 | 0.046 | 0.042 | 0.045 | 0.046 |
| tradeable(price>=5&mcap>=500M&$10M) | 1620 | 0.049 | 0.049 | 0.054 | 0.046 | 0.049 | 0.052 |
| tradeable_and_betas_observed(LP-investable) | 1454 | 0.036 | 0.035 | 0.037 | 0.034 | 0.034 | 0.037 |

The models rank small, low-priced stocks well (IC 0.15 for price < $5, ≈ 0.20 below $250M) and large or priced stocks poorly (0.04–0.05 above $2B; 0.05 on the screened universe; 0.034–0.037 on the LP-investable universe). The overall 0.14–0.15 is a blend dominated by the many small stocks. A rank IC of 0.035 is small, and §5 shows it is not monetized by a long/short book on the names that pass the tradeability screens.

## 4. Sector concentration and the names behind the alpha

The legacy RF book has no sector control. Average sector exposure of its holdings (weights in % of NAV, 68 months):

| GICS sector | Avg. net weight (% NAV) | Avg. gross weight (% NAV) | Share of gross book | Largest |net| in any month (% NAV) |
|---|---:|---:|---:|---:|
| 35 Health Care | -41.6% | 68.8% | 34.4% | 58% |
| 55 Utilities | -1.7% | 2.2% | 1.1% | 17% |
| 50 Communication Services | -0.0% | 7.4% | 3.7% | 7% |
| 15 Materials | +0.2% | 4.5% | 2.2% | 8% |
| 60 Real Estate | +0.4% | 1.4% | 0.7% | 3% |
| 30 Consumer Staples | +1.1% | 4.8% | 2.4% | 6% |
| 10 Energy | +2.4% | 10.3% | 5.1% | 23% |
| 20 Industrials | +4.9% | 27.3% | 13.6% | 23% |
| 40 Financials | +7.6% | 12.9% | 6.4% | 35% |
| 25 Consumer Discretionary | +11.3% | 27.4% | 13.7% | 38% |
| 45 Information Technology | +15.6% | 33.0% | 16.5% | 30% |

**Health Care is on average 34% of the gross book and 42% of NAV *net short*, and reaches 58% of NAV net short in one month** — a huge, unintended sector bet for a book meant to be market-neutral. The long side is Information Technology, Consumer Discretionary, Financials and Industrials. This is the concentrated-sector risk the tips warn about ("performance … not driven by one concentrated sector bet") and is consistent with the IR loss when sector limits are added (§2). The ten largest average shorts, by weight averaged over all months (heavily overlapping the names `experiments/rf_2/README.md` §6 listed for its own book — IOVA, NTLA, AMC, MULN, NKLA, LAZR):

| Ticker | Company | Avg. weight (% NAV, averaged over all 68 months) |
|---|---|---:|
| IOVA | Iovance Biotherapeutics Inc | -0.79% |
| NTLA | Intellia Therapeutics Inc | -0.69% |
| LAZR | Luminar Technologies Inc | -0.65% |
| AMC | A M C Entertainment Holdings Inc | -0.62% |
| NKLA | Nikola Corp | -0.60% |
| ALT | Altimmune Inc | -0.60% |
| MULN | Mullen Automotive Inc | -0.59% |
| PLUG | Plug Power Inc | -0.57% |
| SPCE | Virgin Galactic Holdings Inc | -0.56% |
| FCEL | Fuelcell Energy Inc | -0.56% |

By my own characterization these are biotechs, EV/hydrogen/space names and meme stocks — plausibly hard-to-borrow, high-volatility shorts, i.e. the "untradeable short book" the FIAM brief says it will read the short book for (`docs/FIAM.md` §2, §12). I have not checked borrow availability name by name (no such data in the panel).

## 5. The tradeable-universe signal, two ways

**Decile table.** Mean next-month excess return (%, equal-weighted, all 68 OOS months pooled) by within-month prediction decile, on the universe the LP can hold (price ≥ $5, market cap ≥ $500M, $10M dollar volume, both betas observed):

| Decile (1 = lowest prediction) | rf3 | et | lgbm | cat | xgb2 | ens5 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.67 | 0.83 | 0.91 | 0.88 | 0.69 | 0.75 |
| 2 | 0.51 | 0.39 | 0.49 | 0.55 | 0.62 | 0.61 |
| 3 | 0.72 | 0.71 | 0.79 | 0.71 | 0.94 | 0.73 |
| 4 | 0.74 | 0.99 | 0.87 | 0.68 | 0.80 | 0.71 |
| 5 | 0.85 | 0.93 | 0.97 | 0.75 | 0.64 | 0.86 |
| 6 | 0.73 | 0.85 | 0.65 | 0.86 | 0.70 | 0.84 |
| 7 | 0.93 | 0.71 | 0.57 | 0.78 | 0.59 | 0.75 |
| 8 | 0.80 | 0.69 | 0.83 | 0.73 | 0.83 | 0.76 |
| 9 | 0.86 | 0.85 | 0.85 | 0.85 | 1.08 | 0.83 |
| 10 | 0.91 | 0.78 | 0.82 | 0.96 | 0.86 | 0.90 |
| **D10 − D1** | **+0.24** | **-0.04** | **-0.09** | **+0.07** | **+0.17** | **+0.15** |

The spread between the top and bottom decile ranges from −0.09 to +0.24 %/month across the models — smaller than the monthly T-bill + 4% hurdle (≈ 0.3–0.75 %/month over this period) even before costs, and the decile means are not monotone.

**FIAM §6 baseline book** (`decile_ew_tradeable` rows of §2): long the top decile / short the bottom decile, equal-weighted: gross IR RF −0.04, ET −0.18, LightGBM −0.21, CatBoost −0.15, XGB_2 −0.10, ens5 −0.10 (net −0.20 to −0.35) with realized β −0.1 to −0.4. So the result is not an artifact of the LP's corner solutions or of my constraint set.

**A correction I made along the way.** An earlier version of the decile table did not require both betas to be observed and showed a top-minus-bottom spread of about +0.5 %/month, which contradicted the decile *book*. The difference is universe: recently listed stocks (no beta history) that the model ranked low did poorly; they cannot enter any LP here, so the decile table and the IC table above use the LP-investable universe. The bucket labelled `tradeable(price>=5&mcap>=500M&$10M)` in §3 still includes those beta-less names; the last row of §3 is the LP-investable one.

## 6. Cross-model comparison

Selection is on validation rank IC only; the test numbers are reported, not used to choose:

| Model | Mean val rank IC | Test rank IC | OOS R² | Legacy IR (gross / net) | `pm_t10` IR (gross / net) |
|---|---:|---:|---:|---:|---:|
| rf3 | 0.1068 | 0.1434 | +0.225% | 1.26 / 1.09 | -0.21 / -0.31 |
| et | 0.1259 | 0.1516 | +0.154% | 0.96 / 0.81 | -0.28 / -0.37 |
| lgbm | 0.1312 | 0.1545 | +0.109% | 0.92 / 0.80 | -0.12 / -0.22 |
| cat | 0.1122 | 0.1442 | +0.176% | 0.92 / 0.78 | -0.31 / -0.41 |
| xgb2 | 0.1211 | 0.1480 | +0.140% | 0.90 / 0.76 | -0.17 / -0.27 |

- **Validation IC does not rank the models by their test portfolios.** LightGBM and ET have the highest validation IC but not the best legacy IR; RF has the lowest validation IC and the highest legacy IR. Differences are inside the noise (single seed, one path); do not read a model ranking into this batch.
- **Momentum** (legacy gross IR, `modern` with the 3 momentum factors vs `modern_nomom` without):

| Model | `modern` (15, with momentum) | `modern_nomom` (12) | Δ |
|---|---:|---:|---:|
| rf3 | 1.33 | 1.26 | -0.07 |
| et | 1.13 | 0.96 | -0.17 |
| lgbm | 1.07 | 0.92 | -0.15 |
| cat | 1.19 | 0.92 | -0.27 |
| xgb2 | 1.04 | 0.90 | -0.14 |

  Removing momentum lowers legacy IR by 0.07–0.27 in all five models (the models share data and features, so this is not five independent tests). That is inside the ±0.11 noise band for RF (0.07) and outside it (0.14–0.27) for the other four. On performance alone the financial engineer's suspicion is not supported by this data; the reason to keep momentum out is defensibility (momentum crashes, and its ranking is not explainable from fundamentals), not that it hurt here. `mispricing_perf` still embeds a momentum component.
- Every model's PM book is at or below zero, so the model choice does not matter for the tradeable result.

## 7. What this implies (not run — proposals)

1. **Report this honestly.** The evaluation checks neutrality and short-book tradeability first (`docs/FIAM.md` §14); a strategy whose entire edge is in sub-$5 / sub-$500M biotech shorts does not pass that test however good its legacy IR looks.
2. **The limiting factor is the signal, not the model or the constraints.** Five model families, momentum on/off, feature sets of 12, 15 and (XGB_2 only) 139 characteristics, training on the tradeable universe only (`_trad` arms in each model doc — legacy IR stays 0.7–1.4 but PM IR stays −0.4 to +0.2) and the FIAM baseline weighting all give about zero in the tradeable universe. More model variants on these 147 characteristics are unlikely to change that.
3. Directions that address the actual gap: signals with information in mid/large caps (for example the 8-K event layer — owned by a teammate — or alternative data available to finalists); explicit modelling of borrow cost/availability so the small-cap short edge could be sized honestly; or a long-only-tilt formulation. None of these was tested here.
4. If the small-cap edge is to be shown at all, disclose it as such: ≥ $2B floor row (RF IR 0.67 gross / 0.51 net) is the least-bad tradeable row, but it is one row, unvalidated.

## 8. Limitations

- One prediction path per model (single seed); no confidence intervals on any IR here. `experiments/rf_2/README.md` §1 gives ±0.11 as a 2-sd seed-noise band for RF at fixed config.
- Costs are assumed tiers; borrow availability is not modelled at all.
- The screens ($5, $500M), the 5%/35% sector limits and the 10% cap were pre-specified from the tips; the market-cap-floor rows are a sensitivity sweep and were not used to choose anything. The `_trad` runs and the decile book were added *after* seeing the first negative result, as follow-up diagnostics.
- Sector labels are the panel's `gics` codes as supplied (2-digit); the panel's timing of those labels is not verified.
- 68 months, of which the first two contain the Jan-2021 squeeze; single OOS path.

## Reproduce

```
.venv/bin/python experiments/rf_3/rf_3.py; .venv/bin/python experiments/et/et.py; .venv/bin/python experiments/lgbm/lgbm.py; .venv/bin/python experiments/cat/cat.py; .venv/bin/python experiments/xgb_2/xgb_2.py
.venv/bin/python experiments/pm_ablation/pm_ablation.py
```
Outputs (`output/`): `pm_ablation_summary.csv` (one row per model × portfolio), `pm_ablation_results.json`, `pm_ablation_ic_by_bucket.csv`, `pm_ablation_deciles_tradeable.csv`, `pm_ablation_sector_exposure_legacy_rf3.csv`, `pm_ablation_top_shorts_legacy_rf3.csv`, `oos_predictions_ens5_modern_nomom.csv`.
