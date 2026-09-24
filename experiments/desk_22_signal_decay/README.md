# desk_22 — Signal decay and rebalance speed

**Limitation:** the panel is monthly. Next-day, 5-day, 10-day and 20-day horizons cannot be measured without daily returns (not in the data; FIAM allows CRSP/Alpha Vantage, not pulled here). Horizons are h = 1..12 months with the signal fixed at formation. DEV only. Output: `ic_by_horizon_dev.csv`.

| signal | IC h=1 | h=2 | h=3 | h=6 | h=9 | h=12 | 3-month cum IC (t) | slow? |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| profitability group | +0.020 | +0.017 | +0.015 | +0.017 | +0.023 | +0.026 | +0.024 (1.8) | yes |
| JKP Quality | +0.024 | +0.021 | +0.017 | +0.014 | +0.023 | +0.024 | +0.032 (1.8) | yes |
| JKP Low Leverage | +0.020 | +0.019 | +0.020 | +0.009 | +0.015 | +0.010 | +0.033 (2.1) | no (decays by h=6) |
| value group | −0.016 | −0.021 | −0.017 | −0.005 | −0.008 | −0.001 | −0.036 (−2.0) | – |
| JKP Seasonality | −0.024 | −0.017 | −0.021 | −0.015 | −0.013 | −0.021 | −0.036 (−3.3) | wrong-signed throughout |
| dtc | −0.004 | −0.001 | −0.007 | +0.014 | +0.022 | +0.018 | −0.010 | – |
| B1 composite | +0.013 | +0.002 | −0.001 | −0.000 | +0.003 | +0.009 | +0.006 (0.3) | no |
| text novneg_max | +0.007 | −0.004 | +0.006 | −0.002 | +0.001 | +0.008 | +0.005 (1.2) | no |

Turnover budget (a proxy for rebalance speed; B1): 5% → −0.770, **10% → −0.668**, 20% → −0.589, 40% → −0.731 net IR. All paired |t| < 0.5: no evidence.

**Interpretation:**
- Profitability and quality are slow signals. Their IC at 6–12 months matches the 1-month IC, so they tolerate slow rebalancing.
- The composite's 1-month edge decays within a month, because its short-horizon components (surprise, vol, reversal) decay fast.
- Text shows no Lazy-Prices-style slow accrual in this universe.
- The rebalance speed is not a lever: costs are only ~8 bp/month.
