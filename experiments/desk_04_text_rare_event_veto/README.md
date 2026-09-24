# Rare severe 8-K events as a long-side veto (DESK_04_TEXT_RARE_EVENT_VETO)

Implementation: `desk_04_text_rare_event_veto.py` (seconds). Dev only. Re-run after cleanup: identical results.json.

## Objective / why
FIAM section 4 calls 4.01/4.02 "two of the strongest distress signals". Too rare for a ranking group, but a veto only needs them to select tail losers among names the composite wants to be long. Flag: any of items 4.02, 4.01, 2.06, 2.05 in month t (and a 3-month persistence version). Statistic: month-clustered flagged-minus-unflagged tail rate (y <= -15%) and mean return.

## Results (dev)
| set | flagged/month | tail flagged | tail unflagged | diff (t) | mean-ret diff (t) |
|---|---:|---:|---:|---:|---:|
| severe_t, universe | 6.6 | 3.6% | 4.4% | -0.8pp (-0.76) | +0.51% (+1.03) |
| severe_t, composite top quintile (long candidates) | 3.5 (4 months ok) | 0.0% | 0.9% | n/a | n/a |
| severe_3m, universe | 18.1 | 4.3% | 4.5% | -0.2pp (-0.42) | +0.45% (+1.58) |
| severe_3m, long candidates | 3.9 (37 months ok) | 4.9% | 2.5% | +2.4pp (+1.34) | -0.23% (-0.35) |
| severe_3m, short candidates | 5.4 (64 months ok) | 6.1% | 8.4% | **-2.3pp (-2.06)** | +0.29% (+0.43) |

## Interpretation
**Failed:** in the tradeable universe severe-event names do NOT have fatter left tails; if anything the reverse (bad news is priced immediately in >= $2B names). The only |t| > 2 cell says flagged names among SHORT candidates lose *less*, i.e. the veto idea would apply to shorts, not longs (one cell of six; not adopted). **Carry forward:** rare events stay out of the PM; they appear in the rationale table only via the desk-level flags. **Status: IMPLEMENTED_BUT_FAILED**

## Limitations
Long-candidate cells have 3-4 flagged names/month; underpowered.
