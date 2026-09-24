# POST-HOC: novelty x negative as a single 8th group, both periods (DESK_12_NOVNEG_POSTHOC)

Implementation: `desk_12_novneg_posthoc.py --confirm` (~1 min). Output: `output/results.json`. **Post-hoc: suggested by test-window results already seen (desk_09/desk_11), so it cannot be adopted or counted as confirmation** (PREREGISTRATION rule 4). Ledgered as such.

## Objective
The one text signal with same-sign evidence in both periods (`novneg_max`) - is it worth a properly pre-registered follow-up? Added at weight 1 to the respective base (dev: composite; test: composite + dtc with SI cap).

## Results
| period | novneg alone IC (t) | base IC -> with | paired IC t | perm p | net IR base -> with | paired net t | max DD |
|---|---:|---:|---:|---:|---:|---:|---:|
| dev | +0.0071 (+1.63) | +0.0106 -> +0.0113 | +0.53 | 0.284 | -0.872 -> -0.859 | +0.40 | -36.9% -> -33.0% |
| test | +0.0103 (+2.87) | +0.0455 -> +0.0378 | **-2.09** | 0.104 | +0.637 -> +0.616 | -0.36 | -9.8% -> -9.4% |

## Interpretation
`novneg_max` is a genuine but tiny signal (pooled IC +0.0087, t 3.06 over 139 months) whose information is **smaller than the base's**: at equal weight it dilutes a 0.045-IC base (paired IC t -2.09) and does not repair a base that loses money. It would need a data-driven small weight, which cannot be chosen without further out-of-sample data. **Carry forward:** if new months (or a broader universe) become available, pre-register `novneg_max` at a weight <= 1/4 of a group and test it on the new data only. **Status: POST-HOC, NOT ADOPTED**
