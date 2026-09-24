# DESKS.md — Factors desk + Text desk + Deterministic PM (implemented)

Implements the "information-partitioned desks + deterministic PM" architecture of `docs/AGENTIC.md`. Package: `fiam_desks/`. Experiments: `experiments/desk_00` … `desk_12` (one README each). Nothing here uses an LLM; every decision is a fixed rule of desk outputs, so any position can be re-derived by hand from `rationale.csv`.

```
Factors desk (frozen composite 7 groups + days-to-cover) ─┐
Text desk (txt_v1/txt_v2.1 rules, novelty, tone) ─────────┼─► Deterministic PM (pm.py decide) ─► LP (lp.py, lc_t10 + SI cap) ─► holdings / returns / rationale
                                                           │        modes: factors_only | blend | veto_long | tilt | dial | judge | agree_dial
Audit + devil's advocate scripts (audit.py, desk_11) ─────┘
```

## 1. Package map
| file | role |
|---|---|
| `config.py` | frozen constants: universe, factor groups + signs, LP config, periods, published reference numbers, hashes |
| `panel.py` | the stock-month table (523,125 rows), frozen rank transform, universe mask, dev/test period labels, `truncated(t)` for look-ahead audits |
| `factors_desk.py` | `composite()` (frozen), `add_groups()` (extra equal-weight groups), `urank()`; `DeskOutput` contract |
| `text_desk.py` | loads the teammate's tables (SHA-256 verified), signal registry with pre-registered signs, `desk()` builds score + adverse flag + coverage |
| `si.py` | FINRA short-interest ratio and days-to-cover from the local cache (2020-06 onward) |
| `pm.py` | `decide()`: desk outputs -> LP inputs (`pred`, `veto_long`, `veto_short`, `w_mult`); `cfg_with_si_cap()` |
| `lp.py` | the frozen LP + per-row vetoes / weight multipliers / short cap / extra neutrality; performance and cost statistics are imported from `experiments/largecap` (not copied) |
| `evaluate.py` | IC, paired tests, residual IC, permutation null, kill/pass rule, shuffle placebo |
| `audit.py` | reproducibility / look-ahead agent: target alignment, txt_v1 vs raw 8-Ks, txt_v2 invariants, independent novelty recompute, truncation invariance, determinism |
| `ledger.py` | append-only log of every test-window look (`experiments/desk_test_ledger.csv`) |
| `system.py` | assembles the default system; writes FIAM-format holdings, returns and the rationale table |

**Desk output contract:** `score` in [-1, 1], + = long, 0 = "no view" (never NaN); optional diagnostics (`groups`, `agreement`, `flag`, `coverage`). Values at (permno, eom) may use only data dated <= eom.

## 2. Reproduce
```
.venv/bin/python experiments/desk_00_infra_audit/desk_00_infra_audit.py     # regression gate, ~20 s: run after ANY change to fiam_desks/
.venv/bin/python experiments/desk_10_final_system/desk_10_final_system.py   # default system -> experiments/desk_10_final_system/output/{holdings,returns,rationale}.csv
```
Text tables: `cache/text_lane_2026-09-21/` (copy of the handoff, `SHA256SUMS` checked on every load). Software versions are pinned in `requirements.txt`.

## 3. Protocol (why the results can be trusted, and where they cannot)
The composite and the text signals are fit-free, so 2015-02..2020-12 (71 target months, **DEV**) is a clean development window; 2021-01..2026-08 (68 months, **TEST**) is the FIAM scoring window. All design decisions were made on DEV. TEST was read only in `desk_09` (arms and decision rules fixed in `desk_09/PREREGISTRATION.md` before it ran), in the descriptive `desk_11`, and in the labelled post-hoc `desk_12`; every test-window arm is in `experiments/desk_test_ledger.csv` (19 rows). Caveats: the frozen composite, days-to-cover and the SI cap had already been evaluated on TEST by earlier experiments; the number of DEV looks across desk_01..08 is about 45 (Bonferroni is applied only within each file).

## 4. Evaluation criteria (defined before the runs; results in the READMEs)
| target | test | result |
|---|---|---|
| infrastructure | reproduce published composite/LP; text tables vs raw 8-Ks; causality; determinism | **works** (desk_00, desk_10) |
| Factors desk | dev/test IC, LP IR, candidate additions under the repo kill/pass rule | anchor works on TEST, loses on DEV; candidates all killed/inconclusive (desk_05/07) |
| Text desk | standalone IC on the tradeable universe, additive gain, roles (persistence, gating, tail), size band, placebo | **does not work** as alpha on DEV (desk_01-04); `novneg_max` is a small real signal but dilutes (desk_09/12) |
| Integration | PM-score IC, LP net IR, paired net-return t, constraints, placebo, bootstrap CI | works mechanically; no text mode adds detectable value (desk_08/09/11) |
| Risk | style attribution, time stability, concentration, regime | **two of four criteria fail** (desk_11) |

## 5. What the system uses (settled)
Factors desk: frozen 18-factor composite + days-to-cover 8th group. PM: `lc_t10` LP + short-interest-ratio cap 10%. Text desk: **advisory only** (in `rationale.csv`, no capital effect). Test net IR 0.637 (gross 0.721; bootstrap 90% CI [0.14, 1.16]), beta +0.02, max DD -9.8%; the same system on DEV: net IR -0.76.

## 6. What worked / partly / not / uncertain
- **Demonstrably works:** the harness (exact reproduction of published rows); the audits; the PM layer's constraint handling; the rationale table; the finding that the text lane's IC is real on all stocks conditional on a filing and vanishes on the tradeable universe (desk_01/02).
- **Partly works:** `novneg_max` (pooled IC +0.0087, t 3.06 over 139 months; replicated on TEST) - real but too small to help a 0.045-IC base; `veto_long` / `judge` (+0.09 / +0.02 net IR on TEST, CIs include 0).
- **Does not work:** text as a ranking group at any floor; persistence, news-gating, rare-event veto; momentum, long-term reversal, sector-neutral ranking; deeper neutralisation; `dial`, `agree_dial`, `tilt` (dev-to-test reversal).
- **Uncertain:** whether the TEST-window performance is a stable premium (it fades after mid-2023, the same construction lost in 2015-2020, and ~75% of its variance is explained by value/quality/profitability proxies); the assumed costs; single-path noise (IR s.e. ~0.46).

## 7. Decisions that changed during iteration
1. Text was first planned as a ranking group / overlay on the frozen universe; desk_01/02 showed its edge sits in names failing the price/liquidity screens, so the universe-floor and role were revisited (desk_02, 03, 04) before any PM integration.
2. The PM was expected to repair the factors desk (neutral dial): desk_06 showed the DEV loss is the signal's, not the LP's; the idea was dropped.
3. Factors-desk mining stopped after desk_07 (five ideas killed or inconclusive) to avoid data-dredging.
4. The candidate list for the confirmation was corrected to match the stated rule (only `tilt`; `judge` runs but cannot be adopted), before the test run.
5. The default was decided by the pre-registered rule, not by the best-looking TEST row (veto_long has the highest net IR 0.728 but is not adopted).

## 8. Limitations / next steps
- The system is a regime-dependent style portfolio; the honest deck statement is "IR 0.64 (CI 0.14-1.16) on 2021-26, -0.76 on 2015-20, alpha t 1.07 after style proxies".
- `txt_v2` rule regexes and the Loughran-McDonald tone are the teammate's code (on branch `text-agents`), verified structurally only.
- Costs and borrow are assumptions; no borrow-availability data.
- Next: (a) pre-register `novneg_max` at a small weight and test on new months only; (b) `MAIN.py` generation from `fiam_desks/`; (c) a daily-mark risk report (FIAM section 11); (d) do not spend the remaining time on further text or PM variants - the evidence says the marginal value is below the noise floor.
