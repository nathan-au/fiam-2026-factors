# Desk system infrastructure audit (DESK_00_INFRA_AUDIT)

Implementation: `desk_00_infra_audit.py` (run: `.venv/bin/python experiments/desk_00_infra_audit/desk_00_infra_audit.py`, ~20 s). Everything it writes goes to `output/results.json`.

## Objective
Before any desk is evaluated, prove that the new `fiam_desks/` package (a) reproduces the frozen composite and LP exactly, (b) reads inputs that are what they claim to be, (c) is causal and deterministic.

## Hypothesis
All seven checks pass. A failure anywhere invalidates every later desk experiment, so this file is the gate for `desk_01` onward.

## Research origin
FIAM §10 (zero tolerance for look-ahead; "Python code is submitted and will be checked"), the text lane's own guarantees G1-G6 (`cache/text_lane_2026-09-21/INTERFACE_TEXT_TO_MODEL.md`), and `docs/AGENTIC.md` (reproducibility agent). The checks are a deterministic script, not an LLM.

## Implementation
`fiam_desks/audit.py`. (1) composite + `lc_t10` on the test period vs the published numbers; (2) `ret_exc_lead1m` at (permno, eom) == `ret_exc` at (permno, next eom) on every consecutive pair; (3) SHA-256 of both text tables vs `cache/text_lane_2026-09-21/SHA256SUMS`; `txt_v1` rebuilt from the raw 8-K parquet (permno, filing_date, items, is_amendment) and compared cell by cell; `txt_v2` structural invariants; (4) our own novelty (5-word phrases vs the same firm's previous 24 months, strictly earlier filing dates) recomputed from the raw filing text for a seeded 60-permno sample, compared with the table AND with a deliberately future-inclusive variant; (5) truncation invariance: composite and two text-desk builds (incl. causal decay) rebuilt from rows with `eom <= t` at 3 random dates equal the full-data values; (6) two runs of the LP give identical holdings.

## Experimental setup
Panel = all stock-months with a next-month return (523,125). Universe = frozen large-cap universe (price >= $5, market cap >= $2B, 126d dollar volume >= $10M, both betas). Dev target months 2015-02..2020-12 (71), test 2021-01..2026-08 (68). Nothing is fitted anywhere in this file.

## Baseline
Published: composite universe IC 0.0366 (t 1.91); `lc_t10` gross IR 0.615 / net IR 0.541 (`experiments/largecap`).

## Commands
```
.venv/bin/python experiments/desk_00_infra_audit/desk_00_infra_audit.py
```

## Results
| check | result |
|---|---|
| composite reproduces | universe IC 0.03665 (t 1.91), `lc_t10` IR 0.6148 gross / 0.5411 net: **match** to 3 decimals; all 7 FIAM constraint checks pass on the book |
| target alignment | 522,431 consecutive pairs, 0 mismatches |
| text table hashes | match `SHA256SUMS` (copied from the handoff) |
| `txt_v1` vs raw 8-Ks | 241,058 raw (permno, month) cells: 0 missing from the table, **0 cell mismatches in any of the 13 count columns** |
| `txt_v2` invariants | 13/13 pass (novelty only where a filing exists, every rule count <= filings, unique month-end key, novelty in [0, 1]) |
| novelty recompute (60 permnos, 4,257 permno-months) | past-only Spearman vs table **0.951** (median abs diff 0.032); future-inclusive control **0.849** -> the table follows the past-only definition; no sign of look-ahead in novelty |
| desk truncation invariance | 10,587 values at 3 dates, max abs diff 0.0 for composite and both text builds |
| determinism | identical holdings hash on two runs |

## Interpretation
Everything passes. **Demonstrated:** `txt_v1` is exactly the raw item counts; novelty behaves like a past-only measure; our desk code is causal and deterministic. **Not demonstrated:** the exact rule regexes of `txt_v2` (`abrupt_exit`, `distress`, ...) and the Loughran-McDonald tone: their source is on the teammate's `text-agents` branch, which is not in the zip, so those columns are checked for structure only. The novelty match is 0.95 not 1.0 because our body extraction and tokenizer are approximations of the teammate's; we did not tune ours to match.

**Status: PASSED - gate open for desk_01.**

## Limitations
- 60 permnos (4,257 permno-months) for the novelty recompute, not all 3,687.
- Truncation invariance covers OUR transforms of the tables; the tables' own causality rests on the raw-count and novelty recomputes above.

## Follow-up
Re-run this file after any change to `fiam_desks/` (it is the regression gate) and before the freeze.
