# desk_33 — LLM agents on top of the deterministic pipeline (explainability, devil's advocate), with verifiers

Both agents use the local `qwen3.5:9b-q4_K_M`, temperature 0, cached (`fiam_research/agents.py`). Neither can change a position.

## E. Explainability agent (60 B1 positions: top 5 long + top 5 short in 6 TEST formation months)
Input: one structured rationale row (group scores, composite, SI, days-to-cover). Output: two sentences.
Verifier:
- every number must come from the record;
- only the row's own ticker may appear;
- no forward-looking words;
- qualitative checks on short-interest adjectives and group-sign words.

| check | result |
|---|---:|
| passes verifier | 87% (52/60) |
| invented numbers | 0% |
| forward-looking language | 0% |
| wrong ticker / outside knowledge | 1 case: "long **Coke (KO)**" for COKE (Coca-Cola Consolidated): the model substituted a different company from memory |
| qualitative errors flagged | 11.7% (7). Manual audit: **6 true errors** (e.g. SI of 0.5% / 2.6% / 4.4% called "high", quality −0.09 called "high"), 2 checker false positives (the `volatility_beta` group is signed so + = low vol) |
| errors the verifier missed (manual audit of 10 passing rows) | ≥ 1 (a correct number attributed to the wrong factor group) |
| median latency | 9.3 s |

**Verdict:** the LLM adds fluency but **~10–20% of rationales are wrong** in meaning while being right in numbers. The deterministic template reason in `rationale.csv` is faithful by construction and carries the same information. **Use the template for the deliverable.** LLM prose only with the verifier *and* human review; it is not worth it for a 200-name book.

## D. Devil's-advocate agent (reads the desk_27 + desk_32 result JSONs)
The first attempt was truncated at 200 output tokens (a harness limit) and returned the wrong schema. It was re-run once with 1,200 tokens (`D_note` in results.json).

| | result |
|---|---|
| attacks | 5 |
| path exists and cited value matches ("traceable") | 4/5 |
| claim actually supported by the cited value | **~1/5**. E.g. "fails to maintain profitability" cites combined Sharpe **+0.61**; "declining and negative IC in later years" cites **+0.034** |

**Verdict:** mechanical traceability is necessary but not sufficient. The LLM produced fluent attacks whose evidence contradicts them, which is FIAM §9's warning 1 in miniature. The deterministic devil's advocate (desk_11: style R² 0.75, fading IR; desk_28: the growth-vs-value flip) found the real weaknesses.
