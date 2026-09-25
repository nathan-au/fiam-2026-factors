# desk_31 — Can a local LLM reading masked 8-Ks add information? (analyst / event-triage agent)

**Design, fixed before any output was read** (`run_reader.py`, `analyze.py` docstrings):
- **Sample:** 1,775 news-bearing 8-Ks, 25 per month, filing months 2015-01..2020-11, from universe stocks (items 1.01 / 1.02 / 2.01 / 2.05 / 2.06 / 4.01 / 4.02 / 5.02 / 7.01 / 8.01, no amendments), numpy seed 0.
- **Model:** `qwen3.5:9b-q4_K_M`, local (ollama), temperature 0, seed 0, JSON output.
- **Masking:** the text lane's f1 masker (verbatim copy, `fiam_research/masking_f1.py`) plus a case-insensitive pre-pass over every filer alias in the 8-K file. The first version (d1) was found leaking filer names via the LLM's own evidence quotes, so it was fixed and all reads were redone as d2.
- **Output per filing:** event type, direction −2..+2, surprise 0..2, verbatim evidence.
- **Cost:** 4.3 h wall-clock (two clients), median 15 s per call under memory pressure. JSON valid for 97.9%.

| hypothesis | regression (filing level, month-clustered SE) | LLM coefficient (t) | bar | verdict |
|---|---|---:|---|---|
| H1 comprehension: direction vs the filing month's own return | r_same ~ direction | +0.031 (1.35) | t ≥ 3 | **fail** |
| H2 incremental over LM tone + novelty | r_same ~ direction + neg + nov | +0.030 (1.36) | t ≥ 2 | **fail** (novelty itself t −2.2) |
| H3 alpha: next-month return, controlling for B1, tone, novelty | y_next ~ direction + b1 + neg + nov | **−0.036 (−1.85)** | t ≥ 2 | **fail: wrong sign** |
| H4 risk: surprise vs next-month abs residual | absres ~ surprise + nov + ivol | +0.010 (0.45) | t ≥ 2 | **fail** (ivol t 6.1) |
| H5 look-ahead: attacker names the company (top-1, 150 filings) | – | **22%** | – | masking insufficient |
| H5 attacker guesses the year | – | 10% | uniform = 17% | no date leakage |

Direction distribution: −2: 20, −1: 85, 0: 673, +1: 880, +2: 79. **The model calls 54% of filings good news and 6% bad**; it reads corporate PR language at face value. Correctly re-identified filers include Citigroup, Ford, AIG, Gilead, Zimmer Biomet and Pilgrim's Pride: industry vocabulary, subsidiaries and unmasked people's names survive masking.

**Conclusion:**
- A 9B local LLM reading masked 8-K bodies adds **no** return or risk information beyond the deterministic text features. It barely tracks the market's same-month reaction (t 1.35, vs novelty's t −2.2).
- It also re-identifies one filer in five, so any positive result would have been contaminated by model-side look-ahead.
- Per the pre-registration, H3 failed, so **the LLM was not taken to TEST (desk_32 R4 not run)**.
- This extends the text lane's finding (93% agreement with keyword rules) from "the LLM adds nothing over rules on the judgments it makes" to "the LLM's reading has no predictive content in this universe".
- **Limits:** one small local model and one prompt; 1,775 filings; frontier models were not tested (no API in this environment). A frontier model would read better but would also *remember* more, which makes the look-ahead problem worse.
