# AGENTIC_WORKFLOW.md — where LLMs / agents are (and are not) in the pipeline, with evidence

FIAM §9 lists six agent roles and two warnings: "agents are fluent whether or not they are right", and "agents crowd". This file maps each role to what the project built, what model it used, what it could see, how look-ahead was controlled, and the measured result. **No LLM output moves capital in the submitted system (B1).** Where an LLM was tested as a signal source, the test is pre-registered and reported whatever the result.

| FIAM §9 role | implementation | model | sees | look-ahead control | result | in the submission? |
|---|---|---|---|---|---|---|
| **Signal-discovery agent** | the research loop of desk_13–33: the agent proposes hypotheses (desk_16 literature), writes the experiment code, runs it on DEV, logs every variant, applies FDR / DSR / PBO, and pre-registers TEST looks | Claude Code (Opus 5.5, frontier) | panel, text tables, backtester, ledgers; **TEST only through pre-registered scripts** | decisions by mechanical rules on DEV. TEST sealed behind `PREREGISTRATION.md` + `desk_test_ledger.csv`. Residual: the agent *knows* 2015–2026 market history (e.g. the 2022 growth crash) and could steer hypothesis choice, which the protocol limits but cannot eliminate | ~70 DEV trials; PBO 0.49; its top DEV idea (SI cap 5%, the only FDR survivor) **reversed on TEST** (t −2.0); nothing it found beat B1 on TEST | the *process* (ledgers, pre-registrations, READMEs) is the deliverable |
| **Analyst / event-triage agent** | desk_31: masked 8-K reader (event type, direction −2..+2, surprise 0..2, verbatim evidence) | qwen3.5 9B q4, local (ollama), temperature 0, seed 0 | one masked 8-K body (3,000 chars) | deterministic masker (text lane f1) + filer-alias pre-pass; prompt forbids outside knowledge; attacker probe measures re-identification | **no signal**: comprehension t 1.35, next-month t −1.85 (wrong sign), risk t 0.45; attacker re-identifies **22%** of masked filers | no |
| **Devil's-advocate agent** | desk_11 (deterministic attacks, criteria fixed in code) + desk_33 D (LLM attacks, each must cite a JSON path whose value is verified) | script / qwen3.5 9B | result JSONs only | no market data at all | LLM: 4/5 attacks path-traceable, ~1/5 actually supported by the cited value. Deterministic desk_11/28 found the real weaknesses | desk_11 / desk_28 in the deck; desk_33 D as a candid appendix example |
| **Portfolio-construction / explainability agent** | construction is deterministic (LP, desk_20 showed no QP beats it); desk_33 E: LLM writes two-sentence rationales from the structured rationale row, verified for numbers, tickers, forward language and qualitative correctness | qwen3.5 9B | one rationale row (formation-month desk outputs) | no returns or dates in the input; forward-looking words rejected | 87% pass the verifier; ~10–20% wrong in meaning (e.g. 0.5% SI called "high", COKE rendered as "Coke (KO)"); 0 invented numbers | no: the deterministic template reason is used |
| **Reproducibility agent** | desk_00 regression gate, desk_15 audit, truncation / determinism checks in desk_10/29 | deterministic scripts | code + data | – | found the delisting look-ahead (desk_15) | yes |

## Why the capital-moving layer stays deterministic
1. **Evidence.** The text lane's LLM agreed with a one-line keyword rule 93% of the time on the judgments it made (`handoff/reports/agent_vs_rules.md`). desk_31 tests whether reading the text adds return information beyond rules and dictionaries.
2. **Look-ahead.** A model trained after 2021 has seen the outcome. Masking reduces but does not remove re-identification (desk_31 H5 measures it). Every LLM feature therefore carries a residual look-ahead risk that the deterministic features do not.
3. **Cost / reproducibility.** Around 8 s per filing locally means ~370k filings ≈ 35 days of compute. A sample-based test is the most that is honest here.

## Measured verdict (desk_31, desk_33)
| question | answer | evidence |
|---|---|---|
| Can an LLM add alpha by reading 8-Ks? | **No** (with a local 9B model, masked text) | desk_31 H1–H4 all fail; H3 has the wrong sign |
| Can masking remove model-side look-ahead? | **Not fully** | 22% of filers re-identified top-1 after an improved mask; year not identified (10% vs 17% chance) |
| Can an LLM explain positions? | Fluently, but **10–20% wrong in meaning** | desk_33 E, manual audit |
| Can an LLM play devil's advocate? | It produces traceable-looking attacks that its own evidence contradicts | desk_33 D |
| Where did "agentic" work actually pay off? | **The research loop itself** (signal discovery with multiple-testing discipline, reproducibility audits), run by a frontier coding agent with ledgers and pre-registration | found the delisting look-ahead (desk_15); refuted dtc (desk_14); caught its own best DEV idea reversing on TEST (desk_27) |

## What to put in the deck (FIAM §9 asks for architecture, visibility, tools, step limit, model, look-ahead control, prompts)
1. **Signal-discovery agent = the research loop.** Tools: Python over the panel and text tables, the LP backtester, the research ledger, a TEST-look ledger. Step budget: desk_13–33 (~100 ledgered variants). Look-ahead: TEST reachable only through pre-registered scripts (2 looks, both logged; the second Bonferroni-penalised). Candid result: PBO 0.49 on DEV; the only FDR survivor reversed on TEST; nothing beat B1. This is exactly "how the top-ranked idea holds up in a period it never saw".
2. **Event-triage / analyst agent = a documented negative result.** Prompt (`fiam_research/llm.py` SYSTEM, prompt_version d2), masker, the 22% re-identification rate, and the pre-registered tests. It shows how model-side look-ahead was measured, not just asserted.
3. **Explainability agent.** Show the verifier and one caught error (the COKE → "Coke (KO)" substitution) as the argument for why the submitted rationale table is template-generated.
4. **Reproducibility and devil's-advocate agents** are deterministic scripts (desk_00, desk_11, desk_15, desk_28). Say why: they must be right, not fluent.

Prompts and tool definitions for the appendix: `fiam_research/llm.py` (SYSTEM, ATTACK), `fiam_research/agents.py` (EXPLAIN_SYS, DEVIL_SYS), the text lane's `FIAM-2026/agents/prompts/*.md`.
