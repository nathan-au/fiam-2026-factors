# AGENTIC.md — candidate agent architectures

Every box ends in a number, not an opinion. All comparisons are paired IC (t-stat) on identical rows, tradeable universe, after the [[NEGATIVE_RESULT|memorisation probe]]. Diagrams only; prose kept to one evidence line per architecture.

---

## Famous Investor Agents

```mermaid
flowchart TD
    F["Filings + factors"]
    F --> G["Graham"]
    F --> L["Lynch"]
    F --> B["Buffett"]
    F --> J["Bogle"]
    F --> M["Burry"]
    G & L & B & J & M --> P["Portfolio"]
```
- Graham: deep value, margin of safety
- Lynch: growth at reasonable price
- Buffett: quality moat, owner earnings
- Bogle: low-cost, broad diversification
- Burry: contrarian, balance-sheet stress

## N Agents

```mermaid
flowchart TD
    F["Filings + factors"] --> A1["Agent 1"]
    F --> A2["Agent 2"]
    F --> A3["Agent 3"]
    F --> A4["..."]
    F --> A10["Agent 10"]
    A1 & A2 & A3 & A4 & A10 --> P["Portfolio"]
```
- Agents can either all use the same model or each use a different one

## Information-partitioned desks + deterministic PM

```mermaid
flowchart TD
    D1["Factors"]
    D2["Text/8-K"]
    D1 & D2 --> PM["Portfolio manager"]
```
- Portfolio manager is deterministic
- Disagreement between desks can be used as a sizing dial

## Prosecutor / defense / judge (short trial)

```mermaid
flowchart TD
    P1["Prosecutor"] --> D1["Defense"]
    D1 --> P2["Prosecutor"]
    P2 --> D2["Defense"]
    D2 --> J["Judge"]
    J -->|"guilty"| PORT["Portfolio"]
    J -->|"not guilty"| DROP["No trade"]
```
- Round 1: Prosecutor accuses (why this stock fails), Defense defends (the stock's well-being)
- Round 2: Prosecutor and Defense each rebut once more
- Judge is deterministic: guilty if days-to-cover > X or residual IC < 0
- Guilty → short enters the Portfolio ("prison"); not guilty → acquitted, no trade

## Memorisation-probe gate

```mermaid
flowchart TD
    F["Masked filing"] --> M["LLM guesses"]
    M -->|"right"| DROP["Drop"]
    M -->|"wrong"| USE["Usable"]
```
- LLM tries to guess the company + date from the masked filing
- Guesses right → drop / down-weight (recall, not forecasting)
- Can't guess → usable for forecast

## Cost-aware triage cascade

```mermaid
flowchart TD
    A["Filings"] --> R["Regex rules"]
    R -->|"clear"| OUT1["Scored"]
    R -->|"ambiguous"| S["Small model"]
    S -->|"clear"| OUT2["Scored"]
    S -->|"still ambiguous"| L["Large model"]
    L --> OUT3["Scored"]
```
- 373k filings enter through regex rules (item code, keyword match)
- Clear cases scored cheaply and directly
- Ambiguous cases go to a small local model, then a large model only for the remaining top few %

## FACTORS.md-sectioned desks

```mermaid
flowchart TD
    V["Value/Size"] & Q["Quality"] & G["Growth/Invest"] & RL["Risk/Liquidity"] --> PM["Portfolio manager"]
```
- Value/Size desk: FACTORS.md §1,2
- Quality desk: §7,10-13
- Growth/Invest desk: §3,8,9
- Risk/Liquidity desk: §14-16

## Material triage then cost triage

```mermaid
flowchart TD
    A["Filings"] --> IT["Classifier"]
    IT -->|"routine"| SKIP["Skip"]
    IT -->|"material"| R["Cost triage"]
```
- Classifier: item-code / materiality regex (e.g. dividend decl. = routine, auditor resignation / 5.02 = material)
- Routine → scored 0, no LLM call
- Material → scored by the cost-aware triage cascade

## Single-agent collapse test

```mermaid
flowchart TD
    F["Filing"] --> ONE["Single agent"]
    ONE --> R["Referee"]
    R -->|"pass"| QP["Portfolio"]
    R -->|"veto"| DROP["Dropped"]
```
- Single agent: one prompt, one call — thesis → self-critique → verdict
- Referee is a deterministic rule, not an LLM

## Analyst-agent score as a 19th factor

```mermaid
flowchart TD
    F["Filing + chars"] --> A["Analyst agent"]
    A --> SCORE["Score"]
    SCORE --> ORTH["Orthogonalize"]
    ORTH --> RESID["Significant?"]
    RESID -->|"no"| KILL["Killed"]
    RESID -->|"yes"| ADD["Add factor"]
```
- Analyst agent: FIAM §9 use case, outputs a signed conviction score
- Orthogonalize the score vs. the 18-factor composite
- No significant residual IC → killed (same bar as any feat_* block)
- Significant residual IC → add as 19th factor, judged like composite_overlap

## FIAM's own reference architecture

```mermaid
flowchart TD
    F["8-K filing"] --> ET["Event triage"]
    ET -->|"material"| AN["Analyst agent"]
    ET -->|"routine"| SKIP["Skip"]
    AN --> FACTOR["Factor"]
    SD["Signal discovery"] --> FACTOR
    FACTOR --> DA["Devil's advocate"]
    DA --> PC["Portfolio construction"]
    PC --> WHY["Rationale"]
    WHY --> REPRO["Reproducibility"]
    REPRO -->|"clean"| SUBMIT["Submit"]
    REPRO -->|"flags issue"| FIX["Back to pipeline"]
```
- Event-triage agent: routine (e.g. dividend decl.) vs. material (e.g. auditor resignation) — only material filings reach the analyst
- Analyst agent: ticker + 8-Ks + char history → structured verdict (what happened, expected?, signed conviction)
- Signal-discovery agent proposes/codes/backtests 147-characteristic combos; multiple-testing discipline is the team's responsibility
- Devil's-advocate agent attacks the thesis: disconfirming filings, crowding, factor exposures that explain the alpha away
- Portfolio-construction agent: forecasts → positions (100-500 names, 200% gross, ±50% net, turnover budget, sector limits)
- Rationale: writes down why each trade was made (explainability is a deliverable)
- Reproducibility agent re-runs the pipeline hunting for look-ahead bias, survivorship, leakage — clean → submission, flags issue → back into pipeline
