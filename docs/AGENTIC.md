# AGENTIC.md — candidate agent architectures

Every box ends in a number, not an opinion. All comparisons are paired IC (t-stat) on identical rows, tradeable universe, after the [[NEGATIVE_RESULT|memorisation probe]]. Diagrams only; prose kept to one evidence line per architecture.

---

## 0. Rejected

```mermaid
flowchart LR
    A["Named-investor personas<br/>(Buffett agent, etc.)"] -->|"recalls real 2021-2025<br/>outcomes = look-ahead"| X["❌ not used"]
    B["50 agents, majority vote"] -->|"shared training data →<br/>ρ high → effective n ≈ 1-2"| X
```
Evidence: FIAM §10 look-ahead rule; docs N31/tree-ensemble equal-weight never beat best member.

---

## 1. Information-partitioned desks + rule-based CIO

```mermaid
flowchart TD
    subgraph Desks["Desks — disjoint information, same model"]
        D1["Chars-only desk"]
        D2["Text/8-K desk"]
        D3["Peer/industry desk"]
    end
    D1 & D2 & D3 --> AGG["CIO (deterministic)<br/>weight = validation IC<br/>shrink toward 0.5x if desks disagree"]
    AGG --> QP["Portfolio QP"]
```
Evidence: N31 diverse-feature ensembles = best member, not better; disagreement→sizing is untested, cheap.

---

## 2. Prosecutor / defense / referee (short veto)

```mermaid
sequenceDiagram
    participant P as Prosecutor<br/>(thesis: why short)
    participant D as Defense<br/>(devil's advocate)
    participant R as Referee (rule, not LLM)
    P->>R: signed thesis
    D->>R: disconfirming filings, crowding, factor exposure
    R->>R: veto if days-to-cover > X<br/>or residual IC < 0
    R-->>QP: pass / veto
```
Evidence: FIAM §9 devil's-advocate role; days-to-cover the one block that passed (t 2.67, SI>10% filter halved max DD).

---

## 3. Memorisation-probe gate

```mermaid
flowchart LR
    F["Masked filing<br/>(names/dates/$ → placeholders)"] --> M["Same LLM:<br/>guess company + date"]
    M -->|"guesses right"| DROP["drop / down-weight<br/>(= recall)"]
    M -->|"can't guess"| USE["usable for forecast"]
    SH["Shuffled-date placebo"] -.check.-> M
```
Evidence: FIAM §10 — "teams that cannot explain how they ruled out model-side look-ahead will be treated as having used it."

---

## 4. Cost-aware triage cascade

```mermaid
flowchart LR
    A["373k filings"] --> R["Regex rules<br/>(item code, keyword)"]
    R -->|"clear"| OUT1["scored, cheap"]
    R -->|"ambiguous"| S["Small local model"]
    S -->|"clear"| OUT2["scored"]
    S -->|"still ambiguous<br/>~top few %"| L["Large model"]
    L --> OUT3["scored, ~25 GPU-h"]
```
Evidence: full-corpus LLM ≈ 620-930 GPU-h vs D72's 25 GPU-h partial block.

---

## 5. Auditor guild (signal discovery, policed)

```mermaid
flowchart TD
    SD["Signal-discovery agent<br/>proposes feature code"] --> REG{"Typed causal<br/>operator registry"}
    REG -->|"only ≤t operators allowed"| LH["Leakage-hunter agent<br/>writes truncation test"]
    LH -->|"pass"| LEDGER["Hypothesis ledger<br/>(Bonferroni budget)"]
    LH -->|"fail"| KILL["❌ killed"]
    LEDGER -->|"budget spent"| STOP["no more tests"]
```
Evidence: IDEA_STATUS.md §3 — ≈185 variants tested project-wide; days-to-cover's p rises 0.015→0.23 after correction.

---

## 6. Explainer-only (math decides, LLM narrates)

```mermaid
flowchart LR
    SCORE["Model score"] --> QP["QP sets weight<br/>(no LLM in this path)"]
    QP --> TRADE["trade + facts:<br/>score, sector, β, events"]
    TRADE --> EXP["LLM writes rationale<br/>(cannot change weight)"]
```
Evidence: FIAM §9 — "explainability is a deliverable"; keeps LLM off the number that's graded.

---

## 7. FACTORS.md-sectioned desks + Delphi collaboration

```mermaid
flowchart TD
    V["Value/Size desk<br/>(§1,2)"] & Q["Quality desk<br/>(§7,10-13)"] & G["Growth/Invest desk<br/>(§3,8,9)"] & RL["Risk/Liquidity desk<br/>(§14-16)"] --> BLIND["Round 1: blind scores"]
    BLIND --> SHOW["each desk sees others' scores"]
    SHOW --> REV["Round 2: one revision"]
    REV --> CONF{"Conflict?<br/>e.g. cheap + deteriorating quality"}
    CONF -->|"yes"| VETO["downweight / short veto"]
    CONF -->|"no"| EQ["equal-weight = composite baseline"]
```
Evidence: composite_overlap Shapley — value ≈54% of composite IC, liquidity/surprise ≈0; ridge meta-model over desks overfit (IC −0.0025); IC-weighted groups worse (−0.0126).

---

## Recommended near-term stack

```mermaid
flowchart TD
    F["Filing"] --> P3["#3 Memorisation probe"]
    P3 -->|clears| P2["#2 Prosecutor/Defense/Referee"]
    P2 --> QP["Portfolio QP"]
    QP --> P6["#6 Explainer"]
    P6 --> DECK["Deck"]
```
Fits the 28 Sept freeze: touches only ~150 candidate names/month, reuses the existing masked extractor, each arm gets a pre-registered ΔIC/t-stat before it's trusted.

---

## Test protocol (applies to every architecture above)

```mermaid
flowchart LR
    A["Pre-register expected ΔIC"] --> B["Run on tradeable universe,<br/>identical rows/folds/seeds"]
    B --> C["Paired t-stat"]
    C --> D{"survives Bonferroni<br/>within-file?"}
    D -->|no| E["report as negative result"]
    D -->|yes| F["adds to model/portfolio"]
```
