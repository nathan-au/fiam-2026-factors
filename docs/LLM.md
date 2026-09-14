# 8-K Text Signal — Investigation and Plan

This is a planning document, not an implementation — nothing described here
is built yet. It records what was investigated about `fiam/8k_20150101_20260831_identified.parquet`
and the numerical panel's join constraints, the options considered for turning
8-K text into a predictive signal, and the concrete next steps chosen.
Context: `docs/FIAM.md` (competition spec, esp. §4, §8-9, §10), the existing
baseline in `ols.py` / `docs/OLS.md`, and `fiam/readme.md` (dataset guide).

---

## 1. Why 8-Ks, and where they fit

`ols.py` is a pure numeric-characteristics OLS baseline. The competition brief
explicitly wants that baseline extended, and its own §8-9 point at 8-K text
and agentic/LLM extraction as the intended next layer. The dataset guide
(`fiam/readme.md`) and `docs/FIAM.md` §4 describe the filing archive; this
document works out how to actually use it.

## 2. Data facts established during investigation

- `fiam/8k_20150101_20260831_identified.parquet`: 373,139 filing rows, 3,687
  distinct PERMNOs (of 6,561 in the characteristics panel — most stock-months
  have no 8-K at all; this is an augmentation to the full panel, not a
  text-only universe).
- Key columns: `permno`, `filing_date`, `items` (list of 8-K item codes),
  `text` (cleaned filing body), `word_count`, `is_amendment`.
- Item-code frequency: `9.01` (routine exhibit, 296k), `2.02` (earnings,
  120k), `7.01`/`8.01` (Reg FD/other events, 88k/84k), `5.02` (officer/director
  changes, 71k), `1.01` (material agreements, 48k), down to the rare-but-loaded
  distress signals `4.01`/`4.02` (auditor change/non-reliance, 2,190/698).
- Filing frequency per permno-month: 148k months have exactly 1 filing, 65k
  have 2, and a long tail up to 10+ — any feature needs an aggregation rule
  across multiple same-month filings.
- Text length: median 595 words; right tail out to 183k words (naive
  averaging needs to guard against a handful of huge filings dominating).
- **Item-code filtering alone barely reduces volume**: 90.4% of all filings
  carry at least one "material" item code (`2.02/1.01/1.02/2.01/2.05/2.06/
  5.02/4.01/4.02/7.01/8.01`). "Skip the routine ones" doesn't shrink the
  corpus much by itself.
- Two filters that *do* shrink it meaningfully, combined:

  | Filter | Filing count |
  |---|---|
  | All filings | 373,139 |
  | Restricted to permnos ever investable (`dolvol_126d ≥ $10M`, matches the existing OLS screen) | 289,133 |
  | + auditor/M&A/impairment only (`4.01/4.02/2.01/2.05/2.06`) | **6,851** |
  | + material agreements (`1.01/1.02`) | 45,529 |
  | + exec turnover (`5.02`) | 101,417 |

- Local compute reality check: machine is an Apple M3 / 16GB RAM. `ollama` is
  already installed with `qwen3.5:9b-mlx`, `qwen3.5:9b-q4_K_M`, and `gemma4`
  pulled. No `sentence-transformers`/`torch`/embeddings library installed yet,
  and no LLM API key configured — **the plan below assumes local-only
  inference**, no hosted API. A 9B local model at this scale realistically
  runs full-corpus (373k) structured extraction in days, not hours, hence the
  staged item-code + investability filtering above.

## 3. Timing / join convention (must mirror the existing OLS pipeline)

`ols.py` keys everything off `eom` (month *t*) → `target_month` (*t*+1),
trading at the start of *t*+1. Any 8-K feature must slot into the same
convention:

- Aggregate all filings with `filing_date` in calendar month *t* into one
  feature vector for `(permno, t)`, joined the same way the 147 characteristics
  are keyed. Since `eom` is month-end, any filing dated within month *t* is by
  construction available before the month *t*+1 trade — this alignment is
  safe under the existing convention.
- `filing_date` is a provider date, not a verified SEC-acceptance timestamp,
  and `filing_timestamp_utc` is a midnight placeholder (`fiam/readme.md` §3) —
  don't attempt intraday/same-day cutoffs; a monthly-bucket join is the
  conservative convention `docs/FIAM.md` asks to state explicitly.
- Stock-months with no filing stay in the panel with zeroed/null text
  features — `docs/FIAM.md` §4 is explicit that stock-months without a filing
  must be kept unless the strategy deliberately restricts its universe.
- **Model-side look-ahead** (`docs/FIAM.md` §10): a frontier/local model may
  recognize a company and recall what actually happened to its stock
  afterward, contaminating any "conviction score" with hindsight rather than
  genuine analysis of the filing text. Mitigation: strip `ticker`/
  `company_name` from any extraction prompt, and as a robustness check, run a
  redacted-vs-unredacted sample comparison — a meaningful difference is
  evidence of hindsight leakage and should be reported either way.
- **A leakage nuance specific to text features**: a raw per-document feature
  (an embedding vector, an LLM-extracted field) is a deterministic function of
  that one document — safe to compute once for the whole corpus and cache,
  like `word_count`. Any step that is *fit* across documents (PCA/PLS
  loadings, KMeans centroids, an embedding-to-return regression) must instead
  be refit inside each walk-forward fold using only that fold's
  training-period text, exactly like the OLS coefficients themselves. This is
  the most likely place to accidentally introduce look-ahead bias in this
  path.

## 4. Options considered

### Tier 1 — structured/metadata features (no NLP)
Per-permno-month filing count, item-code dummies/counts (especially
`4.01/4.02`, `5.02`, `2.01/2.05/2.06`), `is_amendment`, `log(word_count)`,
days since last filing. Cheapest, most defensible, bolts directly onto the
existing 147-column design matrix. (Not the focus of this document, but the
natural first step if starting over.)

### Tier 2 — lexicon-based sentiment
Loughran-McDonald finance word lists (or FinBERT sentiment) scored against
`text`, aggregated per month. No embedding/LLM infra needed.

### Tier 3 — embeddings (chosen focus)
Turn each filing's text into a fixed-length vector, aggregate to one vector
per stock-month, feed into the model as extra characteristics.

- **Sentiment/tone score** (FinBERT or Loughran-McDonald) — 1-3 dims, cheapest
  and most interpretable of the embedding-adjacent options.
- **General sentence embedding + supervised reduction** — encode with a local
  `sentence-transformers` model (no API needed, CPU-feasible at this volume:
  full 373k corpus should encode in well under an hour on this machine).
  Raw embeddings (384-768 dims) are far too many features for the available
  training history and would overfit immediately, so reduce via **PLS
  regression against training-period returns** (not plain PCA) to get a
  handful (2-5) of dense "text factors" that are actually correlated with the
  target. Must be refit per walk-forward fold (§3, leakage nuance above).
- **Unsupervised clustering as a cheap event-tagger** — KMeans the embeddings
  into ~20-50 clusters, use each cluster's historical mean subsequent return
  as a feature. No LLM required. Doubles as an optional triage signal for
  tier 4 (§5) by flagging filings that sit far from a "boilerplate centroid,"
  independent of item code.

### Tier 4 — LLM / agentic extraction (chosen focus)
Prompt a local model with a filing's text (+ item codes), get back a
structured record — a content summary, not a return forecast — that becomes a
feature.

- **Single-pass structured extraction** — one prompt per filing →
  `{event_type, surprise: bool, severity: 0-1, signed_tone: -1..1,
  one_line_summary}`. `event_type` → one-hot/dummy, `severity`/`tone` →
  continuous features, joined like the tier-1 item-code dummies.
- **Two-stage triage → analyst** (`docs/FIAM.md` §9's own suggested pattern)
  — a fast, short-output pass (routine vs material, or a single 0-1 severity
  float, on a truncated input) run over a much larger slice; only the flagged
  fraction gets the full extraction prompt. Cheaper per document since output
  is ~5 tokens instead of ~150.
- **Devil's-advocate pass** — a second, *different* local model critiques the
  first's conviction score (e.g. `gemma4` critiques `qwen3.5`'s output, or vice
  versa). Better-motivated than same-model self-critique and "free" since both
  are already pulled locally. Higher effort; a stretch goal.
- **Fundamentals framing instead of returns** (`docs/FIAM.md` §7) — ask
  "does this filing imply an earnings/revenue surprise next quarter" instead
  of "will the stock go up," which reframes the model's job as reading
  disclosed facts rather than prophesying markets and sidesteps the
  model-side-look-ahead concern more cleanly.

## 5. Chosen direction

Build tier 3 (embeddings) and tier 4 (LLM extraction) **in parallel as
independent feature sets**, both feeding the same augmented-OLS design matrix
as extra columns (same integration pattern as tier 1's item-code dummies).
Not merged into a single dependent pipeline, though tier 3's KMeans clusters
are available as an optional triage signal for tier 4 if the staged rollout
below needs a bigger net (§4, tier 3 clustering note).

### Tier 4 staged rollout (volume-driven, see §2 table)

1. **Stage 1 — validate cheap, ~6.9k filings.** Full structured extraction on
   auditor-change/M&A/impairment filings (`4.01/4.02/2.01/2.05/2.06`)
   restricted to the ever-investable universe. Highest-conviction category
   `docs/FIAM.md` itself flags ("two of the strongest distress signals in the
   corpus"), and small enough to run overnight. Purpose: get an early read on
   whether the signal is there before spending more compute.
2. **Stage 2 — expand if Stage 1 shows lift.** Add material agreements
   (`1.01/1.02`) → ~45.5k total.
3. **Stage 3 — stretch.** Add exec turnover (`5.02`) → ~101k; switch to a
   cheaper extraction prompt (shorter expected output) and/or the faster
   `qwen3.5:9b-mlx` build, run as a multi-day background job.
4. **Left to tier 3 instead of tier 4**: `7.01/8.01` (Reg FD/other events,
   ~172k combined) and `2.02` (earnings, 120k) are too voluminous for full LLM
   treatment and are also where a cheap sentiment/embedding score is a
   reasonable substitute — earnings-adjacent text is exactly what
   FinBERT-style tone scoring targets.

### File/feature layout (planned)

Both tiers are one-time, cacheable precompute (§3, leakage nuance — raw
per-document features don't need refitting per fold):

- `fiam/8k_embeddings.parquet` — keyed by `document_id`, one row per filing,
  raw embedding vector + metadata.
- `fiam/8k_llm_extract.parquet` — keyed by `document_id`, structured fields
  from whichever stage(s) have been run.
- A separate aggregation step (permno + month) joins both into the existing
  `ols.py` model table as extra `stock_vars` columns.

### Evaluation plan

Same ablation logic as the existing baseline: OOS R² and portfolio IR with vs.
without each feature block (tier 3 alone, tier 4 alone, combined), so the deck
can show whether either tier — or their combination — actually earns its
place. A tier that doesn't move R² is a legitimate, reportable result per
`docs/FIAM.md` §9's own framing ("a candid account of an agent that did not
work is worth more than a polished account of one that supposedly did").

## 6. Open decisions / not yet resolved

1. Exact embedding model choice for tier 3 (a local `sentence-transformers`
   model needs to be picked and installed — not yet decided which).
2. Exact extraction prompt/schema for tier 4 Stage 1 — not yet drafted.
3. Whether the two-stage triage-then-analyst pattern (§4) is worth building
   before or after Stage 1's item-code-filtered validation run.
4. Whether the devil's-advocate pass is in scope given time budget, or a
   stretch goal only if Stage 1/2 show enough lift to justify it.
5. Redacted-vs-unredacted look-ahead robustness check (§3) — designed here,
   not yet run.
