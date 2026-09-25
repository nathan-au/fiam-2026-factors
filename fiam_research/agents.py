"""LLM agents ON TOP of the deterministic pipeline (experiments/desk_33). Neither agent can move a position; each output is checked by a deterministic
verifier before it may be used, and every call is cached (fiam_research.llm).

  explainer        turns one rationale row (structured desk outputs for a held position) into two plain-English sentences for the deck / PM log.
                   Look-ahead: it sees only formation-month fields, no names of future events, no returns; the prompt forbids predictions.
                   Verifier: every number in the text must equal a field value (to its printed precision); the only ticker allowed is the row's own;
                   no forward-looking language.
  devils_advocate  reads the run's result JSON and writes attacks, each citing a JSON path and the value found there.
                   Verifier: the path must exist and the value must match (so an attack is traceable to data, FIAM sec 9 warning 1).
"""

import json
import re

from . import llm

EXPLAIN_SYS = """You write the trade rationale for ONE position of a quantitative market-neutral equity portfolio, for an investment committee.
You receive a JSON record of what the model's desks said about this stock at the formation date. Write at most TWO sentences, plain English.
Rules: use ONLY the fields given; copy every number exactly as printed in the record (same decimals); mention the ticker, the side (long/short) and the weight;
name the two strongest supporting factor groups and, if present, the strongest opposing one; mention short interest if given. Do NOT predict prices or returns,
do NOT use words like will, expect, forecast, outperform, likely; do NOT mention any other company or any event not in the record.
Return ONLY JSON: {"rationale": "..."}"""

DEVIL_SYS = """You are the devil's advocate on a quantitative research team. You receive the JSON results of the team's research run. Your only job is to attack
the claim that the strategy has a robust, exploitable edge. Write exactly 5 attacks. Each attack MUST cite one JSON path (keys joined by '/', list indices as
numbers) whose value supports it, and copy that value exactly. Use only the JSON; no outside knowledge.
Return ONLY JSON: {"attacks": [{"claim": "...", "path": "a/b/c", "value": ...}, ...]}"""

FORWARD = re.compile(r"\b(will|expect\w*|forecast\w*|outperform\w*|likely|predict\w*|going to|should rise|should fall)\b", re.IGNORECASE)
NUM = re.compile(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?")
TICK = re.compile(r"\b[A-Z]{2,5}\b")
ALLOWED_CAPS = {"LONG", "SHORT", "NAV", "SI", "CEO", "CFO", "USD", "US", "EPS", "IC", "LP", "PM", "JSON", "OK", "AND", "OR", "THE", "DTC", "GICS"}


def explain(record: dict) -> dict:
    return llm.cached("explain", json.dumps(record, sort_keys=True))


def _llm_with(system: str, user: str, kind: str, num_predict: int = 200) -> dict:
    import hashlib
    llm.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(f"{llm.MODEL}|{kind}|{system}|{user}".encode()).hexdigest()[:24]
    f = llm.CACHE_DIR / f"{kind}_{h}.json"
    if f.exists():
        return json.loads(f.read_text())
    out = llm._call(system, user, num_predict=num_predict)
    f.write_text(json.dumps(out))
    return out


def explain_row(record: dict) -> dict:
    return _llm_with(EXPLAIN_SYS, json.dumps(record), "explain")


def verify_explanation(text: str, record: dict) -> dict:
    vals = []
    for v in record.values():
        if isinstance(v, (int, float)):
            vals.append(float(v))
        elif isinstance(v, str):
            vals += [float(x) for x in NUM.findall(v)]
    bad_nums = []
    for n in NUM.findall(text or ""):
        x = float(n)
        if not any(abs(x - v) < 1e-9 or abs(abs(x) - abs(v)) < 1e-9 for v in vals):
            bad_nums.append(n)
    ticks = {t for t in TICK.findall(text or "") if t not in ALLOWED_CAPS and t != record.get("ticker")}
    fwd = FORWARD.findall(text or "")
    sem = semantic_errors(text or "", record)
    return {"ok": not bad_nums and not ticks and not fwd and not sem, "unsupported_numbers": bad_nums, "foreign_tickers": sorted(ticks), "forward_language": fwd,
            "semantic_errors": sem}


def semantic_errors(text: str, record: dict) -> list[str]:
    """Qualitative claims checked against the record (partial by design - catches the failure types seen in the smoke test):
    short interest called high/elevated when < 5% of shares (or low when > 10%); a factor group called strong/positive/supportive when its score < 0,
    or weak/negative when > 0."""
    errs = []
    t = text.lower()
    si = record.get("short_interest_pct_shares")
    if si is not None:
        if re.search(r"\b(high|elevated|heavy|significant)\b[^.]{0,40}short interest|short interest[^.]{0,25}\b(high|elevated|heavy)\b", t) and si < 5:
            errs.append(f"short interest called high at {si}%")
        if re.search(r"\b(low|light|minimal)\b[^.]{0,40}short interest|short interest[^.]{0,25}\b(low|light|minimal)\b", t) and si > 10:
            errs.append(f"short interest called low at {si}%")
    for k, v in record.items():
        if not k.startswith("g_"):
            continue
        g = k[2:].replace("_", " ")
        name = g.split()[0]
        if re.search(r"\b(strong|positive|supportive|favorable|high)\b[^.]{0,30}\b" + re.escape(name), t) and v < 0 and not re.search(r"\b(weak|negative|low)\b[^.]{0,30}\b" + re.escape(name), t):
            errs.append(f"{g} described as positive but score {v}")
        if re.search(r"\b(weak|negative|poor|low)\b[^.]{0,30}\b" + re.escape(name), t) and v > 0 and not re.search(r"\b(strong|positive|high)\b[^.]{0,30}\b" + re.escape(name), t):
            errs.append(f"{g} described as negative but score {v}")
    return errs


def devils_advocate(results: dict) -> dict:
    return _llm_with(DEVIL_SYS, json.dumps(results, default=str)[:12000], "devil2", num_predict=1200)


def resolve(obj, path: str):
    cur = obj
    for k in str(path).strip("/").split("/"):
        if isinstance(cur, list):
            cur = cur[int(k)]
        else:
            cur = cur[k]
    return cur


def verify_attack(results: dict, attack: dict) -> dict:
    try:
        v = resolve(results, attack.get("path", ""))
    except (KeyError, IndexError, ValueError, TypeError):
        return {"ok": False, "why": "path does not exist"}
    a = attack.get("value")
    try:
        ok = abs(float(v) - float(a)) < 1e-6 * max(1.0, abs(float(v)))
    except (TypeError, ValueError):
        ok = str(v) == str(a)
    return {"ok": bool(ok), "why": "" if ok else f"value at path is {v!r}, attack cites {a!r}"}
