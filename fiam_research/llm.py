"""Local-LLM reader of MASKED 8-K text (experiments/desk_31). No API, no network: ollama on localhost, temperature 0, fixed seed, JSON output.

Look-ahead controls:
  * the text is masked with the text lane's deterministic masker (FIAM-2026/agents/masking.py, mask_version f1): filer -> CompanyX, other listed
    companies -> Company_k (gazetteer = every company name in the 8-K file), people, absolute dates, fiscal years, large dollar amounts, locations;
  * the prompt forbids outside knowledge and asks only what the text states;
  * a separate attacker probe (same model) tries to re-identify company and period; its hit rate is reported (desk_31).
Every call is cached on disk by (model, prompt_version, doc hash) so a run is reproducible and resumable.
"""

import hashlib
import json
import re
import urllib.request
from datetime import date
from pathlib import Path

from fiam_desks import config as C

from . import masking_f1 as M  # verbatim copy of the text lane's masker (FIAM-2026/agents/masking.py, mask_version f1)

OLLAMA = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:9b-q4_K_M"  # 6 GB build: the 13 GB MLX build swaps on this 16 GB machine
CACHE_DIR = C.CACHE / "llm"
PROMPT_VERSION = "d2"  # d2: filer aliases masked case-insensitively (d1 leaked filer names), q4 model

SYSTEM = """You read one anonymized SEC Form 8-K. CompanyX is the filer; other companies are Company_1, Company_2...; people are Person_k; dates are relative
(day_-3 = three days before filing); fiscal periods are relative (Q_-1, FY_0); large dollar figures are buckets (usd_bucket_...); places are region_k.
Judge ONLY from the text. Do not use outside knowledge about any company, industry, or time period. Do not try to identify anyone.
Return ONLY this JSON:
{"event_type": "earnings|guidance|material_agreement|m_and_a|impairment_restructuring|officer_change|auditor_change|litigation|capital_return|financing|shareholder_vote|other",
 "is_routine": true|false,
 "direction": -2|-1|0|1|2,
 "surprise": 0|1|2,
 "evidence": "one short verbatim quote"}
direction = how the reported development affects CompanyX's shareholders AS THE TEXT DESCRIBES IT (-2 clearly bad ... +2 clearly good, 0 neutral/unclear/routine).
surprise = 0 scheduled/boilerplate, 1 some new information, 2 major unexpected development."""

ATTACK = """You are a forensic analyst. The text is a masked excerpt of a real SEC Form 8-K (filer = CompanyX). Use every residual cue and everything you know
about U.S. public companies to identify the filer and the calendar year of the filing. Return ONLY JSON: {"company": "name or ticker", "year": 2015}"""

_ITEM = re.compile(r"Item\s+\d\.\d\d", re.IGNORECASE)


def body(text: str, n_chars: int = 3000) -> str:
    """Skip the cover page: start at the first 'Item X.XX' heading, keep n_chars."""
    m = _ITEM.search(text or "")
    s = text[m.start():] if m else (text or "")
    return s[:n_chars]


_JUNK = re.compile(r"(/[A-Z]{2,3}/?$)|(;.*$)", re.IGNORECASE)


def alias_cores(names, tickers=()) -> list[str]:
    """Suffix-stripped cores of every name the filings file gives the filer (company, content, CRSP historical, provider, Compustat), plus tickers."""
    out = set()
    for n in names:
        if not isinstance(n, str) or not n.strip():
            continue
        n = _JUNK.sub("", n.strip()).strip(" ,.")
        prev = None
        while prev != n:
            prev = n
            n = M._SUFFIX_RE.sub("", n).strip(" ,.")
        if len(n) >= 3:
            out.add(n)
    for t in tickers:
        if isinstance(t, str) and len(t) >= 2:
            out.add(t)
    return sorted(out, key=len, reverse=True)


def mask(text: str, filer: str, filing_date: date, gazetteer: list[str], doc_id: str = "", aliases=()) -> str:
    for a in aliases:  # case-insensitive pre-pass (the f1 masker is case-sensitive and only knows one filer name)
        text = re.sub(r"\b" + re.escape(a) + r"(?:'s)?\b", "CompanyX", text, flags=re.IGNORECASE if len(a) > 4 or not a.isupper() else 0)
    return M.mask_document(doc_id=doc_id, text=text, filer_name=filer, filing_date=filing_date, company_gazetteer=gazetteer).masked_text


def _call(system: str, user: str, model: str = MODEL, timeout: int = 600, num_predict: int = 200) -> dict:
    payload = {"model": model, "stream": False, "format": "json", "think": False,
               "options": {"temperature": 0, "seed": 0, "num_ctx": 8192 if num_predict > 200 else 4096, "num_predict": num_predict},
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    req = urllib.request.Request(OLLAMA, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    txt = out["message"]["content"]
    try:
        return {"ok": True, **json.loads(txt), "_ms": out.get("total_duration", 0) / 1e6}
    except json.JSONDecodeError:
        return {"ok": False, "raw": txt[:500], "_ms": out.get("total_duration", 0) / 1e6}


def cached(kind: str, masked_text: str, model: str = MODEL) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(f"{model}|{kind}|{PROMPT_VERSION}|{masked_text}".encode()).hexdigest()[:24]
    f = CACHE_DIR / f"{kind}_{h}.json"
    if f.exists():
        return json.loads(f.read_text())
    res = _call(SYSTEM if kind == "read" else ATTACK, masked_text, model)
    f.write_text(json.dumps(res))
    return res
