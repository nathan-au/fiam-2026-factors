# VERBATIM COPY of FIAM-2026/agents/masking.py (text lane, branch nathan-refactor @ e35fcbc); sha256 83b65557e2e82cfc. Do not edit: copied so fiam_research runs without the other repo.
"""agents/masking.py — deterministic masking of 8-K text before any LLM sees it.

Spec: docs/AGENT_MASKING.md §2 (mask_version f1). Adapted from the QUARCC
`LLM relation graph` project (v2 §5c) and KTD-FIN's data-side masking protocol.

WHAT IS MASKED
  filer                     -> CompanyX (name, name without legal suffix, "the Company", "the Registrant")
  other gazetteer companies -> Company_1, Company_2, ... consistent within a document (first-appearance order)
  tickers / exchange tags   -> removed        "(NYSE: ACM)", "Nasdaq: ACM"
  CIK / file / accession    -> removed
  people                    -> Person_1, ... ("Mr./Ms./Mrs./Dr. Surname", plus an optional person gazetteer)
  absolute dates            -> day_{k}        relative to filing_date (day_-4 = four days before filing)
  quarters / fiscal years   -> Q_{k} / FY_{k} relative to the filing quarter / year
  bare years                -> FY_{k}
  dollar amounts >= $1M     -> usd_bucket_*   coarse buckets (scale figures re-identify a firm)
  locations                 -> region_{k}     US states, common cities, countries (gazetteer, extendable)
  brands / products         -> product_{k}    only via an explicit gazetteer (residual risk measured by the probe)

WHAT IS KEPT VERBATIM
  percentages, ratios, small counts, item codes (2.02, 4.01 ...), exhibit references, generic words.

PROPERTIES (tests/test_masking.py, SYNTHETIC companies only)
  * no gazetteer name, ticker, or absolute date survives
  * same entity -> same placeholder throughout the document
  * percentages survive
  * idempotent: mask(mask(x)) == mask(x)
  * the alias map never travels with the masked text (it is a separate field, stored separately)

NO LLM IN THE MASKER. Pure function of (text, gazetteers, filing_date, mask_version).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

MASK_VERSION = "f1"

# ----------------------------------------------------------------------------- gazetteers
LEGAL_SUFFIXES = [
    "Incorporated", "Inc\\.?", "Corporation", "Corp\\.?", "Company", "Co\\.?", "Limited", "Ltd\\.?",
    "LLC", "L\\.L\\.C\\.", "LP", "L\\.P\\.", "plc", "PLC", "N\\.V\\.", "S\\.A\\.", "SA", "AG", "SE", "Holdings?",
    "Group", "Trust", "Bancorp", "Technologies", "International",
]
_SUFFIX_RE = re.compile(r"[,\s]+(?:" + "|".join(LEGAL_SUFFIXES) + r")\s*$", re.IGNORECASE)

US_STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida",
    "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine",
    "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska",
    "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio",
    "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas",
    "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming", "District of Columbia",
]
US_CITIES = [
    "New York City", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego",
    "Dallas", "San Jose", "Austin", "Jacksonville", "San Francisco", "Columbus", "Charlotte", "Indianapolis",
    "Seattle", "Denver", "Boston", "Nashville", "Detroit", "Portland", "Las Vegas", "Memphis", "Louisville",
    "Baltimore", "Milwaukee", "Albuquerque", "Tucson", "Fresno", "Sacramento", "Atlanta", "Miami", "Minneapolis",
    "Cleveland", "Pittsburgh", "Cincinnati", "St. Louis", "Kansas City", "Raleigh", "Omaha", "Oakland", "Tampa",
    "Orlando", "Salt Lake City", "Cupertino", "Redmond", "Mountain View", "Menlo Park", "Palo Alto", "Santa Clara",
    "Sunnyvale", "Irvine", "Duluth", "Cambridge", "Princeton", "Stamford", "Greenwich", "Hartford", "Wilmington",
]
COUNTRIES = [
    "United States", "U\\.S\\.A\\.", "Canada", "Mexico", "United Kingdom", "England", "Ireland", "Germany", "France",
    "Netherlands", "Switzerland", "Sweden", "Israel", "China", "Hong Kong", "Taiwan", "Japan", "South Korea",
    "Korea", "India", "Singapore", "Australia", "Brazil", "Bermuda", "Cayman Islands", "Luxembourg",
]

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December|" \
         "Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
_MONTH_NUM = {m.lower(): i % 12 + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
     "november", "december"])}
_MONTH_NUM.update({"jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9,
                   "oct": 10, "nov": 11, "dec": 12})

_DATE_LONG = re.compile(r"\b(" + MONTHS + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b")
_DATE_DMY = re.compile(r"\b(\d{1,2})\s+(" + MONTHS + r")\.?,?\s+(\d{4})\b")
_DATE_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DATE_US = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
_MONTH_YEAR = re.compile(r"\b(" + MONTHS + r")\.?\s+(\d{4})\b")
_QUARTER = re.compile(
    r"\b(?:(first|second|third|fourth|1st|2nd|3rd|4th)\s+(?:fiscal\s+)?quarter(?:\s+of)?(?:\s+fiscal)?\s+(?:year\s+)?(\d{4})"
    r"|Q([1-4])\s*(?:of\s+)?(?:FY\s*)?(\d{4})|(\d{4})\s*Q([1-4]))\b", re.IGNORECASE)
_FISCAL_YEAR = re.compile(r"\b(?:fiscal(?:\s+year)?|FY)\s*(\d{4})\b", re.IGNORECASE)
_BARE_YEAR = re.compile(r"\b(19[89]\d|20[0-4]\d)\b")

_TICKER_TAG = re.compile(
    r"\(?\b(?:NYSE(?:\s+American)?|NASDAQ|Nasdaq|NYSE\s+Arca|OTC(?:QB|QX)?|TSX|AMEX)\s*(?:Global\s+(?:Select\s+)?Market)?"
    r"\s*[:\-]?\s*\"?[A-Z]{1,5}(?:\.[A-Z])?\"?\)?")
_CIK = re.compile(r"\b(?:CIK|Central Index Key)\s*(?:No\.?|Number)?\s*[:#]?\s*\d{4,10}\b", re.IGNORECASE)
_FILE_NO = re.compile(r"\b(?:Commission\s+)?File\s+(?:No\.?|Number)\s*[:#]?\s*\d{1,3}-\d{3,6}\b", re.IGNORECASE)
_ACCESSION = re.compile(r"\b\d{10}-\d{2}-\d{6}\b")

_MONEY = re.compile(
    r"(?:US\$|USD\s?|\$)\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?\s*(million|billion|thousand|mm|bn|m|b|k)?\b",
    re.IGNORECASE)

_PERSON = re.compile(r"\b(Mr|Ms|Mrs|Dr|Messrs)\.\s+([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){0,2})")

_PLACEHOLDER = re.compile(r"\b(CompanyX|Company_\d+|Person_\d+|region_\d+|product_\d+|day_-?\d+|Q_-?\d+|FY_-?\d+|usd_bucket_\w+)\b")


# ----------------------------------------------------------------------------- output type
@dataclass(frozen=True)
class MaskedDoc:
    doc_id: str
    masked_text: str
    alias_map: dict[str, str] = field(repr=False)  # placeholder -> raw span; NEVER shown to any LLM
    mask_version: str = MASK_VERSION


# ----------------------------------------------------------------------------- helpers
def _name_variants(name: str) -> list[str]:
    """Return the name and its legal-suffix-stripped form (longest first)."""
    name = name.strip()
    variants = {name}
    stripped = _SUFFIX_RE.sub("", name).strip()
    while stripped and stripped != name and len(stripped) > 2:
        variants.add(stripped)
        name, stripped = stripped, _SUFFIX_RE.sub("", stripped).strip()
    return sorted(variants, key=len, reverse=True)


def _esc(name: str) -> str:
    """Escape a name for regex, letting any whitespace in it match line breaks too."""
    return r"\s+".join(re.escape(part) for part in name.split())


def _bucket_usd(amount: float) -> str:
    if amount < 1e6:
        return None
    if amount < 1e7:
        return "usd_bucket_1M_10M"
    if amount < 1e8:
        return "usd_bucket_10M_100M"
    if amount < 1e9:
        return "usd_bucket_100M_1B"
    if amount < 1e10:
        return "usd_bucket_1B_10B"
    return "usd_bucket_10B_plus"


def _money_value(num: str, frac: str | None, unit: str | None) -> float:
    v = float(num.replace(",", "") + ("." + frac if frac else ""))
    mult = {"thousand": 1e3, "k": 1e3, "million": 1e6, "mm": 1e6, "m": 1e6, "billion": 1e9, "bn": 1e9, "b": 1e9}
    return v * mult.get((unit or "").lower(), 1.0)


def _quarter_index(year: int, q: int) -> int:
    return year * 4 + (q - 1)


# ----------------------------------------------------------------------------- main
def mask_document(
    doc_id: str,
    text: str,
    filing_date: date,
    filer_name: str,
    company_gazetteer: list[str] | None = None,
    person_gazetteer: list[str] | None = None,
    product_gazetteer: list[str] | None = None,
    mask_version: str = MASK_VERSION,
) -> MaskedDoc:
    """Apply mask_version f1. Pure and idempotent. See module docstring for the spec."""
    if mask_version != MASK_VERSION:
        raise ValueError(f"unknown mask_version {mask_version}")
    alias: dict[str, str] = {}
    counters = {"company": 0, "person": 0, "region": 0, "product": 0}
    out = text

    def _new(kind: str, raw: str) -> str:
        counters[kind] += 1
        ph = {"company": f"Company_{counters['company']}", "person": f"Person_{counters['person']}",
              "region": f"region_{counters['region']}", "product": f"product_{counters['product']}"}[kind]
        alias[ph] = raw
        return ph

    # 1. filer -> CompanyX (longest variant first so "Acme Corp" is not left as "Acme")
    for v in _name_variants(filer_name):
        out = re.sub(r"\b" + _esc(v) + r"\b", "CompanyX", out)
    out = re.sub(r"\bthe (Company|Registrant|Corporation)\b", "CompanyX", out, flags=re.IGNORECASE)
    alias["CompanyX"] = filer_name

    # 2. other companies (gazetteer), consistent placeholder per entity, first-appearance order
    entities = []
    for name in (company_gazetteer or []):
        if _SUFFIX_RE.sub("", name).strip().lower() == _SUFFIX_RE.sub("", filer_name).strip().lower():
            continue
        first = min((m.start() for v in _name_variants(name) for m in re.finditer(r"\b" + _esc(v) + r"\b", out)),
                    default=None)
        if first is not None:
            entities.append((first, name))
    for _, name in sorted(entities):
        ph = _new("company", name)
        for v in _name_variants(name):
            out = re.sub(r"\b" + _esc(v) + r"\b", ph, out)

    # 3. tickers, CIK, file numbers, accession numbers -> removed
    out = _TICKER_TAG.sub("", out)
    out = _CIK.sub("", out)
    out = _FILE_NO.sub("", out)
    out = _ACCESSION.sub("", out)

    # 4. people
    people: dict[str, str] = {}
    for name in (person_gazetteer or []):
        if re.search(r"\b" + _esc(name) + r"\b", out):
            people[name] = _new("person", name)
            out = re.sub(r"\b" + _esc(name) + r"\b", people[name], out)
            surname = name.split()[-1]
            out = re.sub(r"\b(Mr|Ms|Mrs|Dr)\.\s+" + re.escape(surname) + r"\b", people[name], out)

    def _person_sub(m: re.Match) -> str:
        raw = m.group(2)
        if raw not in people:
            people[raw] = _new("person", m.group(0))
        return people[raw]
    out = _PERSON.sub(_person_sub, out)

    # 5. dates -> relative day index; quarters/fiscal years/bare years -> relative labels
    fy, fq = filing_date.year, (filing_date.month - 1) // 3 + 1

    def _rel_day(y: int, mo: int, d: int) -> str:
        try:
            return f"day_{(date(y, mo, d) - filing_date).days}"
        except ValueError:
            return f"FY_{y - fy}"
    out = _DATE_LONG.sub(lambda m: _rel_day(int(m.group(3)), _MONTH_NUM[m.group(1).lower()], int(m.group(2))), out)
    out = _DATE_DMY.sub(lambda m: _rel_day(int(m.group(3)), _MONTH_NUM[m.group(2).lower()], int(m.group(1))), out)
    out = _DATE_ISO.sub(lambda m: _rel_day(int(m.group(1)), int(m.group(2)), int(m.group(3))), out)
    out = _DATE_US.sub(lambda m: _rel_day(int(m.group(3)), int(m.group(1)), int(m.group(2))), out)

    def _quarter_sub(m: re.Match) -> str:
        words = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4}
        if m.group(1):
            q, y = words[m.group(1).lower()], int(m.group(2))
        elif m.group(3):
            q, y = int(m.group(3)), int(m.group(4))
        else:
            y, q = int(m.group(5)), int(m.group(6))
        return f"Q_{_quarter_index(y, q) - _quarter_index(fy, fq)}"
    out = _QUARTER.sub(_quarter_sub, out)
    out = _FISCAL_YEAR.sub(lambda m: f"FY_{int(m.group(1)) - fy}", out)
    out = _MONTH_YEAR.sub(lambda m: f"FY_{int(m.group(2)) - fy}", out)
    out = _BARE_YEAR.sub(lambda m: f"FY_{int(m.group(1)) - fy}", out)

    # 6. dollar amounts >= $1M -> buckets (smaller amounts kept: they are rarely identifying)
    def _money_sub(m: re.Match) -> str:
        b = _bucket_usd(_money_value(m.group(1), m.group(2), m.group(3)))
        return b if b else m.group(0)
    out = _MONEY.sub(_money_sub, out)

    # 7. locations -> region_k (consistent per document), longest names first
    regions: dict[str, str] = {}
    for loc in sorted(US_CITIES + US_STATES + COUNTRIES, key=len, reverse=True):
        pat = re.compile(r"\b" + loc + r"\b")
        if pat.search(out):
            key = loc.replace("\\", "")
            regions.setdefault(key, _new("region", key))
            out = pat.sub(regions[key], out)
    out = re.sub(r"(region_\d+),\s*region_\d+", r"\1", out)          # "Duluth, Minnesota" -> one region

    # 8. brands / products via explicit gazetteer only
    for prod in sorted(product_gazetteer or [], key=len, reverse=True):
        if re.search(r"\b" + _esc(prod) + r"\b", out):
            ph = _new("product", prod)
            out = re.sub(r"\b" + _esc(prod) + r"\b", ph, out)

    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"(?:\s*\.){2,}", ".", out)                          # ". ." left by removed identifiers
    return MaskedDoc(doc_id=doc_id, masked_text=out, alias_map=alias, mask_version=mask_version)


def contains_placeholders_only(text: str) -> bool:
    """Cheap sanity check used by extract.py before a prompt is sent: no residual ticker tags / absolute dates."""
    return not (_TICKER_TAG.search(text) or _DATE_LONG.search(text) or _DATE_ISO.search(text) or _DATE_US.search(text))
