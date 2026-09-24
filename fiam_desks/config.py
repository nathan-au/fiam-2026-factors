"""Shared constants of the desk system. Everything here is FIXED and PRE-REGISTERED (docs/DESKS.md); an experiment that needs
a different value passes it explicitly and says so in its README. Nothing in this file is tuned on 2021-2026 data."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # project root (fiam/ data and cache/ live here)
FIAM_DIR = ROOT / "fiam"
CACHE = ROOT / "cache"
CHARS_FILE = FIAM_DIR / "chars_final_with_names.parquet"
FILINGS_FILE = FIAM_DIR / "8k_20150101_20260831_identified.parquet"
TEXT_DIR = CACHE / "text_lane_2026-09-21"  # copy of the teammate's handoff tables (SHA256SUMS next to them)
EXPERIMENTS = ROOT / "experiments"
LEDGER = EXPERIMENTS / "desk_test_ledger.csv"

TARGET_COL = "ret_exc_lead1m"

# Periods, by TARGET month. DEV is the development period: every design decision is made on it. TEST is the FIAM scoring window;
# it is touched only by confirmation runs, and every such run is appended to LEDGER (docs/DESKS.md sec 3).
DEV = ("2015-02-28", "2020-12-31")
TEST = ("2021-01-31", "2026-08-31")

# Universe (frozen, experiments/largecap): price >= $5, market cap >= $2B, 126d dollar volume >= $10M, both betas observed.
FLOOR_MCAP = 2000.0
MIN_PRICE = 5.0
MIN_DOLVOL = 10_000_000.0

# Frozen factor set of experiments/largecap (18 factors in 7 economic groups, signs fixed a priori from docs/FACTORS.md).
FACTOR_GROUPS = {
    "value": {"be_me": +1, "ni_me": +1, "fcf_me": +1},
    "profitability": {"gp_at": +1, "ni_be": +1, "ebit_sale": +1},
    "investment_issuance_accruals": {"at_gr1": -1, "chcsho_12m": -1, "oaccruals_at": -1},
    "quality": {"qmj": +1, "f_score": +1},
    "surprise": {"niq_su": +1, "saleq_su": +1},
    "volatility_beta": {"ivol_capm_21d": -1, "rmax5_21d": -1, "betabab_1260d": -1},
    "liquidity": {"ami_126d": +1, "turnover_126d": -1},
}
SIGNS = {f: s for g in FACTOR_GROUPS.values() for f, s in g.items()}
FEATURES = list(SIGNS)

# Frozen portfolio configuration (`lc_t10` of experiments/largecap).
LP_BASE = dict(min_dolvol=MIN_DOLVOL, min_price=MIN_PRICE, min_mcap=FLOOR_MCAP, beta_cols=("raw_beta_60m", "raw_betabab_1260d"),
               max_weight=0.01, sector_net=0.05, sector_gross=0.70, turnover=0.10)
GROSS = 2.0
RELAX_LADDER = (1.0, 1.5, 2.0, 3.0, 5.0, None)

# Published reference numbers the system must reproduce (experiments/largecap/README.md, $2B floor, test period).
PUBLISHED = {"comp_universe_ic": 0.0366, "comp_ic_t": 1.91, "lc_t10_ir_gross": 0.615, "lc_t10_ir_net": 0.541}

# Short-interest cap adopted by experiments/feat_si_followup (a priori level, not tuned): shorts only where SI ratio <= 10%.
SI_CAP = 0.10


def config_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
