"""Append-only ledger of every look at the TEST window (docs/DESKS.md sec 3): one row per test-period arm, so the number of looks is auditable."""

import csv
import datetime as dt

from . import config as C

FIELDS = ["utc", "experiment", "arm", "config_hash", "note"]


def record(experiment: str, arm: str, config: dict, note: str = "") -> None:
    new = not C.LEDGER.exists()
    with open(C.LEDGER, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(FIELDS)
        w.writerow([dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), experiment, arm, C.config_hash(config), note])
