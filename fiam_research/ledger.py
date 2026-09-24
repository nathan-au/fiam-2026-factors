"""Research ledger: one row per variant evaluated in this research run (desk_13 onward), on ANY window, so the number of trials behind every
reported result is countable (deflated Sharpe / PBO / FDR use it). TEST-window evaluations are additionally recorded in the frozen
experiments/desk_test_ledger.csv via fiam_desks.ledger (the project's single log of test looks)."""

import csv
import datetime as dt
import json

from fiam_desks import config as C, ledger as TL

PATH = C.EXPERIMENTS / "desk_research_ledger.csv"
FIELDS = ["utc", "experiment", "family", "variant", "hypothesis", "params", "discovery_window", "selection_window", "eval_window", "test_touched",
          "n_variants_in_family", "ic", "ic_t", "ir_net", "ir_gross", "max_dd_net", "paired_net_t_vs_base", "decision", "reason"]


def _f(x):
    try:
        return f"{float(x):.4f}"
    except (TypeError, ValueError):
        return ""


def record(experiment, family, variant, hypothesis="", params=None, discovery="dev", selection="dev", eval_window="dev", n_variants=1, m=None,
           paired_t=None, decision="", reason=""):
    m = m or {}
    test = "test" in eval_window
    new = not PATH.exists()
    with open(PATH, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(FIELDS)
        w.writerow([dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), experiment, family, variant, hypothesis,
                    json.dumps(params or {}, sort_keys=True, default=str), discovery, selection, eval_window, test, n_variants, _f(m.get("ic")),
                    _f(m.get("ic_t")), _f(m.get("ir_net")), _f(m.get("ir_gross")), _f(m.get("max_dd_net")), _f(paired_t), decision, reason])
    if test:
        TL.record(experiment, f"{family}:{variant}", params or {}, reason or decision)
