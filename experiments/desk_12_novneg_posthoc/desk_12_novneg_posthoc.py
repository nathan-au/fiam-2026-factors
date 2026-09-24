"""
FIAM 2026 - POST-HOC exploratory: novelty x negative as a single 8th group, both periods (desk_12_novneg_posthoc).

STATUS: post-hoc. The hypothesis (novneg_max alone) was suggested by desk_09/desk_11 results on the TEST window, which were already seen, so nothing here can be adopted into the
default system or counted as confirmation (PREREGISTRATION.md rule 4). Its purpose is to tell the user whether the one text signal with same-sign evidence in both periods is worth a
properly pre-registered follow-up when new data exist. Reported: dev and test standalone IC, paired IC gain over the respective base, permutation p, LP net IR and paired net t.
Bases: dev = frozen composite (SI data do not exist before 2020-06); test = default factors desk (composite + dtc) with the SI cap.
Run:  .venv/bin/python experiments/desk_12_novneg_posthoc/desk_12_novneg_posthoc.py --confirm
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from fiam_desks import config as C, evaluate as E, factors_desk as FD, ledger, lp as L, pm as PM, system as S, text_desk as TD  # noqa: E402
from fiam_desks.panel import Panel  # noqa: E402

OUT = HERE / "output"
EXP = "desk_12_novneg_posthoc"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true")
    if not ap.parse_args().confirm:
        sys.exit("--confirm required")
    P = Panel()
    comp, Sd, raw, txt = S.desks(P)
    nn = TD.signal_score(P, raw, "novneg_max")
    res = {}
    for period, fac_kind, cfg in (("dev", "composite", C.LP_BASE), ("test", "composite+dtc", PM.cfg_with_si_cap())):
        if period == "test":
            ledger.record(EXP, "F1_cap vs F1_cap+novneg (post-hoc)", {"signal": "novneg_max", "w": 1}, "post-hoc, test already seen")
        base = S.factors_output(P, comp, Sd, fac_kind)
        with_nn = FD.add_groups(base, {"novneg": nn}, "base+novneg")
        ic_b, ic_n = E.ic_series(P, base.score, period), E.ic_series(P, with_nn.score, period)
        pr = E.paired(ic_n, ic_b)
        # permutation null for the gain over THIS base (base already has k groups)
        Gsum = base.extras["groups"].sum(axis=1).to_numpy() + (-FD.urank(Sd["dtc"], P) if fac_kind == "composite+dtc" else 0.0)
        nb = 7 + (1 if fac_kind == "composite+dtc" else 0)
        perm = E.perm_null(P, Gsum, nb, nn, period, w=1.0, n_perm=200, seed=5)
        books = {}
        for nm, fac in (("base", base), ("base+novneg", with_nn)):
            dec = PM.decide(fac, None, "factors_only")
            books[nm] = L.evaluate_book(PM.lp_frame(P, dec, period, {"sir": Sd["sir"]}), cfg, keep_frames=True)
        t = E.paired_net(books["base+novneg"]["frame"], books["base"]["frame"])["t"]
        res[period] = {"base": fac_kind, "novneg_alone_ic": E.ic_summary(E.ic_series(P, nn, period)), "ic_base": ic_b.mean(), "ic_with": ic_n.mean(), "paired_ic_gain": pr["mean_diff"], "paired_ic_t": pr["t"],
                       "perm_p": perm["p_value"], "ir_net_base": books["base"]["ir_net"], "ir_net_with": books["base+novneg"]["ir_net"], "paired_net_t": t,
                       "maxdd_base": books["base"]["max_dd_net"], "maxdd_with": books["base+novneg"]["max_dd_net"]}
        r = res[period]
        print(f"{period}: novneg alone IC {r['novneg_alone_ic']['ic']:+.4f} (t {r['novneg_alone_ic']['t']:+.2f}) | base IC {r['ic_base']:+.4f} -> {r['ic_with']:+.4f} (paired t {r['paired_ic_t']:+.2f}, perm p {r['perm_p']:.3f}) | "
              f"net IR {r['ir_net_base']:+.3f} -> {r['ir_net_with']:+.3f} (paired net t {r['paired_net_t']:+.2f}) | maxDD {100 * r['maxdd_base']:.1f}% -> {100 * r['maxdd_with']:.1f}%")
    E.dump(OUT / "results.json", res)


if __name__ == "__main__":
    main()
