"""
FIAM 2026 - Deterministic devil's advocate on the assembled system (desk_11_devils_advocate). Descriptive: nothing here selects or changes a configuration.

FIAM section 9 asks for a second agent whose only job is to attack the thesis. Here it is a script with fixed attacks, each stated with the outcome that would count as a failure:
  1 time stability     net IR in each half and each calendar year; leave-one-year-out; FAIL if the sign flips in more than one calendar year
  2 time concentration IR after removing the 3 best months; FAIL if it falls below 0.25
  3 factor attribution regress the book's net excess return on the market and six universe long-short characteristic spreads (size, value, momentum, low-vol, quality, profitability):
                       FAIL if alpha t < 2 (the return is explained by known style exposures)
  4 random-score placebo the identical LP, screens and costs run on 30 score vectors shuffled within (month, sector): FAIL if the real net IR is not above the 90th percentile
  5 bootstrap          circular block bootstrap (block 4, 5000 draws, seed 0) 90% interval for the net IR, and for the net-IR DIFFERENCE of every text mode vs the default base
                       (the desk_09 arms): a mode is distinguishable only if the interval excludes 0
  6 regime disclosure  the same system on 2015-2020 (desk_10)
  7 text lane claim    pooled 2015-2026 universe IC of novneg_max / novelty_max (descriptive, both periods, after-the-fact)
Run:  .venv/bin/python experiments/desk_11_devils_advocate/desk_11_devils_advocate.py --confirm
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from fiam_desks import config as C, evaluate as E, factors_desk as FD, ledger, lp as L, pm as PM, system as S, text_desk as TD  # noqa: E402
from fiam_desks.panel import Panel  # noqa: E402

OUT = HERE / "output"
EXP = "desk_11_devils_advocate"
RF_HALF = 0.04 / 12.0


def ir(active):
    a = np.asarray(active, float)
    return float(np.sqrt(12) * a.mean() / a.std(ddof=1))


def boot_idx(n, draws, block, seed):
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(draws, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return idx.reshape(draws, -1)[:, :n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--placebos", type=int, default=30)
    args = ap.parse_args()
    if not args.confirm:
        sys.exit("--confirm required (test-window descriptive run)")
    t0 = time.time()
    P = Panel()
    parts = S.desks(P)
    base = S.build(P, "test", parts=parts)
    fr = base["result"]["frame"].set_index("target_month")
    net = fr["net_port_excess_ret"]
    act = net - RF_HALF
    res = {"default_ir_net": ir(act)}
    ledger.record(EXP, "default system attack (descriptive)", S.DEFAULT, "no selection")

    # 1 time stability
    yrs = {int(y): float(v) for y, v in net.groupby(net.index.year).apply(lambda s: (1 + s).prod() - 1).items()}
    halves = {"2021-01..2023-06": ir(act[:"2023-06-30"]), "2023-07..2026-08": ir(act["2023-07-31":])}
    loyo = {int(y): ir(act[act.index.year != y]) for y in sorted(set(act.index.year))}
    res["time_stability"] = {"calendar_year_net_return": yrs, "half_ir_net": halves, "leave_one_year_out_ir_net": loyo,
                             "years_negative": int(sum(v < 0 for v in yrs.values())), "fail": bool(sum(v < 0 for v in yrs.values()) > 1)}
    # 2 concentration
    best3 = net.nlargest(3).index
    res["time_concentration"] = {"best_months": [str(d.date()) for d in best3], "ir_without_best3": ir(act.drop(best3)), "fail": bool(ir(act.drop(best3)) < 0.25)}
    # 3 factor attribution
    m = P.mask("test")
    d = pd.DataFrame({"m": P.months[m], "y": P.y[m]})
    proxies = {"size_SMB": -P.df["x_size"].to_numpy()[m], "value": P.df["be_me"].to_numpy()[m], "momentum": P.df["x_mom"].to_numpy()[m],
               "lowvol": -P.df["x_vol"].to_numpy()[m], "quality": P.df["qmj"].to_numpy()[m], "profitability": P.df["gp_at"].to_numpy()[m]}
    F = {}
    for k, x in proxies.items():
        dd = d.assign(x=x)
        q = dd.groupby("m")["x"].rank(pct=True)
        top, bot = dd[q > 0.8].groupby("m")["y"].mean(), dd[q <= 0.2].groupby("m")["y"].mean()
        F[k] = top - bot
    F = pd.DataFrame(F)
    mk = (fr["sp500_ret"] - fr["rf_monthly"])
    X = sm.add_constant(pd.concat([mk.rename("mkt"), F], axis=1).reindex(net.index))
    ols = sm.OLS(net, X).fit()
    res["factor_attribution"] = {"alpha_annual": float(12 * ols.params["const"]), "alpha_t": float(ols.tvalues["const"]), "r2": float(ols.rsquared),
                                 "loadings": {k: {"beta": float(ols.params[k]), "t": float(ols.tvalues[k])} for k in X.columns if k != "const"},
                                 "fail": bool(abs(ols.tvalues["const"]) < 2)}
    # 4 placebo
    comp, Sd, raw, txt = parts
    fac = S.factors_output(P, comp, Sd, "composite+dtc")
    irs = []
    for k in range(args.placebos):
        sc = E.shuffle_within(P, fac.score, seed=k)
        dec = {"pred": sc, "veto_long": np.zeros(len(sc), bool), "veto_short": np.zeros(len(sc), bool), "w_mult": np.ones(len(sc))}
        irs.append(L.evaluate_book(PM.lp_frame(P, dec, "test", {"sir": Sd["sir"]}), base["lp_cfg"])["ir_net"])
    ledger.record(EXP, f"{args.placebos} random-score placebos", {"seeds": list(range(args.placebos))}, "descriptive")
    res["random_score_placebo"] = {"placebo_ir_net_mean": float(np.mean(irs)), "placebo_ir_net_sd": float(np.std(irs, ddof=1)), "p90": float(np.quantile(irs, 0.9)), "max": float(np.max(irs)),
                                   "share_ge_real": float((np.array(irs) >= res["default_ir_net"]).mean()), "fail": bool(res["default_ir_net"] <= np.quantile(irs, 0.9))}
    # 5 bootstrap
    idx = boot_idx(len(net), 5000, 4, 0)
    a = act.to_numpy()
    bs = np.array([ir(a[i]) for i in idx])
    res["bootstrap_default"] = {"ir_net": ir(a), "ci90": [float(np.quantile(bs, 0.05)), float(np.quantile(bs, 0.95))], "share_draws_ir_le_0": float((bs <= 0).mean())}
    arms = {"blend": ("blend", {"w": 1.0}), "veto_long": ("veto_long", {}), "tilt": ("tilt", {"lam": 0.25}), "dial": ("dial", {"kappa": 0.5}), "judge": ("judge", {"a_min": 5, "use_text": True})}
    diffs = {}
    for name, (mode, kw) in arms.items():
        o = S.build(P, "test", {"text_mode": mode, "text_params": kw}, parts=parts)["result"]["frame"].set_index("target_month")["net_port_excess_ret"] - RF_HALF
        o = o.reindex(act.index).to_numpy()
        dd_ = np.array([ir(o[i]) - ir(a[i]) for i in idx])
        diffs[name] = {"ir_diff": ir(o) - ir(a), "ci90": [float(np.quantile(dd_, 0.05)), float(np.quantile(dd_, 0.95))], "distinguishable": bool(np.quantile(dd_, 0.05) > 0 or np.quantile(dd_, 0.95) < 0)}
    res["bootstrap_text_modes_vs_default"] = diffs
    # 7 pooled text-lane claim
    pooled = {}
    for n in ("novneg_max", "novelty_max", "abrupt_exit"):
        s = TD.signal_score(P, raw, n)
        ic = E.ic_series(P, s, "dev")
        ic2 = E.ic_series(P, s, "test")
        allic = pd.concat([ic, ic2])
        pooled[n] = {"dev": E.ic_summary(ic), "test": E.ic_summary(ic2), "pooled_ic": float(allic.mean()), "pooled_t": E.tstat(allic)}
    res["text_pooled_descriptive"] = pooled
    E.dump(OUT / "results.json", res)

    print(f"default system net IR {res['default_ir_net']:+.3f}; bootstrap 90% CI {res['bootstrap_default']['ci90']} (share of draws <= 0: {res['bootstrap_default']['share_draws_ir_le_0']:.3f})")
    ts = res["time_stability"]
    print(f"1 time stability: calendar-year net returns {ts['calendar_year_net_return']} | halves {halves} | leave-one-year-out {loyo} -> {'FAIL' if ts['fail'] else 'pass'}")
    print(f"2 concentration: without best 3 months ({res['time_concentration']['best_months']}) IR {res['time_concentration']['ir_without_best3']:+.3f} -> {'FAIL' if res['time_concentration']['fail'] else 'pass'}")
    fa = res["factor_attribution"]
    print(f"3 factor attribution: alpha {100 * fa['alpha_annual']:+.1f}%/yr (t {fa['alpha_t']:+.2f}), R2 {fa['r2']:.2f}; loadings " + ", ".join(f"{k} {v['beta']:+.2f} (t {v['t']:+.1f})" for k, v in fa['loadings'].items()) + f" -> {'FAIL' if fa['fail'] else 'pass'}")
    rp = res["random_score_placebo"]
    print(f"4 random-score placebo: {rp['placebo_ir_net_mean']:+.3f} +- {rp['placebo_ir_net_sd']:.3f} (p90 {rp['p90']:+.3f}, max {rp['max']:+.3f}); share >= real {rp['share_ge_real']:.2f} -> {'FAIL' if rp['fail'] else 'pass'}")
    for k, v in diffs.items():
        print(f"5 text mode {k:10s}: net IR diff vs default {v['ir_diff']:+.3f}, 90% CI [{v['ci90'][0]:+.3f}, {v['ci90'][1]:+.3f}] -> {'distinguishable' if v['distinguishable'] else 'NOT distinguishable from 0'}")
    for k, v in pooled.items():
        print(f"7 pooled IC {k:12s}: dev {v['dev']['ic']:+.4f} (t {v['dev']['t']:+.2f}), test {v['test']['ic']:+.4f} (t {v['test']['t']:+.2f}), pooled {v['pooled_ic']:+.4f} (t {v['pooled_t']:+.2f})")
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
