"""
desk_30_text_untested_uses - the uses of txt_v1 / txt_v2 (the handoff tables; byte-identical to cache/text_lane_2026-09-21) that no earlier
experiment tested. DEV only. Economic hypotheses, signs fixed before running:
  X1  EARNINGS FRESHNESS (txt_v1 item 2.02; Bernard-Thomas PEAD is concentrated right after the announcement): the surprise group's IC is higher among
      names with a 2.02 filing in month t or t-1 than among names without.            test: fresh - stale conditional IC > 0
  X2  PENDING MERGER (txt_v2 n_101_merger in months t-5..t): a merger-agreement filing pins the price to a deal; the factor view is uninformative and
      shorting a target earns the negative of the arb spread.                           test: B1 IC among flagged names < IC among others; flagged names'
                                                                                        next-month return > universe mean
  X3  FRESH-NEWS ABSTENTION (txt_v2 novelty_max >= 0.5 in month t): characteristics are stale right after genuinely new disclosures.
                                                                                        test: B1 IC among news names < among the rest
  X4  BOOK TEXT RISK (desk_23 C3: novneg predicts idiosyncratic risk): the B1 book's formation-month |weight|-weighted novneg rank predicts next-month
      |book return| (portfolio risk timing).                                            test: rank corr > 0
BH-FDR q = 0.10 over the four one-sided tests. Book arms (only interpretable next to their test; pre-registered):
  X1b B1 with the surprise group set to 0 where no 2.02 in t or t-1      X2b B1 with flagged names barred on both sides      X3b same for news names
Rule 'promising': DEV net IR >= B1 + 0.15 AND paired monthly net t >= 1 AND max DD not worse by > 5 pp.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from fiam_desks import evaluate as E, factors_desk as FD, text_desk as TD  # noqa: E402
from fiam_research import core, ledger, stats as ST  # noqa: E402

OUT = HERE / "output"
EXP = "desk_30_text_untested_uses"


def rolling_any(P, x, months):
    """1 if x > 0 for the same permno in any of the last `months` months including t (causal)."""
    df = pd.DataFrame({"permno": P.df["permno"].to_numpy(), "eom": P.df["eom"].to_numpy(), "x": (np.nan_to_num(x) > 0).astype(float)})
    W = df.pivot(index="eom", columns="permno", values="x")
    W = W.reindex(pd.date_range(W.index.min(), W.index.max(), freq="ME")).fillna(0.0)
    R = W.rolling(months, min_periods=1).max()
    return R.to_numpy()[R.index.get_indexer(df["eom"]), R.columns.get_indexer(df["permno"])] > 0


def cond_diff(P, s, a, b):
    ia, ib = E.ic_series(P, s, "dev", a), E.ic_series(P, s, "dev", b)
    d = (ia - ib).dropna()
    return {"ic_A": float(ia.mean()), "ic_B": float(ib.mean()), "diff": float(d.mean()), "t": ST.tstat(d), "n": int(len(d))}


def main():
    t0 = time.time()
    ctx = core.research_ctx(text=True)
    P, raw = ctx.P, ctx.raw
    u, dev = P.u, P.mask("dev")
    b1 = core.b1_score(ctx)
    G = ctx.comp.extras["groups"]
    fresh = rolling_any(P, raw["txt_item_2_02"], 2)
    merger = rolling_any(P, raw["txt_v2_n_101_merger"], 6)
    news = np.nan_to_num(raw["txt_v2_novelty_max"]) >= 0.5
    res = {"shares_dev_universe": {"fresh_2.02": float(fresh[dev].mean()), "merger_6m": float(merger[dev].mean()), "news_novelty_ge_0.5": float(news[dev].mean())}}
    print(res, flush=True)
    tests = {}
    tests["X1_fresh_minus_stale_surprise_ic"] = cond_diff(P, G["surprise"].to_numpy(), fresh, ~fresh)
    x2 = cond_diff(P, b1, ~merger, merger)  # predicted > 0 (others better than flagged)
    d = pd.DataFrame({"m": P.months[dev], "y": P.y[dev], "f": merger[dev]})
    spr = d.groupby("m").apply(lambda g: g.loc[g.f, "y"].mean() - g.loc[~g.f, "y"].mean(), include_groups=False).dropna()
    x2["flagged_minus_rest_ret"], x2["ret_t"] = float(spr.mean()), ST.tstat(spr)
    tests["X2_nonmerger_minus_merger_b1_ic"] = x2
    tests["X3_nonnews_minus_news_b1_ic"] = cond_diff(P, b1, ~news, news)
    # X4 book text risk
    base = core.book(ctx, b1, "dev", **core.b1_kw())
    h = base["holdings"]
    nq = pd.Series(np.where(u, np.nan_to_num(raw["txt_v2_novel_negative_max"], nan=0.0), np.nan)).groupby(P.df["eom"].to_numpy()).rank(pct=True)
    key = P.df[["permno", "eom"]].assign(nq=nq.to_numpy())
    hk = h.merge(key, on=["permno", "eom"], how="left")
    expo = (hk["weight"].abs() * hk["nq"]).groupby(hk["target_month"]).sum() / hk["weight"].abs().groupby(hk["target_month"]).sum()
    fr = base["frame"].set_index("target_month")
    absr = fr["port_excess_ret"].abs()
    z = pd.concat([expo.rename("x"), absr.rename("v")], axis=1).dropna()
    rc = float(np.corrcoef(rankdata(z.x), rankdata(z.v))[0, 1])
    tests["X4_book_text_risk_vs_abs_return"] = {"rank_corr": rc, "t": rc * np.sqrt((len(z) - 2) / max(1e-9, 1 - rc ** 2)), "n": int(len(z))}
    T = pd.DataFrame(tests).T
    stat = T["t"].astype(float).copy()
    stat["X2_nonmerger_minus_merger_b1_ic"] = tests["X2_nonmerger_minus_merger_b1_ic"]["t"]
    T["p"] = [ST.p_from_t(v) for v in stat]
    T["bh_q10"] = ST.bh_fdr(T["p"].to_numpy(), 0.10)
    print(T.round(4).to_string(), flush=True)
    for k, r in T.iterrows():
        ledger.record(EXP, "text_untested_test", k, "untested text use (economic hypothesis)", {}, "dev", "dev", "dev", 4, {"ic": r.get("diff", r.get("rank_corr")), "ic_t": r["t"]},
                      decision="discovery" if r["bh_q10"] else "no evidence")
    # book arms
    s1 = FD.add_groups(ctx.comp, {"dtc": ctx.dtc_block(True)}).score  # = b1; X1b replaces the surprise group
    G1 = G.copy()
    G1["surprise"] = np.where(fresh, G["surprise"], 0.0)
    x1b = (G1.sum(axis=1).to_numpy() + ctx.dtc_block(True)) / 8.0
    arms = {"B1": (b1, {}), "X1b_surprise_fresh_only": (x1b, {}), "X2b_bar_merger": (b1, {"extras": {"veto_long": merger, "veto_short": merger}}),
            "X3b_bar_news": (b1, {"extras": {"veto_long": news, "veto_short": news}})}
    B = core.run_many(ctx, [(k, s, "dev", core.b1_kw(**kw)) for k, (s, kw) in arms.items()])
    f0 = B["B1"]["frame"].set_index("target_month")["net_port_excess_ret"]
    m0 = core.metrics(ctx, B["B1"], b1, "dev", boot=False)
    rows = []
    for k, (s, kw) in arms.items():
        m = core.metrics(ctx, B[k], s, "dev", boot=False)
        f = B[k]["frame"].set_index("target_month")["net_port_excess_ret"]
        pt = ST.paired(f, f0)["t"] if k != "B1" else np.nan
        ok = k != "B1" and m["ir_net"] >= m0["ir_net"] + 0.15 and pt >= 1 and m["max_dd_net"] >= m0["max_dd_net"] - 0.05
        dec = "reference" if k == "B1" else ("promising" if ok else ("rejected" if pt <= -1 else "no evidence"))
        rows.append({"arm": k, "ic": m["ic"], "ir_net": m["ir_net"], "sharpe": m["sharpe_net"], "max_dd": m["max_dd_net"], "paired_t": pt, "relaxed": m["months_turnover_relaxed"], "decision": dec})
        if k != "B1":
            ledger.record(EXP, "text_untested_book", k, "untested text use as book rule", {}, "dev", "dev", "dev", 3, m, paired_t=pt, decision=dec)
    R = pd.DataFrame(rows)
    print(R.round(4).to_string())
    T.to_csv(OUT / "tests_dev.csv")
    R.to_csv(OUT / "books_dev.csv", index=False)
    E.dump(OUT / "results.json", {"shares": res, "tests": tests, "books": rows})
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
