"""
desk_17_regime_attribution - why does the baseline construction lose on 2015-2020 and earn on 2021-2026?

Descriptive attribution of B1 (baseline + desk_15 corrections) over BOTH periods (TEST already seen; nothing here selects a configuration, and any
design idea it suggests must be validated on DEV-only criteria later - ledgered as descriptive).
  A  universe IC by year of each of the 7 groups + dtc; Fama-MacBeth group factor returns by year
  B  book return decomposition: monthly holdings exposure to each group score x that month's FM group return; residual = stock-specific
  C  long / short legs and sector contribution by year
  D  is DEV vs TEST different beyond noise? Welch t of monthly IC and of net returns; bootstrap of the IR difference
  E  ex-ante predictability of the composite's monthly IC (DEV discovery, TEST check): AR(1); trailing-12m IC; VIX, trailing market return,
     value spread (log BM top minus bottom quintile), cross-sectional return dispersion, trailing 6m market vol (all known at formation)
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from fiam_desks import config as C, evaluate as E, factors_desk as FD  # noqa: E402
from fiam_research import core, ledger, stats as ST  # noqa: E402

OUT = HERE / "output"
EXP = "desk_17_regime_attribution"


def fm_returns(P, G: pd.DataFrame):
    """Monthly cross-sectional OLS of next-month return on the group scores (universe rows): the return per unit of group score."""
    u = P.u
    X = G[u].copy()
    X["y"], X["m"] = P.y[u], P.months[u]
    out = {}
    for m, g in X.groupby("m"):
        Z = sm.add_constant(g.drop(columns=["y", "m"]).to_numpy())
        b = np.linalg.lstsq(Z, g["y"].to_numpy(), rcond=None)[0]
        out[m] = b[1:]
    return pd.DataFrame(out, index=G.columns).T


def main():
    t0 = time.time()
    ctx = core.research_ctx(text=False)
    P = ctx.P
    res = {}
    G = ctx.comp.extras["groups"].copy()
    G["dtc"] = ctx.dtc_block(True)
    score = core.b1_score(ctx)
    # A: IC by year per group, both periods
    ics = {}
    for g in G.columns:
        s = pd.concat([E.ic_series(P, G[g].to_numpy(), "dev"), E.ic_series(P, G[g].to_numpy(), "test")])
        ics[g] = s
    ics["composite_B1"] = pd.concat([E.ic_series(P, score, "dev"), E.ic_series(P, score, "test")])
    ICY = pd.DataFrame(ics).groupby(lambda d: d.year).mean().round(4)
    ICY.to_csv(OUT / "group_ic_by_year.csv")
    ICP = pd.DataFrame(ics)
    res["group_ic_dev_test"] = {g: {"dev": float(ICP[g][:"2020-12-31"].mean()), "dev_t": ST.tstat(ICP[g][:"2020-12-31"]),
                                    "test": float(ICP[g]["2021-01-31":].mean()), "test_t": ST.tstat(ICP[g]["2021-01-31":])} for g in ICP.columns}
    print(ICY.to_string())
    F = fm_returns(P, G)
    FY = F.groupby(F.index.year).sum().round(4)
    FY.to_csv(OUT / "fm_group_returns_by_year.csv")
    print("FM group returns (sum per year):\n", FY.to_string())
    # B/C: book decomposition by period
    for period in ("dev", "test"):
        r = core.b1_book(ctx, score, period)
        h = r["holdings"]
        Gi = G.copy()
        Gi["permno"], Gi["eom"] = P.df["permno"].to_numpy(), P.df["eom"].to_numpy()
        hk = h[["permno", "eom", "target_month", "weight", "sector", C.TARGET_COL]].merge(Gi, on=["permno", "eom"], how="left")
        expo = hk[list(G.columns)].mul(hk["weight"], axis=0).groupby(hk["target_month"]).sum()
        contrib = expo * F.reindex(expo.index)
        gross = (hk["weight"] * hk[C.TARGET_COL]).groupby(hk["target_month"]).sum()
        contrib["residual_stock_specific"] = gross - contrib.sum(axis=1)
        cy = contrib.groupby(contrib.index.year).sum().round(4)
        cy["gross_total"] = gross.groupby(gross.index.year).sum().round(4)
        cy.to_csv(OUT / f"book_decomposition_{period}.csv")
        ey = expo.groupby(expo.index.year).mean().round(3)
        ey.to_csv(OUT / f"book_group_exposure_{period}.csv")
        fr = r["frame"].set_index("target_month")
        legs = pd.DataFrame({"long": fr["long_leg_ret"], "short": fr["short_leg_ret"]}).groupby(fr.index.year).sum().round(4)
        secy = (hk["weight"] * hk[C.TARGET_COL]).groupby([hk["target_month"].dt.year, hk["sector"]]).sum().unstack().round(4)
        secy.to_csv(OUT / f"sector_by_year_{period}.csv")
        res[f"book_{period}"] = {"decomposition_total": contrib.sum().round(4).to_dict(), "mean_exposure": expo.mean().round(3).to_dict(), "legs_by_year": legs.to_dict()}
        print(f"\n[{period}] decomposition by year (sum of monthly gross contributions):\n", cy.to_string(), "\nexposure by year:\n", ey.to_string(), "\nlegs:\n", legs.to_string())
        res[f"frame_{period}"] = fr
        ledger.record(EXP, "attribution", f"B1_{period}", "descriptive attribution", {}, "n/a", "n/a", period, 1, {}, decision="descriptive", reason="descriptive, no selection")
    # D: distinguishability
    fd, ft = res.pop("frame_dev"), res.pop("frame_test")
    icd, ict = ICP["composite_B1"][:"2020-12-31"], ICP["composite_B1"]["2021-01-31":]
    from scipy.stats import ttest_ind
    tt = ttest_ind(ict, icd, equal_var=False)
    tn = ttest_ind(ft["net_port_excess_ret"], fd["net_port_excess_ret"], equal_var=False)
    # IR difference bootstrap (independent block bootstraps)
    a, b = (ft["net_port_excess_ret"] - core.RF_HALF).to_numpy(), (fd["net_port_excess_ret"] - core.RF_HALF).to_numpy()
    ia, ib = ST.boot_idx(len(a), 5000, 4, 1), ST.boot_idx(len(b), 5000, 4, 2)
    d = np.array([ST.ir(a[i]) - ST.ir(b[j]) for i, j in zip(ia, ib)])
    res["dev_vs_test"] = {"ic_dev": float(icd.mean()), "ic_test": float(ict.mean()), "ic_welch_t": float(tt.statistic), "ic_p": float(tt.pvalue),
                          "net_dev": float(fd["net_port_excess_ret"].mean()), "net_test": float(ft["net_port_excess_ret"].mean()), "net_welch_t": float(tn.statistic), "net_p": float(tn.pvalue),
                          "ir_diff": float(ST.ir(a) - ST.ir(b)), "ir_diff_ci90": [float(np.quantile(d, .05)), float(np.quantile(d, .95))]}
    print("\nD dev vs test:", res["dev_vs_test"])
    # E: ex-ante predictability of monthly IC
    bm = core.load_chars(ctx, ["be_me"])["be_me"].to_numpy()
    u = P.u
    dd = pd.DataFrame({"m": P.months[u], "bm": bm[u], "ret": P.df["raw_ret_1_0"].to_numpy()[u]})
    dd = dd[dd["bm"] > 0]
    q = dd.groupby("m")["bm"].rank(pct=True)
    vs = np.log(dd[q > 0.8].groupby("m")["bm"].median()) - np.log(dd[q <= 0.2].groupby("m")["bm"].median())
    disp = dd.groupby("m")["ret"].std()
    R = ctx.regimes.copy()
    R["value_spread"], R["xs_dispersion"] = vs, disp
    ic = ICP["composite_B1"]
    R["ic"] = ic
    R["ic_lag1"] = ic.shift(1)  # IC of the previous target month: realised at the end of month t = formation eom -> causal
    R["ic_trail12"] = ic.shift(1).rolling(12, min_periods=12).mean()
    R = R.dropna(subset=["ic"])
    preds = ["ic_lag1", "ic_trail12", "vix_eom", "mkt_ret_12m_trailing", "mkt_vol_6m_trailing", "value_spread", "xs_dispersion", "baa10y_eom", "t10y2y_eom"]
    rows = {}
    for p_ in preds:
        out = {}
        for per, sl in (("dev", slice(None, "2020-12-31")), ("test", slice("2021-01-31", None))):
            z = R.loc[sl, ["ic", p_]].dropna()
            x = (z[p_] - z[p_].mean()) / z[p_].std()
            o = sm.OLS(z["ic"], sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
            out[per] = {"slope_per_sd": float(o.params.iloc[1]), "nw_t": float(o.tvalues.iloc[1]), "n": int(len(z))}
        rows[p_] = out
        print(f"E {p_:22s} dev slope {out['dev']['slope_per_sd']:+.4f} (t {out['dev']['nw_t']:+.2f})  test slope {out['test']['slope_per_sd']:+.4f} (t {out['test']['nw_t']:+.2f})")
    res["ic_predictability"] = rows
    R.to_csv(OUT / "regime_table.csv")
    E.dump(OUT / "results.json", res)
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
