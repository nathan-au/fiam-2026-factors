"""
desk_25_model_classes - do more expressive (but causal, deterministic, auditable) models beat the fit-free composites?

Walk-forward INSIDE DEV (expanding window, annual refit, 1-month purge): for forecast year Y in 2017..2020, train on universe rows with target month
<= Dec(Y-1) minus 1 month, predict Y. OOS window = 2017-01..2020-12 (48 months). No hyperparameter search (fixed a priori, so there is no validation
multiple-testing): inputs = 146 JKP-signed universe ranks (or 13 theme scores); target = within-month rank of next-month return in [-1, 1].
  M0  J13z composite (fit-free reference)            M1  B1 (fit-free reference)
  M2  ridge on 146 signed ranks, alpha = 1e4 x n_months (heavy shrinkage)
  M3  theme ridge with an EQUAL-WEIGHT PRIOR: b = 1/13 + ridge fit of (y - T 1/13) on the 13 themes (Bayesian shrinkage to the fit-free composite)
  M4  monotone gradient boosting (xgboost, depth 2, 300 rounds, eta 0.03, subsample 0.5, colsample 0.5, seed 0), monotone +1 in every signed input
      (i.e. monotone in the JKP literature direction): nonlinear but cannot flip a published sign
Rule 'promising': OOS paired IC gain vs M0 t >= 1 AND positive in both 2017-2018 and 2019-2020 AND book net IR >= M0 + 0.15.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from fiam_desks import evaluate as E  # noqa: E402
from fiam_research import core, ledger, stats as ST, themes as TH  # noqa: E402

OUT = HERE / "output"
EXP = "desk_25_model_classes"
_RERUN = True  # second run adds per-year diagnostics only; the 3 variants were ledgered by the first run
YEARS = [2017, 2018, 2019, 2020]


def ridge(X, y, alpha, prior=None):
    p = X.shape[1]
    b0 = np.zeros(p) if prior is None else prior
    r = y - X @ b0
    b = np.linalg.solve(X.T @ X + alpha * np.eye(p), X.T @ r)
    return b0 + b


def main():
    t0 = time.time()
    ctx = core.research_ctx(text=False)
    P = ctx.P
    D = TH.signed_scores(ctx, "rank")
    Tz = TH.theme_scores(ctx, "z")
    u = P.u
    tm = pd.DatetimeIndex(P.months)
    yr = pd.Series(P.y).groupby(P.df["eom"].to_numpy()).rank(pct=True).to_numpy() * 2 - 1
    preds = {k: np.zeros(len(P.df)) for k in ("M2_ridge146", "M3_theme_ridge_prior", "M4_mono_gbm")}
    import xgboost as xgb
    info = {}
    for Y in YEARS:
        tr = u & (tm <= pd.Timestamp(f"{Y - 1}-11-30")) & np.isfinite(yr)
        te = u & (tm.year == Y)
        X, Xt = D.to_numpy()[tr], D.to_numpy()[te]
        n_m = len(np.unique(tm[tr]))
        b = ridge(X, yr[tr], 1e4 * n_m)
        preds["M2_ridge146"][te] = Xt @ b
        T, Tt = Tz.to_numpy()[tr], Tz.to_numpy()[te]
        prior = np.full(T.shape[1], 1 / T.shape[1])
        bt = ridge(T, yr[tr], 1e4 * n_m, prior)
        preds["M3_theme_ridge_prior"][te] = Tt @ bt
        info[Y] = {"train_rows": int(tr.sum()), "theme_weights": dict(zip(Tz.columns, np.round(bt, 3)))}
        mdl = xgb.XGBRegressor(n_estimators=300, max_depth=2, learning_rate=0.03, subsample=0.5, colsample_bytree=0.5, random_state=0, n_jobs=4,
                               monotone_constraints="(" + ",".join(["1"] * X.shape[1]) + ")", tree_method="hist")
        mdl.fit(X, yr[tr])
        preds["M4_mono_gbm"][te] = mdl.predict(Xt)
        print(Y, "trained", info[Y]["train_rows"], flush=True)
    oos = (tm.year >= 2017) & (tm.year <= 2020)
    S = {"M0_J13z": Tz.mean(axis=1).to_numpy(), "M1_B1": core.b1_score(ctx), **preds}
    S = {k: np.where(oos, v, 0.0) for k, v in S.items()}
    ic = {k: E.ic_series(P, v, "dev", oos) for k, v in S.items()}
    # books restricted to OOS months: score 0 before 2017 -> run the dev book and evaluate 2017..2020 frames
    B = core.run_many(ctx, [(k, v, "dev", core.b1_kw()) for k, v in S.items()])
    f = {k: B[k]["frame"].set_index("target_month")["net_port_excess_ret"]["2017-01-31":] for k in S}
    rows = []
    for k in S:
        d = ic[k] - ic["M0_J13z"]
        pn = ST.paired(f[k], f["M0_J13z"])["t"] if k != "M0_J13z" else np.nan
        irn = ST.ir(f[k] - core.RF_HALF)
        dd = float(((1 + f[k]).cumprod() / (1 + f[k]).cumprod().cummax() - 1).min())
        ok = k not in ("M0_J13z", "M1_B1") and ST.tstat(d) >= 1 and d[:"2018-12-31"].mean() > 0 and d["2019-01-31":].mean() > 0 and irn >= ST.ir(f["M0_J13z"] - core.RF_HALF) + 0.15
        dec = "reference" if k in ("M0_J13z", "M1_B1") else ("promising" if ok else ("rejected" if ST.tstat(d) <= -1 else "no evidence"))
        rows.append({"model": k, "ic_oos": float(ic[k].mean()), "ic_t": ST.tstat(ic[k]), "paired_ic_vs_M0": float(d.mean()), "paired_ic_t": ST.tstat(d),
                     "ir_net_oos": irn, "max_dd_oos": dd, "paired_net_t": pn, "gain_17_18": float(d[:"2018-12-31"].mean()), "gain_19_20": float(d["2019-01-31":].mean()),
                     **{f"ic_{y}": float(ic[k][str(y)].mean()) for y in YEARS}, "decision": dec})
        if dec != "reference" and not getattr(sys.modules[__name__], "_RERUN", False):
            ledger.record(EXP, "model_class", k, "walk-forward ML inside DEV", {"years": YEARS}, "dev(train<=Y-1)", "none (fixed hyperparameters)", "dev 2017-2020",
                          3, {"ic": rows[-1]["ic_oos"], "ic_t": rows[-1]["ic_t"], "ir_net": irn, "max_dd_net": dd}, paired_t=pn, decision=dec)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "models_oos_2017_2020.csv", index=False)
    print(R.round(4).to_string())
    for Y, v in info.items():
        print(Y, {k: v_ for k, v_ in sorted(v["theme_weights"].items(), key=lambda x: -x[1])})
    E.dump(OUT / "results.json", {"rows": rows, "info": info})
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
