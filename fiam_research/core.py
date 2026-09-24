"""Research context and the full metric suite. Imports the frozen desk system; never modifies it.

Ctx          panel + frozen desk parts + extended short interest + market proxy + style spreads + regime table
book()       any score vector -> the frozen LP (optionally with per-row controls / cfg overrides) -> result dict with frames
metrics()    IC, IR (net / gross), Sharpe, drawdown, beta, turnover, positions, exposure, trading and borrow cost, style exposures (holdings and returns
             based), by year, by regime, by sector, long/short contribution, bootstrap interval
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from fiam_desks import config as C, evaluate as E, factors_desk as FD, lp as L, system as S
from fiam_desks.panel import Panel

from . import si_ext, stats as ST

RF_HALF = 0.04 / 12.0
STYLE_PROXIES = {  # characteristic (panel column, already rank-transformed over all stocks) and the sign of the long side
    "size_small": ("x_size", -1), "value": ("be_me", +1), "momentum": ("x_mom", +1), "lowvol": ("x_vol", -1),
    "quality": ("qmj", +1), "profitability": ("gp_at", +1), "investment_low": ("at_gr1", -1),
}


def _fred(name, col):
    d = pd.read_csv(C.CACHE / f"{name}.csv", parse_dates=["observation_date"]).rename(columns={"observation_date": "date", name: col})
    d[col] = pd.to_numeric(d[col], errors="coerce")
    return d.dropna()


class Ctx:
    def __init__(self, smoke: bool = False, text: bool = True, panel=None):
        self.P = P = panel if panel is not None else Panel(smoke=smoke)
        self.comp = FD.composite(P)
        from fiam_desks import si as SI0
        self.S0 = SI0.features(P)            # frozen (2020-06 onward)
        self.S = si_ext.features(P)          # extended (2018-01 onward)
        if text:
            _, _, self.raw, self.txt = S.desks(P, need_si=False)
        self._market()
        self._styles()
        self._regimes()

    # ---- shared series -----------------------------------------------------------------------------------------------------------------
    def _market(self):
        df = self.P.df
        w = df["raw_me"].to_numpy()
        ok = np.isfinite(w) & np.isfinite(df["y"].to_numpy())
        d = pd.DataFrame({"m": df["target_month"].to_numpy()[ok], "w": w[ok], "y": df["y"].to_numpy()[ok]})
        self.mkt = d.groupby("m").apply(lambda g: np.average(g["y"], weights=g["w"]), include_groups=False).rename("mkt_vw")

    def _styles(self):
        P = self.P
        u = P.u
        d = pd.DataFrame({"m": P.months[u], "y": P.y[u]})
        F = {}
        for k, (col, sgn) in STYLE_PROXIES.items():
            x = sgn * P.df[col].to_numpy()[u]
            q = pd.Series(x).groupby(d["m"].to_numpy()).rank(pct=True).to_numpy()
            F[k] = d[q > 0.8].groupby("m")["y"].mean() - d[q <= 0.2].groupby("m")["y"].mean()
        self.styles = pd.DataFrame(F)

    def _regimes(self):
        months = pd.DatetimeIndex(sorted(set(self.P.months)))
        eoms = months - pd.offsets.MonthEnd(1)
        vix = _fred("VIXCLS", "vix").set_index("date")["vix"]
        baa = _fred("BAA10Y", "baa10y").set_index("date")["baa10y"]
        crv = _fred("T10Y2Y", "t10y2y").set_index("date")["t10y2y"]
        last = lambda s, t: float(s[:t].iloc[-1]) if len(s[:t]) else np.nan
        mkt = self.mkt.copy()
        mk_hist = mkt.copy()  # realised VW market excess return by target month (known after that month ends)
        R = pd.DataFrame(index=months)
        R["vix_eom"] = [last(vix, t) for t in eoms]
        R["baa10y_eom"] = [last(baa, t) for t in eoms]
        R["t10y2y_eom"] = [last(crv, t) for t in eoms]
        # trailing market return / vol over the 12 / 6 months ENDING at the formation eom (i.e. target months <= eom)
        R["mkt_ret_12m_trailing"] = [float((1 + mk_hist[(mk_hist.index <= t) & (mk_hist.index > t - pd.DateOffset(months=12))]).prod() - 1)
                                     if (mk_hist.index <= t).sum() >= 12 else np.nan for t in eoms]
        R["mkt_vol_6m_trailing"] = [float(mk_hist[(mk_hist.index <= t) & (mk_hist.index > t - pd.DateOffset(months=6))].std() * np.sqrt(12))
                                    if (mk_hist.index <= t).sum() >= 6 else np.nan for t in eoms]
        R["mkt_ret_contemp"] = mkt.reindex(months).to_numpy()  # descriptive only (not known at formation)
        self.regimes = R

    # ---- scoring helpers ------------------------------------------------------------------------------------------------------------------
    def dtc_block(self, extended: bool = True):
        S_ = self.S if extended else self.S0
        return -FD.urank(S_["dtc"], self.P)

    def baseline_score(self, extended_si: bool = False):
        """The frozen default factors desk: 7-group composite + days-to-cover 8th group."""
        return FD.add_groups(self.comp, {"dtc": self.dtc_block(extended_si)}).score


def book(ctx: Ctx, score, period: str, si_cap: bool | float = True, extended_si: bool = False, cfg: dict | None = None, extras: dict | None = None,
         keep_frames: bool = True):
    """Run the frozen LP on `score` for `period` ('dev' or 'test'). si_cap True -> 10% cap (as the baseline); a float sets the level; False = none."""
    P = ctx.P
    S_ = ctx.S if extended_si else ctx.S0
    ex = {"sir": S_["sir"]}
    ex.update(extras or {})
    dec = {"pred": np.asarray(score, float), "veto_long": ex.pop("veto_long", np.zeros(len(P.df), bool)),
           "veto_short": ex.pop("veto_short", np.zeros(len(P.df), bool)), "w_mult": ex.pop("w_mult", np.ones(len(P.df)))}
    from fiam_desks import pm as PM
    preds = PM.lp_frame(P, dec, period, ex)
    base_cfg = dict(C.LP_BASE)
    if si_cap is not False:
        base_cfg["short_cap_val"] = C.SI_CAP if si_cap is True else float(si_cap)
    base_cfg.update(cfg or {})
    res = L.evaluate_book(preds, base_cfg, keep_frames=keep_frames)
    res["cfg"] = base_cfg
    return res


def active(res) -> pd.Series:
    fr = res["frame"].set_index("target_month")
    return fr["net_port_excess_ret"] - RF_HALF


def metrics(ctx: Ctx, res: dict, score=None, period: str = "dev", boot: bool = True) -> dict:
    P = ctx.P
    fr = res["frame"].set_index("target_month")
    net, gross = fr["net_port_excess_ret"], fr["port_excess_ret"]
    act = net - RF_HALF
    out = {k: res[k] for k in ("ir_gross", "ir_net", "sharpe_net", "cagr_net", "cagr_gross", "alpha_t_net", "beta", "beta_t", "max_dd_net", "max_dd_gross",
                               "one_way_turnover", "min_positions", "max_positions", "avg_gross", "max_gross", "avg_net_exposure", "min_net_exposure",
                               "max_net_exposure", "months_turnover_relaxed", "long_leg_cagr", "short_leg_cagr")}
    out["n_months"] = int(len(fr))
    out["sharpe_gross"] = float(np.sqrt(12) * gross.mean() / gross.std())
    out["mean_net_monthly"] = float(net.mean())
    out["avg_positions"] = float(fr["n_positions"].mean())
    out["avg_n_long"], out["avg_n_short"] = float(fr["n_long"].mean()), float(fr["n_short"].mean())
    out["trade_cost_bp_month"] = float(1e4 * fr["trade_cost"].mean())
    out["borrow_cost_bp_month"] = float(1e4 * fr["borrow_cost"].mean())
    out["long_contrib_ann"] = float(12 * fr["long_leg_ret"].mean())
    out["short_contrib_ann"] = float(12 * fr["short_leg_ret"].mean())
    out["calendar_year_net"] = {int(y): float((1 + s).prod() - 1) for y, s in net.groupby(net.index.year)}
    out["ir_net_by_year"] = {int(y): ST.ir(s) for y, s in act.groupby(act.index.year)}
    # market beta on the panel's value-weighted market (covers 2015-16, where the FRED S&P series is missing)
    mk = ctx.mkt.reindex(net.index)
    ols = sm.OLS(net.to_numpy(), sm.add_constant(mk.to_numpy()), missing="drop").fit()
    out["beta_vw_mkt"], out["beta_vw_mkt_t"] = float(ols.params[1]), float(ols.tvalues[1])
    # IC of the score on the universe
    if score is not None:
        ic = E.ic_series(P, score, period)
        out["ic"], out["ic_t"], out["ic_by_year"] = float(ic.mean()), E.tstat(ic), E.by_year(ic)
    # returns-based style attribution
    X = pd.concat([mk.rename("mkt"), ctx.styles.reindex(net.index)], axis=1)
    o2 = sm.OLS(net, sm.add_constant(X), missing="drop").fit()
    out["style_attr"] = {"alpha_ann": float(12 * o2.params["const"]), "alpha_t": float(o2.tvalues["const"]), "r2": float(o2.rsquared),
                         "loadings": {k: [round(float(o2.params[k]), 3), round(float(o2.tvalues[k]), 2)] for k in X.columns}}
    # holdings-based exposures (weighted average of all-stock rank characteristics, per month, then averaged)
    h = res["holdings"]
    hk = h[["permno", "eom", "target_month", "weight", "sector", C.TARGET_COL]].merge(
        P.df[["permno", "eom"] + list({c for c, _ in STYLE_PROXIES.values()})], on=["permno", "eom"], how="left")
    xs = P.df.groupby("eom")["x_size"].rank(pct=True) * 2 - 1
    hk = hk.merge(P.df[["permno", "eom"]].assign(size_rank=xs.to_numpy()), on=["permno", "eom"], how="left")
    expo = {}
    for k, (col, sgn) in STYLE_PROXIES.items():
        c = "size_rank" if col == "x_size" else col
        expo[k] = float((hk[c] * hk["weight"] * sgn).groupby(hk["target_month"]).sum().mean())
    out["holdings_style_exposure"] = expo
    # sector P&L contribution (annualised, gross)
    hk["pnl"] = hk["weight"] * hk[C.TARGET_COL]
    sec = hk.groupby("sector")["pnl"].sum() / len(fr) * 12
    out["sector_contrib_ann"] = {str(k): round(float(v), 4) for k, v in sec.sort_values().items()}
    # regimes (VIX tercile at formation uses fixed 2015-2026 terciles; descriptive)
    R = ctx.regimes.reindex(net.index)
    vq = pd.qcut(ctx.regimes["vix_eom"], 3, labels=["low", "mid", "high"]).reindex(net.index)
    out["by_regime"] = {
        "vix_tercile": {str(k): {"mean_net": float(net[vq == k].mean()), "ir": ST.ir(act[vq == k]), "n": int((vq == k).sum())} for k in ["low", "mid", "high"]},
        "mkt_contemp_up_down": {lab: {"mean_net": float(net[m].mean()), "n": int(m.sum())} for lab, m in
                                (("up", R["mkt_ret_contemp"] > 0), ("down", R["mkt_ret_contemp"] <= 0))},
        "trailing12_mkt": {lab: {"mean_net": float(net[m].mean()), "ir": ST.ir(act[m]), "n": int(m.sum())} for lab, m in
                           (("up", R["mkt_ret_12m_trailing"] > 0), ("down", R["mkt_ret_12m_trailing"] <= 0))},
    }
    if boot:
        out["boot_ir_net"] = ST.boot_ir(act.to_numpy())
    return out


def short(m: dict) -> str:
    return (f"IC {m.get('ic', np.nan):+.4f} (t {m.get('ic_t', np.nan):+.2f}) | IR net {m['ir_net']:+.3f} gross {m['ir_gross']:+.3f} | SR {m['sharpe_net']:+.2f} | "
            f"DD {100 * m['max_dd_net']:.1f}% | beta {m['beta_vw_mkt']:+.2f} | TO {100 * m['one_way_turnover']:.1f}% | pos {m['min_positions']}-{m['max_positions']} | "
            f"cost {m['trade_cost_bp_month'] + m['borrow_cost_bp_month']:.1f}bp/m")


# ---- research reference (desk_15 decision) -----------------------------------------------------------------------------------------------
def research_ctx(text: bool = True) -> Ctx:
    """The research context from desk_15 on: panel keeps delisting rows (y = 0), short interest from 2018-01."""
    from .panel_ext import PanelX
    return Ctx(text=text, panel=PanelX(0.0))


def b1_score(ctx: Ctx):
    """B1 = the baseline's signal with the audit corrections: composite + days-to-cover (extended SI) as 8th group."""
    return ctx.baseline_score(extended_si=True)


def b1_book(ctx: Ctx, score, period, **kw):
    """Book with the B1 PM settings (lc_t10 LP + 10% SI cap with extended SI)."""
    kw.setdefault("si_cap", True)
    kw.setdefault("extended_si", True)
    return book(ctx, score, period, **kw)


def load_chars(ctx: Ctx, cols) -> pd.DataFrame:
    """Raw (untransformed) characteristic columns aligned to ctx.P.df rows."""
    cols = [c for c in dict.fromkeys(cols)]
    d = pd.read_parquet(C.CHARS_FILE, columns=["permno", "eom"] + cols)
    d["eom"] = pd.to_datetime(d["eom"])
    d = d.drop_duplicates(["permno", "eom"])
    m = ctx.P.df[["permno", "eom"]].merge(d, on=["permno", "eom"], how="left", validate="one_to_one")
    assert len(m) == len(ctx.P.df)
    return m[cols].reset_index(drop=True)


# ---- parallel books (fork: workers share the context copy-on-write) -------------------------------------------------------------------------
_CTX = None


def _job(item):
    name, score, period, kw = item
    import warnings
    warnings.filterwarnings("ignore")
    return name, book(_CTX, score, period, **kw)


def run_many(ctx: Ctx, jobs: list, workers: int = 7) -> dict:
    """jobs: list of (name, score, period, book-kwargs). Returns {name: result}. Deterministic (each LP solve is deterministic)."""
    global _CTX
    import multiprocessing as mp
    _CTX = ctx
    if workers <= 1 or len(jobs) == 1:
        return dict(_job(j) for j in jobs)
    with mp.get_context("fork").Pool(min(workers, len(jobs))) as pool:
        return dict(pool.map(_job, jobs, chunksize=1))


def b1_kw(**kw):
    kw.setdefault("si_cap", True)
    kw.setdefault("extended_si", True)
    return kw
