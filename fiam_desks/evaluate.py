"""Evaluation toolkit shared by every desk experiment: universe rank IC with paired tests, residual IC, size-tercile split, within-(month, sector)
permutation null, placebo lags, and the pre-registered kill/pass rule of docs/EXPERIMENTS.md ("KILL if paired t < 1 or the gain is confined to the
smallest tercile; PASS needs paired t >= 2 AND Bonferroni permutation p <= 0.05 AND not small-tercile-only")."""

import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from . import config as C
from .lp import LC

tstat = LC.tstat
paired = LC.paired
monthly_rank_ic = LC.monthly_rank_ic


def ic_series(panel, score, period, mask=None) -> pd.Series:
    m = panel.mask(period) if mask is None else (panel.mask(period) & mask)
    return monthly_rank_ic(panel.months[m], np.asarray(score, dtype=float)[m], panel.y[m])


def ic_summary(ic: pd.Series) -> dict:
    return {"ic": float(ic.mean()), "t": tstat(ic), "share_pos": float((ic > 0).mean()), "n_months": int(ic.notna().sum())}


def by_year(ic: pd.Series) -> dict:
    return {int(y): round(float(v), 4) for y, v in ic.groupby(ic.index.year).mean().items()}


def resid_ic(panel, block, base, period):
    """Rank IC of the part of `block` orthogonal to `base`, month by month (OLS within the universe)."""
    m = panel.mask(period)
    d = pd.DataFrame({"m": panel.months[m], "b": np.asarray(block)[m], "c": np.asarray(base)[m], "y": panel.y[m]})
    out, corr = {}, {}
    for mo, g in d.groupby("m"):
        c = g["c"].to_numpy() - g["c"].mean()
        b = g["b"].to_numpy() - g["b"].mean()
        den = float(c @ c)
        r = b - (float(c @ b) / den) * c if den > 0 else b
        if np.std(r) < 1e-12:
            continue
        out[mo] = np.corrcoef(rankdata(r), rankdata(g["y"].to_numpy()))[0, 1]
        if np.std(g["b"]) > 0 and np.std(g["c"]) > 0:
            corr[mo] = np.corrcoef(rankdata(g["b"].to_numpy()), rankdata(g["c"].to_numpy()))[0, 1]
    return pd.Series(out), pd.Series(corr)


def perm_null(panel, base_sum, n_base, block, period, w=1.0, n_perm=200, seed=0):
    """Null distribution of the IC gain of adding `block` (weight w) to the composite, obtained by shuffling the block within (month, sector) over
    universe rows. Keeps the block's distribution and industry mix, breaks its link to the stock."""
    m = np.flatnonzero(panel.mask(period))
    base = np.asarray(base_sum)[m]
    b = np.asarray(block)[m]
    mo, sec, y = panel.months[m], panel.sector[m], panel.y[m]
    mg = [np.flatnonzero(mo == x) for x in np.unique(mo)]
    sg = [np.flatnonzero((mo == x) & (sec == s)) for x in np.unique(mo) for s in np.unique(sec[mo == x])]
    yr = [rankdata(y[g]) - rankdata(y[g]).mean() for g in mg]

    def mean_ic(score):
        v = []
        for g, y_ in zip(mg, yr):
            r = rankdata(score[g])
            r = r - r.mean()
            den = np.sqrt((r @ r) * (y_ @ y_))
            v.append((r @ y_) / den if den > 0 else 0.0)
        return float(np.mean(v))

    obs = mean_ic((base + w * b) / (n_base + w)) - mean_ic(base / n_base)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        bp = b.copy()
        for g in sg:
            bp[g] = b[rng.permutation(g)]
        null[k] = mean_ic((base + w * bp) / (n_base + w)) - mean_ic(base / n_base)
    return {"observed_gain": obs, "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
            "p_value": float((1 + np.sum(null >= obs)) / (1 + n_perm)), "n_perm": n_perm}


def block_report(panel, comp_out, block, period, name, w=1.0, n_perm=200, lag_block=None, seed=0) -> dict:
    """Standalone + additive tests of one signed block (in [-1, 1], + = long) against the frozen composite on `period` (universe rows)."""
    G = comp_out.extras["groups"]
    base_sum = G.sum(axis=1).to_numpy()
    base = comp_out.score
    comb = (base_sum + w * block) / (7 + w)
    ic_b, ic_c, ic_k = ic_series(panel, block, period), ic_series(panel, base, period), ic_series(panel, comb, period)
    ri, cor = resid_ic(panel, block, base, period)
    m = panel.mask(period)
    res = {"name": name, "period": period, "weight": w, "nonzero_share_universe": float((np.asarray(block)[m] != 0).mean()),
           "block": ic_summary(ic_b), "block_ic_by_year": by_year(ic_b), "comp_ic": float(ic_c.mean()),
           "comp_plus_block": ic_summary(ic_k), "paired_gain": paired(ic_k, ic_c),
           "residual_ic": float(ri.mean()), "residual_ic_t": tstat(ri), "rank_corr_with_comp": float(cor.mean())}
    for t, lab in enumerate(("small", "mid", "large")):
        mk = panel.terc == t
        bt, ct, kt = ic_series(panel, block, period, mk), ic_series(panel, base, period, mk), ic_series(panel, comb, period, mk)
        res[f"tercile_{lab}"] = {"block_ic": float(bt.mean()), "block_t": tstat(bt), "paired_gain": float((kt - ct).mean()), "paired_t": tstat(kt - ct)}
    if lag_block is not None:
        ic_l = ic_series(panel, lag_block, period)
        res["placebo_lag"] = ic_summary(ic_l)
    if n_perm:
        res["perm"] = perm_null(panel, base_sum, 7, block, period, w=w, n_perm=n_perm, seed=seed)
    return res


def kill_pass(res: dict, n_tests: int = 1) -> str:
    """The project's pre-registered rule (docs/EXPERIMENTS.md), applied to a block_report."""
    g = res["paired_gain"]
    small_only = (res["tercile_small"]["paired_gain"] > 0 and res["tercile_mid"]["paired_gain"] <= 0 and res["tercile_large"]["paired_gain"] <= 0)
    if g["t"] < 1 or small_only:
        return "kill"
    p = res.get("perm", {}).get("p_value", 1.0) * n_tests
    if g["t"] >= 2 and p <= 0.05:
        return "pass"
    return "inconclusive"


def paired_net(frame_a: pd.DataFrame, frame_b: pd.DataFrame) -> dict:
    a = frame_a.set_index("target_month")["net_port_excess_ret"]
    b = frame_b.set_index("target_month")["net_port_excess_ret"]
    return paired(a, b)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
        return None
    if isinstance(o, (pd.DataFrame, pd.Series, np.ndarray)):
        return None
    return o


def dump(path, obj):
    path.write_text(json.dumps(clean(obj), indent=2))


def shuffle_within(panel, x, seed: int) -> np.ndarray:
    """Permute x among UNIVERSE rows within (eom, sector): keeps its distribution and industry mix, breaks the link to the stock (placebo for any desk output)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x).copy()
    idx = np.flatnonzero(panel.u)
    key = pd.Series(list(zip(panel.df["eom"].to_numpy()[idx], panel.sector[idx])))
    for _, g in key.groupby(key):
        ii = idx[g.index.to_numpy()]
        x[ii] = x[rng.permutation(ii)]
    return x
