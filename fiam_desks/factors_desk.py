"""Factors desk: the frozen 7-group, 18-factor composite of experiments/largecap (no fitting), plus the per-stock diagnostics the PM
uses as desk-internal confidence. Output contract (every desk): `DeskOutput.score` in [-1, 1] (+ = long), 0 outside the universe;
NaN never appears in `score` (a desk with no view says 0); everything else is optional diagnostics.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C


@dataclass
class DeskOutput:
    name: str
    score: np.ndarray
    extras: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)


def group_scores(panel, u=None, key=None, min_group=5) -> pd.DataFrame:
    """Seven factor-group scores for every panel row (0 outside the universe): re-rank each factor within (month[, key]) over
    universe rows, flip by the pre-registered sign, average within group. `key` (e.g. sector) gives a within-key ranking."""
    df = panel.df
    u = panel.u if u is None else u
    X = df.loc[u, C.FEATURES]
    grp = [df.loc[u, "eom"]] + ([df.loc[u, key]] if key else [])
    R = X.groupby(grp).rank(pct=True) * 2 - 1
    if key:
        n = X.groupby(grp)[C.FEATURES[0]].transform("count")
        R = R.where(n >= min_group)
    R = R * pd.Series(C.SIGNS)
    G = pd.DataFrame({g: R[list(fs)].mean(axis=1) for g, fs in C.FACTOR_GROUPS.items()})
    out = pd.DataFrame(0.0, index=df.index, columns=list(C.FACTOR_GROUPS))
    out.loc[u] = G.fillna(0.0).to_numpy()
    return out


def composite(panel, variant: str = "frozen") -> DeskOutput:
    """variant `frozen` = the published composite. `sector_neutral` re-ranks within GICS sector (an experiment, not the default)."""
    G = group_scores(panel, key="sector" if variant == "sector_neutral" else None)
    score = G.mean(axis=1).to_numpy()
    # desk-internal confidence: how many of the 7 groups agree with the composite's sign (only meaningful for non-zero scores)
    agree = (np.sign(G.to_numpy()) == np.sign(score)[:, None]).sum(axis=1) / G.shape[1]
    agree = np.where(panel.u, agree, np.nan)
    disp = G.std(axis=1).to_numpy()
    return DeskOutput("factors", score, {"groups": G, "agreement": agree, "group_dispersion": disp},
                      {"variant": variant, "features": C.FEATURES})


def urank(values, panel, key=None, min_group=5) -> np.ndarray:
    """Within-month percentile rank to [-1, 1] over UNIVERSE rows; 0 outside the universe, for NaN values, and for groups smaller than
    `min_group` (the project convention: missing = the honest 'no view' value)."""
    v = pd.Series(np.asarray(values, dtype=float))
    d = pd.DataFrame({"m": panel.df["eom"].to_numpy(), "v": v.to_numpy()})
    u = panel.u
    grp = ["m"]
    if key is not None:
        d["k"] = np.asarray(key)
        grp.append("k")
    sub = d[u & np.isfinite(d["v"].to_numpy())]
    r = sub.groupby(grp)["v"].rank(pct=True) * 2 - 1
    r = r.where(sub.groupby(grp)["v"].transform("count") >= min_group)
    out = np.zeros(len(d))
    out[r.index.to_numpy()] = r.fillna(0.0).to_numpy()
    return out


def add_groups(base: DeskOutput, blocks: dict, name: str = "factors+") -> DeskOutput:
    """Frozen 7-group composite plus extra equal-weight groups (each in [-1, 1], + = long, 0 = no view): score = (sum of the 7 group scores + sum of blocks) / (7 + k).
    Diagnostics (group agreement) stay those of the 7 frozen groups."""
    G7 = base.extras["groups"].sum(axis=1).to_numpy()
    k = len(blocks)
    score = (G7 + sum(blocks.values())) / (7 + k)
    return DeskOutput(name, score, {**base.extras, "blocks": blocks}, {**base.config, "extra_groups": list(blocks)})
