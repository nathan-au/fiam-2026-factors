"""Statistics for skeptical evaluation: block bootstrap, paired tests, Newey-West t, deflated Sharpe ratio, probability of backtest overfitting (CSCV),
Benjamini-Hochberg FDR.

References (see experiments/desk_16_literature/README.md):
  Bailey & Lopez de Prado (2014) "The Deflated Sharpe Ratio", J. Portfolio Management 40(5).
  Bailey, Borwein, Lopez de Prado & Zhu (2017) "The Probability of Backtest Overfitting", J. Computational Finance 20(4).
  Benjamini & Hochberg (1995) JRSS-B 57(1).  Politis & Romano (1994) stationary / circular block bootstrap.
"""

import itertools

import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

EULER = 0.5772156649


def ir(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.sqrt(12) * x.mean() / x.std(ddof=1)) if len(x) > 2 and x.std(ddof=1) > 0 else np.nan


def tstat(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 and x.std(ddof=1) > 0 else np.nan


def nw_t(x, lags: int = 3) -> float:
    """Newey-West (Bartlett) t-statistic of the mean."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 5:
        return np.nan
    e = x - x.mean()
    v = e @ e / n
    for l in range(1, lags + 1):
        v += 2 * (1 - l / (lags + 1)) * (e[l:] @ e[:-l]) / n
    return float(x.mean() / np.sqrt(v / n)) if v > 0 else np.nan


def boot_idx(n, draws=5000, block=4, seed=0):
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(draws, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return idx.reshape(draws, -1)[:, :n]


def boot_ir(active, draws=5000, block=4, seed=0) -> dict:
    a = np.asarray(active, float)
    idx = boot_idx(len(a), draws, block, seed)
    bs = np.array([ir(a[i]) for i in idx])
    return {"ir": ir(a), "ci90": [float(np.nanquantile(bs, 0.05)), float(np.nanquantile(bs, 0.95))], "p_le_0": float(np.mean(bs <= 0))}


def boot_ir_diff(a, b, draws=5000, block=4, seed=0) -> dict:
    """IR(a) - IR(b) on the same months, circular block bootstrap (paired: same resampled months for both)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx = boot_idx(len(a), draws, block, seed)
    d = np.array([ir(a[i]) - ir(b[i]) for i in idx])
    return {"diff": ir(a) - ir(b), "ci90": [float(np.nanquantile(d, 0.05)), float(np.nanquantile(d, 0.95))], "p_le_0": float(np.mean(d <= 0))}


def paired(a: pd.Series, b: pd.Series) -> dict:
    d = (a - b).dropna()
    return {"diff": float(d.mean()), "t": tstat(d), "nw_t": nw_t(d), "n": int(len(d))}


# ---- deflated Sharpe ratio -------------------------------------------------------------------------------------------------------------------
def expected_max_sr(n_trials: int, var_sr: float) -> float:
    """E[max SR] of n_trials independent zero-skill strategies whose (per-period) SR estimates have variance var_sr (Bailey & LdP 2014 eq. 6)."""
    if n_trials <= 1:
        return 0.0
    return float(np.sqrt(var_sr) * ((1 - EULER) * norm.ppf(1 - 1 / n_trials) + EULER * norm.ppf(1 - 1 / (n_trials * np.e))))


def psr(returns, sr_star: float = 0.0) -> float:
    """Probabilistic Sharpe ratio: P(true per-period SR > sr_star) given skew/kurtosis of the sample."""
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    T = len(r)
    sr = r.mean() / r.std(ddof=1)
    g3, g4 = skew(r), kurtosis(r, fisher=False)
    den = np.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    return float(norm.cdf((sr - sr_star) * np.sqrt(T - 1) / den))


def deflated_sr(returns, trial_srs) -> dict:
    """DSR of `returns` given the per-period Sharpe ratios of ALL trials in the family (the selected one included)."""
    trial_srs = np.asarray(trial_srs, float)
    trial_srs = trial_srs[np.isfinite(trial_srs)]
    n = len(trial_srs)
    v = float(np.var(trial_srs, ddof=1)) if n > 1 else 0.0
    s0 = expected_max_sr(n, v)
    r = np.asarray(returns, float)
    return {"n_trials": n, "sr_annual": float(np.sqrt(12) * r.mean() / r.std(ddof=1)), "sr0_annual": float(np.sqrt(12) * s0), "dsr": psr(r, s0), "psr0": psr(r, 0.0)}


# ---- probability of backtest overfitting --------------------------------------------------------------------------------------------------
def pbo_cscv(M: np.ndarray, n_blocks: int = 8, metric=None) -> dict:
    """Combinatorially symmetric cross-validation (Bailey et al. 2017). M: T x N matrix of per-period returns of N strategy variants.
    For every split of the n_blocks time blocks into halves, pick the variant best in-sample and record its relative OOS rank.
    PBO = share of splits where the in-sample winner ranks below the OOS median."""
    metric = metric or (lambda x: x.mean(axis=0) / np.where(x.std(axis=0, ddof=1) > 0, x.std(axis=0, ddof=1), np.nan))
    M = np.asarray(M, float)
    T, N = M.shape
    blocks = np.array_split(np.arange(T), n_blocks)
    logits, deg = [], []
    for comb in itertools.combinations(range(n_blocks), n_blocks // 2):
        is_idx = np.concatenate([blocks[i] for i in comb])
        oos_idx = np.concatenate([blocks[i] for i in range(n_blocks) if i not in comb])
        s_is, s_oos = metric(M[is_idx]), metric(M[oos_idx])
        best = int(np.nanargmax(s_is))
        rank = (np.sum(s_oos < s_oos[best]) + 0.5 * np.sum(s_oos == s_oos[best])) / N  # in (0, 1)
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(np.log(rank / (1 - rank)))
        deg.append(s_oos[best] - np.nanmedian(s_oos))
    logits = np.array(logits)
    return {"pbo": float(np.mean(logits <= 0)), "n_splits": len(logits), "median_logit": float(np.median(logits)),
            "mean_oos_minus_median": float(np.mean(deg))}


def bh_fdr(pvals, q: float = 0.10) -> np.ndarray:
    """Benjamini-Hochberg: boolean array of discoveries at FDR q."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    thr = q * np.arange(1, n + 1) / n
    ok = p[order] <= thr
    k = np.max(np.flatnonzero(ok)) + 1 if ok.any() else 0
    out = np.zeros(n, bool)
    out[order[:k]] = True
    return out


def p_from_t(t: float) -> float:
    """One-sided normal p-value of a t-statistic (H1: > 0)."""
    return float(1 - norm.cdf(t)) if np.isfinite(t) else 1.0
