"""
FIAM 2026 - Gated score-formation model (operationalizes "RankGLU: Residual
Gated Score Formation for Cross-Sectional Stock Prediction", arXiv
2606.08930, Jun 2026).

The paper's idea (per docs/PAPERS.md §1): a prediction-head architecture with
a direct linear scoring pathway PLUS a bounded, gated nonlinear branch,
designed to capture some nonlinear interaction without overfitting unstable
return magnitudes -- validated in the paper only on Chinese equity indices
(CSI300/CSI800), so transfer to this panel is explicitly untested going in.

This does NOT reproduce the paper's exact architecture/training details (not
available beyond the abstract/summary, and no PyTorch/deep-learning
framework is used anywhere else in this project). What IS implemented, a
small hand-written network trained with plain numpy (forward pass + manual
backprop + Adam, no autodiff library):

    z      = X @ W1 + b1                    (hidden, dim H)
    gate   = sigmoid(X @ Wg + bg)            (hidden, dim H, in [0,1])
    nonlin = tanh(z) * gate                  (bounded branch, in [-1,1])
    score  = X @ w_lin + b_lin + nonlin @ w_out + b_out

-- a direct linear pathway (X @ w_lin) plus a bounded multiplicative gated
branch (nonlin @ w_out), summed residually, matching the paper's description
("maintains a direct linear scoring pathway while adding a bounded
multiplicative branch"). Hidden width H and L2 regularization are tuned on
the validation fold, same discipline as every other script here.

Self-contained: data loading, rank transform, walk-forward schedule,
investability screen, beta-neutral LP, and evaluation code are ported from
ols.py / xgb.py (not imported).

Run: .venv/bin/python experiments/rankglu/rankglu.py
Outputs (all in output/): oos_predictions_rankglu.csv,
    portfolio_holdings_beta_neutral_rankglu.csv,
    portfolio_returns_beta_neutral_rankglu.csv, rankglu_results.json
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import linprog

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent  # experiments/<name>/
BASE = HERE.parents[1]  # project root: fiam/ data and cache/ are shared
FIAM_DIR = BASE / "fiam"
CACHE = BASE / "cache"
OUTPUT = HERE / "output"
CACHE.mkdir(exist_ok=True)
OUTPUT.mkdir(exist_ok=True)

CHARS_FILE = FIAM_DIR / "chars_final_with_names.parquet"
FACTOR_LIST = FIAM_DIR / "factor_char_list.csv"

TARGET_COL = "ret_exc_lead1m"
OOS_START = pd.Timestamp("2021-01-01")
OOS_END = pd.Timestamp("2026-08-31")
RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# 1. Load data and build the model table (ported from ols.py)
# ---------------------------------------------------------------------------


def load_model_table() -> tuple[pd.DataFrame, list[str]]:
    stock_vars = pd.read_csv(FACTOR_LIST)["variable"].tolist()
    keep_cols = stock_vars + ["permno", "eom", "ticker", "company_name", "me", TARGET_COL]
    df = pd.read_parquet(CHARS_FILE, columns=keep_cols)
    df["eom"] = pd.to_datetime(df["eom"])
    df["raw_prc"] = df["prc"]
    df["raw_dolvol_126d"] = df["dolvol_126d"]
    df["raw_me"] = df["me"]
    df["raw_beta_60m"] = df["beta_60m"]
    df["target_month"] = (df["eom"] + pd.offsets.MonthBegin(1)).values.astype("datetime64[M]")
    df["target_month"] = pd.to_datetime(df["target_month"]) + pd.offsets.MonthEnd(0)
    df = df[df[TARGET_COL].notna()].copy()
    return df, stock_vars


def cross_sectional_rank_transform(df: pd.DataFrame, stock_vars: list[str]) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("eom")
    for var in stock_vars:
        med = g[var].transform("median")
        out[var] = out[var].fillna(med).fillna(0.0)
    g = out.groupby("eom")
    for var in stock_vars:
        r = g[var].rank(method="dense")
        rmax = r.groupby(out["eom"]).transform("max")
        out[var] = np.where(rmax > 0, (r / rmax) * 2 - 1, 0.0)
    return out


# ---------------------------------------------------------------------------
# 2. RankGLU: hand-written forward/backward pass + Adam
# ---------------------------------------------------------------------------


def init_params(n_features: int, h_dim: int) -> dict:
    scale1 = np.sqrt(2.0 / n_features)
    return {
        "W1": RNG.normal(0, scale1, size=(n_features, h_dim)),
        "b1": np.zeros(h_dim),
        "Wg": RNG.normal(0, scale1, size=(n_features, h_dim)),
        "bg": np.zeros(h_dim),
        "w_lin": np.zeros(n_features),
        "b_lin": 0.0,
        "w_out": RNG.normal(0, np.sqrt(2.0 / h_dim), size=h_dim),
        "b_out": 0.0,
    }


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def forward(params, X):
    z = X @ params["W1"] + params["b1"]
    g_pre = X @ params["Wg"] + params["bg"]
    gate = sigmoid(g_pre)
    tanh_z = np.tanh(z)
    nonlin = tanh_z * gate
    score = X @ params["w_lin"] + params["b_lin"] + nonlin @ params["w_out"] + params["b_out"]
    cache = {"X": X, "z": z, "gate": gate, "tanh_z": tanh_z, "nonlin": nonlin}
    return score, cache


def backward(params, cache, y_pred, y_true, l2):
    n = len(y_true)
    dscore = (2.0 / n) * (y_pred - y_true)  # d(MSE)/d(score)

    grads = {}
    grads["w_lin"] = cache["X"].T @ dscore + 2 * l2 * params["w_lin"]
    grads["b_lin"] = dscore.sum()
    grads["w_out"] = cache["nonlin"].T @ dscore + 2 * l2 * params["w_out"]
    grads["b_out"] = dscore.sum()

    dnonlin = np.outer(dscore, params["w_out"])  # (n, H)
    dtanh_z = dnonlin * cache["gate"]
    dgate = dnonlin * cache["tanh_z"]

    dz = dtanh_z * (1 - cache["tanh_z"] ** 2)
    dg_pre = dgate * cache["gate"] * (1 - cache["gate"])

    grads["W1"] = cache["X"].T @ dz + 2 * l2 * params["W1"]
    grads["b1"] = dz.sum(axis=0)
    grads["Wg"] = cache["X"].T @ dg_pre + 2 * l2 * params["Wg"]
    grads["bg"] = dg_pre.sum(axis=0)
    return grads


class Adam:
    def __init__(self, params, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) if isinstance(v, np.ndarray) else 0.0 for k, v in params.items()}
        self.v = {k: np.zeros_like(v) if isinstance(v, np.ndarray) else 0.0 for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (grads[k] ** 2)
            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] = params[k] - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def train_rankglu(X_train, y_train, X_val, y_val, h_dim, l2, epochs=20, batch_size=4096, lr=5e-3):
    params = init_params(X_train.shape[1], h_dim)
    opt = Adam(params, lr=lr)
    n = len(y_train)
    for epoch in range(epochs):
        idx = RNG.permutation(n)
        for start in range(0, n, batch_size):
            batch_idx = idx[start:start + batch_size]
            Xb, yb = X_train[batch_idx], y_train[batch_idx]
            y_pred, cache = forward(params, Xb)
            grads = backward(params, cache, y_pred, yb, l2)
            opt.step(params, grads)
    val_pred, _ = forward(params, X_val)
    val_mse = float(np.mean((y_val - val_pred) ** 2))
    return params, val_mse


HDIM_GRID = [8, 16, 32]
L2_GRID = [1e-4, 1e-3]


def fit_best_rankglu(X_train, y_train, X_val, y_val):
    best_params, best_config, best_val_mse = None, None, np.inf
    for h_dim in HDIM_GRID:
        for l2 in L2_GRID:
            params, val_mse = train_rankglu(X_train, y_train, X_val, y_val, h_dim, l2)
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_params = params
                best_config = {"h_dim": h_dim, "l2": l2}
    return best_params, best_config, best_val_mse


# ---------------------------------------------------------------------------
# 3. Expanding-window walk-forward
# ---------------------------------------------------------------------------


def year_bounds(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")


def run_walk_forward(df: pd.DataFrame, stock_vars: list[str]):
    test_years = range(2021, 2027)
    preds, fold_params = [], []

    for test_year in test_years:
        test_start, test_end = year_bounds(test_year)
        test_end = min(test_end, OOS_END)
        val_start, _ = year_bounds(test_year - 2)
        _, val_end = year_bounds(test_year - 1)

        train_mask = df["target_month"] < val_start
        val_mask = (df["target_month"] >= val_start) & (df["target_month"] <= val_end)
        test_mask = (df["target_month"] >= test_start) & (df["target_month"] <= test_end)
        train, val, test = df[train_mask], df[val_mask], df[test_mask]
        if train.empty or val.empty or test.empty:
            continue

        X_train, y_train = train[stock_vars].values, train[TARGET_COL].values
        X_val, y_val = val[stock_vars].values, val[TARGET_COL].values

        params, config, val_mse = fit_best_rankglu(X_train, y_train, X_val, y_val)
        config.update({"test_year": test_year, "val_mse": val_mse})
        fold_params.append(config)

        X_test = test[stock_vars].values
        y_pred, _ = forward(params, X_test)

        fold = test[
            ["permno", "target_month", "ticker", "company_name", TARGET_COL,
             "raw_prc", "raw_dolvol_126d", "raw_me", "raw_beta_60m"]
        ].copy()
        fold["pred"] = y_pred
        fold["test_year"] = test_year
        fold["n_train"] = len(train)
        fold["n_val"] = len(val)
        preds.append(fold)

        print(
            f"[fold {test_year}] train={len(train):,}, val={len(val):,}, test={len(test):,}, "
            f"h_dim={config['h_dim']} l2={config['l2']:g} val_mse={val_mse:.6f}"
        )

    return pd.concat(preds, ignore_index=True), fold_params


# ---------------------------------------------------------------------------
# 4. Evaluation, portfolio construction (ported from ols.py / xgb.py, unchanged)
# ---------------------------------------------------------------------------


def oos_r2(actual, predicted):
    return 1.0 - np.sum((actual - predicted) ** 2) / np.sum(actual**2)


MIN_DOLLAR_VOLUME = 10_000_000.0
MAX_WEIGHT = 0.01


def build_beta_neutral_portfolio(preds):
    holdings, monthly_stats = [], []
    for month, grp in preds.groupby("target_month"):
        grp = grp[grp["raw_dolvol_126d"] >= MIN_DOLLAR_VOLUME]
        grp = grp.dropna(subset=["raw_beta_60m"]).reset_index(drop=True)
        n = len(grp)
        if n < 2 * int(1.0 / MAX_WEIGHT):
            continue
        pred = grp["pred"].values
        beta = grp["raw_beta_60m"].values
        c = np.concatenate([-pred, pred])
        A_eq = np.array([
            np.concatenate([np.ones(n), -np.ones(n)]),
            np.concatenate([beta, -beta]),
            np.concatenate([np.ones(n), np.ones(n)]),
        ])
        b_eq = [0.0, 0.0, 2.0]
        bounds = [(0.0, MAX_WEIGHT)] * (2 * n)
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if not res.success:
            print(f"  [warn] LP infeasible for {month.date()}, skipping month")
            continue
        w = res.x[:n] - res.x[n:]
        keep = np.abs(w) > 1e-8
        both = grp.loc[keep].copy()
        both["weight"] = w[keep]
        both["target_month"] = month
        holdings.append(both[["target_month", "permno", "ticker", "company_name", "weight", TARGET_COL, "pred", "raw_beta_60m"]])
        port_ret = (both["weight"] * both[TARGET_COL]).sum()
        long_mask = both["weight"] > 0
        long_ret = (both.loc[long_mask, "weight"] * both.loc[long_mask, TARGET_COL]).sum()
        short_ret = (both.loc[~long_mask, "weight"] * both.loc[~long_mask, TARGET_COL]).sum()
        gross = both["weight"].abs().sum()
        net = both["weight"].sum()
        beta_exposure = (both["weight"] * both["raw_beta_60m"]).sum()
        monthly_stats.append({
            "target_month": month, "port_excess_ret": port_ret, "long_leg_ret": long_ret,
            "short_leg_ret": short_ret, "gross_exposure": gross, "net_exposure": net,
            "beta_exposure": beta_exposure, "n_long": int((both["weight"] > 0).sum()),
            "n_short": int((both["weight"] < 0).sum()), "n_positions": int(keep.sum()),
        })
    holdings_df = pd.concat(holdings, ignore_index=True)
    stats_df = pd.DataFrame(monthly_stats).sort_values("target_month").reset_index(drop=True)
    return holdings_df, stats_df


def load_fred_series():
    tb3ms = pd.read_csv(CACHE / "TB3MS.csv", parse_dates=["observation_date"])
    tb3ms = tb3ms.rename(columns={"observation_date": "month", "TB3MS": "tb3ms_annual_pct"})
    tb3ms["month"] = tb3ms["month"].values.astype("datetime64[M]")
    tb3ms["month"] = pd.to_datetime(tb3ms["month"]) + pd.offsets.MonthEnd(0)
    tb3ms["rf_monthly"] = tb3ms["tb3ms_annual_pct"] / 100.0 / 12.0
    sp = pd.read_csv(CACHE / "SP500.csv", parse_dates=["observation_date"])
    sp = sp.rename(columns={"observation_date": "date", "SP500": "close"})
    sp = sp.dropna(subset=["close"])
    sp["month"] = sp["date"].values.astype("datetime64[M]")
    sp["month"] = pd.to_datetime(sp["month"]) + pd.offsets.MonthEnd(0)
    sp_monthly = sp.sort_values("date").groupby("month")["close"].last().reset_index()
    sp_monthly["sp500_ret"] = sp_monthly["close"].pct_change()
    return tb3ms[["month", "tb3ms_annual_pct", "rf_monthly"]], sp_monthly[["month", "sp500_ret"]]


def compute_turnover_and_concentration(holdings_df):
    piv = holdings_df.pivot_table(index="target_month", columns="permno", values="weight", fill_value=0.0).sort_index()
    diffs = piv.diff().abs().sum(axis=1, min_count=1) / 2.0
    turnover = (diffs.iloc[1:] / 2.0).rename("turnover")
    abs_w = holdings_df["weight"].abs()
    top10_share_by_month = holdings_df.groupby("target_month")["weight"].apply(lambda w: w.abs().nlargest(10).sum() / 2.0)
    return {
        "avg_monthly_turnover": float(turnover.mean()), "min_monthly_turnover": float(turnover.min()),
        "max_monthly_turnover": float(turnover.max()), "avg_position_weight_abs": float(abs_w.mean()),
        "max_position_weight_abs": float(abs_w.max()), "avg_top10_share_of_gross": float(top10_share_by_month.mean()),
    }


def compute_short_book_characteristics(holdings_df):
    chars = pd.read_parquet(CHARS_FILE, columns=["permno", "eom", "me", "dolvol_126d"])
    chars["eom"] = pd.to_datetime(chars["eom"])
    h = holdings_df.copy()
    h["target_month"] = pd.to_datetime(h["target_month"])
    h["char_month"] = (h["target_month"] - pd.offsets.MonthBegin(1)).values.astype("datetime64[M]")
    h["char_month"] = pd.to_datetime(h["char_month"]) + pd.offsets.MonthEnd(0)
    m = h.merge(chars, left_on=["permno", "char_month"], right_on=["permno", "eom"], how="left")
    short = m[m["weight"] < 0]
    return {
        "short_book_avg_market_cap_musd": float(short["me"].mean()), "short_book_median_market_cap_musd": float(short["me"].median()),
        "short_book_avg_dollar_volume_usd": float(short["dolvol_126d"].mean()),
        "short_book_share_below_2000m_cap": float((short["me"] < 2000).mean()),
        "short_book_share_below_1000m_cap": float((short["me"] < 1000).mean()),
    }


def drawdown_series(rets):
    wealth = (1 + rets).cumprod()
    peak = wealth.cummax().clip(lower=1.0)
    return wealth / peak - 1


def compute_drawdown_stats(rets, dates, cagr):
    rets = rets.reset_index(drop=True)
    dates = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    dd = drawdown_series(rets)
    trough_i = int(dd.idxmin())
    max_dd = float(dd.iloc[trough_i])
    if max_dd < 0:
        at_high = dd.iloc[: trough_i + 1] >= 0
        peak_i = int(at_high[at_high].index.max()) if at_high.any() else -1
        recovered = dd.iloc[trough_i + 1:] >= 0
        recovery_i = int(recovered[recovered].index.min()) if recovered.any() else None
    else:
        peak_i, recovery_i = trough_i, trough_i

    def date_at(i):
        if i is None:
            return None
        if i < 0:
            return str((dates.iloc[0] - pd.offsets.MonthEnd(1)).date())
        return str(dates.iloc[i].date())

    underwater = (dd < 0).astype(int)
    runs = underwater.groupby((underwater == 0).cumsum()).sum()
    longest_underwater = int(runs.max()) if len(runs) else 0
    return {
        "max_drawdown": max_dd, "max_drawdown_peak_date": date_at(peak_i),
        "max_drawdown_trough_date": date_at(trough_i), "max_drawdown_recovery_date": date_at(recovery_i),
        "max_drawdown_peak_to_trough_months": int(trough_i - peak_i),
        "max_drawdown_recovery_months": None if recovery_i is None else int(recovery_i - trough_i),
        "longest_underwater_months": longest_underwater, "current_drawdown": float(dd.iloc[-1]),
        "avg_drawdown_when_underwater": float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0,
        "calmar_ratio": float(cagr / abs(max_dd)) if max_dd < 0 else None,
    }


def compute_performance(stats_df, tag):
    tb3ms, sp500 = load_fred_series()
    perf = stats_df.merge(tb3ms, left_on="target_month", right_on="month", how="left")
    perf = perf.merge(sp500, left_on="target_month", right_on="month", how="left", suffixes=("", "_sp"))
    perf["benchmark_monthly"] = perf["rf_monthly"] + 0.04 / 12.0
    perf["active_ret"] = perf["port_excess_ret"] - 0.04 / 12.0
    ir = np.sqrt(12) * perf["active_ret"].mean() / perf["active_ret"].std()
    sharpe = np.sqrt(12) * perf["port_excess_ret"].mean() / perf["port_excess_ret"].std()
    reg_df = perf.dropna(subset=["sp500_ret", "rf_monthly"]).copy()
    y = perf.loc[reg_df.index, "port_excess_ret"].values
    x = (reg_df["sp500_ret"] - reg_df["rf_monthly"]).values
    X = sm.add_constant(x)
    ols_res = sm.OLS(y, X).fit()
    alpha_monthly, beta = ols_res.params
    alpha_se, beta_se = ols_res.bse
    alpha_t, beta_t = ols_res.tvalues
    cum_ret = (1 + perf["port_excess_ret"]).prod() - 1
    n_months = len(perf)
    ann_ret_arith = perf["port_excess_ret"].mean() * 12
    ann_ret_geo = (1 + perf["port_excess_ret"]).prod() ** (12 / n_months) - 1
    hit_rate = (perf["active_ret"] > 0).mean()
    best_month = perf.loc[perf["port_excess_ret"].idxmax()]
    worst_month = perf.loc[perf["port_excess_ret"].idxmin()]

    def calendar_year_of(col):
        d = perf.dropna(subset=[col]).assign(year=lambda x: x["target_month"].dt.year)
        return d.groupby("year").apply(lambda g: (1 + g[col]).prod() - 1, include_groups=False).to_dict()

    calendar_year = calendar_year_of("port_excess_ret")
    calendar_year_benchmark = calendar_year_of("benchmark_monthly")
    calendar_year_sp500 = calendar_year_of("sp500_ret")
    long_leg_cagr = (1 + perf["long_leg_ret"]).prod() ** (12 / n_months) - 1
    short_leg_cagr = (1 + perf["short_leg_ret"]).prod() ** (12 / n_months) - 1
    perf["drawdown"] = drawdown_series(perf["port_excess_ret"])
    perf["sp500_drawdown"] = drawdown_series(perf["sp500_ret"].fillna(0.0))
    sp500_cagr = (1 + perf["sp500_ret"].fillna(0.0)).prod() ** (12 / n_months) - 1
    dd_stats = compute_drawdown_stats(perf["port_excess_ret"], perf["target_month"], ann_ret_geo)
    sp500_dd_stats = compute_drawdown_stats(perf["sp500_ret"].fillna(0.0), perf["target_month"], sp500_cagr)

    results = {
        "n_months": int(n_months), "avg_monthly_return": float(perf["port_excess_ret"].mean()),
        "annualized_return_arith": float(ann_ret_arith), "annualized_return_geo_cagr": float(ann_ret_geo),
        "cumulative_return": float(cum_ret), "hit_rate_active": float(hit_rate),
        "information_ratio": float(ir), "sharpe_ratio": float(sharpe),
        "alpha_monthly": float(alpha_monthly), "alpha_annualized": float(alpha_monthly * 12),
        "alpha_tstat": float(alpha_t), "alpha_se_monthly": float(alpha_se),
        "beta": float(beta), "beta_se": float(beta_se), "beta_tstat": float(beta_t),
        "corr_with_sp500": float(perf["port_excess_ret"].corr(perf["sp500_ret"])),
        "avg_gross_exposure": float(stats_df["gross_exposure"].mean()),
        "max_gross_exposure": float(stats_df["gross_exposure"].max()),
        "avg_net_exposure": float(stats_df["net_exposure"].mean()),
        "min_net_exposure": float(stats_df["net_exposure"].min()),
        "max_net_exposure": float(stats_df["net_exposure"].max()),
        "avg_n_positions": float(stats_df["n_positions"].mean()),
        "avg_n_long": float(stats_df["n_long"].mean()), "avg_n_short": float(stats_df["n_short"].mean()),
        "best_month": {"date": str(best_month["target_month"].date()), "ret": float(best_month["port_excess_ret"])},
        "worst_month": {"date": str(worst_month["target_month"].date()), "ret": float(worst_month["port_excess_ret"])},
        "calendar_year_returns": {int(k): float(v) for k, v in calendar_year.items()},
        "calendar_year_returns_benchmark": {int(k): float(v) for k, v in calendar_year_benchmark.items()},
        "calendar_year_returns_sp500": {int(k): float(v) for k, v in calendar_year_sp500.items()},
        "long_leg_avg_monthly_ret": float(perf["long_leg_ret"].mean()), "long_leg_cagr": float(long_leg_cagr),
        "short_leg_avg_monthly_ret": float(perf["short_leg_ret"].mean()), "short_leg_cagr": float(short_leg_cagr),
        **dd_stats, "sp500_drawdown": sp500_dd_stats,
    }
    perf.to_csv(OUTPUT / f"portfolio_returns_{tag}.csv", index=False)
    return results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading model table...")
    df, stock_vars = load_model_table()
    print(f"  {len(df):,} stock-month rows, {len(stock_vars)} predictors")

    print("Cross-sectional preprocessing...")
    df = cross_sectional_rank_transform(df, stock_vars)

    print("Walk-forward expanding-window RankGLU (h_dim, l2 tuned on validation)...")
    preds, fold_params = run_walk_forward(df, stock_vars)
    preds = preds[(preds["target_month"] >= OOS_START) & (preds["target_month"] <= OOS_END)]
    preds.to_csv(OUTPUT / "oos_predictions_rankglu.csv", index=False)

    r2 = oos_r2(preds[TARGET_COL].values, preds["pred"].values)
    print(f"\nOOS R^2 (full 2021-01 to 2026-08 sample): {r2 * 100:.4f}%")

    all_results = {"oos_r2": float(r2), "n_predictor_columns": len(stock_vars),
                    "oos_prediction_rows": int(len(preds)), "fold_hyperparameters": fold_params}

    print("\nBuilding portfolio: Beta-neutral...")
    bn_holdings, bn_stats = build_beta_neutral_portfolio(preds)
    bn_holdings.to_csv(OUTPUT / "portfolio_holdings_beta_neutral_rankglu.csv", index=False)

    print("Computing performance...")
    bn_results = compute_performance(bn_stats, tag="beta_neutral_rankglu")
    bn_results["avg_beta_exposure_at_formation"] = float(bn_stats["beta_exposure"].mean())
    bn_results["max_abs_beta_exposure_at_formation"] = float(bn_stats["beta_exposure"].abs().max())
    bn_results.update(compute_turnover_and_concentration(bn_holdings))
    bn_results.update(compute_short_book_characteristics(bn_holdings))
    all_results["beta_neutral"] = bn_results

    print("\n=== Summary: Beta-neutral (RankGLU) ===")
    for k, v in bn_results.items():
        if not isinstance(v, dict):
            print(f"  {k}: {v}")

    with open(OUTPUT / "rankglu_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nFull results written to experiments/rankglu/output/rankglu_results.json")
