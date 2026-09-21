"""
FIAM 2026 - End-to-end weight learning (operationalizes "AlphaZeroBeta: Deep
Reinforcement Learning for Market-Neutral Portfolios", arXiv 2607.18001,
Jul 2026).

The paper's idea (per docs/PAPERS.md §2): instead of the two-stage pipeline
`ols.py`/`xgb.py`/`rf.py`/`sparse.py`/`joint.py` all use -- fit a model to
minimize pointwise return error, THEN separately re-solve a fresh LP for
weights each month -- train the characteristics-to-weights mapping directly
against a portfolio-level objective (Sharpe), end to end.

The paper's actual system (per its verbatim abstract, read directly -- see
docs/PAPERS.md sec 2): a CNN-GRU policy trained end-to-end via Recurrent PPO
(a specific policy-gradient RL algorithm), evaluated across seven equity
indices 2014-2024. NO CNN, GRU, PPO, or RL of any kind is built here (this
project has no RL/deep-learning infrastructure) -- theta is a plain linear
vector, optimized by SPSA (Simultaneous Perturbation Stochastic
Approximation, Spall 1992), a classic zeroth-order method that needs only
objective evaluations, not gradients.

FIX (see docs/E2E.md for the full before/after): the first version's
training objective was Sharpe alone. This version implements the paper's
actual REWARD STRUCTURE -- "a composite reward function that balances
risk-adjusted excess return, benchmark correlation, and transaction
costs" -- as composite_objective():

  1. score_i = X_i . theta               (same 147 rank-transformed inputs)
  2. within each month, orthogonalize the score against beta_60m via a
     closed-form linear projection (exact, not learned) -- this is the smooth
     stand-in for the LP's hard beta-neutrality constraint
  3. center the residual score (smooth stand-in for dollar-neutrality)
  4. squash through tanh and scale by the per-name cap (smooth stand-in for
     the LP's gross-exposure/per-name constraints) -- soft_portfolio_weights()
  5. objective = Sharpe(monthly returns) - lambda_tc * avg_turnover
                - lambda_corr * |corr(monthly returns, S&P 500 returns)|
     where avg_turnover is the mean one-way soft-weight turnover between
     CONSECUTIVE months (aligned by permno) and the correlation term uses
     actual monthly S&P 500 returns (same FRED series used everywhere else
     in this project). Training-month sampling was changed from a random
     bag of 12 months to a random CONTIGUOUS 12-month block, since turnover
     is only meaningful between adjacent months.

theta is optimized directly against this composite objective using SPSA.
theta is warm-started from the training fold's closed-form OLS coefficients
(standard practice for zeroth-order refinement) and refined for a capped
number of iterations; validation-fold composite objective (not training)
selects the best checkpoint, matching this project's existing validation
discipline. Still no RL, no neural network, no CNN-GRU -- only the reward
STRUCTURE (three named terms) now matches the paper; the optimizer and
function class remain unrelated to PPO/CNN-GRU.

**For comparability with every other script in this project, the final
theta's scores are NOT used with the soft/smooth portfolio above for OOS
reporting.** They are fed through the exact same hard beta-neutral LP used
by ols.py/xgb.py/rf.py/sparse.py/joint.py, so the final portfolio numbers
are apples-to-apples across all 8 models. The soft portfolio only exists
inside the training loop, as the (differentiable-free) training signal.

Self-contained: data loading, rank transform, walk-forward schedule,
investability screen, beta-neutral LP, and evaluation code are ported from
ols.py / xgb.py (not imported).

Run: .venv/bin/python e2e.py
Outputs (all in output/): oos_predictions_e2e.csv,
    portfolio_holdings_beta_neutral_e2e.csv,
    portfolio_returns_beta_neutral_e2e.csv, e2e_training_curve.csv,
    e2e_results.json
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import linprog
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
FIAM_DIR = BASE / "fiam"
CACHE = BASE / "cache"
OUTPUT = BASE / "output"
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
# 2. Smooth portfolio objective + SPSA optimizer
# ---------------------------------------------------------------------------

SHARPNESS = 3.0       # tanh sharpness in the soft weight formula
SOFT_CAP = 0.01        # matches MAX_WEIGHT below, for scale consistency
MAX_STOCKS_PER_MONTH = 1500  # subsample cap per month, for speed
N_MONTHS_PER_EVAL = 12       # training months subsampled per objective call
SPSA_ITERATIONS = 40
CHECKPOINT_EVERY = 5


TRANSACTION_COST_LAMBDA = 0.5    # penalty weight on month-over-month soft turnover
BENCHMARK_CORR_LAMBDA = 0.5      # penalty weight on |correlation with S&P 500|


def month_groups_for_objective(sub_df: pd.DataFrame, stock_vars: list[str]):
    """Pre-split a dataframe into CHRONOLOGICALLY ORDERED per-month (X, y,
    beta, permno, month) tuples, capped in size per month, for fast
    repeated objective evaluation. Chronological order (not just a bag of
    months) is required now so the composite objective can compute
    month-over-month turnover between adjacent months -- see
    composite_objective()."""
    groups = []
    for month, grp in sub_df.sort_values("target_month").groupby("target_month", sort=True):
        grp = grp.dropna(subset=["raw_beta_60m"])
        if len(grp) > MAX_STOCKS_PER_MONTH:
            grp = grp.sample(n=MAX_STOCKS_PER_MONTH, random_state=0)
        if len(grp) < 20:
            continue
        groups.append((
            grp[stock_vars].values,
            grp[TARGET_COL].values,
            grp["raw_beta_60m"].values,
            grp["permno"].values,
            month,
        ))
    return groups


def soft_portfolio_weights(theta: np.ndarray, X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    scores = X @ theta
    scores = scores - scores.mean()
    var_beta = beta.var()
    b = (np.cov(scores, beta, bias=True)[0, 1] / var_beta) if var_beta > 1e-12 else 0.0
    resid = scores - b * beta
    resid = resid - resid.mean()
    scale = resid.std() + 1e-8
    return SOFT_CAP * np.tanh(SHARPNESS * resid / scale)


def composite_objective(
    theta: np.ndarray,
    month_groups: list,
    sp500_by_month: dict,
    lambda_tc: float = TRANSACTION_COST_LAMBDA,
    lambda_corr: float = BENCHMARK_CORR_LAMBDA,
) -> float:
    """The paper's actual reward per its verbatim abstract (docs/PAPERS.md
    sec 2): "a composite reward function that balances risk-adjusted excess
    return, benchmark correlation, and transaction costs." month_groups
    must be chronologically CONSECUTIVE (see month_groups_for_objective) so
    turnover between adjacent months is meaningful."""
    rets, weights_by_month, permnos_by_month, months = [], [], [], []
    for X, y, beta, permno, month in month_groups:
        w = soft_portfolio_weights(theta, X, beta)
        rets.append(float(np.sum(w * y)))
        weights_by_month.append(w)
        permnos_by_month.append(permno)
        months.append(month)

    rets = np.array(rets)
    sharpe = 0.0 if rets.std() < 1e-12 else float(rets.mean() / rets.std())

    # transaction-cost proxy: mean one-way turnover between adjacent months,
    # aligned by permno (names not held in both months count as full turnover)
    turnovers = []
    for i in range(1, len(weights_by_month)):
        prev = pd.Series(weights_by_month[i - 1], index=permnos_by_month[i - 1])
        curr = pd.Series(weights_by_month[i], index=permnos_by_month[i])
        aligned = pd.concat([prev.rename("prev"), curr.rename("curr")], axis=1).fillna(0.0)
        turnovers.append(float((aligned["curr"] - aligned["prev"]).abs().sum()) / 2.0)
    avg_turnover = float(np.mean(turnovers)) if turnovers else 0.0

    # benchmark-correlation proxy: |corr(soft portfolio returns, S&P 500 returns)|
    sp_rets = np.array([sp500_by_month.get(m, np.nan) for m in months])
    valid = ~np.isnan(sp_rets)
    if valid.sum() >= 6 and rets[valid].std() > 1e-12 and sp_rets[valid].std() > 1e-12:
        corr = float(np.corrcoef(rets[valid], sp_rets[valid])[0, 1])
    else:
        corr = 0.0

    return sharpe - lambda_tc * avg_turnover - lambda_corr * abs(corr)


def spsa_train(
    theta_init: np.ndarray, train_groups: list, val_groups: list, sp500_by_month: dict
) -> tuple[np.ndarray, list]:
    theta = theta_init.copy()
    n = len(theta)

    def objective(th, groups):
        return composite_objective(th, groups, sp500_by_month)

    # calibrate perturbation size relative to theta's own scale
    theta_scale = np.abs(theta_init).mean() if np.abs(theta_init).mean() > 1e-8 else 1e-4
    c = 0.15 * theta_scale

    best_theta = theta.copy()
    best_val = objective(theta, val_groups)
    curve = [{"iteration": 0, "val_objective": best_val, "train_objective": objective(theta, train_groups)}]

    def sample_contiguous_block(groups, n_months):
        """Random CONTIGUOUS block of months (not a random bag) -- turnover
        between adjacent months only means something if they're adjacent."""
        n_months = min(n_months, len(groups))
        start = RNG.integers(0, len(groups) - n_months + 1)
        return groups[start:start + n_months]

    # pilot step to calibrate the learning rate `a` so the first update is a
    # modest fraction of theta's own scale
    delta = RNG.choice([-1.0, 1.0], size=n)
    eval_groups = sample_contiguous_block(train_groups, N_MONTHS_PER_EVAL)
    y_plus = objective(theta + c * delta, eval_groups)
    y_minus = objective(theta - c * delta, eval_groups)
    ghat = (y_plus - y_minus) / (2 * c * delta)
    ghat_scale = np.abs(ghat).mean() + 1e-12
    a = 0.05 * theta_scale / ghat_scale

    for k in range(1, SPSA_ITERATIONS + 1):
        ak = a / (k ** 0.602)
        ck = c / (k ** 0.101)
        delta = RNG.choice([-1.0, 1.0], size=n)
        eval_groups = sample_contiguous_block(train_groups, N_MONTHS_PER_EVAL)

        y_plus = objective(theta + ck * delta, eval_groups)
        y_minus = objective(theta - ck * delta, eval_groups)
        ghat = (y_plus - y_minus) / (2 * ck * delta)

        theta = theta + ak * ghat  # ascent: maximize composite objective

        if k % CHECKPOINT_EVERY == 0 or k == SPSA_ITERATIONS:
            val_obj = objective(theta, val_groups)
            train_obj = objective(theta, train_groups)
            curve.append({"iteration": k, "val_objective": val_obj, "train_objective": train_obj})
            if val_obj > best_val:
                best_val = val_obj
                best_theta = theta.copy()

    return best_theta, curve


# ---------------------------------------------------------------------------
# 3. Expanding-window walk-forward: OLS warm start + SPSA refinement
# ---------------------------------------------------------------------------


def year_bounds(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")


def run_walk_forward(df: pd.DataFrame, stock_vars: list[str]):
    test_years = range(2021, 2027)
    preds, fold_curves, fold_params = [], [], []

    _, sp500_monthly = load_fred_series()
    sp500_by_month = dict(zip(sp500_monthly["month"], sp500_monthly["sp500_ret"]))

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

        # warm start: closed-form OLS on the training fold
        ols = LinearRegression()
        ols.fit(train[stock_vars].values, train[TARGET_COL].values)
        theta_init = ols.coef_.copy()

        train_groups = month_groups_for_objective(train, stock_vars)
        val_groups = month_groups_for_objective(val, stock_vars)

        theta_final, curve = spsa_train(theta_init, train_groups, val_groups, sp500_by_month)
        for row in curve:
            row["test_year"] = test_year
        fold_curves.extend(curve)

        ols_val_obj = composite_objective(theta_init, val_groups, sp500_by_month)
        best_val_obj = max(c["val_objective"] for c in curve)
        fold_params.append({
            "test_year": test_year,
            "ols_warmstart_val_composite_objective": ols_val_obj,
            "best_spsa_val_composite_objective": best_val_obj,
            "n_train_months_used": len(train_groups),
            "n_val_months_used": len(val_groups),
        })

        X_test = test[stock_vars].values
        y_pred = X_test @ theta_final

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
            f"OLS-warmstart val composite obj={ols_val_obj:.4f} -> "
            f"best SPSA val composite obj={best_val_obj:.4f}"
        )

    return pd.concat(preds, ignore_index=True), pd.DataFrame(fold_curves), fold_params


# ---------------------------------------------------------------------------
# 4. Evaluation, portfolio construction (ported from ols.py / xgb.py, unchanged
#    -- same hard LP used by every other script, for comparability, see
#    module docstring)
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

    print("Walk-forward: OLS warm start + SPSA end-to-end refinement against smooth portfolio Sharpe...")
    preds, curve_df, fold_params = run_walk_forward(df, stock_vars)
    preds = preds[(preds["target_month"] >= OOS_START) & (preds["target_month"] <= OOS_END)]
    preds.to_csv(OUTPUT / "oos_predictions_e2e.csv", index=False)
    curve_df.to_csv(OUTPUT / "e2e_training_curve.csv", index=False)

    r2 = oos_r2(preds[TARGET_COL].values, preds["pred"].values)
    print(f"\nOOS R^2 (full 2021-01 to 2026-08 sample): {r2 * 100:.4f}%")

    all_results = {"oos_r2": float(r2), "n_predictor_columns": len(stock_vars),
                    "oos_prediction_rows": int(len(preds)), "fold_training": fold_params}

    print("\nBuilding portfolio: Beta-neutral (same hard LP as every other script)...")
    bn_holdings, bn_stats = build_beta_neutral_portfolio(preds)
    bn_holdings.to_csv(OUTPUT / "portfolio_holdings_beta_neutral_e2e.csv", index=False)

    print("Computing performance...")
    bn_results = compute_performance(bn_stats, tag="beta_neutral_e2e")
    bn_results["avg_beta_exposure_at_formation"] = float(bn_stats["beta_exposure"].mean())
    bn_results["max_abs_beta_exposure_at_formation"] = float(bn_stats["beta_exposure"].abs().max())
    bn_results.update(compute_turnover_and_concentration(bn_holdings))
    bn_results.update(compute_short_book_characteristics(bn_holdings))
    all_results["beta_neutral"] = bn_results

    print("\n=== Summary: Beta-neutral (End-to-End SPSA) ===")
    for k, v in bn_results.items():
        if not isinstance(v, dict):
            print(f"  {k}: {v}")

    with open(OUTPUT / "e2e_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nFull results written to output/e2e_results.json")
