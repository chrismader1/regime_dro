# evaluate.py

import numpy as np
import pandas as pd

from regime_dro.arrays import asnumpy_strict, asxp, _to_xp
from regime_dro.optimizer import psd_factor_LtL

try:
    import cupy as xp
    GPU = True
except Exception:
    import numpy as xp
    GPU = False


# =============================================================================
# Existing core stats (unchanged)
# =============================================================================

def stats_from_series(port_daily, config):
    AF = int(config.get("annualization_factor", 252))
    rf_annual = float(config.get("risk_free_rate", 0.0))
    x = asxp(port_daily, dtype=float).ravel()
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rf_daily = (1.0 + rf_annual) ** (1.0 / AF) - 1.0
    sigma_daily = xp.std(x, ddof=1)
    sigma_annual = sigma_daily * xp.sqrt(AF)
    mu_annual_geom = xp.exp(AF * xp.mean(xp.log1p(x))) - 1.0
    sharpe_annual = (xp.mean(x) - rf_daily) / sigma_daily * xp.sqrt(AF) if sigma_daily > 0 else xp.nan
    return float(mu_annual_geom), float(sigma_annual), float(sharpe_annual)


def _max_drawdown_from_series(port_daily):
    x = xp.asarray(port_daily, float)
    if x.size == 0:
        return float("nan")

    try:
        import cupy as _cp
        is_gpu = isinstance(x, _cp.ndarray) or hasattr(x, "__cuda_array_interface__")
    except Exception:
        is_gpu = False

    if is_gpu:
        x_np = asnumpy_strict(x, dtype=float)
        equity_np = np.cumprod(1.0 + x_np)
        peak_np = np.maximum.accumulate(equity_np)
        dd_np = equity_np / peak_np - 1.0
        return float(np.min(dd_np))
    else:
        equity = np.cumprod(1.0 + np.asarray(x, dtype=float))
        peak = np.maximum.accumulate(equity)
        dd = equity / peak - 1.0
        return float(np.min(dd))


def portfolio_stats(weights, returns, config):
    R = np.asarray(returns, dtype=float)
    w = asnumpy_strict(weights, dtype=float).reshape(-1)
    port_daily = R @ w
    mu_annual_geom, sigma_annual, sharpe_annual = stats_from_series(port_daily, config)
    vol_breach = max(sigma_annual - config["risk_budget"], 0.0)
    max_dd = _max_drawdown_from_series(port_daily)
    return {
        "mu_ann": mu_annual_geom,
        "sigma_ann": sigma_annual,
        "sharpe_ann": sharpe_annual,
        "vol_breach": vol_breach,
        "max_dd": max_dd,
    }


def portfolio_stats_multipiece(w_list, taus, returns, config):
    import numpy as _np
    taus = [int(x) for x in list(taus)]
    R = _np.asarray(returns, dtype=float)
    n_days = int(config["n_days"])
    assert taus[0] == 0 and taus[-1] == n_days and len(w_list) == len(taus) - 1

    port_daily_np = _np.empty(n_days, dtype=float)
    for k in range(len(w_list)):
        a, b = taus[k], taus[k + 1]
        w_np = asnumpy_strict(w_list[k], dtype=float).reshape(-1)
        port_daily_np[a:b] = R[a:b] @ w_np

    mu_annual_geom, sigma_annual, sharpe_annual = stats_from_series(port_daily_np, config)
    vol_breach = max(sigma_annual - config["risk_budget"], 0.0)
    max_dd = _max_drawdown_from_series(port_daily_np)
    return {
        "mu_ann": mu_annual_geom,
        "sigma_ann": sigma_annual,
        "sharpe_ann": sharpe_annual,
        "vol_breach": vol_breach,
        "max_dd": max_dd,
    }


def _avg_holding_period_from_marks(rebal_marks):
    if rebal_marks is None:
        return float("nan")
    r = [int(x) for x in rebal_marks]
    if len(r) <= 1:
        return float("nan")
    return float(max(r) / (len(r) - 1))


def evaluate_portfolio(fit, data, G):

    train, test = data["train"], data["test"]
    n_days = data["n_days"]
    AF = int(data.get("ann_factor", 252))

    if fit["type"] == "static":
        stats_oos = portfolio_stats(
            fit["w"], test, {"n_days": n_days, "risk_free_rate": G["risk_free_rate"],
                             "risk_budget": G["risk_budget"], "annualization_factor": AF})
        ge = float(xp.sum(xp.abs(fit["w"])))
        port_tr = train @ fit["w"]
        _, sigma_train_ann, _ = stats_from_series(
            port_tr,
            {"n_days": n_days, "risk_free_rate": G["risk_free_rate"], "annualization_factor": AF})
        train_soc = float("nan")
        if isinstance(data, dict) and ("Sigma_ann_full" in data) and (data["Sigma_ann_full"] is not None):
            L = psd_factor_LtL(data["Sigma_ann_full"], G["epsilon_sigma"])
            L_xp = _to_xp(L)
            train_soc = float(xp.linalg.norm(L_xp @ fit["w"]))

        stats_oos["gross_exp"] = ge
        stats_oos["sigma_train_ann"] = float(sigma_train_ann)
        stats_oos["sigma_oos_ann"] = float(stats_oos["sigma_ann"])
        stats_oos["train_soc_risk"] = train_soc
        stats_oos["train_constraint_slack"] = float(G["risk_budget"] - train_soc) if xp.isfinite(train_soc) else xp.nan
        stats_oos["kappa"] = float(fit.get("kappa", xp.nan))
        stats_oos["delta"] = float(fit.get("delta", xp.nan))
        stats_oos["delta_uncond"] = xp.nan
        stats_oos["delta_gap"] = xp.nan
        rebal = [0, n_days]
        stats_oos["avg_holding_per"] = _avg_holding_period_from_marks(rebal)
        return stats_oos

    else:  # piecewise
        cfg = {"n_days": n_days, "risk_free_rate": G["risk_free_rate"], "risk_budget": G["risk_budget"], "annualization_factor": AF}

        stats_oos = portfolio_stats_multipiece(fit["w_list"], fit["segs"], test, cfg)
        seg_lengths = xp.diff(xp.array(fit["segs"]))
        ge = float(xp.sum(seg_lengths * xp.array([xp.sum(xp.abs(wk)) for wk in fit["w_list"]])) / n_days)

        port_tr = xp.zeros(n_days)
        for (a, b), wk in zip(zip(fit["segs"][:-1], fit["segs"][1:]), fit["w_list"]):
            port_tr[a:b] = train[a:b] @ wk

        _, sigma_train_ann, _ = stats_from_series(
            port_tr, {"n_days": n_days, "risk_free_rate": G["risk_free_rate"], "annualization_factor": AF})
        stats_oos["gross_exp"] = ge
        stats_oos["sigma_train_ann"] = float(sigma_train_ann)
        stats_oos["sigma_oos_ann"]  = float(stats_oos["sigma_ann"])
        stats_oos["train_soc_risk"] = xp.nan
        stats_oos["train_constraint_slack"] = xp.nan
        stats_oos["kappa"] = float(fit.get("kappa", xp.nan))

        dlist = xp.asarray(fit.get("delta_list", []), dtype=float)
        if dlist.size:
            mask = xp.isfinite(dlist)
            stats_oos["delta"] = float(xp.mean(dlist[mask])) if int(mask.sum()) else xp.nan
        else:
            stats_oos["delta"] = xp.nan
        stats_oos["delta_uncond"] = xp.nan
        stats_oos["delta_gap"] = xp.nan

        rebal = list(fit.get("segs", []))
        if not rebal:
            rebal = [0, n_days]
        stats_oos["avg_holding_per"] = _avg_holding_period_from_marks(rebal)
        return stats_oos


def evaluate_regime_independently(fit, data, G):
    n_days = int(data["n_days"])
    test = data["test"]

    R_test = np.asarray(test, dtype=float)

    stats_oos = {}
    dlist = list(map(float, fit.get("delta_list", [])))
    for j, dj in enumerate(dlist, start=1):
        stats_oos[f"delta_k{j}"] = dj

    for k, (a, b) in enumerate(zip(fit["segs"][:-1], fit["segs"][1:])):
        wk = fit["w_list"][k]
        seg_length = int(b - a)

        mu_seg = sigma_seg = sharpe_seg = vol_breach_seg = np.nan
        gross_exp_seg = float(np.sum(np.abs(asnumpy_strict(wk, dtype=float))))

        if seg_length > 1:
            wk_np = asnumpy_strict(wk, dtype=float).reshape(-1)
            seg_series_oos = R_test[a:b] @ wk_np

            seg_config = dict(G)
            seg_config["n_days"] = n_days
            seg_config["annualization_factor"] = int(data.get("ann_factor", 252))

            mu_seg, sigma_seg, sharpe_seg = stats_from_series(seg_series_oos, seg_config)
            vol_breach_seg = max(sigma_seg - G["risk_budget"], 0.0)

        stats_oos[f"mu_ann_k{k+1}"] = mu_seg
        stats_oos[f"sigma_ann_k{k+1}"] = sigma_seg
        stats_oos[f"sharpe_ann_k{k+1}"] = sharpe_seg
        stats_oos[f"vol_breach_k{k+1}"] = vol_breach_seg
        stats_oos[f"gross_exp_k{k+1}"] = gross_exp_seg

    return stats_oos


# =============================================================================
# Extended portfolio analytics
# =============================================================================
# All functions below operate on plain numpy / pandas inputs. No GPU, no internal
# state, no dependency on the fit/data structures of evaluate_portfolio. Inputs:
#
#   daily_returns : 1D numeric series or array of arithmetic daily returns
#   weights_df    : DataFrame indexed by date with one column per asset
#   AF            : annualization factor (252 for daily, 52 for weekly, etc.)
#
# All return scalars, pandas Series, or pandas DataFrames as documented.


def _to_1d_float(daily_returns):
    """Coerce to 1D float64 numpy array, dropping NaN."""
    x = np.asarray(pd.Series(daily_returns).dropna().to_numpy(), dtype=float).ravel()
    return x


def cagr_geometric(daily_returns, AF):
    """Compound annual growth rate from a daily-returns series.

    cagr = prod(1 + r_t) ** (AF / T) - 1
    """
    x = _to_1d_float(daily_returns)
    T = x.size
    if T == 0:
        return float("nan")
    final = float(np.prod(1.0 + x))
    if final <= 0.0:
        return float("nan")
    return float(final ** (float(AF) / float(T)) - 1.0)


def drawdown_series(daily_returns):
    """Pointwise drawdown series d_t = equity_t / running_peak_t - 1.

    Returns a pandas Series indexed identically to the input (when input is a
    Series). All values in (-1, 0].
    """
    s = pd.Series(daily_returns).astype(float).fillna(0.0)
    equity = (1.0 + s).cumprod()
    peak = equity.cummax()
    return equity / peak - 1.0


def time_under_water(daily_returns):
    """Maximum length of consecutive observations spent below the prior peak.

    Returned as an integer count of observations (e.g. days, weeks).
    """
    dd = drawdown_series(daily_returns)
    under = (dd < 0.0).astype(int).to_numpy()
    if under.size == 0:
        return 0
    # Longest run of 1s.
    max_run = 0
    cur_run = 0
    for v in under:
        if v == 1:
            cur_run += 1
            if cur_run > max_run:
                max_run = cur_run
        else:
            cur_run = 0
    return int(max_run)


def calmar_ratio(daily_returns, AF):
    """Calmar = CAGR / |MaxDrawdown|. NaN if MaxDD is zero."""
    cagr = cagr_geometric(daily_returns, AF)
    max_dd = _max_drawdown_from_series(_to_1d_float(daily_returns))
    if not np.isfinite(max_dd) or max_dd == 0.0:
        return float("nan")
    return float(cagr / abs(max_dd))


def downside_deviation(daily_returns, mar, AF):
    """Annualised downside deviation relative to a minimum acceptable return (MAR).

    mar is given in DAILY units (caller's responsibility). For an annual MAR of
    rf, pass mar = (1+rf)**(1/AF) - 1.
    """
    x = _to_1d_float(daily_returns)
    if x.size == 0:
        return float("nan")
    shortfall = np.minimum(x - float(mar), 0.0)
    dd_daily = float(np.sqrt(np.mean(shortfall ** 2)))
    return float(dd_daily * np.sqrt(float(AF)))


def sortino_ratio(daily_returns, mar, AF):
    """Annualised Sortino = (mean_daily - mar) * AF / downside_deviation_ann.

    mar in DAILY units.
    """
    x = _to_1d_float(daily_returns)
    if x.size == 0:
        return float("nan")
    dd_ann = downside_deviation(x, mar=mar, AF=AF)
    if not np.isfinite(dd_ann) or dd_ann == 0.0:
        return float("nan")
    excess_ann = (float(np.mean(x)) - float(mar)) * float(AF)
    return float(excess_ann / dd_ann)


def var_historical(daily_returns, level):
    """Historical Value-at-Risk at the given level (e.g. level=0.05 = 95% VaR).

    Returned as a non-negative loss magnitude. NaN if no data.
    """
    x = _to_1d_float(daily_returns)
    if x.size == 0:
        return float("nan")
    q = float(np.quantile(x, float(level)))
    return float(-q) if q < 0 else 0.0


def cvar_historical(daily_returns, level):
    """Historical Conditional VaR (expected shortfall) at the given level.

    Returned as a non-negative loss magnitude.
    """
    x = _to_1d_float(daily_returns)
    if x.size == 0:
        return float("nan")
    q = float(np.quantile(x, float(level)))
    tail = x[x <= q]
    if tail.size == 0:
        return float("nan")
    m = float(np.mean(tail))
    return float(-m) if m < 0 else 0.0


def rolling_sharpe(daily_returns, window, AF):
    """Rolling annualised Sharpe over a window of observations.

    Returns a pandas Series aligned to the input index (NaN for the first
    window-1 entries). Risk-free rate is assumed zero — pass excess returns
    in if you need otherwise.
    """
    s = pd.Series(daily_returns).astype(float)
    mu = s.rolling(int(window)).mean()
    sd = s.rolling(int(window)).std(ddof=1)
    sharpe = (mu / sd) * np.sqrt(float(AF))
    return sharpe


def effective_n_holdings(weights_df):
    """Effective number of holdings per date: 1 / sum(w_i^2).

    Inverse Herfindahl. Returns a pandas Series indexed by date.
    """
    W = pd.DataFrame(weights_df).astype(float).fillna(0.0)
    denom = (W ** 2).sum(axis=1)
    out = pd.Series(np.where(denom > 0, 1.0 / denom, np.nan), index=W.index, name="effective_n")
    return out


def cash_exposure(holdings_df):
    """Cash residual per date: 1 - sum(abs(w_i)).

    Returns a pandas Series indexed by date. Can be negative if the portfolio
    is over-allocated (which shouldn't happen under no_leverage=True).
    """
    H = pd.DataFrame(holdings_df).astype(float).fillna(0.0)
    return pd.Series(1.0 - H.abs().sum(axis=1), index=H.index, name="cash")


def turnover_series(holdings_df):
    """One-way turnover per rebalance: 0.5 * sum |w_t - w_{t-1}|.

    Returns a pandas Series aligned to the input index. First row is NaN
    (no prior weights).
    """
    H = pd.DataFrame(holdings_df).astype(float).fillna(0.0)
    diff = H.diff().abs().sum(axis=1) * 0.5
    diff.iloc[0] = np.nan
    diff.name = "turnover_one_way"
    return diff


def regime_conditional_stats(daily_returns, regime_labels, AF):
    """Per-regime annualised stats for a daily-returns series.

    Both inputs must be pandas Series indexed by the same dates. The function
    inner-joins on the index, groups by regime label, and computes mu/sigma/
    sharpe/n_obs per regime.

    Returns a DataFrame with one row per regime label and columns:
        n_obs, mu_ann, sigma_ann, sharpe_ann
    """
    s_ret = pd.Series(daily_returns).astype(float)
    s_reg = pd.Series(regime_labels)
    df = pd.concat({"r": s_ret, "z": s_reg}, axis=1).dropna()
    if df.empty:
        return pd.DataFrame(columns=["n_obs", "mu_ann", "sigma_ann", "sharpe_ann"])

    rows = []
    for z, g in df.groupby("z"):
        x = g["r"].to_numpy(dtype=float)
        if x.size == 0:
            continue
        mu_d = float(np.mean(x))
        sd_d = float(np.std(x, ddof=1)) if x.size > 1 else float("nan")
        mu_ann = mu_d * float(AF)
        sigma_ann = sd_d * float(np.sqrt(float(AF))) if np.isfinite(sd_d) else float("nan")
        sharpe = (mu_ann / sigma_ann) if (np.isfinite(sigma_ann) and sigma_ann > 0) else float("nan")
        rows.append({"regime": z, "n_obs": int(x.size), "mu_ann": mu_ann,
                     "sigma_ann": sigma_ann, "sharpe_ann": sharpe})
    out = pd.DataFrame(rows).set_index("regime").sort_index()
    return out


def enrich_summary(summary, daily_returns, AF, mar=0.0):
    """Augment an existing summary dict with extended portfolio analytics.

    Adds (without overwriting existing keys):
        cagr_geom, calmar, sortino, downside_dev_ann,
        var_95, cvar_95, var_99, cvar_99,
        time_under_water

    Returns a new dict (does not mutate input).
    """
    out = dict(summary or {})

    daily = pd.Series(daily_returns).astype(float)

    add = {
        "cagr_geom":         cagr_geometric(daily, AF=AF),
        "calmar":            calmar_ratio(daily, AF=AF),
        "sortino":           sortino_ratio(daily, mar=float(mar), AF=AF),
        "downside_dev_ann":  downside_deviation(daily, mar=float(mar), AF=AF),
        "var_95":            var_historical(daily, level=0.05),
        "cvar_95":           cvar_historical(daily, level=0.05),
        "var_99":            var_historical(daily, level=0.01),
        "cvar_99":           cvar_historical(daily, level=0.01),
        "time_under_water":  time_under_water(daily),
    }
    for k, v in add.items():
        if k not in out:
            out[k] = v
    return out
