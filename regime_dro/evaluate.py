# evaluate.py

import numpy as np

from regime_dro.arrays import asnumpy_strict, asxp, _to_xp
from regime_dro.optimizer import psd_factor_LtL

try:
    import cupy as xp
    GPU = True
except Exception:
    import numpy as xp
    GPU = False


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
