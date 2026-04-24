# report.py

import numpy as np
import pandas as pd

from regime_dro.arrays import asnumpy_strict, asxp
from regime_dro.optimizer import psd_factor_LtL
from regime_dro.evaluate import stats_from_series

try:
    import cupy as xp
    GPU = True
except Exception:
    import numpy as xp
    GPU = False


def oos_summary(results: dict, model_order=None) -> pd.DataFrame:

    base_rows = [
        "mu_ann",
        "sigma_ann",
        "sharpe_ann",
        "vol_breach",
        "max_dd",
        "gross_exp",
        "delta",
        "delta_uncond",
        "delta_gap",
    ]

    ALLOW_CI = {
        "mu_ann",
        "sigma_ann",
        "sharpe_ann",
        "vol_breach",
        "delta",
        "delta_uncond",
        "delta_gap",
    }

    NO_CI_MODELS = {"SPX"}

    if model_order is None:
        model_order = list(results.keys())

    out = pd.DataFrame(index=base_rows, columns=model_order, dtype=object)

    def _fmt_value_only(v):
        try:
            vf = float(v)
            return "" if not np.isfinite(vf) else f"{vf:.3f}"
        except Exception:
            return ""

    for m in model_order:
        if m not in results or len(results[m]) == 0:
            continue
        row0 = results[m].iloc[0]
        for col in base_rows:
            if col not in results[m].columns:
                continue
            v  = row0.get(col, np.nan)
            lo = row0.get(f"{col}_ci_low",  np.nan)
            hi = row0.get(f"{col}_ci_high", np.nan)

            use_ci = (
                (m not in NO_CI_MODELS) and
                (col in ALLOW_CI) and
                pd.notna(lo) and pd.notna(hi) and
                np.isfinite(float(lo)) and np.isfinite(float(hi))
            )

            out.at[col, m] = (
                f"{float(v):.3f} ({float(lo):.3f}, {float(hi):.3f})"
                if use_ci else _fmt_value_only(v)
            )

    def _all_blank(series):
        return all((isinstance(x, str) and x == "") or (x is None) for x in series.values)
    out = out.loc[~out.apply(_all_blank, axis=1)]
    return out


def print_oos_table(results_dict, model_order):
    model_order = [m for m in model_order if m in results_dict and len(results_dict[m]) > 0]
    if not model_order:
        print("\nNo models to display."); return
    print("\n" + "=" * 108)
    print("OOS Portfolio Performance")
    print("=" * 108)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(oos_summary(results_dict, model_order=model_order))


def _print_mu_by_name(names, mu_vec, prefix="   "):
    names = list(names)
    mu_vec = xp.asarray(mu_vec, float).ravel()
    s = ", ".join(f"{names[i]}:{float(mu_vec[i]):+.4f}" for i in range(len(names)))
    print(prefix + "mu_ann: [" + s + "]")


def _section(title: str):
    print("\n" + "=" * 72)
    print(str(title))
    print("=" * 72 + "\n")


def print_single_portfolio_block(label, w, returns_train, returns_eval, rho, Sigma_ann, config, rtol=1e-6, atol=1e-9):
    Rtr = asxp(returns_train, dtype=float)
    Rev = asxp(returns_eval,  dtype=float)
    wxp = asxp(w, dtype=float).reshape(-1)

    n_days, n_assets = Rtr.shape
    AF = int(config.get("annualization_factor", 252))

    mu_train_ann_assets    = AF * xp.mean(Rtr, axis=0)
    sigma_train_ann_assets = xp.sqrt(AF) * xp.std(Rtr, axis=0, ddof=1)

    L_np = psd_factor_LtL(Sigma_ann, config["epsilon_sigma"])
    L_xp = asxp(L_np, dtype=float)
    risk_train_ann = float(xp.linalg.norm(L_xp @ wxp))
    tol = max(atol, rtol * max(rho, risk_train_ann))
    ok_train = bool(risk_train_ann <= rho + tol)

    ret_train_ann = float(mu_train_ann_assets @ wxp)
    port_eval = Rev @ wxp
    _, risk_eval_ann, _ = stats_from_series(port_eval, dict(config, annualization_factor=AF))
    mu_eval_ann_assets = AF * xp.mean(Rev, axis=0)
    ret_eval_ann = float(mu_eval_ann_assets @ wxp)

    gross_exposure = float(xp.sum(xp.abs(wxp)))
    top_idx = xp.argsort(wxp)[-3:][::-1]
    nz = xp.where(wxp != 0)[0]
    bot_idx = nz[xp.argsort(wxp[nz])[:3]] if nz.size else xp.array([], dtype=int)

    return {
        "ret_train_ann": ret_train_ann,
        "risk_train_ann": risk_train_ann,
        "ret_eval_ann": ret_eval_ann,
        "risk_eval_ann": float(risk_eval_ann),
        "mu_train_ann_assets": asnumpy_strict(mu_train_ann_assets).tolist(),
        "sigma_train_ann_assets": asnumpy_strict(sigma_train_ann_assets).tolist(),
        "gross_exposure": gross_exposure,
        "ok_train_budget": ok_train,
    }


def print_regime_block(label, returns_train, returns_eval, w_list, segs, rho,
                       taus_display, seg_deltas, config=None):
    Rtr = asxp(returns_train, dtype=float)
    Rev = asxp(returns_eval,  dtype=float)
    n_days = Rtr.shape[0]
    AF = int((config or {}).get("annualization_factor", 252))

    port_train = xp.zeros(n_days, dtype=float)
    port_eval  = xp.zeros(n_days, dtype=float)
    for k, w in enumerate(w_list):
        a, b = segs[k], segs[k+1]
        wxp = asxp(w, dtype=float).reshape(-1)
        port_train[a:b] = (Rtr[a:b] @ wxp)
        port_eval[a:b]  = (Rev[a:b] @ wxp)

    cfg = {
        "n_days": n_days,
        "risk_free_rate": float((config or {}).get("risk_free_rate", 0.0)),
        "annualization_factor": AF,
    }
    ret_train_ann, risk_train_ann, _ = stats_from_series(port_train, cfg)
    ret_eval_ann,  risk_eval_ann,  _ = stats_from_series(port_eval,  cfg)

    mu_train_ann_assets    = AF * xp.mean(Rtr, axis=0)
    sigma_train_ann_assets = xp.sqrt(AF) * xp.std(Rtr, axis=0, ddof=1)

    return {
        "ret_train_ann": float(ret_train_ann),
        "risk_train_ann": float(risk_train_ann),
        "ret_eval_ann": float(ret_eval_ann),
        "risk_eval_ann": float(risk_eval_ann),
        "mu_train_ann_assets": asnumpy_strict(mu_train_ann_assets).tolist(),
        "sigma_train_ann_assets": asnumpy_strict(sigma_train_ann_assets).tolist(),
        "taus_display": list(taus_display),
        "seg_deltas": list(seg_deltas),
    }
