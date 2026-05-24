# fit.py

import numpy as np
import pandas as pd

from regime_dro.arrays import asnumpy_strict, asxp
from regime_dro.delta import compute_delta
from regime_dro.optimizer import solve_optimizer
from regime_dro.windows import compute_mean_from_window, compute_cov_from_window, _window_start

try:
    import cupy as xp
    GPU = True
except Exception:
    import numpy as xp
    GPU = False


def _feasible_placeholder(N, G):
    import numpy as _np
    N = int(N)
    if N <= 0:
        return _np.zeros(0, dtype=float)

    no_short = bool(G.get("no_shorting", False))
    no_lev   = bool(G.get("no_leverage", False))
    max_cash = G.get("max_cash", None)
    cap      = G.get("max_pos_size", None)

    if no_short and no_lev and (max_cash is not None):
        s = max(0.0, 1.0 - float(max_cash))
        if s <= 0.0:
            return _np.zeros(N, dtype=float)

        if cap is None or not _np.isfinite(cap) or cap < 0.0:
            return _np.full(N, s / max(N, 1), dtype=float)

        cap = float(cap)
        per = min(cap, s / max(N, 1))
        w = _np.full(N, per, dtype=float)
        invested = float(w.sum())

        rem = max(0.0, s - invested)
        if rem > 0 and cap > 0:
            for i in range(N):
                if rem <= 0:
                    break
                add = min(cap - w[i], rem)
                if add > 0:
                    w[i] += add
                    rem  -= add
        return w

    return _np.zeros(N, dtype=float)


def _all_zero_weights(w, tol=1e-12) -> bool:
    w = xp.asarray(w, float).ravel()
    return bool(xp.all(xp.abs(w) <= tol))


def fit_mvo_rebalanced(R_df: pd.DataFrame, G, ann: int, marks: list,
                       min_lb: int, max_lb: int, lam_shr: float, verbose: bool = False):
    """Piecewise MVO over rebalance marks using rolling windows with min/max lookback."""
    from regime_dro.report import _section, _print_mu_by_name

    idx = R_df.index
    w_list, segs = [], marks
    N = R_df.shape[1]

    if verbose:
        _section("MVO")

    for a, b in zip(marks[:-1], marks[1:]):
        if a == 0:
            w_list.append(_feasible_placeholder(N, G))
            continue
        ws = _window_start(a, min_lb, max_lb)
        R_win = R_df.iloc[ws:a].dropna(how="any")
        if len(R_win) < max(2, min_lb):
            w_list.append(w_list[-1] if w_list else _feasible_placeholder(N, G))
            continue

        mask_all = np.ones(R_win.shape[0], dtype=bool)
        mu_ann = compute_mean_from_window(R_win, mask_all, min_obs=min_lb, ann=ann)
        Sig_ann = compute_cov_from_window(R_win, ann=ann, shrink_lambda=lam_shr, min_obs=min_lb)

        w = solve_optimizer(mu_ann, Sig_ann, delta=0.0, config=G, verbose=bool(verbose))
        if verbose:
            dt = idx[a]
            print(f"[MVO] t={a} {getattr(dt, 'date', lambda: dt)()}  delta=0.0000")
            _print_mu_by_name(R_win.columns.tolist(), mu_ann)
        w_list.append(w)

    return {"type": "piecewise", "w_list": w_list, "segs": segs,
            "kappa": xp.nan, "delta_list": []}


def fit_dro_rebalanced(R_df: pd.DataFrame, params, G, ann: int, marks: list,
                       min_lb: int, max_lb: int, lam_shr: float, verbose: bool = False):
    """Piecewise static DRO over rebalance marks using rolling windows with min/max lookback."""
    from regime_dro.report import _section, _print_mu_by_name

    idx = R_df.index
    w_list, segs, delta_list = [], marks, []
    N = R_df.shape[1]

    if verbose:
        _section("DRO")

    for a, b in zip(marks[:-1], marks[1:]):
        if a == 0:
            w_list.append(_feasible_placeholder(N, G)); delta_list.append(xp.nan)
            continue
        ws = _window_start(a, min_lb, max_lb)
        R_win = R_df.iloc[ws:a].dropna(how="any")
        if len(R_win) < max(2, min_lb):
            w_list.append(w_list[-1] if w_list else _feasible_placeholder(N, G))
            delta_list.append(delta_list[-1] if delta_list else xp.nan)
            continue

        mask_all = np.ones(R_win.shape[0], dtype=bool)
        mu_ann = compute_mean_from_window(R_win, mask_all, min_obs=min_lb, ann=ann)
        Sig_ann = compute_cov_from_window(R_win, ann=ann, shrink_lambda=lam_shr, min_obs=min_lb)

        delta = compute_delta(params.get("kappa", 1.0),
                              mu_ann, Sig_ann,
                              R=R_win.to_numpy(dtype=float),
                              params=params)
        w = solve_optimizer(mu_ann, Sig_ann, delta, config=G, verbose=bool(verbose))
        if verbose:
            dt = idx[a]
            print(f"[DRO] t={a} {getattr(dt, 'date', lambda: dt)()}  delta={float(delta):.4f}")
            _print_mu_by_name(R_win.columns.tolist(), mu_ann)

        w_list.append(w); delta_list.append(float(delta))

    return {"type": "piecewise", "w_list": w_list, "segs": segs,
            "kappa": params.get("kappa", xp.nan), "delta_list": delta_list}


def fit_mvo(data, params, G):
    from regime_dro.report import _print_mu_by_name

    delta = 0.0
    if bool(params.get("verbose", False)):
        print(f"[MVO] delta = {delta:.6g}")
        _print_mu_by_name(list(data.get("px_cols", range(len(data["mu_ann_full"])))), data["mu_ann_full"])
    w = solve_optimizer(
        data["mu_ann_full"], data["Sigma_ann_full"],
        delta, G, verbose=bool(params.get("verbose", False)),)
    return {"type": "static", "w": w, "kappa": xp.nan, "delta": float(delta)}


def fit_dro(data, params, G):
    from regime_dro.report import _print_mu_by_name

    delta = compute_delta(params.get("kappa", 1.0),
                          data["mu_ann_full"], data["Sigma_ann_full"], data["train"], params)
    if bool(params.get("verbose", False)):
        print(f"[DRO] delta = {float(delta):.6g}")
        _print_mu_by_name(list(data.get("px_cols", range(len(data["mu_ann_full"])))), data["mu_ann_full"])
    w = solve_optimizer(data["mu_ann_full"], data["Sigma_ann_full"], delta,
                        G, verbose=bool(params.get("verbose", False)))
    return {"type": "static", "w": w, "kappa": params.get("kappa", xp.nan), "delta": float(delta)}


def fit_regime_dro(data, params, G):
    from regime_dro.report import _section, _print_mu_by_name

    n_days = data["n_days"]
    AF = int(params.get("annualization_factor", data.get("ann_factor", 252)))

    segs = params.get("segs")
    if segs is None:
        segs_fn = params.get("segs_fn", None)
        if segs_fn is not None:
            segs = segs_fn(data, params, G)
        else:
            taus  = data.get("taus_true", [0, n_days])
            delay = int(params.get("delay", 0))
            mids  = [int((taus[k-1] + taus[k]) / 2) for k in range(1, len(taus) - 1)]
            dets  = [min(m + delay, n_days - 1) for m in mids]
            for i in range(1, len(dets)):
                if dets[i] <= dets[i - 1]:
                    dets[i] = min(dets[i - 1] + 1, n_days - 1)
            segs = [0] + dets + [n_days]

    if bool(params.get("verbose", False)):
        _section("RegDRO")

    w_list, deltas = [], []

    for a, b in zip(segs[:-1], segs[1:]):
        R_seg = data["train"][a:b]
        if (b - a) < 2:
            mu_est = data["mu_ann_full"]
        else:
            log_seg = xp.log1p(R_seg)
            mu_est  = xp.expm1(log_seg.mean(axis=0) * AF)

        min_obs = int(params.get("min_lookback", 21))
        max_lb  = int(params.get("max_lookback", 1260))
        lam_shr = float(params.get("sigma_shrinkage_lambda", 0.0))

        import numpy as _np
        R_df_full = pd.DataFrame(_np.asarray(data["train"], dtype=float),
                                 columns=list(data.get("px_cols", range(data["train"].shape[1]))))

        t_for_sigma = max(0, min(int(b) - 1, int(data["n_days"]) - 1))
        ws = _window_start(t_for_sigma + 1, min_obs, max_lb)
        R_win_df = R_df_full.iloc[ws : t_for_sigma + 1]

        try:
            Sigma_est = xp.asarray(
                compute_cov_from_window(R_win_df, ann=AF, shrink_lambda=lam_shr, min_obs=min_obs),
                dtype=float
            )
        except Exception:
            Sigma_est = xp.asarray(data["Sigma_ann_full"], float)

        R_source  = R_seg

        params_k = dict(params); params_k["n_ref"] = (b - a)
        delta_k = compute_delta(params_k.get("kappa", 1.0), mu_est, Sigma_est, R_source, params_k)
        if bool(params.get("verbose", False)):
            t_fit = max(0, min(int(b) - 1, int(n_days) - 1))
            D_pos = t_fit
            dt = data.get("index", None)
            dt_str = ""
            if dt is not None and 0 <= t_fit < len(dt):
                d = dt[t_fit]
                dt_str = f"{getattr(d, 'date', lambda: d)()}"
            print(f"[RegDRO] t={D_pos} {dt_str}  seg=[{a},{b})  delta={float(delta_k):.4f}")
            names = list(data.get("px_cols", range(len(mu_est))))
            _print_mu_by_name(names, mu_est)
        w_k = solve_optimizer(mu_est, Sigma_est, delta_k, G, verbose=bool(params.get("verbose", False)))
        deltas.append(float(delta_k)); w_list.append(w_k)

    return {"type": "piecewise", "w_list": w_list, "segs": segs,
            "kappa": params.get("kappa", xp.nan),
            "delta_list": deltas,
            "delta": xp.nan}


def fit_dro_reverse(data, params, G):
    delta = float(params["delta"])
    w = solve_optimizer(
        data["mu_ann_full"], data["Sigma_ann_full"], delta,
        G, verbose=bool(params.get("verbose", False)))
    return {"type": "static", "w": w, "delta": delta, "kappa": xp.nan}


def fit_regime_dro_rev_constSigma(data, params, G):
    segs = params["segs"]
    Sigma_fix = data["Sigma_ann_full"]
    w_list = []
    for j, (a, b) in enumerate(zip(segs[:-1], segs[1:])):
        R_seg = data["train"][a:b]
        log_seg = xp.log1p(R_seg)
        AF = int(params.get("annualization_factor", data.get("ann_factor", 252)))
        mu_est = xp.expm1(log_seg.mean(axis=0) * AF)
        w = solve_optimizer(mu_est, Sigma_fix, float(params["delta_list"][j]),
                            G, verbose=bool(params.get("verbose", False)))
        w_list.append(w)
    return {"type": "piecewise", "w_list": w_list, "segs": segs, "delta_list": params["delta_list"]}
