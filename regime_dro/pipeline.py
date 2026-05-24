# pipeline.py

import numpy as np
import pandas as pd

from regime_dro.arrays import asnumpy_strict
from regime_dro.optimizer import solve_dro
from regime_dro.evaluate import stats_from_series, _max_drawdown_from_series
from regime_dro.fit import _feasible_placeholder
from regime_dro.io import save_out

from regime_dro.artifacts import (
    load_artifacts,
    map_labels_to_calendar,
    select_best_config,
    labels_from_segments_df,
    permissible_tuples_from_CONFIG,
    tuples_in_results_csv,
)
from regime_dro.context import (
    regdro_decision_context,
    make_solver_cfg_from_CONFIG,
    gross_exp_on_window,
    make_index_union,
    pnl_with_delay_and_cost,
    period_ends,
)


def run_regdro(prices_df: pd.DataFrame, CONFIG: dict, verbose: bool = True) -> dict:
    """
    Generic production pipeline: solve RegDRO at each regime-change date on a user-supplied
    price panel and regime labels. No comparison models, no benchmark, no paper experiments.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Daily prices. Index = DatetimeIndex. Columns = tickers. User's responsibility
        (source it from Bloomberg, Yahoo, CSV, etc.).
    CONFIG : dict
        Required keys:
          - MODELS             : list of {'config', 'n_regimes', 'dim_latent'} dicts (permissible tuples)
          - DATA               : {'start_dt', 'end_dt'}
          - REBAL              : {'min_lookback', 'max_lookback', 'rebalance_period', 'freq', 'regdro_rebal_method'}
          - PORTFOLIO          : risk_budget, risk_free_rate, epsilon_sigma, sigma_shrinkage_lambda,
                                 delta_name, no_shorting, no_leverage, max_cash, max_pos_size
          - EXECUTION          : {'execution_delay', 'trading_cost'}
          - DELTA_DEFAULTS     : dict of δ-method parameter templates
          - results_csv        : path to gridsearch_results.csv
          - segments_parquet   : path to gridsearch_segments.parquet
          - dro_pickle         : (optional) path to save output pickle
          - annualization_factor (optional, default 252)

    Returns
    -------
    dict with keys:
      - fit       : piecewise fit {type, w_list, segs, names, delta_list}
      - summary   : dict of OOS stats (mu_ann, sigma_ann, sharpe_ann, vol_breach, max_dd, gross_exp, delta, ...)
      - daily     : pd.Series, daily PnL (net of costs, with execution delay)
      - holdings  : pd.DataFrame, month-end effective weights
      - Z_labels  : dict[str, np.ndarray] regime labels per security
      - securities: list[str] investable universe
      - returns   : pd.DataFrame full returns panel used for fitting
      - G         : solver config
    """
    G = make_solver_cfg_from_CONFIG(CONFIG)

    # Load + validate artifacts
    arts = load_artifacts(CONFIG["results_csv"], CONFIG["segments_parquet"])
    df_res = arts["df_res"]
    df_seg = arts["df_seg"]

    # Intersect user price panel with artifact universe
    have_px  = set(map(str, prices_df.columns))
    have_res = set(df_res["security"].astype(str).str.strip().unique())
    have_seg = set(df_seg["security"].astype(str).str.strip().unique())
    px_cols = sorted(have_px & have_res & have_seg)
    if not px_cols:
        raise RuntimeError("No securities intersect between prices_df, results_csv, and segments_parquet.")

    df_raw = prices_df[px_cols].astype(float)

    # Date range
    s = CONFIG["DATA"].get("start_dt")
    e = CONFIG["DATA"].get("end_dt")

    idx  = df_raw.index
    s_dt = pd.to_datetime(s) if s is not None else idx[0]
    e_dt = pd.to_datetime(e) if e is not None else idx[-1]

    i_start = idx.get_indexer([s_dt], method="nearest")[0]
    i_end   = idx.get_indexer([e_dt], method="nearest")[0]
    i_hist  = max(0, i_start - int(CONFIG["REBAL"]["max_lookback"]))

    df_raw_slice_hist = df_raw.iloc[i_hist : i_end + 1]
    if df_raw_slice_hist.shape[0] < 2:
        raise RuntimeError("Not enough rows after slicing prices_df with pre-start history.")

    df_returns_full = df_raw_slice_hist.pct_change().fillna(0.0)
    full_index_fit  = df_returns_full.index

    oos_index = full_index_fit[(full_index_fit >= s_dt) & (full_index_fit <= e_dt)]
    if len(oos_index) == 0:
        raise RuntimeError("Empty OOS index after applying [start_dt, end_dt].")

    df_returns = df_returns_full
    full_index = oos_index

    # Solver params
    lam    = float(CONFIG["PORTFOLIO"]["sigma_shrinkage_lambda"])
    min_lb = int(CONFIG["REBAL"]["min_lookback"])
    max_lb = int(CONFIG["REBAL"]["max_lookback"])
    AF     = int(CONFIG.get("annualization_factor", 252))

    # rSLDS labels on OOS + FIT calendars
    Z_labels     = {}
    Z_labels_fit = {}

    perm_tuples = permissible_tuples_from_CONFIG(CONFIG)
    present     = tuples_in_results_csv(df_res, perm_tuples)
    missing     = sorted(set(perm_tuples) - present)
    if missing:
        raise RuntimeError(f"results_csv is missing REQUIRED tuples: {missing}")

    df_res_perm = df_res.copy()
    df_res_perm["config"]     = df_res_perm["config"].astype(str).str.strip()
    df_res_perm["n_regimes"]  = pd.to_numeric(df_res_perm["n_regimes"],  errors="coerce").astype("Int64")
    df_res_perm["dim_latent"] = pd.to_numeric(df_res_perm["dim_latent"], errors="coerce").astype("Int64")
    perm_set = set(perm_tuples)
    mask_perm = df_res_perm.apply(
        lambda r: (str(r["config"]), int(r["n_regimes"]) if pd.notna(r["n_regimes"]) else -1,
                   int(r["dim_latent"]) if pd.notna(r["dim_latent"]) else -1) in perm_set,
        axis=1
    )
    df_res_perm = df_res_perm[mask_perm]

    for sec in px_cols:
        best = select_best_config(df_res_perm, sec)
        if best is None:
            continue
        c_name, K, D = best
        z_ser = labels_from_segments_df(df_seg, sec, c_name, K, D)
        if z_ser is None:
            continue
        Z_labels[sec]     = map_labels_to_calendar(z_ser, full_index)
        Z_labels_fit[sec] = map_labels_to_calendar(z_ser, full_index_fit)

    names_all = [t for t in px_cols if t in Z_labels]
    if not names_all:
        raise RuntimeError("No assets produced regime labels → cannot run RegDRO.")

    # Build rebalance schedule: periodic (default) or regime_change (legacy)
    method = str(CONFIG["REBAL"].get("regdro_rebal_method", "periodic"))
    if method == "periodic":
        rebal_period = int(CONFIG["REBAL"]["rebalance_period"])
        freq         = str(CONFIG["REBAL"]["freq"])
        _, taus = make_index_regdro_periodic(full_index, s, e, rebal_period, freq)
    elif method == "regime_change":
        _, taus = make_index_union(full_index, {k: np.asarray(v, float) for k, v in Z_labels.items()}, s, e)
    else:
        raise ValueError(f"REBAL.regdro_rebal_method must be 'periodic' or 'regime_change', got {method!r}")
    taus = [int(x) for x in taus]
    
    params_reg = dict(CONFIG["DELTA_DEFAULTS"][CONFIG["PORTFOLIO"]["delta_name"]])

    # Solve RegDRO at each regime-change date
    w_reg_list, del_reg_list = [], []

    for a, b in zip(taus[:-1], taus[1:]):
        t_mid = min(max(a, 0), len(full_index) - 1)
        D     = full_index[t_mid]
        D_pos = full_index_fit.get_loc(D)

        ok, keep, pos_map, X_win_df, mu_cond, mu_uncond, Sig, X_win, mask_cond_all = regdro_decision_context(
            D_pos=D_pos,
            full_index_fit=full_index_fit,
            df_returns_full=df_returns_full,
            names_all=names_all,
            Z_labels_fit=Z_labels_fit,
            AF=AF,
            min_obs=min_lb,
            max_lb=max_lb,
            lam_shr=lam,
            G=G,)

        if not ok:
            w_reg_list.append(np.asarray(_feasible_placeholder(len(names_all), G), float))
            del_reg_list.append(np.nan)
            continue

        use_cond = bool(mask_cond_all.size) and int(mask_cond_all.sum()) > 0
        X_cond = X_win[mask_cond_all, :] if use_cond else X_win
        w_reg_sub, delta_k = solve_dro(mu_cond, Sig, params_reg, G, R=X_cond, verbose=bool(verbose))

        w_reg_full = np.zeros(len(names_all))
        w_reg_sub_np = asnumpy_strict(w_reg_sub, float).ravel()
        idxs = [pos_map[n] for n in keep]
        w_reg_full[idxs] = w_reg_sub_np
        w_reg_list.append(w_reg_full); del_reg_list.append(float(delta_k))

    fit_reg = {
        "type": "piecewise",
        "w_list": [np.asarray(w, float) for w in w_reg_list],
        "segs":   np.asarray(taus, dtype=int),
        "names":  names_all,
        "delta_list": [float(d) if np.isfinite(d) else np.nan for d in del_reg_list],
    }

    # Daily PnL
    k_delay = int(CONFIG["EXECUTION"].get("execution_delay", 0))
    tc      = float(CONFIG["EXECUTION"].get("trading_cost", 0.0))

    rebal_dates = full_index[np.asarray(fit_reg["segs"], int)[:-1]]
    oos_start = full_index[0]
    rows = []
    for dt, w in zip(rebal_dates, fit_reg["w_list"]):
        if dt >= oos_start:
            rows.append(pd.Series(asnumpy_strict(w, float).ravel(), index=names_all, name=dt))
    W_on_dates = pd.DataFrame(rows).sort_index()

    R_oos_panel = df_returns.loc[full_index, names_all]
    daily, W_daily, W_eff = pnl_with_delay_and_cost(W_on_dates, full_index, R_oos_panel, k_delay, tc, "RegDRO_daily")

    # Summary stats
    n_aligned = len(full_index)
    x = daily.to_numpy(float)
    mu, sig, sh = stats_from_series(x, {
        "n_days": n_aligned,
        "risk_free_rate": G["risk_free_rate"],
        "annualization_factor": AF,
    })
    summary = {
        "mu_ann": mu,
        "sigma_ann": sig,
        "sharpe_ann": sh,
        "vol_breach": max(sig - G["risk_budget"], 0.0),
        "max_dd": _max_drawdown_from_series(x),
        "gross_exp": gross_exp_on_window(fit_reg, n_aligned),
        "delta": float(np.nanmean(fit_reg["delta_list"])) if len(fit_reg["delta_list"]) else np.nan,
    }

    # Monthly holdings
    me_idx = period_ends(full_index, "M")
    holdings = W_eff.reindex(me_idx).ffill().rename_axis("date")

    # Print summary
    if verbose:
        print("\n" + "=" * 60)
        print("RegDRO — OOS Summary")
        print("=" * 60)
        for k, v in summary.items():
            print(f"  {k:<14s} {v:.4f}" if np.isfinite(float(v)) else f"  {k:<14s} NaN")
        print(f"  n_securities   {len(names_all)}")
        print(f"  n_rebal_dates  {len(fit_reg['w_list'])}")

    out = {
        "fit": fit_reg,
        "summary": summary,
        "daily": daily,
        "holdings": holdings,
        "Z_labels": {k: np.asarray(v) for k, v in Z_labels.items()},
        "securities": names_all,
        "returns": df_returns,
        "G": G,
    }

    if "dro_pickle" in CONFIG and CONFIG["dro_pickle"]:
        save_out(out, CONFIG["dro_pickle"])

    return out
