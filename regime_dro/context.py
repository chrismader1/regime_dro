# context.py

import numpy as np
import pandas as pd

from regime_dro.arrays import asnumpy_strict
from regime_dro.windows import compute_mean_from_window, compute_cov_from_window


def regdro_decision_context(
    *,
    D_pos: int,
    full_index_fit: pd.DatetimeIndex,
    df_returns_full: pd.DataFrame,
    names_all: list,
    Z_labels_fit: dict,
    AF: int,
    min_obs: int,
    max_lb: int,
    lam_shr: float,
    G: dict,
):
    """
    Build the optimisation context at a single decision date (position on FIT index).
    """
    a_win = max(0, int(D_pos) - int(max_lb))
    b_win = int(D_pos)
    if b_win - a_win < max(2, int(min_obs)):
        return False, [], {}, pd.DataFrame(), np.array([]), np.array([]), np.zeros((0, 0)), np.zeros((0, 0)), np.zeros(0, dtype=bool)

    win_idx = full_index_fit.take(np.arange(a_win, b_win, dtype=int))

    keep, masks = [], []
    for n in names_all:
        z_ser = np.asarray(Z_labels_fit[n], dtype=float)
        z_now = z_ser[D_pos] if (0 <= D_pos < len(z_ser)) else np.nan
        if not np.isfinite(z_now):
            continue
        m = (z_ser[a_win:b_win] == z_now)
        if not np.any(m):
            continue
        x = df_returns_full.loc[win_idx, n].to_numpy(float)
        m = m & np.isfinite(x)
        if int(m.sum()) >= int(min_obs):
            keep.append(n)
            masks.append(m)

    if not keep:
        return False, [], {}, pd.DataFrame(), np.array([]), np.array([]), np.zeros((0, 0)), np.zeros((0, 0)), np.zeros(0, dtype=bool)

    cap_applies = bool(G.get("no_shorting", False)) and np.isfinite(G.get("max_pos_size", np.nan)) and (G.get("max_pos_size", 0.0) > 0.0)

    if cap_applies:
        c_max = float(G.get("max_cash", 0.0))
        u     = float(G.get("max_pos_size", 1.0))
        N_req = int(np.ceil((1.0 - c_max) / max(u, 1e-12)))
        if len(keep) < N_req:
            return False, [], {}, pd.DataFrame(), np.array([]), np.array([]), np.zeros((0, 0)), np.zeros((0, 0)), np.zeros(0, dtype=bool)

    X_win_df = df_returns_full.loc[win_idx, keep]
    mu_cond = np.asarray(
        [compute_mean_from_window(X_win_df[[n]], masks[j], min_obs=min_obs, ann=AF)[0]
         for j, n in enumerate(keep)],
        dtype=float
    )
    Sig = compute_cov_from_window(X_win_df[keep], ann=AF, shrink_lambda=lam_shr, min_obs=min_obs)

    mask_all = np.ones(X_win_df.shape[0], dtype=bool)
    mu_uncond = compute_mean_from_window(X_win_df, mask_all, min_obs=min_obs, ann=AF)

    mask_cond_all = np.logical_and.reduce(masks) if len(masks) else np.zeros(X_win_df.shape[0], dtype=bool)

    X_win = X_win_df.to_numpy(float)
    pos_map = {n: i for i, n in enumerate(names_all)}
    return True, keep, pos_map, X_win_df, mu_cond, mu_uncond, Sig, X_win, mask_cond_all


def make_solver_cfg_from_CONFIG(CONFIG):
    P = CONFIG["PORTFOLIO"]

    max_cash = P.get("max_cash", None)
    max_cash = None if max_cash is None else float(max_cash)

    max_pos_size = P.get("max_pos_size", None)
    max_pos_size = None if max_pos_size is None else float(max_pos_size)

    return {
        "risk_budget":    float(P["risk_budget"]),
        "risk_free_rate": float(P["risk_free_rate"]),
        "epsilon_sigma":  float(P["epsilon_sigma"]),
        "no_shorting":    bool(P.get("no_shorting", False)),
        "no_leverage":    bool(P.get("no_leverage", False)),
        "max_cash":       max_cash,
        "max_pos_size":   max_pos_size,
    }


def gross_exp_on_window(fit, T_req, win=None):
    """Average gross exposure over a reporting window of length T_req (optionally [a,b) slice)."""
    import numpy as _np

    def _ge_vec(w):
        w_np = asnumpy_strict(w, dtype=float).ravel()
        return float(_np.sum(_np.abs(w_np)))

    if fit["type"] == "static":
        return _ge_vec(fit["w"])

    segs = [int(x) for x in fit["segs"]]
    a0, b0 = (0, 10**12) if win is None else (int(win[0]), int(win[1]))
    num = 0.0
    for (a, b), w in zip(zip(segs[:-1], segs[1:]), fit["w_list"]):
        L = max(0, min(b, b0) - max(a, a0))
        if L > 0:
            num += L * _ge_vec(w)
    return num / max(T_req, 1)


def make_index_rebal(
    intersection_index: pd.DatetimeIndex,
    start_dt,
    end_dt,
    rebalance_period: int,
    freq: str,
):
    """Fixed-period rebalancing dates on the INTERSECTION calendar.
    Builds a target grid at `freq` (e.g. "B", "W-FRI", "M"), takes every
    `rebalance_period`-th point, then snaps each target date to the last
    available date on `intersection_index` that is <= the target."""
    idx_req = pd.DatetimeIndex(
        pd.Series(True, index=intersection_index).loc[start_dt:end_dt].index
    )
    if len(idx_req) == 0:
        return idx_req, [0, 0]
    target_grid = pd.date_range(start=idx_req[0], end=idx_req[-1], freq=freq)
    if len(target_grid) == 0:
        target_grid = pd.DatetimeIndex([idx_req[-1]])
    k = int(max(1, rebalance_period))
    take = np.arange(0, len(target_grid), k, dtype=int)
    targets = target_grid.take(take)
    if (len(targets) == 0) or (targets[-1] != target_grid[-1]):
        targets = targets.append(target_grid[-1:])
    pos = intersection_index.get_indexer(targets, method="pad")
    pos = [p for p in pos if p >= 0]
    idx_rebal = intersection_index.take(sorted(set(pos)))
    pos_full = intersection_index.get_indexer(idx_rebal)
    marks = sorted(set([0] + [int(p) for p in pos_full if p >= 0] + [len(intersection_index)]))
    return pd.DatetimeIndex(idx_rebal), marks


def make_index_union(
    union_index: pd.DatetimeIndex,
    Z_labels: dict,
    start_dt,
    end_dt,
):
    """Union of regime-change dates on the UNION calendar (RegDRO)."""
    T = len(union_index)
    if T == 0:
        return pd.DatetimeIndex([]), [0]

    chg_any = np.zeros(T, dtype=bool)
    for _, z in (Z_labels or {}).items():
        z = np.asarray(z, float)
        finite = np.isfinite(z)
        c = np.zeros(T, dtype=bool)
        if T >= 2:
            c[1:] = finite[1:] & finite[:-1] & (z[1:] != z[:-1])
        chg_any |= c

    idx_req = pd.DatetimeIndex(
        pd.Series(True, index=union_index).loc[start_dt:end_dt].index
    )
    in_req = pd.Index(union_index).isin(idx_req)
    index_union = pd.DatetimeIndex(union_index[chg_any & in_req])

    if len(index_union) == 0:
        taus = [0, T]
    else:
        pos = union_index.get_indexer(index_union)
        pos = [int(p) for p in pos if p >= 0]
        taus = sorted(set([0] + pos + [T]))

    return index_union, taus


def make_index_regdro_periodic(
    intersection_index: pd.DatetimeIndex,
    start_dt,
    end_dt,
    rebalance_period: int,
    freq: str,
):
    """Periodic rebalancing dates for RegDRO. Identical schedule to make_index_rebal;
    regime lookup happens per-rebalance-date in the pipeline."""
    return make_index_rebal(intersection_index, start_dt, end_dt, rebalance_period, freq)


def expand_daily_weights(weights_on_dates: pd.DataFrame, full_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Forward-fill piecewise weights to every day in full_index."""
    w = weights_on_dates.sort_index()
    if len(w.index) == 0:
        return pd.DataFrame(0.0, index=full_index, columns=weights_on_dates.columns)
    if w.index[0] > full_index[0]:
        zero_row = pd.DataFrame([np.zeros(w.shape[1])], index=[full_index[0]], columns=w.columns)
        w = pd.concat([zero_row, w], axis=0)
    return w.reindex(full_index).ffill().fillna(0.0)


def pnl_with_delay_and_cost(
    W_on_dates: pd.DataFrame,
    full_index: pd.DatetimeIndex,
    R_df: pd.DataFrame,
    delay: int,
    tc: float,
    name: str,
):
    """Expand weights to daily, apply execution delay and transaction costs, return PnL series."""
    W_daily = expand_daily_weights(W_on_dates, full_index)
    W_eff   = W_daily.shift(int(delay)).fillna(0.0)
    pnl_g   = (W_eff * R_df).sum(axis=1)
    to      = W_eff.diff().abs().sum(axis=1)
    if len(to):
        to.iloc[0] = W_eff.iloc[0].abs().sum()
    return (pnl_g - float(tc) * to).rename(name), W_daily, W_eff


def period_ends(idx: pd.DatetimeIndex, freq: str = "M") -> pd.DatetimeIndex:
    """Return last available date per period on `idx`."""
    return pd.DatetimeIndex(pd.Series(idx).groupby(idx.to_period(freq)).max())
