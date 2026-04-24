# windows.py

import numpy as np
import pandas as pd


def _window_start(t_end_exclusive: int, min_lb: int, max_lb: int) -> int:
    te = int(t_end_exclusive)
    a = max(0, te - int(max_lb))
    if te - a < int(min_lb):
        a = max(0, te - int(min_lb))
    return a


def compute_mean_from_window(
    R_win,
    mask,
    *,
    min_obs: int = 252,
    ann: int = 252,
):
    """
    Mean-only estimator for SIMPLE returns (from pct_change()).
    For MVO/DRO pass an all-True mask; for RegDRO pass the in-regime mask.
    """
    X = R_win.to_numpy(np.float64, copy=False) if isinstance(R_win, pd.DataFrame) else np.asarray(R_win, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("R_win must be 2D (T,d).")
    m = np.asarray(mask)
    if m.dtype != bool or m.ndim != 1 or m.shape[0] != X.shape[0]:
        raise ValueError(f"mask must be (T,) bool matching R_win.shape[0]; got mask {m.shape}, R {X.shape}")
    if not np.any(m):
        raise ValueError("in-regime mask is empty.")

    Xm = X[m, :]
    finite = np.isfinite(Xm)
    counts = finite.sum(axis=0)
    if np.any(counts < min_obs):
        raise ValueError(f"Insufficient in-regime observations: min {int(counts.min())} < required {min_obs}")

    Xm = np.where(finite, Xm, np.nan)
    mu_periodic = np.nanmean(Xm, axis=0)
    mu_ann = mu_periodic * float(ann)

    if not np.all(np.isfinite(mu_ann)):
        raise ValueError("Non-finite annualized mean encountered.")
    return mu_ann


def compute_cov_from_window(
    R_win,
    *,
    ann: int = 252,
    shrink_lambda: float = 0.0,
    min_obs: int = 2,
):
    """
    Unconditional covariance for SIMPLE returns on the full lookback window.
    Shrinkage towards scaled identity: (1-λ)Σ + λ * s2_bar * I.
    """
    if isinstance(R_win, pd.DataFrame):
        X = R_win.to_numpy(np.float64, copy=False)
    else:
        X = np.asarray(R_win, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError(f"R_win must be 2D (T,d); got shape {X.shape}")

    row_ok = np.isfinite(X).all(axis=1)
    Xc = X[row_ok, :]
    if Xc.shape[0] < min_obs:
        raise ValueError(
            f"Not enough observations for covariance: {Xc.shape[0]} < {min_obs}"
        )

    Sig = np.cov(Xc.T, ddof=1)
    Sig = np.asarray(Sig, dtype=np.float64)

    if Sig.ndim == 0:
        Sig = Sig.reshape(1, 1)
    elif Sig.ndim != 2:
        raise ValueError(f"Unexpected covariance ndim={Sig.ndim}")

    Sig_ann = Sig * float(ann)

    lam = float(np.clip(shrink_lambda, 0.0, 1.0))
    if lam > 0.0:
        N = Sig_ann.shape[0]
        s2_bar = float(np.trace(Sig_ann) / max(N, 1))
        Sig_ann = (1.0 - lam) * Sig_ann + lam * s2_bar * np.eye(N, dtype=np.float64)

    if not np.all(np.isfinite(Sig_ann)):
        raise ValueError("Non-finite covariance encountered.")
    if Sig_ann.shape[0] != Sig_ann.shape[1]:
        raise ValueError(f"Covariance must be square; got {Sig_ann.shape}")

    return Sig_ann
