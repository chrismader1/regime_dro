# covariance.py - Risk-budget covariance Sigma_t: the four design variants.
#
# Shared discipline (design, Risk-budget covariance):
#   * correlations are always shrunk toward the regime-agnostic Ledoit-Wolf
#     correlation R_LW from the training window (intensity `gamma`, a
#     select-span choice);
#   * any non-PSD assembly is repaired by projection to the nearest PSD
#     matrix, and repairs are counted.
#
# Variants:
#   1 regime-agnostic  : Ledoit-Wolf covariance of the window
#   2 argmax buckets   : pairwise correlations bucketed by the argmax label
#                        pair (z_i, z_j), shrunk toward R_LW, bucket selected
#                        by the current label pair
#   3 soft blend       : bucket correlations estimated with product posterior
#                        date weights and blended with the current product
#                        posterior weights
#   4 mixture diagonal : per-stock mixture variances (Section regdro risk
#                        budget) on the diagonal, correlations from 1/2/3
#
# No sklearn dependency: Ledoit-Wolf (2004, shrink toward scaled identity)
# is implemented directly.

import numpy as np


# ---------------------------------------------------------------------------
# Ledoit-Wolf and helpers
# ---------------------------------------------------------------------------

def ledoit_wolf_cov(X):
    """Ledoit-Wolf (2004) shrinkage toward the scaled identity.

    X : (T, N) demeaned-or-not return panel (rows with any non-finite entry
        are dropped). Returns (Sigma_lw, shrinkage in [0, 1]).
    """
    X = np.asarray(X, dtype=float)
    ok = np.isfinite(X).all(axis=1)
    Xc = X[ok]
    T, N = Xc.shape
    if T < 2:
        raise ValueError(f"ledoit_wolf_cov: need T >= 2 finite rows, got {T}")
    Xc = Xc - Xc.mean(axis=0, keepdims=True)
    S = (Xc.T @ Xc) / T
    mu = float(np.trace(S)) / N
    d2 = float(np.sum((S - mu * np.eye(N)) ** 2))
    b2_bar = 0.0
    for t in range(T):
        xt = Xc[t][:, None]
        b2_bar += float(np.sum((xt @ xt.T - S) ** 2))
    b2_bar /= T ** 2
    b2 = min(b2_bar, d2)
    rho = 0.0 if d2 <= 0.0 else b2 / d2
    Sigma_lw = rho * mu * np.eye(N) + (1.0 - rho) * S
    return Sigma_lw, float(rho)


def cov_to_corr(S):
    """Correlation matrix from a covariance; zero-variance rows get corr 0
    off-diagonal, 1 on."""
    S = np.asarray(S, dtype=float)
    d = np.sqrt(np.clip(np.diag(S), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        R = S / np.outer(d, d)
    R[~np.isfinite(R)] = 0.0
    np.fill_diagonal(R, 1.0)
    return np.clip(R, -1.0, 1.0)


def corr_to_cov(R, variances):
    """Covariance from a correlation matrix and a variance vector."""
    s = np.sqrt(np.clip(np.asarray(variances, dtype=float), 0.0, None))
    return np.asarray(R, dtype=float) * np.outer(s, s)


def nearest_psd(S, eps=1e-10):
    """Project a symmetric matrix onto the PSD cone by eigenvalue clipping.
    Returns (S_psd, repaired: bool)."""
    S = np.asarray(S, dtype=float)
    S_sym = 0.5 * (S + S.T)
    vals, vecs = np.linalg.eigh(S_sym)
    if vals.min() >= float(eps):
        return S_sym, False
    vals_c = np.clip(vals, float(eps), None)
    return (vecs * vals_c) @ vecs.T, True


# ---------------------------------------------------------------------------
# Weighted pairwise correlations (variants 2 and 3)
# ---------------------------------------------------------------------------

def _weighted_corr_pair(x, y, w):
    """Weighted Pearson correlation of two series (weights >= 0)."""
    w = np.asarray(w, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0.0)
    if int(ok.sum()) < 2:
        return np.nan
    x, y, w = x[ok], y[ok], w[ok]
    wsum = w.sum()
    if wsum <= 0.0:
        return np.nan
    mx = np.sum(w * x) / wsum
    my = np.sum(w * y) / wsum
    cxy = np.sum(w * (x - mx) * (y - my)) / wsum
    vx = np.sum(w * (x - mx) ** 2) / wsum
    vy = np.sum(w * (y - my) ** 2) / wsum
    if vx <= 0.0 or vy <= 0.0:
        return np.nan
    return float(np.clip(cxy / np.sqrt(vx * vy), -1.0, 1.0))


def corr_argmax_buckets(X, Z, R_lw, z_now, gamma, min_obs=63):
    """Variant 2: per pair (i, j), correlations bucketed by the argmax label
    pair, shrunk toward R_lw[i, j]; the current label pair selects the bucket.

    X     : (T, N) training-window returns
    Z     : (T, N) argmax labels in {0, 1} (NaN = unknown date)
    R_lw  : (N, N) Ledoit-Wolf correlation of the window
    z_now : (N,) current labels
    gamma : shrink intensity toward R_lw in [0, 1]
    """
    X = np.asarray(X, dtype=float)
    Z = np.asarray(Z, dtype=float)
    R_lw = np.asarray(R_lw, dtype=float)
    z_now = np.asarray(z_now, dtype=float)
    N = X.shape[1]
    gamma = float(np.clip(gamma, 0.0, 1.0))
    R = np.eye(N)
    for i in range(N):
        for j in range(i + 1, N):
            r_lw = R_lw[i, j]
            zi, zj = z_now[i], z_now[j]
            r = np.nan
            if np.isfinite(zi) and np.isfinite(zj):
                m = (Z[:, i] == zi) & (Z[:, j] == zj)
                if int(np.sum(m)) >= int(min_obs):
                    r = _weighted_corr_pair(X[:, i], X[:, j],
                                            m.astype(float))
            r_ij = r_lw if not np.isfinite(r) else (1.0 - gamma) * r + gamma * r_lw
            R[i, j] = R[j, i] = r_ij
    return R


def corr_soft_blend(X, P, R_lw, p_now, gamma, min_eff=63.0):
    """Variant 3: per pair (i, j) and regime pair (k, l), correlations
    estimated with product posterior date weights P[:, i, k] * P[:, j, l],
    shrunk toward R_lw[i, j], then blended with the current product weights
    p_now[i, k] * p_now[j, l].

    P     : (T, N, 2) posterior paths (P[..., 0] = regime-1 probability)
    p_now : (N, 2) current posteriors
    min_eff : minimum effective sample size (sum of weights) per bucket;
              thin buckets fall back to R_lw.
    """
    X = np.asarray(X, dtype=float)
    P = np.asarray(P, dtype=float)
    R_lw = np.asarray(R_lw, dtype=float)
    p_now = np.asarray(p_now, dtype=float)
    N = X.shape[1]
    gamma = float(np.clip(gamma, 0.0, 1.0))
    R = np.eye(N)
    for i in range(N):
        for j in range(i + 1, N):
            r_lw = R_lw[i, j]
            acc = 0.0
            for k in (0, 1):
                for l in (0, 1):
                    w_t = P[:, i, k] * P[:, j, l]
                    w_t = np.where(np.isfinite(w_t), w_t, 0.0)
                    if float(w_t.sum()) >= float(min_eff):
                        r_kl = _weighted_corr_pair(X[:, i], X[:, j], w_t)
                    else:
                        r_kl = np.nan
                    if not np.isfinite(r_kl):
                        r_kl = r_lw
                    else:
                        r_kl = (1.0 - gamma) * r_kl + gamma * r_lw
                    acc += p_now[i, k] * p_now[j, l] * r_kl
            R[i, j] = R[j, i] = float(np.clip(acc, -1.0, 1.0))
    return R


# ---------------------------------------------------------------------------
# Variant 4 diagonal and assembly
# ---------------------------------------------------------------------------

def mixture_variance_diag(p_now, m_k, Q_k):
    """Per-stock mixture variance (risk budget of Section regdro):
    Sigma_i = sum_k pi_ik Q_ik + sum_k pi_ik (m_ik - m_bar_i)^2.

    p_now : (N, 2) current posteriors
    m_k   : (N, 2) current per-regime predictions
    Q_k   : (N, 2) per-regime innovation variances
    All in consistent (annualized or daily) units; unit choice is the
    caller's.
    """
    p = np.asarray(p_now, dtype=float)
    m = np.asarray(m_k, dtype=float)
    Q = np.asarray(Q_k, dtype=float)
    m_bar = np.sum(p * m, axis=1, keepdims=True)
    return np.sum(p * Q, axis=1) + np.sum(p * (m - m_bar) ** 2, axis=1)


def assemble_sigma(variant, *, X, ann, gamma, Z=None, z_now=None, P=None,
                   p_now=None, m_k=None, Q_k=None, corr_variant=1,
                   min_obs=63, psd_eps=1e-10):
    """Assemble the budget matrix Sigma_t for one rebalance date.

    variant : 1..4 per the design; for variant 4, `corr_variant` in {1, 2, 3}
              selects the correlation source.
    X       : (T, N) training-window daily returns
    ann     : annualization factor (variances scaled by `ann`)
    Returns (Sigma_ann, info) with info = {"repaired": bool,
    "lw_shrinkage": float}.
    """
    variant = int(variant)
    S_lw_daily, rho_lw = ledoit_wolf_cov(X)
    R_lw = cov_to_corr(S_lw_daily)

    def _corr(which):
        if which == 1:
            return R_lw
        if which == 2:
            if Z is None or z_now is None:
                raise ValueError("variant 2 needs Z and z_now")
            return corr_argmax_buckets(X, Z, R_lw, z_now, gamma, min_obs=min_obs)
        if which == 3:
            if P is None or p_now is None:
                raise ValueError("variant 3 needs P and p_now")
            return corr_soft_blend(X, P, R_lw, p_now, gamma, min_eff=float(min_obs))
        raise ValueError(f"unknown correlation variant {which}")

    if variant == 1:
        Sigma = S_lw_daily * float(ann)
    elif variant in (2, 3):
        R = _corr(variant)
        var_daily = np.clip(np.diag(S_lw_daily), 0.0, None)
        Sigma = corr_to_cov(R, var_daily) * float(ann)
    elif variant == 4:
        if p_now is None or m_k is None or Q_k is None:
            raise ValueError("variant 4 needs p_now, m_k, Q_k")
        R = _corr(int(corr_variant))
        var_ann = mixture_variance_diag(p_now, m_k, Q_k)
        Sigma = corr_to_cov(R, var_ann)  # inputs already annualized by caller
    else:
        raise ValueError(f"unknown Sigma variant {variant}")

    Sigma_psd, repaired = nearest_psd(Sigma, eps=psd_eps)
    return Sigma_psd, {"repaired": bool(repaired), "lw_shrinkage": float(rho_lw)}
