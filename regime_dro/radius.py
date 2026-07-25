# radius.py - Wasserstein radius decomposition eps = D + e (regime + estimation).
#
# Maps 1:1 to the theory section:
#   g(pi_max)              bracket of Eq. (dreg), Prop. posteriorweighting
#   pi_dagger              Eq. (pidagger), Cor. crossover
#   pooled_moments         Eqs. (stdmeanlimit)-(stdvarlimit)
#   pooling_cost           Prop. stdexact  (c(lambda), d_k, min_k d_k, b(lambda))
#   regime_radius_path     eps_{i,t} = g(pi_max_i(t)) * delta_tilde_i + e_i(alpha)
#   standard_radius        eps_i     = c(lambda_i) * delta_star_i + e_std_i(alpha)
#   assignment_entropy     Remark louis (H-bar)
#
# All quantities are scalar per stock (Assumption 3: distances along the
# portfolio direction; per-stock fits are univariate).

import numpy as np
from scipy.optimize import brentq

from regime_dro.delta import wasserstein2_gaussian

_INV_SQRT2 = 1.0 / np.sqrt(2.0)


# ---------------------------------------------------------------------------
# g and its inverse (Cor. crossover)
# ---------------------------------------------------------------------------

def g_certainty(pi_max):
    """g(p) = p*sqrt(1-p) + (1-p)*sqrt(p) on [1/2, 1]; vectorized.

    Strictly decreasing on [1/2, 1] with g(1/2) = 1/sqrt(2), g(1) = 0.
    Inputs are clipped into [1/2, 1] (posteriors can carry float noise).
    """
    p = np.clip(np.asarray(pi_max, dtype=float), 0.5, 1.0)
    return p * np.sqrt(1.0 - p) + (1.0 - p) * np.sqrt(p)


def g_inverse(y):
    """Inverse of g on [1/2, 1]. Scalar in, scalar out.

    y >= 1/sqrt(2) -> 1/2;  y <= 0 -> 1.0.
    """
    y = float(y)
    if y >= _INV_SQRT2:
        return 0.5
    if y <= 0.0:
        return 1.0
    return float(brentq(lambda p: g_certainty(p) - y, 0.5, 1.0 - 1e-15))


def pi_dagger(b_floor):
    """Crossover certainty of Eq. (pidagger):
    g^{-1}(b) if b < 1/sqrt(2), else 1/2.
    """
    b = float(b_floor)
    if not np.isfinite(b):
        return np.nan
    if b >= _INV_SQRT2:
        return 0.5
    return g_inverse(b)


# ---------------------------------------------------------------------------
# Separations and pooled moments (Prop. stdexact and its inputs)
# ---------------------------------------------------------------------------

def _w2_scalar(m, Q, m_prime, Q_prime):
    """Scalar Gelbrich distance, delegated to the package's general routine."""
    return wasserstein2_gaussian(
        np.asarray([float(m)]), np.asarray([[float(Q)]]),
        np.asarray([float(m_prime)]), np.asarray([[float(Q_prime)]]),
    )


def separations(m1, m2, Q1, Q2):
    """(delta_star, delta_tilde): mean separation and effective separation.

    delta_star  = |m1 - m2|
    delta_tilde = sqrt(delta_star^2 + (sqrt(Q1) - sqrt(Q2))^2)
    """
    d_star = abs(float(m1) - float(m2))
    v_mis = (np.sqrt(max(float(Q1), 0.0)) - np.sqrt(max(float(Q2), 0.0))) ** 2
    d_tilde = float(np.sqrt(d_star ** 2 + v_mis))
    return d_star, d_tilde


def pooled_moments(lam, m1, m2, Q1, Q2):
    """Probability limits of the pooled moments, Eqs. (stdmeanlimit)-(stdvarlimit)."""
    lam = float(lam)
    d_star, _ = separations(m1, m2, Q1, Q2)
    m_std = lam * float(m1) + (1.0 - lam) * float(m2)
    Q_std = lam * float(Q1) + (1.0 - lam) * float(Q2) + lam * (1.0 - lam) * d_star ** 2
    return m_std, Q_std


def pooling_cost(lam, m1, m2, Q1, Q2):
    """Per-regime distances to the pooled nominal (Prop. stdexact) and derived
    constants.

    Returns dict with:
      d1, d2       : Gelbrich distances of each true conditional to the pooled
                     nominal, exact (Eq. dk)
      d_std        : max_k d_k  (= c(lambda) * delta_star, the D component of
                     standard DRO)
      c            : d_std / delta_star (NaN when delta_star == 0)
      min_dk       : min_k d_k (Cor. meanreach)
      b_floor      : min_k d_k / delta_tilde (Eq. bfloor; NaN when
                     delta_tilde == 0)
      pi_dagger    : crossover certainty from b_floor (Eq. pidagger)
      delta_star, delta_tilde
    """
    m_std, Q_std = pooled_moments(lam, m1, m2, Q1, Q2)
    d1 = _w2_scalar(m1, Q1, m_std, Q_std)
    d2 = _w2_scalar(m2, Q2, m_std, Q_std)
    d_star, d_tilde = separations(m1, m2, Q1, Q2)
    d_std = max(d1, d2)
    min_dk = min(d1, d2)
    c = d_std / d_star if d_star > 0.0 else np.nan
    b_floor = min_dk / d_tilde if d_tilde > 0.0 else np.nan
    return {
        "d1": d1, "d2": d2,
        "d_std": d_std, "c": c,
        "min_dk": min_dk, "b_floor": b_floor,
        "pi_dagger": pi_dagger(b_floor) if np.isfinite(b_floor) else np.nan,
        "delta_star": d_star, "delta_tilde": d_tilde,
    }


# ---------------------------------------------------------------------------
# Radius assembly (Radii paragraph of the design)
# ---------------------------------------------------------------------------

def regime_radius_path(pi_max_path, delta_tilde, e_alpha):
    """eps_t = g(pi_max(t)) * delta_tilde + e(alpha), elementwise.

    Returns (D_path, eps_path). NaNs in pi_max propagate to both.
    """
    p = np.asarray(pi_max_path, dtype=float)
    D = np.where(np.isfinite(p), g_certainty(p) * float(delta_tilde), np.nan)
    return D, D + float(e_alpha)


def standard_radius(pool, e_alpha):
    """eps = d_std + e(alpha) from a pooling_cost() dict. Returns (D, eps)."""
    D = float(pool["d_std"])
    return D, D + float(e_alpha)


def assignment_entropy(P):
    """Average assignment entropy H-bar (Remark louis), in nats.

    P : (T,) array of regime-1 probabilities, or (T, K) array of posteriors.
    NaN rows are skipped.
    """
    P = np.asarray(P, dtype=float)
    if P.ndim == 1:
        P = np.stack([P, 1.0 - P], axis=1)
    ok = np.all(np.isfinite(P), axis=1)
    if not np.any(ok):
        return np.nan
    Pc = np.clip(P[ok], 1e-12, 1.0)
    H = -np.sum(Pc * np.log(Pc), axis=1)
    return float(np.mean(H))
