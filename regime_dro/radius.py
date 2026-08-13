# radius.py - Wasserstein radius decomposition eps = D + e (regime + estimation).
#
# Maps 1:1 to the theory section (dro_radius.tex, revised):
#   separations            delta_star, delta_tilde (Sec. stdtworegimes)
#   pooled_moments         Eqs. (stdmeanlimit)-(stdvarlimit)
#   pooling_cost           Prop. stdexact (d_k, c(lambda), max_k d_k = D_std,
#                          Eq. dstdmax) and Cor. meanreach (min_k d_k floor)
#   ambiguity_bounds       Prop. ambiguity: (1-pi_k) delta* <= D_k(pi)
#                                            <= sqrt(1-pi_k) delta~*
#   mixture_w2_distances   Eq. (dkexact): D_k(pi) exact via the 1-D quantile
#                          representation Eq. (quantilew2), and the ordering
#                          of Lemma majorityminority (majority regime nearer)
#   calibrate_tau          Eq. (taucal): select-span threshold at budget
#                          alpha_z
#   branch_component       Eq. (dregrule): confident branch D_{k_max},
#                          ambiguous branch D_{k_min}
#   realized_miss_frequency Eq. (missfreq)
#   occupancy              Eq. (occupancy): confident-branch occupancy q
#   confident_win          Eq. (confidentwin): tau < (max_k d_k / delta~*)^2
#   q_cutoff               Eq. (qcutoff): occupancy cutoff for
#                          bar D_reg < D_std
#   regime_radius_path     eps(t) = D_reg(t) + e(alpha_e), branch rule,
#                          with alpha_z = alpha_e = alpha/2 (split_alpha)
#   standard_radius        eps_std = D_std + e_std(alpha_e)
#   product_wedge          Eq. (productwedge), Cor. productjoint
#   assignment_entropy     Remark louis (H-bar)
#
# All quantities are scalar per stock (Assumption 3: distances along the
# portfolio direction; per-stock fits are univariate).
#
# LEGACY (previous revision, superseded): the g(pi_max) bracket and the
# pi-dagger crossover are no longer part of the theory. g_certainty,
# g_inverse, pi_dagger and the b_floor/pi_dagger keys of pooling_cost are
# retained at the bottom of this module solely so existing pipeline code
# keeps running; new code must use the branch rule above.

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri

from regime_dro.delta import wasserstein2_gaussian

_INV_SQRT2 = 1.0 / np.sqrt(2.0)


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
      d_std        : max_k d_k = D_std (Eq. dstdmax; = c(lambda) * delta_star)
      c            : d_std / delta_star (NaN when delta_star == 0);
                     c(lambda) >= max(lambda, 1-lambda) >= 1/2 (Prop. stdexact)
      min_dk       : min_k d_k, floored by min(lambda, 1-lambda) * delta_star
                     (Cor. meanreach)
      delta_star, delta_tilde
      b_floor, pi_dagger : LEGACY keys of the previous revision's crossover
                     construction, retained for compatibility only; the
                     current theory replaces them with the branch rule
                     (calibrate_tau / branch_component).
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
# Posterior-weighted ambiguity: bounds and exact distances
# (Prop. ambiguity, Eq. dkexact, Lemma majorityminority)
# ---------------------------------------------------------------------------

def ambiguity_bounds(pi_k, delta_star, delta_tilde):
    """Prop. ambiguity: bounds on the distance from regime k's conditional to
    the posterior-weighted mixture,

        (1 - pi_k) * delta_star <= D_k(pi) <= sqrt(1 - pi_k) * delta_tilde.

    Vectorized in pi_k; returns (lower, upper).
    """
    p = np.clip(np.asarray(pi_k, dtype=float), 0.0, 1.0)
    lo = (1.0 - p) * float(delta_star)
    hi = np.sqrt(1.0 - p) * float(delta_tilde)
    return lo, hi


def _mixture_inverse_cdf(u, pi1, m1, s1, m2, s2, n_x=4001):
    """Inverse CDF of the two-component Gaussian mixture on a quantile grid,
    by monotone interpolation of the CDF on a fine x-grid."""
    lo = min(m1 - 10.0 * s1, m2 - 10.0 * s2)
    hi = max(m1 + 10.0 * s1, m2 + 10.0 * s2)
    if hi <= lo:
        hi = lo + 1e-12
    x = np.linspace(lo, hi, int(n_x))
    F = pi1 * ndtr((x - m1) / max(s1, 1e-300)) \
        + (1.0 - pi1) * ndtr((x - m2) / max(s2, 1e-300))
    # enforce strict monotonicity against float ties in the tails
    F = np.maximum.accumulate(F)
    F += np.linspace(0.0, 1e-12, int(n_x))
    return np.interp(u, F, x)


def mixture_w2_distances(pi1, m1, m2, Q1, Q2, n_u=2048, n_x=4001):
    """Exact one-dimensional distances of Eq. (dkexact),

        D_k(pi) = [ int_0^1 (Phi_k^{-1}(u) - F_pi^{-1}(u))^2 du ]^{1/2},

    with F_pi = pi1 * Phi_1 + (1 - pi1) * Phi_2, evaluated by midpoint
    quadrature on the quantile representation Eq. (quantilew2). Returns
    (D_1, D_2). By Lemma majorityminority the majority regime is the nearer:
    D_{k_max} = min_k D_k and D_{k_min} = max_k D_k.
    """
    pi1 = float(np.clip(pi1, 0.0, 1.0))
    m1, m2 = float(m1), float(m2)
    s1 = float(np.sqrt(max(float(Q1), 0.0)))
    s2 = float(np.sqrt(max(float(Q2), 0.0)))

    # degenerate mixtures: single Gaussian, Gelbrich exact
    if pi1 >= 1.0:
        return 0.0, _w2_scalar(m2, Q2, m1, Q1)
    if pi1 <= 0.0:
        return _w2_scalar(m1, Q1, m2, Q2), 0.0

    u = (np.arange(int(n_u)) + 0.5) / float(n_u)
    z = ndtri(u)
    Finv = _mixture_inverse_cdf(u, pi1, m1, s1, m2, s2, n_x=n_x)
    D1 = float(np.sqrt(np.mean((m1 + s1 * z - Finv) ** 2)))
    D2 = float(np.sqrt(np.mean((m2 + s2 * z - Finv) ** 2)))
    return D1, D2


def w2_empirical_mixture_1d(x, pi1, m1, m2, Q1, Q2, n_u=512, n_x=2001):
    """W2 distance between an empirical sample and the two-component Gaussian
    mixture F_pi = pi1 Phi_1 + (1 - pi1) Phi_2, via the quantile
    representation Eq. (quantilew2):

        W2^2 = int_0^1 (F_emp^{-1}(u) - F_pi^{-1}(u))^2 du,

    with the empirical quantile function evaluated as a step function on a
    fine u-grid (exact for any sample size, including a single observation,
    where the formula reduces to E_F[(y - X)^2]). pi1 = 1 gives the
    empirical-vs-Gaussian case. This is the ball-coverage distance of the
    design: hold-out empirical measure against the train nominal.
    """
    v = np.asarray(x, dtype=float).ravel()
    v = np.sort(v[np.isfinite(v)])
    n = v.size
    if n == 0:
        return np.nan
    s1 = float(np.sqrt(max(float(Q1), 0.0)))
    s2 = float(np.sqrt(max(float(Q2), 0.0)))
    pi1 = float(np.clip(pi1, 0.0, 1.0))
    u = (np.arange(int(n_u)) + 0.5) / float(n_u)
    # empirical quantile function F_emp^{-1}(u) = x_(ceil(u n)), 1-indexed
    emp_q = v[np.clip(np.ceil(u * n).astype(int) - 1, 0, n - 1)]
    Finv = _mixture_inverse_cdf(u, pi1, float(m1), s1, float(m2), s2, n_x=n_x)
    return float(np.sqrt(np.mean((emp_q - Finv) ** 2)))


_FLOOR_CACHE = {}


def coverage_floor_quantile(n, q, AF, kind="w2", n_sim=4000, seed=0):
    """Finite-sample floor of the ball-coverage statistic, in units of the
    nominal's annualized standard deviation.

    Even when the hold-out sample is drawn EXACTLY from the nominal, the
    coverage statistic is bounded away from zero by sampling noise: an
    n-point empirical measure sits a positive W2 distance from its own
    population law, and the moment-consistent annualization map
    y = AF*m_h + sqrt(AF)*(r - m_h) amplifies the mean-estimation error by
    AF. For a Gaussian nominal N(m_ann, s_ann^2) the standardized statistic
    under H0 is distribution-free:

        y' = (z - zbar) + sqrt(AF) * zbar,   z ~ N(0, 1)^n,

    kind="w2"       : W2(empirical measure of y', N(0, 1))   (Eq. quantilew2)
    kind="gelbrich" : sqrt(AF * zbar^2 + (sd(z; ddof=1) - 1)^2), the Gelbrich
                      distance between the hold-out moment estimates and the
                      nominal moments (NaN floor for n < 2)

    Returns the q-quantile of the simulated H0 distribution; multiply by the
    record's annualized nominal std. Exact for a Gaussian nominal; for a
    two-component mixture it is applied with the mixture std (the mixture is
    within-regime Gaussian, and on confident dates -- the bulk of the panel
    -- it is a single Gaussian to numerical accuracy). Deterministic in
    (n, q, AF, kind, n_sim, seed); results are cached.
    """
    n = int(n)
    key = (n, float(q), float(AF), str(kind), int(n_sim), int(seed))
    if key in _FLOOR_CACHE:
        return _FLOOR_CACHE[key]
    if n < 1 or (kind == "gelbrich" and n < 2):
        _FLOOR_CACHE[key] = np.nan
        return np.nan
    rng = np.random.default_rng(int(seed))
    z = rng.normal(size=(int(n_sim), n))
    zbar = z.mean(axis=1)
    if kind == "gelbrich":
        sd = z.std(axis=1, ddof=1)
        d = np.sqrt(float(AF) * zbar ** 2 + (sd - 1.0) ** 2)
    elif kind == "w2":
        y = (z - zbar[:, None]) + np.sqrt(float(AF)) * zbar[:, None]
        d = np.array([w2_empirical_mixture_1d(y[i], 1.0, 0.0, 0.0, 1.0, 1.0)
                      for i in range(int(n_sim))])
    else:
        raise ValueError(f"coverage_floor_quantile: unknown kind {kind!r}")
    out = float(np.quantile(d, float(q)))
    _FLOOR_CACHE[key] = out
    return out


def mixture_branch_distances(pi1, m1, m2, Q1, Q2, n_u=2048, n_x=4001):
    """(D_{k_max}, D_{k_min}): the majority- and minority-regime distances
    entering the branch rule Eq. (dregrule), with k_max = argmax_k pi_k.

    Lemma majorityminority identifies these with (min_k D_k, max_k D_k).
    That identification is exact for Q1 = Q2; with unequal regime variances
    and pi close to 1/2 the ordering can invert (the mixture sits nearer the
    wider component), in which case this function still follows the tex's
    rule --- the branch values are the majority/minority distances, not the
    numerical min/max.
    """
    D1, D2 = mixture_w2_distances(pi1, m1, m2, Q1, Q2, n_u=n_u, n_x=n_x)
    return (D1, D2) if float(pi1) >= 0.5 else (D2, D1)


def mixture_minmax_distances(pi1, m1, m2, Q1, Q2, n_u=2048, n_x=4001):
    """(min_k D_k, max_k D_k) of the exact distances --- the values the
    coverage bookkeeping of Sec. regvsstd assigns to the two branches via
    Lemma majorityminority."""
    D1, D2 = mixture_w2_distances(pi1, m1, m2, Q1, Q2, n_u=n_u, n_x=n_x)
    return min(D1, D2), max(D1, D2)


# ---------------------------------------------------------------------------
# Branch rule, threshold calibration, occupancy (Sec. regvsstd)
# ---------------------------------------------------------------------------

def split_alpha(alpha):
    """The design's budget split: alpha_z = alpha_e = alpha / 2."""
    a = float(alpha)
    return a / 2.0, a / 2.0


def calibrate_tau(pi_min_select, alpha_z):
    """Eq. (taucal): the select-span threshold

        tau = sup{ s : (1/T_sel) sum_t pi_min(t) 1{pi_min(t) <= s} <= alpha_z },

    evaluated over attained thresholds (between order statistics the admitted
    set, hence the budget, is unchanged). Returns 0.0 when even the smallest
    pi_min busts the budget (no confident dates), and the maximum of the
    support (<= 1/2) when the whole span fits inside it.
    """
    p = np.asarray(pi_min_select, dtype=float).ravel()
    p = p[np.isfinite(p)]
    if p.size == 0:
        return np.nan
    T = p.size
    order = np.sort(p)
    csum = np.cumsum(order)
    u = np.unique(order)
    # budget used at threshold u_k: sum of all p <= u_k (O(T log T))
    cost = csum[np.searchsorted(order, u, side="right") - 1] / T
    ok = cost <= float(alpha_z) + 1e-15
    if not np.any(ok):
        return 0.0
    return float(u[np.flatnonzero(ok)[-1]])


def branch_component(D_kmax, D_kmin, pi_min, tau):
    """Eq. (dregrule): the regime-uncertainty component at each date,

        D_reg(t) = D_{k_max}(pi_t)   if pi_min(t) <= tau  (confident branch)
                 = D_{k_min}(pi_t)   otherwise            (ambiguous branch),

    with (D_kmax, D_kmin) the majority/minority distances of
    mixture_branch_distances. Vectorized; NaNs in pi_min propagate.
    """
    pm = np.asarray(pi_min, dtype=float)
    Dc = np.asarray(D_kmax, dtype=float)
    Da = np.asarray(D_kmin, dtype=float)
    # 1e-12 tolerance: pi_min values that equal tau analytically can differ
    # in the last float bit when computed via different routes (1 - 0.95 >
    # 0.05 in float); the boundary date belongs on the confident branch
    out = np.where(pm <= float(tau) + 1e-12, Dc, Da)
    return np.where(np.isfinite(pm), out, np.nan)


def realized_miss_frequency(pi_min, tau):
    """LHS of Eq. (missfreq): (1/T) sum_t pi_min(t) 1{pi_min(t) <= tau};
    the guarantee is that this does not exceed alpha_z."""
    p = np.asarray(pi_min, dtype=float).ravel()
    p = p[np.isfinite(p)]
    if p.size == 0:
        return np.nan
    return float(p[p <= float(tau) + 1e-12].sum() / p.size)


def occupancy(pi_min, tau):
    """Eq. (occupancy): confident-branch occupancy
    q = (1/T) sum_t 1{pi_min(t) <= tau}."""
    p = np.asarray(pi_min, dtype=float).ravel()
    p = p[np.isfinite(p)]
    if p.size == 0:
        return np.nan
    return float(np.mean(p <= float(tau) + 1e-12))


def confident_win(tau, d_std, delta_tilde):
    """Eq. (confidentwin): on the confident branch D_reg < D_std whenever
    tau < (max_k d_k / delta_tilde)^2 --- select-span constants only.
    Returns (holds: bool, threshold on tau)."""
    if float(delta_tilde) <= 0.0:
        return False, np.nan
    thr = (float(d_std) / float(delta_tilde)) ** 2
    return bool(float(tau) < thr), float(thr)


def q_cutoff(dkmin_bar, d_std, tau, delta_tilde):
    """Eq. (qcutoff): bar D_reg < D_std whenever

        q > (bar D_{k_min} - max_k d_k) / (bar D_{k_min} - sqrt(tau) delta~*).

    Returns the cutoff (NaN when the denominator is not positive, i.e. the
    span-average bound cannot certify a win at any occupancy).
    """
    num = float(dkmin_bar) - float(d_std)
    den = float(dkmin_bar) - np.sqrt(float(tau)) * float(delta_tilde)
    if den <= 0.0:
        return np.nan
    return num / den


def regime_radius_path(pi1_path, m1, m2, Q1, Q2, tau, e_alpha_e,
                       n_u=2048, n_x=4001):
    """The revised regime-DRO radius path (Sec. regvsstd):

        eps(t) = D_reg(t) + e(alpha_e),

    with D_reg(t) from the branch rule Eq. (dregrule) at threshold `tau`
    (calibrated by calibrate_tau at budget alpha_z on the select span) and
    the per-date distances computed exactly per Eq. (dkexact).

    pi1_path : (T,) TRUE regime-1 posterior path. The mixture of
        Eq. (dkexact) is F_pi = pi1*Phi_1 + (1-pi1)*Phi_2; passing the
        posterior maximum instead mirrors the mixture whenever regime 2 is
        the majority, which is wrong for Q1 != Q2.
    m1, m2, Q1, Q2 : per-regime conditional moments (Assumption 3: scalars)
    e_alpha_e   : estimation component at level alpha_e (alpha/2 split)

    Returns (D_path, eps_path, info) where info carries the per-date
    majority/minority distances (D_kmax, D_kmin), pi_min, occupancy q and
    realized miss frequency.
    """
    p1 = np.clip(np.asarray(pi1_path, dtype=float), 0.0, 1.0)
    pmin = np.where(np.isfinite(p1), np.minimum(p1, 1.0 - p1), np.nan)

    # cache on the rounded TRUE regime-1 weight; mixture_branch_distances
    # selects the majority branch internally
    Dkmax = np.full(p1.shape, np.nan)
    Dkmin = np.full(p1.shape, np.nan)
    cache = {}
    flatp = p1.ravel()
    fc, fa = Dkmax.ravel(), Dkmin.ravel()
    for i, p in enumerate(flatp):
        if not np.isfinite(p):
            continue
        key = round(float(p), 6)
        if key not in cache:
            cache[key] = mixture_branch_distances(key, m1, m2, Q1, Q2,
                                                  n_u=n_u, n_x=n_x)
        fc[i], fa[i] = cache[key]

    D = branch_component(Dkmax, Dkmin, pmin, tau)
    eps = D + float(e_alpha_e)
    info = {"D_kmax": Dkmax, "D_kmin": Dkmin, "pi_min": pmin,
            "q": occupancy(pmin, tau),
            "miss_frequency": realized_miss_frequency(pmin, tau)}
    return D, eps, info


def standard_radius(pool, e_alpha):
    """eps_std = D_std + e_std(alpha_e) from a pooling_cost() dict
    (Eq. dstdmax). Returns (D, eps)."""
    D = float(pool["d_std"])
    return D, D + float(e_alpha)


# ---------------------------------------------------------------------------
# Transfer and the product wedge (Cor. transfer, Cor. productjoint)
# ---------------------------------------------------------------------------

def product_wedge(eps_vec, w):
    """Eq. (productwedge): the nonnegative gap between the product-set and
    joint-ball certificates,

        sum_i eps_i |w_i| - eps_min ||w||_2  >=  0,

    with equality only when w has a single nonzero coordinate."""
    e = np.asarray(eps_vec, dtype=float).ravel()
    w = np.asarray(w, dtype=float).ravel()
    if e.shape != w.shape:
        raise ValueError(f"eps_vec and w must match: {e.shape} vs {w.shape}")
    return float(np.sum(e * np.abs(w)) - np.min(e) * np.linalg.norm(w))


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


# ---------------------------------------------------------------------------
# LEGACY (previous revision) --- superseded by the branch rule above.
# g(pi_max) bracket and pi-dagger crossover: no longer part of the paper's
# theory; retained only so existing pipeline code keeps importing. Do not use
# in new code.
# ---------------------------------------------------------------------------

def g_certainty(pi_max):
    """LEGACY. g(p) = p*sqrt(1-p) + (1-p)*sqrt(p) on [1/2, 1]; the previous
    revision's operational bracket, superseded by mixture_w2_distances /
    branch_component."""
    p = np.clip(np.asarray(pi_max, dtype=float), 0.5, 1.0)
    return p * np.sqrt(1.0 - p) + (1.0 - p) * np.sqrt(p)


def g_inverse(y):
    """LEGACY. Inverse of g on [1/2, 1]. Scalar in, scalar out."""
    y = float(y)
    if y >= _INV_SQRT2:
        return 0.5
    if y <= 0.0:
        return 1.0
    return float(brentq(lambda p: g_certainty(p) - y, 0.5, 1.0 - 1e-15))


def pi_dagger(b_floor):
    """LEGACY. Crossover certainty of the previous revision, superseded by
    calibrate_tau / confident_win."""
    b = float(b_floor)
    if not np.isfinite(b):
        return np.nan
    if b >= _INV_SQRT2:
        return 0.5
    return g_inverse(b)
