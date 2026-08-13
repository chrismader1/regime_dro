# delta.py

import numpy as np

try:
    import cupy as xp
    GPU = True
except Exception:
    import numpy as xp
    GPU = False


def _rng_from_params(params):
    seed = None if params is None else params.get("seed", None)
    return np.random.default_rng(seed)


def _sqrtm_psd(A, eps=1e-12):
    """Symmetric PSD square root.

    `eps` clamps negative eigenvalues arising from round-off; it is NOT added
    to the positive ones. Adding it (the previous behaviour) inflates every
    eigenvalue by eps/(2*sqrt(lambda)), which at daily-return scale
    (variances ~1e-4) exceeds the Gelbrich trace term itself and drove
    wasserstein2_gaussian to return exactly 0. Clamping preserves the same
    protection against indefiniteness without the bias.
    """
    vals, vecs = xp.linalg.eigh(0.5 * (A + A.T))
    vals = xp.clip(vals, float(eps), None) if eps else xp.clip(vals, 0.0, None)
    return (vecs * xp.sqrt(vals)) @ vecs.T


def wasserstein2_gaussian(mu1, Sigma1, mu2, Sigma2, eps=1e-12):
    """
    Gelbrich formula: W2^2(N(mu1,S1), N(mu2,S2)) =
      ||mu1-mu2||^2 + tr(S1 + S2 - 2 (S2^{1/2} S1 S2^{1/2})^{1/2})
    """
    Sigma1 = xp.atleast_2d(xp.asarray(Sigma1, dtype=float))
    Sigma2 = xp.atleast_2d(xp.asarray(Sigma2, dtype=float))
    dmu2 = float(xp.dot(mu1 - mu2, mu1 - mu2))

    # Scalar case: the matrix route is exact but loses all precision to
    # round-off when variances are ~1e-4 (daily returns), so use the closed
    # form W2^2 = (m1-m2)^2 + (s1-s2)^2 directly.
    if Sigma1.shape == (1, 1) and Sigma2.shape == (1, 1):
        s1 = float(xp.sqrt(max(float(Sigma1[0, 0]), 0.0)))
        s2 = float(xp.sqrt(max(float(Sigma2[0, 0]), 0.0)))
        return float(xp.sqrt(max(dmu2 + (s1 - s2) ** 2, 0.0)))

    S2h = _sqrtm_psd(Sigma2, eps=float(eps))
    mid = S2h @ Sigma1 @ S2h
    midh = _sqrtm_psd(mid, eps=float(eps))
    trpart = float(xp.trace(Sigma1 + Sigma2 - 2.0 * midh))
    w2_sq = max(dmu2 + trpart, 0.0)
    return float(xp.sqrt(w2_sq))


def sliced_w2_empirical(X, Y, n_proj=256, rng=None, U=None):
    """
    1D sliced W2 between empirical measures using random projections.
    """
    X = xp.asarray(X, dtype=xp.float32)
    Y = xp.asarray(Y, dtype=xp.float32)
    n, d = X.shape
    m = Y.shape[0]

    if U is None:
        if (rng is None) and hasattr(xp.random, "standard_normal"):
            try:
                U = xp.random.standard_normal((int(n_proj), d), dtype=X.dtype)
            except TypeError:
                U = xp.random.standard_normal((int(n_proj), d)).astype(X.dtype, copy=False)
        else:
            rng = _rng_from_params({}) if rng is None else rng
            U = xp.asarray(rng.normal(size=(int(n_proj), d)), dtype=X.dtype)
        U = U / xp.maximum(xp.linalg.norm(U, axis=1, keepdims=True), 1e-12)
    else:
        U = xp.asarray(U, dtype=X.dtype)

    XU = X @ U.T
    YU = Y @ U.T

    if m == n:
        XU = xp.sort(XU, axis=0)
        YU = xp.sort(YU, axis=0)
        diff = XU - YU
        w2_sq = xp.mean(diff * diff)
        return float(xp.sqrt(xp.maximum(w2_sq, 0.0)))
    else:
        k = int(min(n, m))
        if k <= 1:
            XU = xp.mean(XU, axis=0, keepdims=True)
            YU = xp.mean(YU, axis=0, keepdims=True)
        else:
            q = (xp.arange(1, k + 1, dtype=XU.dtype) - 0.5) / k
            XU = xp.quantile(XU, q, axis=0)
            YU = xp.quantile(YU, q, axis=0)
        diff = XU - YU
        w2_sq = xp.mean(diff * diff)
        return float(xp.sqrt(xp.maximum(w2_sq, 0.0)))


def w2_empirical_gaussian_1d(x, m=None, Q=None):
    """Gaussianity misspecification slack (design, Gaussianity paragraph):
    the one-dimensional W2 distance between the empirical distribution of a
    sample and the fitted (or supplied) Gaussian, via the quantile
    representation Eq. (quantilew2),

        W2^2 = int_0^1 (F_emp^{-1}(u) - F_N^{-1}(u))^2 du,

    evaluated at the plotting positions u_i = (i - 1/2)/n, where the
    empirical quantile function is the order statistic x_(i). When m / Q are
    omitted they are fitted to the sample (mean, ddof-1 variance).
    """
    from scipy.special import ndtri
    v = np.asarray(x, dtype=float).ravel()
    v = v[np.isfinite(v)]
    n = v.size
    if n < 2:
        return np.nan
    if m is None:
        m = float(np.mean(v))
    if Q is None:
        Q = float(np.var(v, ddof=1))
    s = float(np.sqrt(max(float(Q), 0.0)))
    u = (np.arange(n) + 0.5) / n
    gq = float(m) + s * ndtri(u)
    return float(np.sqrt(np.mean((np.sort(v) - gq) ** 2)))


def _mbb_indices(T: int, m: int, L: int, rng=None) -> np.ndarray:
    """
    Moving-block bootstrap: draw start positions U(0..T-1), take L-length
    circular blocks until we have m indices. Returns shape (m,).
    """
    rng = np.random.default_rng(None) if rng is None else rng
    L = int(max(1, L)); T = int(T); m = int(m)
    idx = np.empty(m, dtype=np.int64)
    filled = 0
    while filled < m:
        s = int(rng.integers(0, T))
        block = (s + np.arange(L)) % T
        k = min(L, m - filled)
        idx[filled:filled + k] = block[:k]
        filled += k
    return idx


def bootstrap_np_block_delta(
    R, n_proj=512, B=100, block_len=55, alpha=0.05, seed=None, n_sample=512,
    standardize=False, U=None):
    """
    Moving-block bootstrap of the empirical daily panel.
    Return the (1-alpha) quantile (daily).
    """
    R_xp = xp.asarray(R, dtype=xp.float32)
    T = int(R_xp.shape[0])
    n = int(n_sample)

    scale = 1.0

    if standardize:
        finite = xp.isfinite(R_xp)
        n_j = xp.maximum(finite.sum(axis=0, dtype=xp.float32), 1.0)
        sum_j = xp.where(finite, R_xp, 0.0).sum(axis=0)
        mu_j  = sum_j / n_j
        Xc    = xp.where(finite, R_xp - mu_j[None, :], 0.0)
        ss    = (Xc * Xc).sum(axis=0)
        std_j = xp.sqrt(xp.maximum(ss / xp.maximum(n_j - 1.0, 1.0), 0.0))
        std_j = xp.where(std_j < 1e-12, 1.0, std_j)

        scale = float(xp.sqrt(xp.mean(std_j * std_j)))
        pool = xp.where(finite, Xc / std_j[None, :], 0.0)
    else:
        pool = R_xp

    rng = np.random.default_rng(seed)
    dists = xp.empty(int(B), dtype=float)
    for b in range(int(B)):
        i1 = _mbb_indices(T, n, int(block_len), rng=rng)
        i2 = _mbb_indices(T, n, int(block_len), rng=rng)
        X1 = pool[xp.asarray(i1, dtype=xp.int64)]
        X2 = pool[xp.asarray(i2, dtype=xp.int64)]
        dists[b] = sliced_w2_empirical(X1, X2, n_proj=int(n_proj), rng=rng, U=U)

    return float(scale * xp.quantile(dists, 1.0 - float(alpha)))


def bootstrap_gaussian_block_delta(
    R, alpha=0.05, B=100, block_len=55, eps=1e-9, seed=None, n_sample=512,
    standardize=False, ann=1.0):
    """
    Moving-block bootstrap; distance is Gelbrich W2 between the Gaussian fitted to
    the reference pool and the Gaussian fitted to each block-resample.

    ann : annualization factor, applied MOMENT-CONSISTENTLY inside each
    resample distance -- the mean difference scales by ann, the covariance
    difference by sqrt(ann) (i.e. covariance x ann) -- matching the
    annualized Gelbrich distance between annualized moment estimates.
    ann=1.0 keeps daily units.
    """
    X = xp.asarray(R, dtype=float)
    T, d = int(X.shape[0]), int(X.shape[1])
    if T < 2:
        return 0.0

    scale = 1.0

    if bool(standardize):
        finite = xp.isfinite(X)
        n_j = xp.maximum(finite.sum(axis=0, dtype=xp.float32), 1.0)
        sum_j = xp.where(finite, X, 0.0).sum(axis=0)
        mu_j  = sum_j / n_j
        Xc    = xp.where(finite, X - mu_j[None, :], 0.0)
        ss    = (Xc * Xc).sum(axis=0)
        std_j = xp.sqrt(xp.maximum(ss / xp.maximum(n_j - 1.0, 1.0), 0.0))
        std_j = xp.where(std_j < 1e-12, 1.0, std_j)

        scale = float(xp.sqrt(xp.mean(std_j * std_j)))

        X = xp.where(finite, Xc / std_j[None, :], 0.0)

    mu0 = xp.mean(X, axis=0)
    Xc  = X - mu0
    S0  = (Xc.T @ Xc) / max(T - 1, 1)

    n = int(max(2, n_sample))
    rng = np.random.default_rng(seed)

    _a = float(ann)
    deltas = xp.empty(int(B), dtype=float)
    for b in range(int(B)):
        idx = _mbb_indices(T, n, int(block_len), rng=rng)
        Xb  = X[xp.asarray(idx, dtype=xp.int64)]
        mub = xp.mean(Xb, axis=0)
        Xbc = Xb - mub
        Sb  = (Xbc.T @ Xbc) / max(n - 1, 1)
        # annualized moments: mean x ann, covariance x ann (std x sqrt(ann))
        deltas[b] = wasserstein2_gaussian(_a * mu0, _a * S0,
                                          _a * mub, _a * Sb, float(eps))

    return float(scale * xp.quantile(deltas, 1.0 - float(alpha)))


def compute_delta(kappa, mu_est, Sigma=None, R=None, params=None):
    """
    Selectable Wasserstein radius δ.
    Methods (set via params['delta_method']):
      - 'kappa_l2'           : δ = κ‖μ‖₂
      - 'kappa_rate'         : δ = κ · σ̄ · sqrt(d/n)
      - 'fixed'              : δ = params['delta']
      - 'bound_ek'           : Esfahani–Kuhn bound δ_n(α)
      - 'bootstrap_np'       : Nonparametric bootstrap quantile
      - 'bootstrap_gaussian' : Parametric Gaussian bootstrap using Gelbrich W2
    """

    if not isinstance(params, dict) or "delta_method" not in params:
        raise ValueError("delta_method must be provided (no default).")
    method = params["delta_method"]

    AF = int((params or {}).get("annualization_factor", 252))

    if method in ("bootstrap_np", "bootstrap_gaussian") and "B" not in params:
        raise ValueError("Bootstrap delta requires 'B' in params (match legacy value).")

    kappa = float(kappa)

    if method == "kappa_rate":
        if Sigma is None:
            raise ValueError("kappa_rate requires Sigma.")
        d     = int(xp.size(mu_est))
        n_obs = int(R.shape[0]) if (R is not None and hasattr(R, "shape")) else 1
        n_eff = int((params or {}).get("n_ref", n_obs))
        sbar  = float(xp.sqrt(xp.trace(Sigma) / max(d, 1)))
        return kappa * sbar * xp.sqrt(d / max(n_eff, 1))

    if method == "fixed":
        return float((params or {}).get("delta", 0.0))

    if method == "kappa_l2":
        return kappa * float(xp.linalg.norm(mu_est, 2))

    if method == "bound_ek":
        alpha = float((params or {}).get("alpha", 0.05))
        c1    = float((params or {}).get("c1", 3.0))
        C     = float((params or {}).get("c2", 1.0))
        a     = float((params or {}).get("a", 2.0))
        n_obs = int(R.shape[0]) if (R is not None and hasattr(R, "shape")) else 1
        n     = int((params or {}).get("n_ref", n_obs))
        d     = int(xp.size(mu_est))
        num   = xp.log(c1 / max(alpha, 1e-12))
        base  = (C * num) / max(n, 1)
        n0    = float((params or {}).get("n0", 100.0))
        expo  = (1.0 / max(d, 2)) if (n >= n0) else (1.0 / max(a, 1e-12))
        return float(max(base, 1e-12) ** expo)

    if method == "bootstrap_np":
        alpha       = float((params or {}).get("alpha", 0.05))
        B           = int((params or {}).get("B", 256))
        n_proj      = int((params or {}).get("n_proj", 128))
        seed        = (params or {}).get("seed", None)
        L           = int((params or {}).get("block_len", 10))
        n_sample    = int((params or {}).get("n_sample", 252))
        standardize = bool((params or {}).get("standardize", True))
        # NOTE: the np route annualizes the whole daily radius by sqrt(AF)
        # (no closed-form moment split exists for the projected empirical
        # W2). It is therefore NOT on the same scale as the gaussian route
        # and is not used by the production configuration.
        delta_daily = bootstrap_np_block_delta(
            R, n_proj=n_proj, B=B, block_len=L, alpha=alpha, seed=seed,
            n_sample=n_sample, standardize=standardize, U=params.get("U", None))
        return float(np.sqrt(AF)) * float(delta_daily)

    if method == "bootstrap_gaussian":
        assert R is not None, "bootstrap_gaussian needs raw sample matrix R."
        alpha       = float((params or {}).get("alpha", 0.05))
        B           = int((params or {}).get("B", 512))
        eps         = float((params or {}).get("epsilon_sigma", 1e-9))
        seed        = (params or {}).get("seed", None)
        L           = int((params or {}).get("block_len", 10))
        n_sample    = int((params or {}).get("n_sample", 252))
        standardize = bool((params or {}).get("standardize", True))
        # moment-consistent annualization: mean error scales by AF, sigma
        # error by sqrt(AF). The d = 1 case is exact via the scalar route;
        # for d > 1 the same map is applied to the Gelbrich components
        # inside the bootstrap (see bootstrap_gaussian_block_delta ann arg).
        delta_ann = bootstrap_gaussian_block_delta(
            R, alpha=alpha, B=B, block_len=L, eps=eps, seed=seed,
            n_sample=n_sample, standardize=standardize, ann=float(AF))
        return float(delta_ann)

    raise ValueError(f"Unknown delta_method='{method}'")
