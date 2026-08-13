# optimizer.py

import numpy as np
import cvxpy as cp
import warnings

from regime_dro.arrays import asnumpy_strict
from regime_dro.delta import compute_delta

try:
    import cupy as xp
    GPU = True
except Exception:
    import numpy as xp
    GPU = False

warnings.filterwarnings(
    "ignore",
    message=".*ECOS will no longer be installed by default.*",
    category=FutureWarning,
    module="cvxpy.reductions.solvers.solving_chain",
)
warnings.filterwarnings(
    "ignore",
    message=".*Solution may be inaccurate.*",
    category=UserWarning,
    module="cvxpy.problems.problem",
)


def psd_factor_LtL(Sigma, eps):
    """
    Return L (NumPy float64, contiguous) such that Sigma ≈ L.T @ L.
    """
    import numpy as _np

    try:
        import cupy as _cp
        if isinstance(Sigma, _cp.ndarray) or hasattr(Sigma, "__cuda_array_interface__"):
            S = _cp.asnumpy(Sigma)
        elif hasattr(Sigma, "get") and callable(Sigma.get):
            S = Sigma.get()
        else:
            S = _np.asarray(Sigma)
    except Exception:
        S = _np.asarray(Sigma)

    # numpy >= 2: array(copy=False) raises when a copy is unavoidable
    S = _np.asarray(S, dtype=_np.float64, order="C")

    S_sym = 0.5 * (S + S.T)
    try:
        C = _np.linalg.cholesky(S_sym + float(eps) * _np.eye(S_sym.shape[0], dtype=S_sym.dtype))
    except _np.linalg.LinAlgError:
        vals, vecs = _np.linalg.eigh(S_sym)
        vals = _np.clip(vals, float(eps), None)
        S_psd = (vecs * vals) @ vecs.T
        C = _np.linalg.cholesky(S_psd)

    return _np.ascontiguousarray(C.T, dtype=_np.float64)


def solve_optimizer(mu, Sigma, delta, config, verbose=False):

    import numpy as _np

    n   = int(len(mu))
    rho = float(config["risk_budget"])
    eps = float(config["epsilon_sigma"])

    # private copy: never mutate the caller's Sigma, and stay numpy-2 safe
    Sigma_np = _np.array(asnumpy_strict(Sigma, dtype=_np.float64, order="C"),
                         dtype=_np.float64, order="C", copy=True)
    if not _np.all(_np.isfinite(Sigma_np)):
        raise ValueError(
            "Sigma contains non-finite entries -- zeroing them (the old "
            "behavior) makes the affected asset's variance ~0 and the "
            "optimizer concentrates in exactly the asset whose risk model "
            "failed. Fix the covariance input instead.")

    try:
        import cupy as _cp
        if isinstance(Sigma_np, _cp.ndarray) or hasattr(Sigma_np, "__cuda_array_interface__"):
            Sigma_np = _cp.asnumpy(Sigma_np)
    except Exception:
        pass

    assert not hasattr(Sigma_np, "__cuda_array_interface__"), "Sigma_np is still CUDA-backed!"
    assert isinstance(Sigma_np, _np.ndarray), f"Sigma_np type={type(Sigma_np)}"

    L = psd_factor_LtL(Sigma_np, eps)
    mu_np = asnumpy_strict(mu, dtype=_np.float64)

    w = cp.Variable(n)
    t = cp.Variable(nonneg=True)
    objective = cp.Minimize(float(delta) * t - mu_np @ w)

    no_shorting = bool(config.get("no_shorting", False))
    no_leverage = bool(config.get("no_leverage", False))

    constr = [
        cp.norm(L @ w, 2) <= rho,
        cp.norm(w, 2)    <= t,
        t >= 0,
    ]

    if no_shorting:
        constr += [w >= 0]

    if no_leverage:
        constr += [cp.sum(w) <= 1]

    mpos = config.get("max_pos_size", None)
    if mpos is not None:
        mpos = float(mpos)
        if np.isfinite(mpos) and mpos >= 0.0:
            if no_shorting:
                constr += [w <= mpos]
            else:
                constr += [cp.abs(w) <= mpos]

    mc = config.get("max_cash", None)
    if mc is not None and _np.isfinite(float(mc)):
        mc = float(mc)
        mc = max(0.0, min(1.0, mc))
        constr += [cp.sum(w) >= 1.0 - mc]

    prob = cp.Problem(objective, constr)
    try:
        if verbose:
            print(f"[solve_optimizer] delta = {float(delta):.6g}, rho = {rho:.6g}")
        prob.solve(solver=cp.MOSEK, verbose=False)
    except Exception:
        prob.solve(solver=cp.ECOS, verbose=False)

    if (w.value is None) or (prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)):
        raise RuntimeError(f"ECOS/MOSEK failed: status={prob.status}")

    return xp.asarray(_np.asarray(w.value).reshape(-1))


def solve_optimizer_l1(mu, Sigma, eps_vec, config, verbose=False,
                       solver_order=("MOSEK", "ECOS")):
    """Per-asset-ball DRO (Cor. innervector): maximize
        w' mu - sum_i eps_i |w_i|
    subject to the same constraint block as solve_optimizer
    (||L w||_2 <= rho, plus no_shorting / no_leverage / max_pos_size /
    max_cash). `eps_vec` is the per-stock radius vector (eps_{i,t});
    entries must be finite and nonnegative.
    """
    import numpy as _np

    n   = int(len(mu))
    rho = float(config["risk_budget"])
    eps = float(config["epsilon_sigma"])

    eps_np = asnumpy_strict(eps_vec, dtype=_np.float64).reshape(-1)
    if eps_np.shape[0] != n:
        raise ValueError(f"eps_vec length {eps_np.shape[0]} != len(mu) {n}")
    if not _np.all(_np.isfinite(eps_np)) or _np.any(eps_np < 0.0):
        raise ValueError("eps_vec must be finite and nonnegative.")

    # private copy: never mutate the caller's Sigma, and stay numpy-2 safe
    Sigma_np = _np.array(asnumpy_strict(Sigma, dtype=_np.float64, order="C"),
                         dtype=_np.float64, order="C", copy=True)
    if not _np.all(_np.isfinite(Sigma_np)):
        raise ValueError(
            "Sigma contains non-finite entries -- zeroing them (the old "
            "behavior) makes the affected asset's variance ~0 and the "
            "optimizer concentrates in exactly the asset whose risk model "
            "failed. Fix the covariance input instead.")

    L = psd_factor_LtL(Sigma_np, eps)
    mu_np = asnumpy_strict(mu, dtype=_np.float64)

    w = cp.Variable(n)
    objective = cp.Minimize(eps_np @ cp.abs(w) - mu_np @ w)

    no_shorting = bool(config.get("no_shorting", False))
    no_leverage = bool(config.get("no_leverage", False))

    constr = [cp.norm(L @ w, 2) <= rho]

    if no_shorting:
        constr += [w >= 0]

    if no_leverage:
        constr += [cp.sum(w) <= 1]

    mpos = config.get("max_pos_size", None)
    if mpos is not None:
        mpos = float(mpos)
        if np.isfinite(mpos) and mpos >= 0.0:
            if no_shorting:
                constr += [w <= mpos]
            else:
                constr += [cp.abs(w) <= mpos]

    mc = config.get("max_cash", None)
    if mc is not None and _np.isfinite(float(mc)):
        mc = float(mc)
        mc = max(0.0, min(1.0, mc))
        constr += [cp.sum(w) >= 1.0 - mc]

    prob = cp.Problem(objective, constr)
    if verbose:
        print(f"[solve_optimizer_l1] eps in [{eps_np.min():.6g}, "
              f"{eps_np.max():.6g}], rho = {rho:.6g}")
    last_err = None
    for s in solver_order:
        try:
            prob.solve(solver=getattr(cp, s), verbose=False)
            if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                break
        except Exception as ex:
            last_err = ex
    if (w.value is None) or (prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)):
        raise RuntimeError(f"solve_optimizer_l1 failed: status={prob.status} "
                           f"(last error: {last_err})")

    return xp.asarray(_np.asarray(w.value).reshape(-1))


def solve_dro(mu, Sigma, params, G, R=None, *, verbose=None):
    """
    DRO with δ computed from `params`.
    """
    if verbose is None:
        verbose = bool(params.get("verbose", False))
    delta = compute_delta(params.get("kappa", 1.0), mu, Sigma, R=R, params=params)
    w = solve_optimizer(mu, Sigma, delta, config=G, verbose=verbose)
    return w, float(delta)
