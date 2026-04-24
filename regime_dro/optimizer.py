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

    S = _np.array(S, dtype=_np.float64, order="C", copy=False)

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

    Sigma_np = asnumpy_strict(Sigma, dtype=_np.float64, order="C")
    _np.nan_to_num(Sigma_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        import cupy as _cp
        if isinstance(Sigma_np, _cp.ndarray) or hasattr(Sigma_np, "__cuda_array_interface__"):
            Sigma_np = _cp.asnumpy(Sigma_np)
    except Exception:
        pass

    Sigma_np = _np.array(Sigma_np, dtype=_np.float64, order="C", copy=False)

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


def solve_dro(mu, Sigma, params, G, R=None, *, verbose=None):
    """
    DRO with δ computed from `params`.
    """
    if verbose is None:
        verbose = bool(params.get("verbose", False))
    delta = compute_delta(params.get("kappa", 1.0), mu, Sigma, R=R, params=params)
    w = solve_optimizer(mu, Sigma, delta, config=G, verbose=verbose)
    return w, float(delta)
