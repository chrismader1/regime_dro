# calibration.py - Recalibration of one-step regime probabilities.
#
# Design: on the select span, fit one of {platt, isotonic, temperature} per
# regime model (chosen once, by NLL on select), freeze it, verify with
# reliability diagrams on the confirm span (Assumption 4 discharged).
#
# All functions are target-agnostic: they take (p_raw, y) where p_raw is the
# model's regime-1 probability and y in {0,1} is the calibration target
# supplied by the artifact contract (column `z_target`); the definition of the
# target is an upstream (ssm-export) decision, not made here.

import numpy as np
from scipy.optimize import minimize, minimize_scalar

_EPS = 1e-6


def _clip01(p):
    return np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)


def _logit(p):
    p = _clip01(p)
    return np.log(p / (1.0 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def nll(p, y):
    """Mean negative log-likelihood of Bernoulli targets y under p."""
    p = _clip01(p)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


# ---------------------------------------------------------------------------
# Maps
# ---------------------------------------------------------------------------

class CalibrationMap:
    """Frozen recalibration map. `kind` in {identity, platt, temperature,
    isotonic}; `params` holds fitted parameters."""

    def __init__(self, kind, params):
        self.kind = str(kind)
        self.params = params

    def apply(self, p):
        p = _clip01(p)
        if self.kind == "identity":
            return p
        if self.kind == "platt":
            a, b = self.params
            return _sigmoid(a * _logit(p) + b)
        if self.kind == "temperature":
            (T,) = self.params
            return _sigmoid(_logit(p) / max(float(T), _EPS))
        if self.kind == "isotonic":
            x_knots, y_knots = self.params
            return np.interp(p, x_knots, y_knots)
        raise ValueError(f"Unknown calibration map kind={self.kind!r}")

    def __repr__(self):
        return f"CalibrationMap(kind={self.kind!r}, params={self.params!r})"


def fit_platt(p_raw, y):
    """Platt scaling: sigmoid(a * logit(p) + b), (a, b) by NLL."""
    z = _logit(p_raw)
    y = np.asarray(y, dtype=float)

    def obj(theta):
        a, b = theta
        return nll(_sigmoid(a * z + b), y)

    res = minimize(obj, x0=np.array([1.0, 0.0]), method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 2000})
    a, b = res.x
    return CalibrationMap("platt", (float(a), float(b)))


def fit_temperature(p_raw, y):
    """Temperature scaling: sigmoid(logit(p) / T), T > 0 by NLL."""
    z = _logit(p_raw)
    y = np.asarray(y, dtype=float)

    def obj(logT):
        return nll(_sigmoid(z / np.exp(logT)), y)

    res = minimize_scalar(obj, bounds=(-4.0, 4.0), method="bounded")
    return CalibrationMap("temperature", (float(np.exp(res.x)),))


def _pava(y, w):
    """Pool-adjacent-violators (nondecreasing). Returns fitted block means."""
    y = np.asarray(y, dtype=float).copy()
    w = np.asarray(w, dtype=float).copy()
    n = len(y)
    means = list(y)
    weights = list(w)
    sizes = [1] * n
    i = 0
    while i < len(means) - 1:
        if means[i] <= means[i + 1] + 1e-15:
            i += 1
            continue
        tot_w = weights[i] + weights[i + 1]
        means[i] = (means[i] * weights[i] + means[i + 1] * weights[i + 1]) / tot_w
        weights[i] = tot_w
        sizes[i] += sizes[i + 1]
        del means[i + 1], weights[i + 1], sizes[i + 1]
        if i > 0:
            i -= 1
    out = np.empty(n)
    pos = 0
    for m, s in zip(means, sizes):
        out[pos:pos + s] = m
        pos += s
    return out


def fit_isotonic(p_raw, y):
    """Isotonic regression of y on p (PAVA, in-house; no sklearn dependency)."""
    p = _clip01(p_raw)
    y = np.asarray(y, dtype=float)
    order = np.argsort(p, kind="mergesort")
    p_s, y_s = p[order], y[order]
    # collapse ties in p to single knots
    uniq, inv, counts = np.unique(p_s, return_inverse=True, return_counts=True)
    y_bar = np.bincount(inv, weights=y_s) / counts
    fitted = _pava(y_bar, counts.astype(float))
    x_knots = np.concatenate([[0.0], uniq, [1.0]])
    y_knots = np.concatenate([[fitted[0]], fitted, [fitted[-1]]])
    y_knots = np.clip(y_knots, _EPS, 1.0 - _EPS)
    return CalibrationMap("isotonic", (x_knots, y_knots))


_FITTERS = {"platt": fit_platt, "temperature": fit_temperature,
            "isotonic": fit_isotonic}


def choose_recalibration(p_raw, y, methods=("platt", "isotonic", "temperature")):
    """Fit the candidate maps on the select span and return the NLL-best one,
    compared against identity as well. Returns (CalibrationMap, table dict)."""
    p_raw = _clip01(p_raw)
    y = np.asarray(y, dtype=float)
    table = {"identity": nll(p_raw, y)}
    fitted = {"identity": CalibrationMap("identity", ())}
    for name in methods:
        cmap = _FITTERS[name](p_raw, y)
        fitted[name] = cmap
        table[name] = nll(cmap.apply(p_raw), y)
    best = min(table, key=table.get)
    return fitted[best], table


# ---------------------------------------------------------------------------
# Reliability diagnostics
# ---------------------------------------------------------------------------

def reliability_curve(p, y, n_bins=10):
    """Equal-width reliability curve. Returns dict of arrays:
    bin_center, mean_pred, frac_pos, count (empty bins dropped)."""
    p = _clip01(p)
    y = np.asarray(y, dtype=float)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, int(n_bins) - 1)
    centers, mean_pred, frac_pos, count = [], [], [], []
    for b in range(int(n_bins)):
        m = idx == b
        if not np.any(m):
            continue
        centers.append(0.5 * (edges[b] + edges[b + 1]))
        mean_pred.append(float(np.mean(p[m])))
        frac_pos.append(float(np.mean(y[m])))
        count.append(int(np.sum(m)))
    return {"bin_center": np.asarray(centers), "mean_pred": np.asarray(mean_pred),
            "frac_pos": np.asarray(frac_pos), "count": np.asarray(count)}


def ece(p, y, n_bins=10):
    """Expected calibration error (count-weighted |mean_pred - frac_pos|)."""
    rc = reliability_curve(p, y, n_bins=n_bins)
    if rc["count"].size == 0:
        return np.nan
    w = rc["count"] / rc["count"].sum()
    return float(np.sum(w * np.abs(rc["mean_pred"] - rc["frac_pos"])))


def calibration_pass(p, y, n_bins=10, threshold=0.05):
    """Pass/fail per the falsification paragraph: ECE below `threshold`.
    The threshold is a select-span CONFIG decision; 0.05 is a placeholder
    default, not a recommendation. Returns (bool, ece_value)."""
    e = ece(p, y, n_bins=n_bins)
    return (bool(e < float(threshold)) if np.isfinite(e) else False), e
