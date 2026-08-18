# calibration.py - Recalibration of one-step regime probabilities.
#
# Design (Calibration paragraph, revised): the regime label is latent, so
# calibration is assessed through the object the posterior actually
# determines -- the one-step predictive distribution of the return. The
# probability integral transform (PIT) of the realized next return under
# the posterior-weighted mixture,
#
#     u_{i,t} = G_{i,t}(y_{i,t+1}),
#     G_{i,t}(y) = sum_k pi_{i,k}(t) Phi((y - m_{i,k}) / sqrt(Q_{i,k})),
#
# is uniform on (0,1) when the posterior weights are correct (Eq. pit);
# because the per-regime means and variances enter G unchanged, departures
# from uniformity are attributable to the weights. Recalibration applies a
# single temperature theta > 0 to the posterior,
# pi^theta_k propto pi_k^(1/theta) (theta > 1 flattens, theta < 1
# sharpens), one theta per regime model, pooled across the panel on the
# select span, fitted by minimising the Cramer-von Mises distance of {u}
# to the uniform, then frozen and applied to every posterior thereafter,
# including the placebo models'. Verified out of sample on the confirm
# span: PIT histogram, Kolmogorov-Smirnov distance with block-bootstrap
# bands, and the fitted theta. Temperature is used in preference to Platt
# or isotonic maps because it acts on the posterior alone and cannot
# compensate for a misspecified regime mean or variance by distorting the
# other components of G.
#
# LEGACY (superseded by the PIT layer; retained for synthetic runs with a
# z_target column and for older pickles): the target-based
# {platt, isotonic, temperature-on-logits} maps chosen by NLL, and the
# reliability-diagram diagnostics (reliability_curve, ece,
# calibration_pass). These take (p_raw, y) with y in {0,1} from the
# artifact contract's optional `z_target`.

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import ndtr

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
        if self.kind == "pit_temperature":
            (T,) = self.params
            return temperature_posterior(p, T)
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


# ---------------------------------------------------------------------------
# PIT layer (design, Calibration paragraph -- Eq. pit)
# ---------------------------------------------------------------------------

def temperature_posterior(p, theta):
    """Temperature-scaled two-regime posterior, Eq. of the Calibration
    paragraph: pi^theta_k propto pi_k^(1/theta). theta > 1 flattens the
    posterior toward (1/2, 1/2); theta < 1 sharpens it; theta = 1 is the
    identity. Computed in log space for stability; p in {0, 1} maps to
    itself."""
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    th = max(float(theta), _EPS)
    with np.errstate(divide="ignore"):
        a = np.log(p) / th
        b = np.log1p(-p) / th
    # softmax over the two log-numerators
    m = np.maximum(a, b)
    ea, eb = np.exp(a - m), np.exp(b - m)
    return ea / (ea + eb)


def pit_values(p1, m1, m2, Q1, Q2, y_next):
    """PIT of the realized next return under the posterior-weighted mixture
    (Eq. pit): u = p1 Phi((y - m1)/s1) + (1 - p1) Phi((y - m2)/s2), all
    inputs in the same (daily) units, elementwise over aligned arrays.
    Rows with any non-finite input return NaN."""
    p = np.asarray(p1, dtype=float)
    y = np.asarray(y_next, dtype=float)
    s1 = np.sqrt(np.maximum(np.asarray(Q1, dtype=float), 0.0))
    s2 = np.sqrt(np.maximum(np.asarray(Q2, dtype=float), 0.0))
    m1 = np.asarray(m1, dtype=float)
    m2 = np.asarray(m2, dtype=float)
    p, y, m1, m2, s1, s2 = np.broadcast_arrays(p, y, m1, m2, s1, s2)
    ok = (np.isfinite(p) & np.isfinite(y) & np.isfinite(m1) & np.isfinite(m2)
          & (s1 > 0.0) & (s2 > 0.0))
    u = np.full(p.shape, np.nan)
    u[ok] = (p[ok] * ndtr((y[ok] - m1[ok]) / s1[ok])
             + (1.0 - p[ok]) * ndtr((y[ok] - m2[ok]) / s2[ok]))
    return u


def cvm_uniform(u):
    """Cramer-von Mises statistic of a sample against Uniform(0,1):
    W^2 = 1/(12n) + sum_i ((2i-1)/(2n) - u_(i))^2. NaN if empty."""
    u = np.sort(np.asarray(u, dtype=float).ravel())
    u = u[np.isfinite(u)]
    n = u.size
    if n == 0:
        return np.nan
    i = np.arange(1, n + 1)
    return float(1.0 / (12.0 * n) + np.sum(((2.0 * i - 1.0) / (2.0 * n) - u) ** 2))


def ks_uniform(u):
    """Kolmogorov-Smirnov distance of a sample from Uniform(0,1):
    sup_x |F_emp(x) - x|. NaN if empty."""
    u = np.sort(np.asarray(u, dtype=float).ravel())
    u = u[np.isfinite(u)]
    n = u.size
    if n == 0:
        return np.nan
    i = np.arange(1, n + 1)
    return float(max(np.max(i / n - u), np.max(u - (i - 1) / n)))


def fit_temperature_pit(p1, m1, m2, Q1, Q2, y_next, log_bounds=(-4.0, 4.0)):
    """Fit the single posterior temperature of the Calibration paragraph:
    minimise the Cramer-von Mises distance between the PIT sample
    {G^theta(y_next)} and the uniform, over log theta in `log_bounds`.
    The per-regime means and variances enter G unchanged; only the weights
    are tempered. Returns (CalibrationMap('pit_temperature', (theta,)),
    dict(theta, cvm, cvm_raw, n)).

    The fit clips exactly as CalibrationMap.apply does. Without the clip the
    temperature would be optimised against a map that is never used: apply()
    clips to [_EPS, 1-_EPS] BEFORE tempering, so raw posteriors below _EPS are
    floored there and then tempered, and the fitted theta would be tuned to a
    tail the applied map cannot reach."""
    p = _clip01(p1)

    def obj(logT):
        return cvm_uniform(pit_values(temperature_posterior(p, np.exp(logT)),
                                      m1, m2, Q1, Q2, y_next))

    cvm_raw = cvm_uniform(pit_values(p, m1, m2, Q1, Q2, y_next))
    if not np.isfinite(cvm_raw):
        return (CalibrationMap("identity", ()),
                {"theta": np.nan, "cvm": np.nan, "cvm_raw": np.nan, "n": 0})
    res = minimize_scalar(obj, bounds=tuple(log_bounds), method="bounded",
                          options={"xatol": 1e-5})
    theta = float(np.exp(res.x))
    u = pit_values(p, m1, m2, Q1, Q2, y_next)
    return (CalibrationMap("pit_temperature", (theta,)),
            {"theta": theta, "cvm": float(res.fun), "cvm_raw": cvm_raw,
             "n": int(np.isfinite(u).sum())})


# ---------------------------------------------------------------------------
# LEGACY target-based maps (z_target runs and older pickles only)
# ---------------------------------------------------------------------------

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
