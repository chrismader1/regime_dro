# hypothesis.py

import numpy as np
import pandas as pd

from scipy import stats as sp_stats

try:
    import cupy as xp
    GPU = True
except Exception:
    import numpy as xp
    GPU = False


# =============================================================================
# Paired difference helpers (existing)
# =============================================================================

def _paired_diff(x, y):
    # numpy explicitly: scipy.stats rejects cupy arrays, and these tests
    # are CPU-side regardless of whether the fit path used a GPU
    from regime_dro.arrays import asnumpy_strict
    d = asnumpy_strict(x, float) - asnumpy_strict(y, float)
    mask = np.isfinite(d)
    d = d[mask]
    return d, int(d.size)


def paired_onesided_less(x, y):
    """One-sided paired t-test: H1: mean(x - y) < 0."""
    d, n = _paired_diff(x, y)
    if n < 2:
        return float("nan"), float("nan")
    t, p = sp_stats.ttest_1samp(d, popmean=0.0, alternative="less")
    return t, p


def superiority_paired(x, y):
    """One-sided paired t-test: H1: mean(x - y) > 0."""
    d, n = _paired_diff(x, y)
    if n < 2:
        return float("nan"), float("nan")
    t, p = sp_stats.ttest_1samp(d, popmean=0.0, alternative="greater")
    return t, p


def paired_two_sided_test_with_ci(x, y, alpha=0.05):
    """Two-sided paired t-test with confidence interval on mean difference."""
    d, n = _paired_diff(x, y)
    mean_diff = float(xp.mean(d)) if n else float("nan")
    if n < 2:
        return dict(mean_diff=mean_diff, t=float("nan"), p=float("nan"),
                    ci_low=float("nan"), ci_high=float("nan"), n=n)
    sd = float(xp.std(d, ddof=1))
    se = sd / float(xp.sqrt(n))
    t, p = sp_stats.ttest_1samp(d, popmean=0.0, alternative="two-sided")
    tcrit = float(sp_stats.t.ppf(1 - alpha / 2, df=n - 1))
    ci_low  = mean_diff - tcrit * se
    ci_high = mean_diff + tcrit * se
    return dict(mean_diff=mean_diff, t=t, p=p, ci_low=ci_low, ci_high=ci_high, n=n)


# =============================================================================
# Wilcoxon signed-rank (paired non-parametric)
# =============================================================================

def wilcoxon_paired(x, y, alternative="two-sided"):
    """Wilcoxon signed-rank test on paired observations.

    Parameters
    ----------
    x, y : array-like, same length
        Paired observations.
    alternative : {"two-sided", "greater", "less"}
        "greater" tests H1: median(x - y) > 0.

    Returns
    -------
    dict with keys statistic, p, n_pairs, median_diff.
    """
    d, n = _paired_diff(x, y)
    if n < 2:
        return dict(statistic=float("nan"), p=float("nan"), n_pairs=n,
                    median_diff=float("nan"))
    # Drop zero differences (Wilcoxon convention; "wilcox" zero-method).
    d_np = np.asarray(d, dtype=float)
    nonzero = d_np[d_np != 0.0]
    if nonzero.size < 1:
        return dict(statistic=float("nan"), p=float("nan"), n_pairs=int(n),
                    median_diff=0.0)
    res = sp_stats.wilcoxon(nonzero, alternative=alternative, zero_method="wilcox")
    return dict(
        statistic=float(res.statistic),
        p=float(res.pvalue),
        n_pairs=int(nonzero.size),
        median_diff=float(np.median(nonzero)),
    )


# =============================================================================
# Sharpe-ratio difference test
# =============================================================================
# Memmel (2003), Finance Letters 1, 21-23. "Performance Hypothesis Testing with
# the Sharpe Ratio." Closed-form correction to Jobson & Korkie (1981).
#
# Given two return streams r_x, r_y over the SAME time horizon, the Memmel test
# statistic for H0: SR_x = SR_y is:
#
#   z = (SR_x - SR_y) / sqrt(theta / T)
#
# with theta = 2 - 2*rho_xy + 0.5 * (SR_x^2 + SR_y^2 - 2 * SR_x * SR_y * rho_xy^2)
# and SR_i computed on the SAME period (not annualised in the formula; the test
# is invariant to the units of SR as long as the same units are used for both).
#
# Asymptotically N(0, 1) under H0.

def _sharpe_per_period(x, rf_per_period=0.0):
    """Per-period Sharpe ratio (no annualisation factor)."""
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    sd = float(np.std(x, ddof=1))
    if sd == 0.0:
        return float("nan")
    return float((np.mean(x) - float(rf_per_period)) / sd)


def memmel_sharpe_diff_test(x, y, AF=None, rf_per_period=0.0):
    """Memmel (2003) paired Sharpe-ratio difference test.

    Parameters
    ----------
    x, y : array-like, same length
        Two paired return series (e.g., two portfolios over identical dates).
    AF : int or None
        If given, the reported Sharpe ratios are scaled by sqrt(AF). The
        z-statistic and p-value are invariant to this scaling.
    rf_per_period : float
        Risk-free rate per period in the same frequency as the returns.

    Returns
    -------
    dict with keys:
        sharpe_x, sharpe_y      — annualised if AF given, else per-period
        diff                    — sharpe_x - sharpe_y
        z                       — Memmel z-statistic
        p_two_sided
        p_x_greater             — one-sided p for H1: SR_x > SR_y
        T                       — number of paired observations used
    """
    xa = np.asarray(x, dtype=float).ravel()
    ya = np.asarray(y, dtype=float).ravel()
    n = min(xa.size, ya.size)
    xa = xa[:n]; ya = ya[:n]
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa = xa[mask]; ya = ya[mask]
    T = int(xa.size)
    if T < 3:
        return dict(sharpe_x=float("nan"), sharpe_y=float("nan"),
                    diff=float("nan"), z=float("nan"),
                    p_two_sided=float("nan"), p_x_greater=float("nan"), T=T)

    SR_x = _sharpe_per_period(xa, rf_per_period=rf_per_period)
    SR_y = _sharpe_per_period(ya, rf_per_period=rf_per_period)
    if not (np.isfinite(SR_x) and np.isfinite(SR_y)):
        return dict(sharpe_x=SR_x, sharpe_y=SR_y, diff=float("nan"),
                    z=float("nan"), p_two_sided=float("nan"),
                    p_x_greater=float("nan"), T=T)

    rho = float(np.corrcoef(xa, ya)[0, 1])
    theta = 2.0 - 2.0 * rho + 0.5 * (SR_x ** 2 + SR_y ** 2 - 2.0 * SR_x * SR_y * rho ** 2)
    if theta <= 0.0:
        return dict(sharpe_x=SR_x, sharpe_y=SR_y, diff=SR_x - SR_y,
                    z=float("nan"), p_two_sided=float("nan"),
                    p_x_greater=float("nan"), T=T)
    se = float(np.sqrt(theta / T))
    z = float((SR_x - SR_y) / se)
    p_two = float(2.0 * (1.0 - sp_stats.norm.cdf(abs(z))))
    p_grt = float(1.0 - sp_stats.norm.cdf(z))

    if AF is not None:
        scale = float(np.sqrt(float(AF)))
        SR_x_out = SR_x * scale
        SR_y_out = SR_y * scale
    else:
        SR_x_out = SR_x
        SR_y_out = SR_y

    return dict(
        sharpe_x=SR_x_out,
        sharpe_y=SR_y_out,
        diff=SR_x_out - SR_y_out,
        z=z,
        p_two_sided=p_two,
        p_x_greater=p_grt,
        T=T,
    )


# =============================================================================
# Stationary block bootstrap (Politis & Romano 1994)
# =============================================================================
# Politis & Romano (1994), JASA 89(428), 1303-1313. The block lengths are
# geometric with mean 1/p, generating overlapping wrap-around blocks. The
# resulting bootstrap sample is stationary under stationary inputs.

def politis_white_block_length(series):
    """Optimal mean block length L* for the stationary bootstrap.

    Politis & White (2004), "Automatic Block-Length Selection for the Dependent
    Bootstrap." Econometric Reviews 23, 53-70. We use the closed-form
    approximation that requires only the autocorrelation function.

    Returns an integer >= 1.
    """
    x = np.asarray(series, dtype=float).ravel()
    x = x[np.isfinite(x)]
    n = x.size
    if n < 8:
        return 1
    x = x - np.mean(x)
    # Autocovariance via FFT.
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(x, n=nfft)
    acov = np.fft.irfft(f * np.conj(f), n=nfft)[:n].real / n
    if acov[0] <= 0.0:
        return 1
    acorr = acov / acov[0]
    # Banding threshold per Politis-White.
    m = int(np.ceil(2.0 * np.sqrt(np.log10(n))))
    if m < 1:
        m = 1
    # Heuristic AR(1)-based block length used in standard implementations.
    rho1 = float(acorr[1]) if n > 1 else 0.0
    rho1 = float(np.clip(rho1, -0.999, 0.999))
    if abs(rho1) < 1e-6:
        return 1
    # b_SB ~ ( 2 * rho1^2 / (1 - rho1^2)^2 )^(1/3) * n^(1/3)
    num = 2.0 * (rho1 ** 2)
    den = (1.0 - rho1 ** 2) ** 2
    b = (num / den) ** (1.0 / 3.0) * (n ** (1.0 / 3.0))
    b = int(max(1, round(b)))
    return min(b, max(1, n - 1))


def stationary_block_bootstrap_indices(T, block_len, n_boot, rng=None):
    """Generate bootstrap-sample indices via the stationary block bootstrap.

    Parameters
    ----------
    T : int
        Series length.
    block_len : float
        Mean block length L. Geometric block lengths with parameter p = 1/L.
    n_boot : int
        Number of bootstrap samples to generate.
    rng : np.random.Generator or None

    Returns
    -------
    idx : ndarray of shape (n_boot, T)
        Each row contains T indices into the original series (with wrap-around).
    """
    if rng is None:
        rng = np.random.default_rng()
    T = int(T)
    L = float(block_len)
    if L < 1.0:
        L = 1.0
    p = 1.0 / L
    n_boot = int(n_boot)
    out = np.empty((n_boot, T), dtype=np.int64)
    for b in range(n_boot):
        idx = np.empty(T, dtype=np.int64)
        t = 0
        while t < T:
            start = int(rng.integers(0, T))
            # Geometric block length, minimum 1.
            blen = int(rng.geometric(p))
            if blen < 1:
                blen = 1
            end = min(T, t + blen)
            for k in range(end - t):
                idx[t + k] = (start + k) % T
            t = end
        out[b, :] = idx
    return out


def bootstrap_sharpe_diff(x, y, AF, n_boot=2000, block_len=None, seed=0, alpha=0.05,
                          rf_per_period=0.0):
    """Stationary-block-bootstrap test on the annualised Sharpe difference.

    Parameters
    ----------
    x, y : array-like, same length
        Paired return series.
    AF : int
        Annualisation factor for reporting (the bootstrap test is invariant).
    n_boot : int
        Number of bootstrap replications.
    block_len : float or None
        Mean block length. If None, computed via politis_white_block_length on
        the difference series (x - y).
    seed : int
    alpha : float
        Confidence level for the interval (e.g. 0.05 → 95% CI).
    rf_per_period : float
        Risk-free rate per period.

    Returns
    -------
    dict with keys:
        diff                — point estimate SR_x - SR_y (annualised)
        ci_low, ci_high     — percentile CI on the difference
        p_two_sided         — fraction of bootstrap diffs more extreme than 0
        p_x_greater         — one-sided bootstrap p-value for H1: SR_x > SR_y
                              (fraction of centred bootstrap diffs >= the
                              point estimate)
        block_len_used
        n_boot
        T
    """
    xa = np.asarray(x, dtype=float).ravel()
    ya = np.asarray(y, dtype=float).ravel()
    n = min(xa.size, ya.size)
    xa = xa[:n]; ya = ya[:n]
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa = xa[mask]; ya = ya[mask]
    T = int(xa.size)
    if T < 8:
        return dict(diff=float("nan"), ci_low=float("nan"), ci_high=float("nan"),
                    p_two_sided=float("nan"), p_x_greater=float("nan"),
                    block_len_used=float("nan"), n_boot=int(n_boot), T=T)

    if block_len is None:
        diff_series = xa - ya
        L = politis_white_block_length(diff_series)
    else:
        L = float(block_len)

    rng = np.random.default_rng(int(seed))
    idx = stationary_block_bootstrap_indices(T, block_len=L, n_boot=int(n_boot), rng=rng)
    scale = float(np.sqrt(float(AF)))

    diff_boot = np.empty(int(n_boot), dtype=float)
    for b in range(int(n_boot)):
        ix = idx[b]
        xb = xa[ix]; yb = ya[ix]
        sx = _sharpe_per_period(xb, rf_per_period=rf_per_period)
        sy = _sharpe_per_period(yb, rf_per_period=rf_per_period)
        diff_boot[b] = (sx - sy) * scale if (np.isfinite(sx) and np.isfinite(sy)) else np.nan

    diff_point = (_sharpe_per_period(xa, rf_per_period=rf_per_period)
                  - _sharpe_per_period(ya, rf_per_period=rf_per_period)) * scale

    valid = diff_boot[np.isfinite(diff_boot)]
    if valid.size == 0:
        return dict(diff=diff_point, ci_low=float("nan"), ci_high=float("nan"),
                    p_two_sided=float("nan"), p_x_greater=float("nan"),
                    block_len_used=float(L), n_boot=int(n_boot), T=T)

    ci_low  = float(np.quantile(valid, alpha / 2.0))
    ci_high = float(np.quantile(valid, 1.0 - alpha / 2.0))
    # Two-sided p: fraction of bootstrap samples whose diff is at least as
    # extreme (in absolute value) as if the true diff were 0. Use the centred
    # bootstrap: shift bootstrap diffs by -mean(diff_boot) under the null.
    centred = valid - float(np.mean(valid))
    p_two = float(np.mean(np.abs(centred) >= abs(diff_point)))
    p_grt = float(np.mean(centred >= diff_point))

    return dict(
        diff=float(diff_point),
        ci_low=ci_low,
        ci_high=ci_high,
        p_two_sided=p_two,
        p_x_greater=p_grt,
        block_len_used=float(L),
        n_boot=int(n_boot),
        T=T,
    )


# =============================================================================
# Multiple-testing corrections
# =============================================================================
# Bonferroni: p_adj_i = min(1, p_i * m). Strongest control of FWER.
# Holm (1979), Scandinavian Journal of Statistics 6, 65-70. Step-down,
#   uniformly more powerful than Bonferroni at the same FWER.
# Benjamini & Hochberg (1995), JRSS-B 57, 289-300. Controls the false
#   discovery rate (FDR), more powerful when many tests, less stringent.

def _coerce_pvalues(pvalues):
    p = np.asarray(pvalues, dtype=float).ravel()
    return p


def bonferroni_correct(pvalues, alpha=0.05):
    """Bonferroni correction.

    Returns dict with:
        p_adj  : ndarray of adjusted p-values, same order as input
        reject : ndarray of bool, True where p_adj <= alpha
        m      : number of tests
    """
    p = _coerce_pvalues(pvalues)
    m = int(p.size)
    if m == 0:
        return dict(p_adj=p, reject=np.zeros(0, dtype=bool), m=0)
    p_adj = np.minimum(1.0, p * m)
    return dict(p_adj=p_adj, reject=(p_adj <= float(alpha)), m=m)


def holm_correct(pvalues, alpha=0.05):
    """Holm (1979) step-down correction.

    Returns dict with the same fields as bonferroni_correct.
    """
    p = _coerce_pvalues(pvalues)
    m = int(p.size)
    if m == 0:
        return dict(p_adj=p, reject=np.zeros(0, dtype=bool), m=0)
    order = np.argsort(p)
    p_sorted = p[order]
    # adjusted_i = max_{j <= i} (m - j + 1) * p_(j), clipped at 1.
    adj_sorted = np.empty(m, dtype=float)
    running_max = 0.0
    for i in range(m):
        val = (m - i) * p_sorted[i]
        if val > running_max:
            running_max = val
        adj_sorted[i] = min(1.0, running_max)
    p_adj = np.empty(m, dtype=float)
    p_adj[order] = adj_sorted
    return dict(p_adj=p_adj, reject=(p_adj <= float(alpha)), m=m)


def bh_correct(pvalues, alpha=0.05):
    """Benjamini-Hochberg (1995) FDR correction.

    Returns dict with the same fields as bonferroni_correct, where reject
    controls FDR at level alpha (not FWER).
    """
    p = _coerce_pvalues(pvalues)
    m = int(p.size)
    if m == 0:
        return dict(p_adj=p, reject=np.zeros(0, dtype=bool), m=0)
    order = np.argsort(p)
    p_sorted = p[order]
    ranks = np.arange(1, m + 1, dtype=float)
    adj_sorted = p_sorted * m / ranks
    # Enforce monotonicity from the largest p downwards.
    for i in range(m - 2, -1, -1):
        if adj_sorted[i + 1] < adj_sorted[i]:
            adj_sorted[i] = adj_sorted[i + 1]
    adj_sorted = np.minimum(1.0, adj_sorted)
    p_adj = np.empty(m, dtype=float)
    p_adj[order] = adj_sorted
    return dict(p_adj=p_adj, reject=(p_adj <= float(alpha)), m=m)


# =============================================================================
# Original verbose hypothesis-test reporting (kept as-is)
# =============================================================================

def hypothesis_tests(results_dict, tests, alpha=0.05):
    """
    Verbose hypothesis test reporting.
    tests: list of {"kind": "breach_less" | "equality_sharpe" | "superiority_sharpe",
                    "A": "<model name>", "B": "<model name>"}
    """
    COLS = {
        "mu": ["mu_ann", "mu_annual_geom", "Expected Return (CAGR)", "CAGR"],
        "sh": ["sharpe_ann", "Sharpe annual", "Sharpe Ratio"],
        "br": ["vol_breach", "Volatility Breach"],
    }
    def pick_col_any(results_dict, candidates):
        for df in results_dict.values():
            for c in candidates:
                if c in df.columns:
                    return c
        raise KeyError(f"None of {candidates} found in any results DataFrame columns.")

    col_mu = pick_col_any(results_dict, COLS["mu"])
    col_sh = pick_col_any(results_dict, COLS["sh"])
    col_br = pick_col_any(results_dict, COLS["br"])

    print("\n" + "=" * 72)
    print(f"HYPOTHESIS TESTS  (alpha = {alpha:.2f}, confidence = {int((1 - alpha) * 100)}%)")
    print("=" * 72)

    def align_pair(A, B):
        if A not in results_dict or B not in results_dict:
            raise KeyError(f"Missing model in results: needed '{A}' and '{B}'.")
        dfA, dfB = results_dict[A], results_dict[B]
        m = min(len(dfA), len(dfB))
        if m == 0:
            raise ValueError(f"No overlapping trials for pair ({A}, {B}).")
        return (dfA.iloc[:m].reset_index(drop=True),
                dfB.iloc[:m].reset_index(drop=True))

    printed_section1 = False
    printed_section2 = False
    idx1 = 0
    idx2 = 0
    def ab_label(k): return chr(64 + k)

    for t in tests:
        kind = t["kind"]; A, B = t["A"], t["B"]

        if kind == "breach_less":
            if not printed_section1:
                print("\n[1] Risk-budget breaches (vol_breach)")
                printed_section1 = True
            idx1 += 1
            label = f"1{ab_label(idx1)})"
            dfA, dfB = align_pair(A, B)
            x = dfA[col_br].to_numpy()
            y = dfB[col_br].to_numpy()

            print(f"\n{label} {A} vs {B} — vol_breach (paired t-test, one-sided)")
            print(f"   H0: mean({A}_vol_breach - {B}_vol_breach) = 0")
            print(f"   H1: mean({A}_vol_breach - {B}_vol_breach) < 0")
            T, P = paired_onesided_less(x, y)
            _d = np.asarray(x, float) - np.asarray(y, float)
            mean_diff = float(np.mean(_d[np.isfinite(_d)])) if np.isfinite(_d).any() else float("nan")
            print(f"   Test: Paired t-test on differences ({A} - {B})")
            print(f"   alpha={alpha:.2f}, t={T:.3f}, p(one-sided)={P:.4g}, mean diff={mean_diff:.6f}")
            if P < alpha:
                print(f"   Conclusion: REJECT H0 at {int((1 - alpha) * 100)}% confidence → {A} breaches LESS than {B}.")
            else:
                print("   Conclusion: FAIL TO REJECT H0 — No significant reduction in breaches.")

        elif kind == "equality_sharpe":
            if not printed_section2:
                print("\n[2] Performance")
                printed_section2 = True
            idx2 += 1
            label = f"2{ab_label(idx2)})"
            dfA, dfB = align_pair(A, B)
            x = dfA[col_sh].to_numpy()
            y = dfB[col_sh].to_numpy()

            print(f"\n{label} Equality: {A} vs {B} — Sharpe (paired t-test, two-sided)")
            print(f"   H0: mean({A}_sharpe - {B}_sharpe) = 0")
            print(f"   H1: mean({A}_sharpe - {B}_sharpe) ≠ 0")
            res = paired_two_sided_test_with_ci(x, y, alpha=alpha)
            print("   Test: Paired two-sided t-test on "
                  f"({A} - {B})")
            print(
                f"   alpha={alpha:.2f}, t={res['t']:.3f}, p(two-sided)={res['p']:.4g}, "
                f"mean diff={res['mean_diff']:.6f}, 95% CI=({res['ci_low']:.6f}, {res['ci_high']:.6f}), n={res['n']}"
            )
            if res["p"] < alpha:
                direction = f"{A} > {B}" if res["mean_diff"] > 0 else f"{A} < {B}"
                print(f"   Conclusion: REJECT H0 at {int((1 - alpha) * 100)}% confidence → Sharpe differs ({direction}).")
            else:
                print("   Conclusion: FAIL TO REJECT H0 — No statistically significant Sharpe difference.")

        elif kind == "superiority_sharpe":
            if not printed_section2:
                print("\n[2] Performance")
                printed_section2 = True
            idx2 += 1
            label = f"2{ab_label(idx2)})"
            dfA, dfB = align_pair(A, B)
            x = dfA[col_sh].to_numpy()
            y = dfB[col_sh].to_numpy()

            print(f"\n{label} Superiority: {A} vs {B} — Sharpe (paired t-test)")
            print(f"   H0: mean({A}_sharpe - {B}_sharpe) ≤ 0")
            print(f"   H1: mean({A}_sharpe - {B}_sharpe) > 0")
            T, P = superiority_paired(x, y)
            _d = np.asarray(x, float) - np.asarray(y, float)
            mean_diff = float(np.mean(_d[np.isfinite(_d)])) if np.isfinite(_d).any() else float("nan")
            print(f"   alpha={alpha:.2f}, t={T:.3f}, p(one-sided)={P:.4g}, mean diff={mean_diff:.6f}")
            if P < alpha:
                print(f"   Conclusion: REJECT H0 at {int((1 - alpha) * 100)}% confidence → {A} Sharpe is SUPERIOR to {B}.")
            else:
                print("   Conclusion: FAIL TO REJECT H0 — No significant Sharpe improvement detected.")
