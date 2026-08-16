# schema.py

import pandas as pd


REQUIRED_RESULTS_COLS = ("security", "config", "n_regimes", "dim_latent", "score")
REQUIRED_SEGMENTS_COLS = ("security", "date", "config", "n_regimes", "dim_latent", "z")


def _validate_results_df(df_res: pd.DataFrame) -> None:
    if df_res is None or len(df_res) == 0:
        raise ValueError("results_csv: empty DataFrame.")

    missing = [c for c in REQUIRED_RESULTS_COLS if c not in df_res.columns]
    if missing:
        raise ValueError(
            f"results_csv: missing required columns {missing}. "
            f"Required: {list(REQUIRED_RESULTS_COLS)}. Found: {list(df_res.columns)}"
        )

    if not pd.api.types.is_string_dtype(df_res["security"]) and not pd.api.types.is_object_dtype(df_res["security"]):
        raise TypeError(f"results_csv['security'] must be string/object; got {df_res['security'].dtype}")

    n_reg = pd.to_numeric(df_res["n_regimes"], errors="coerce")
    if n_reg.isna().any():
        raise TypeError("results_csv['n_regimes'] contains non-integer values.")

    d_lat = pd.to_numeric(df_res["dim_latent"], errors="coerce")
    if d_lat.isna().any():
        raise TypeError("results_csv['dim_latent'] contains non-integer values.")

    score = pd.to_numeric(df_res["score"], errors="coerce")
    if score.isna().all():
        raise ValueError("results_csv['score'] is all-NaN; cannot select best config.")


def _validate_segments_df(df_seg: pd.DataFrame) -> None:
    if df_seg is None or len(df_seg) == 0:
        raise ValueError("segments_parquet: empty DataFrame.")

    missing = [c for c in REQUIRED_SEGMENTS_COLS if c not in df_seg.columns]
    if missing:
        raise ValueError(
            f"segments_parquet: missing required columns {missing}. "
            f"Required: {list(REQUIRED_SEGMENTS_COLS)}. Found: {list(df_seg.columns)}"
        )

    if not pd.api.types.is_string_dtype(df_seg["security"]) and not pd.api.types.is_object_dtype(df_seg["security"]):
        raise TypeError(f"segments_parquet['security'] must be string/object; got {df_seg['security'].dtype}")

    if df_seg["date"].dtype != "datetime64[ns]":
        dt = pd.to_datetime(df_seg["date"], errors="coerce")
        if dt.isna().any():
            raise TypeError("segments_parquet['date'] contains unparseable dates.")

    z = pd.to_numeric(df_seg["z"], errors="coerce")
    if z.isna().any():
        raise TypeError("segments_parquet['z'] contains non-integer values.")

    n_reg = pd.to_numeric(df_seg["n_regimes"], errors="coerce")
    if n_reg.isna().any():
        raise TypeError("segments_parquet['n_regimes'] contains non-integer values.")

    d_lat = pd.to_numeric(df_seg["dim_latent"], errors="coerce")
    if d_lat.isna().any():
        raise TypeError("segments_parquet['dim_latent'] contains non-integer values.")


# ---------------------------------------------------------------------------
# Posterior artifact contract (regime-DRO experimental design)
# ---------------------------------------------------------------------------
# segments parquet (per security x model x date):
#   security, model, config, n_regimes, dim_latent, date,
#   z   : argmax label (kept for the hard-assignment MAP variant and the
#         variant-2 covariance buckets)
#   p1  : posterior probability of regime 1 at the date (recalibration is
#         applied downstream; this column is the raw model posterior)
#   m1, m2 : per-date regime predictions m_hat_{k,t} (daily-return units)
#   z_target (OPTIONAL) : {0,1} calibration target for the recalibration
#         layer; its definition is an upstream ssm-export decision
#
# results csv (per security x model x window_end):
#   security, model, config, n_regimes, dim_latent, window_end, score,
#   m1_bar, m2_bar : window-level regime means (annualized units are the
#         pipeline's; store daily, annualize downstream)
#   Q1, Q2         : per-regime innovation variances (daily)
#   lambda1        : fraction of the training window assigned to regime 1
#
# delta_star / delta_tilde / c(lambda) / the branch distances and per-stock
# thresholds tau_i are computed downstream (regime_dro.radius) from these
# summaries; they are never stored.

REQUIRED_RESULTS_COLS_POSTERIOR = (
    "security", "model", "config", "n_regimes", "dim_latent",
    "window_end", "score", "m1_bar", "m2_bar", "Q1", "Q2", "lambda1",
)
REQUIRED_SEGMENTS_COLS_POSTERIOR = (
    "security", "model", "config", "n_regimes", "dim_latent",
    "date", "z", "p1", "m1", "m2",
)

VALID_REGIME_MODELS = ("rslds", "slds", "arhmm")


def _valid_model_label(lbl):
    """A model label is valid iff it is a regime-model class, or a named
    configuration of one: '<class>_<suffix>' (e.g. 'rslds_factor1',
    'rslds_fund1' -- two selected configurations of the same class running
    side by side as separate model keys)."""
    lbl = str(lbl).strip()
    return any(lbl == c or lbl.startswith(c + "_")
               for c in VALID_REGIME_MODELS)


def _validate_results_df_posterior(df_res: pd.DataFrame) -> None:
    if df_res is None or len(df_res) == 0:
        raise ValueError("results csv: empty DataFrame.")
    missing = [c for c in REQUIRED_RESULTS_COLS_POSTERIOR if c not in df_res.columns]
    if missing:
        raise ValueError(
            f"results csv: missing required columns {missing}. "
            f"Required: {list(REQUIRED_RESULTS_COLS_POSTERIOR)}. Found: {list(df_res.columns)}"
        )
    bad_models = {m for m in df_res["model"].astype(str).str.strip()
                  if not _valid_model_label(m)}
    if bad_models:
        raise ValueError(f"results csv: unknown models {sorted(bad_models)}; "
                         f"valid: {list(VALID_REGIME_MODELS)} or "
                         f"'<class>_<suffix>' configurations of them")
    for col in ("m1_bar", "m2_bar", "Q1", "Q2", "lambda1", "score"):
        v = pd.to_numeric(df_res[col], errors="coerce")
        if v.isna().any():
            raise TypeError(f"results csv['{col}'] contains non-numeric values.")
    Q1 = pd.to_numeric(df_res["Q1"], errors="coerce")
    Q2 = pd.to_numeric(df_res["Q2"], errors="coerce")
    if (Q1 <= 0).any() or (Q2 <= 0).any():
        raise ValueError("results csv: Q1/Q2 must be strictly positive.")
    lam = pd.to_numeric(df_res["lambda1"], errors="coerce")
    if ((lam <= 0) | (lam >= 1)).any():
        raise ValueError("results csv: lambda1 must lie in (0, 1).")
    we = pd.to_datetime(df_res["window_end"], errors="coerce")
    if we.isna().any():
        raise TypeError("results csv['window_end'] contains unparseable dates.")


def _validate_segments_df_posterior(df_seg: pd.DataFrame) -> None:
    if df_seg is None or len(df_seg) == 0:
        raise ValueError("segments parquet: empty DataFrame.")
    missing = [c for c in REQUIRED_SEGMENTS_COLS_POSTERIOR if c not in df_seg.columns]
    if missing:
        raise ValueError(
            f"segments parquet: missing required columns {missing}. "
            f"Required: {list(REQUIRED_SEGMENTS_COLS_POSTERIOR)}. Found: {list(df_seg.columns)}"
        )
    bad_models = {m for m in df_seg["model"].astype(str).str.strip()
                  if not _valid_model_label(m)}
    if bad_models:
        raise ValueError(f"segments parquet: unknown models {sorted(bad_models)}; "
                         f"valid: {list(VALID_REGIME_MODELS)} or "
                         f"'<class>_<suffix>' configurations of them")
    p1 = pd.to_numeric(df_seg["p1"], errors="coerce")
    if p1.isna().any():
        raise TypeError("segments parquet['p1'] contains non-numeric values.")
    if ((p1 < 0) | (p1 > 1)).any():
        raise ValueError("segments parquet: p1 must lie in [0, 1].")
    for col in ("m1", "m2"):
        v = pd.to_numeric(df_seg[col], errors="coerce")
        if v.isna().any():
            raise TypeError(f"segments parquet['{col}'] contains non-numeric values.")
    dt = pd.to_datetime(df_seg["date"], errors="coerce")
    if dt.isna().any():
        raise TypeError("segments parquet['date'] contains unparseable dates.")
    if "z_target" in df_seg.columns:
        zt = pd.to_numeric(df_seg["z_target"], errors="coerce").dropna()
        if not zt.isin([0, 1]).all():
            raise ValueError("segments parquet: z_target must be in {0, 1}.")


def validate_posterior_artifacts(df_res: pd.DataFrame, df_seg: pd.DataFrame) -> None:
    """Enforce the posterior-era artifact contract (regime-DRO design)."""
    _validate_results_df_posterior(df_res)
    _validate_segments_df_posterior(df_seg)


def validate_artifacts(df_res: pd.DataFrame, df_seg: pd.DataFrame) -> None:
    """
    Enforce the regime-label artifact file-contract.

    results_csv columns:   security, config, n_regimes, dim_latent, score
    segments_parquet cols: security, date, config, n_regimes, dim_latent, z

    Raises ValueError / TypeError on any deviation.
    """
    _validate_results_df(df_res)
    _validate_segments_df(df_seg)
