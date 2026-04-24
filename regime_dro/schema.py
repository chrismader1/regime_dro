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


def validate_artifacts(df_res: pd.DataFrame, df_seg: pd.DataFrame) -> None:
    """
    Enforce the regime-label artifact file-contract.

    results_csv columns:   security, config, n_regimes, dim_latent, score
    segments_parquet cols: security, date, config, n_regimes, dim_latent, z

    Raises ValueError / TypeError on any deviation.
    """
    _validate_results_df(df_res)
    _validate_segments_df(df_seg)
