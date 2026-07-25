# artifacts.py

import os
import numpy as np
import pandas as pd

from regime_dro.schema import validate_artifacts


def load_artifacts(results_csv: str, segments_parquet: str) -> dict:
    """
    Load regime-label artifacts from disk and schema-validate them.

    Parameters
    ----------
    results_csv      : path to gridsearch_results.csv (see REQUIRED_RESULTS_COLS).
    segments_parquet : path to gridsearch_segments.parquet (see REQUIRED_SEGMENTS_COLS).

    Returns
    -------
    dict with keys: df_res, df_seg.

    Raises
    ------
    FileNotFoundError if either path is missing.
    ValueError/TypeError on schema violation.
    """
    if not os.path.exists(results_csv):
        raise FileNotFoundError(results_csv)
    if not os.path.exists(segments_parquet):
        raise FileNotFoundError(segments_parquet)

    df_res = pd.read_csv(results_csv, engine="python")
    df_res["security"] = df_res["security"].astype(str).str.strip()

    df_seg = pd.read_parquet(segments_parquet)
    df_seg["security"] = df_seg["security"].astype(str).str.strip()
    if df_seg["date"].dtype != "datetime64[ns]":
        df_seg["date"] = pd.to_datetime(df_seg["date"], errors="coerce")

    validate_artifacts(df_res, df_seg)
    return dict(df_res=df_res, df_seg=df_seg)


def _num_series(s):
    return pd.to_numeric(s, errors="coerce").astype("float64")


def map_labels_to_calendar(z_ser: pd.Series, cal: pd.DatetimeIndex) -> np.ndarray:
    """
    Map (and forward-fill) regime labels to a daily trading calendar.
    Returns float64 array; NaN only before the first seen label.
    """
    z = pd.Series(z_ser).copy()
    z.index = pd.to_datetime(z.index, errors="coerce")
    z = z[~z.index.isna()].sort_index()

    cal = pd.DatetimeIndex(cal)

    z_cal = z.reindex(cal).ffill()

    return z_cal.to_numpy(dtype="float64")


def posterior_frame_from_segments(df_seg, security, model):
    """Per (security, model), a date-indexed DataFrame with columns
    z, p1, m1, m2 (and z_target if present), sorted by date."""
    m = (df_seg["security"].astype(str).str.strip() == str(security).strip()) & \
        (df_seg["model"].astype(str).str.strip() == str(model).strip())
    sub = df_seg.loc[m]
    if len(sub) == 0:
        return None
    cols = ["z", "p1", "m1", "m2"] + (["z_target"] if "z_target" in sub.columns else [])
    out = sub[["date"] + cols].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").set_index("date")
    for c in cols:
        out[c] = _num_series(out[c])
    return out


def map_frame_to_calendar(frame: pd.DataFrame, cal: pd.DatetimeIndex) -> pd.DataFrame:
    """Map a date-indexed frame to a trading calendar, column by column, with
    the same forward-fill mechanics as map_labels_to_calendar."""
    data = {c: map_labels_to_calendar(frame[c], cal) for c in frame.columns}
    return pd.DataFrame(data, index=pd.DatetimeIndex(cal))


def regime_summary_from_results(df_res, security, model, asof):
    """Latest window-level regime summary with
    window_end <= asof for (security, model). Returns dict with keys
    m1_bar, m2_bar, Q1, Q2, lambda1, window_end -- or None."""
    m = (df_res["security"].astype(str).str.strip() == str(security).strip()) & \
        (df_res["model"].astype(str).str.strip() == str(model).strip())
    sub = df_res.loc[m].copy()
    if len(sub) == 0:
        return None
    sub["window_end"] = pd.to_datetime(sub["window_end"], errors="coerce")
    sub = sub.dropna(subset=["window_end"]).sort_values("window_end")
    sub = sub[sub["window_end"] <= pd.to_datetime(asof)]
    if len(sub) == 0:
        return None
    row = sub.iloc[-1]
    return {
        "m1_bar": float(row["m1_bar"]), "m2_bar": float(row["m2_bar"]),
        "Q1": float(row["Q1"]), "Q2": float(row["Q2"]),
        "lambda1": float(row["lambda1"]),
        "window_end": row["window_end"],
    }


def snap_start_prev(cal: pd.DatetimeIndex, start_dt):
    """
    If start_dt is not on the union calendar, return the closest date
    in `cal` that is <= start_dt.
    """
    if start_dt is None:
        return cal[0]
    s = pd.to_datetime(start_dt)
    i = cal.searchsorted(s, side="right") - 1
    return cal[0] if i < 0 else cal[i]


def select_best_config(results_df, security):
    """
    For a given `security`, select the best (config, n_regimes, dim_latent) tuple.
    Sort: score ↓, n_regimes ↑, dim_latent ↑.
    """
    if results_df is None or len(results_df) == 0:
        return None
    need = {"security", "config", "n_regimes", "dim_latent"}
    if not need.issubset(results_df.columns):
        return None

    df = results_df.copy()
    df["security"]   = df["security"].astype(str).str.strip()
    df["config"]     = df["config"].astype(str).str.strip()
    df["n_regimes"]  = pd.to_numeric(df["n_regimes"],  errors="coerce").astype("Int64")
    df["dim_latent"] = pd.to_numeric(df["dim_latent"], errors="coerce").astype("Int64")
    df["score_num"]  = pd.to_numeric(df.get("score", np.nan), errors="coerce")

    df = df[df["security"] == str(security).strip()]
    if df.empty:
        return None

    df = df.sort_values(
        ["score_num", "n_regimes", "dim_latent"],
        ascending=[False, True, True], na_position="last"
    )
    r0 = df.iloc[0]
    if pd.isna(r0.get("score_num")):
        return None
    return (str(r0["config"]), int(r0["n_regimes"]), int(r0["dim_latent"]))


def labels_from_segments_df(segments_df, security, config, n_regimes, dim_latent):
    df = segments_df.copy()
    df["security"] = df["security"].astype(str).str.strip()
    df["config"]   = df["config"].astype(str).str.strip()
    df = df[
        (df["security"] == str(security).strip()) &
        (df["config"]   == str(config).strip()) &
        (pd.to_numeric(df["n_regimes"],  errors="coerce").astype("Int64") == int(n_regimes)) &
        (pd.to_numeric(df["dim_latent"], errors="coerce").astype("Int64") == int(dim_latent))
    ]
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["date", "z"]).drop_duplicates(subset="date", keep="last")
    return pd.Series(df["z"].astype(int).to_numpy(),
                     index=pd.DatetimeIndex(df["date"]), name="z")


def available_cfg_tuples_from_parquet(seg_parq: str):
    """
    Read available (config, n_regimes, dim_latent) tuples from segments parquet.
    """
    seg = pd.read_parquet(seg_parq, columns=["config", "n_regimes", "dim_latent"])
    seg["config"]     = seg["config"].astype(str).str.strip()
    seg["n_regimes"]  = pd.to_numeric(seg["n_regimes"],  errors="raise").astype(int)
    seg["dim_latent"] = pd.to_numeric(seg["dim_latent"], errors="raise").astype(int)
    tups = set(seg[["config", "n_regimes", "dim_latent"]]
               .drop_duplicates()
               .itertuples(index=False, name=None))
    return tups


def permissible_tuples_from_CONFIG(CONFIG):
    """
    Build all permissible (config, n_regimes, dim_latent) tuples from CONFIG["MODELS"].
    """    
    lst = CONFIG.get("MODELS", [])
    if not isinstance(lst, (list, tuple)) or len(lst) == 0:
        raise ValueError("CONFIG['MODELS'] must be a non-empty list of dicts.")
    out = []
    for x in lst:
        if not isinstance(x, dict) or not all(k in x for k in ("config", "n_regimes", "dim_latent")):
            raise ValueError("Each MODELS entry must be a dict with keys: config, n_regimes, dim_latent.")
        
        out.append((str(x["config"]).strip(), int(x["n_regimes"]), int(x["dim_latent"])))
    return out


def tuples_in_results_csv(df_res, tuples):
    """
    Return the subset of tuples present in results_csv.
    """
    df = df_res.copy()
    df["config"]     = df["config"].astype(str).str.strip()
    df["n_regimes"]  = pd.to_numeric(df.get("n_regimes", np.nan),  errors="coerce").astype("Int64")
    df["dim_latent"] = pd.to_numeric(df.get("dim_latent", np.nan), errors="coerce").astype("Int64")
    seen = set()
    for _, r in df.iterrows():
        c = str(r["config"])
        K = r["n_regimes"]; D = r["dim_latent"]
        if pd.notna(K) and pd.notna(D):
            seen.add((c, int(K), int(D)))
    return {t for t in tuples if t in seen}
