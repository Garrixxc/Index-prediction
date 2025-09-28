# src/features/engineer.py
import pandas as pd
import numpy as np

def _ensure_flat_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(x) for x in col if x != ""]) for col in df.columns]
    return df

def to_weekly_last(df: pd.DataFrame) -> pd.DataFrame:
    return _ensure_flat_datetime_index(df).resample("W-FRI").last()

def compute_weekly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_flat_datetime_index(prices.copy())
    df.columns = [c.lower().strip() for c in df.columns]
    if "close" not in df.columns:
        raise ValueError(f"Expected 'close' column, found: {df.columns.tolist()}")
    w_close = df["close"].resample("W-FRI").last().rename("close")
    ret_w   = w_close.pct_change().rename("ret_w")
    rv_w    = np.log(w_close).diff().rolling(5, min_periods=3).std().rename("rv_w")
    return pd.concat([w_close, ret_w, rv_w], axis=1)

def rf_from_tbill(tb3ms_weekly: pd.DataFrame) -> pd.Series:
    r_month = tb3ms_weekly["TB3MS"] / 100.0
    r_week  = (1 + r_month) ** (1/4.33) - 1
    return r_week.rename("rf_w")

def macro_transforms(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    if "DGS2" in df and "DGS10" in df: out["term_spread"] = df["DGS10"] - df["DGS2"]
    if "T10Y2Y" in df: out["t10y2y"] = df["T10Y2Y"]
    if "FEDFUNDS" in df: out["fedfunds"] = df["FEDFUNDS"]
    if "UNRATE" in df:
        out["unemp"] = df["UNRATE"]; out["d_unemp"] = df["UNRATE"].diff()
    if "INDPRO" in df: out["indpro_yoy"] = df["INDPRO"].pct_change(12)
    if "CPIAUCSL" in df:
        out["cpi_yoy"] = df["CPIAUCSL"].pct_change(12); out["cpi_mom"] = df["CPIAUCSL"].pct_change()
    if "T5YIFR" in df: out["infl_exp"] = df["T5YIFR"]
    if "BAA10Y" in df: out["cred_spread"] = df["BAA10Y"]
    return out

def assemble_panel(prices_w: pd.DataFrame, fred_w: pd.DataFrame) -> pd.DataFrame:
    prices_w = _ensure_flat_datetime_index(prices_w)
    fred_w   = _ensure_flat_datetime_index(fred_w)
    feats = macro_transforms(fred_w)
    feats = feats[~feats.index.duplicated(keep="last")]
    if "TB3MS" in fred_w: rf = rf_from_tbill(fred_w[["TB3MS"]])
    else: rf = pd.Series(index=feats.index, data=0.0, name="rf_w")
    df = prices_w.join(feats, how="inner")
    df = df.join(rf, how="left").ffill()
    missing = [c for c in ["ret_w", "rv_w"] if c not in df.columns]
    if missing: raise KeyError(f"Missing required columns after join: {missing}")
    df["excess_ret_next"] = df["ret_w"].shift(-1) - df["rf_w"].shift(-1)
    return df.dropna()
