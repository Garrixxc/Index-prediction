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
    ret_w = w_close.pct_change().rename("ret_w")
    rv_w = np.log(w_close).diff().rolling(5, min_periods=3).std().rename("rv_w")
    return pd.concat([w_close, ret_w, rv_w], axis=1)


def rf_from_tbill(tb3ms_weekly: pd.DataFrame) -> pd.Series:
    r_month = tb3ms_weekly["TB3MS"] / 100.0
    r_week = (1 + r_month) ** (1 / 4.33) - 1
    return r_week.rename("rf_w")


def macro_transforms(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    # ── Interest Rate / Yield Curve ──────────────────────────────────────
    if "DGS2" in df and "DGS10" in df:
        out["term_spread"] = df["DGS10"] - df["DGS2"]
    if "T10Y2Y" in df:
        out["t10y2y"] = df["T10Y2Y"]
    if "FEDFUNDS" in df:
        out["fedfunds"] = df["FEDFUNDS"]
    # ── Labor Market ─────────────────────────────────────────────────────
    if "UNRATE" in df:
        out["unemp"] = df["UNRATE"]
        out["d_unemp"] = df["UNRATE"].diff()
    # ── Real Economy ─────────────────────────────────────────────────────
    if "INDPRO" in df:
        out["indpro_yoy"] = df["INDPRO"].pct_change(52)   # weekly series → 52-period YoY
    # ── Inflation ────────────────────────────────────────────────────────
    if "CPIAUCSL" in df:
        out["cpi_yoy"] = df["CPIAUCSL"].pct_change(52)
        out["cpi_mom"] = df["CPIAUCSL"].pct_change(4)     # ~monthly in weekly freq
    if "T5YIFR" in df:
        out["infl_exp"] = df["T5YIFR"]
    # ── Credit ───────────────────────────────────────────────────────────
    if "BAA10Y" in df:
        out["cred_spread"] = df["BAA10Y"]
    return out


def technical_transforms(prices_w: pd.DataFrame) -> pd.DataFrame:
    """
    Add momentum and volatility technical features from weekly price data.
    Inputs: DataFrame with columns [close, ret_w, rv_w].
    """
    out = pd.DataFrame(index=prices_w.index)
    close = prices_w["close"]
    ret_w = prices_w["ret_w"]

    # ── RSI (14-period) ────────────────────────────────────────────────
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=7).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=7).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # ── 4-week Momentum ────────────────────────────────────────────────
    out["mom_4w"] = ret_w.rolling(4, min_periods=2).sum()

    # ── 12-week Momentum (trend) ───────────────────────────────────────
    out["mom_12w"] = ret_w.rolling(12, min_periods=6).sum()

    # ── Price / 52-week SMA ────────────────────────────────────────────
    sma52 = close.rolling(52, min_periods=26).mean()
    out["price_sma52"] = (close / sma52 - 1).replace([np.inf, -np.inf], np.nan)

    # ── Realized Vol Z-score (52w) ─────────────────────────────────────
    rv = prices_w["rv_w"]
    out["rv_zscore"] = (rv - rv.rolling(52, min_periods=26).mean()) / (
        rv.rolling(52, min_periods=26).std().replace(0, np.nan)
    )

    return out


def assemble_panel(prices_w: pd.DataFrame, fred_w: pd.DataFrame) -> pd.DataFrame:
    prices_w = _ensure_flat_datetime_index(prices_w)
    fred_w = _ensure_flat_datetime_index(fred_w)

    macro = macro_transforms(fred_w)
    macro = macro[~macro.index.duplicated(keep="last")]

    tech = technical_transforms(prices_w)

    if "TB3MS" in fred_w:
        rf = rf_from_tbill(fred_w[["TB3MS"]])
    else:
        rf = pd.Series(index=macro.index, data=0.0, name="rf_w")

    df = prices_w.join(macro, how="inner")
    df = df.join(tech, how="left")
    df = df.join(rf, how="left").ffill()

    missing = [c for c in ["ret_w", "rv_w"] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns after join: {missing}")

    df["excess_ret_next"] = df["ret_w"].shift(-1) - df["rf_w"].shift(-1)
    return df.dropna()
