import os
import time
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path
from fredapi import Fred


def _tznaive_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = None
    return df


def get_prices(symbol="^GSPC", start="1980-01-01", end=None, interval="1d") -> pd.DataFrame:
    """
    Return tz-naive index + a single 'close' column.
    Works for flat columns and MultiIndex in either order: [Price, Ticker] or [Ticker, Price].
    """
    end = end or datetime.today().strftime("%Y-%m-%d")

    df = yf.download(
        symbol,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if df is None or len(df) == 0:
        raise ValueError(f"yfinance returned empty data for {symbol}")

    df = _tznaive_index(df)

    if isinstance(df.columns, pd.MultiIndex):
        lv0 = set(df.columns.get_level_values(0))
        lv1 = set(df.columns.get_level_values(1))
        fields = {"Close", "Adj Close", "close", "adj close"}

        if fields & lv0:
            price_level = 0
        elif fields & lv1:
            price_level = 1
        else:
            raise ValueError(f"Close/Adj Close not in MultiIndex columns: {df.columns}")

        target_field = "Close" if "Close" in (lv0 if price_level == 0 else lv1) else "Adj Close"
        close = df.xs(target_field, axis=1, level=price_level)

        if isinstance(close, pd.DataFrame):
            candidates = [symbol, symbol.replace("^", "")]
            pick = None
            for c in candidates:
                if c in close.columns:
                    pick = c
                    break
            close = close[pick] if pick else close.iloc[:, 0]

        out = close.rename("close").to_frame()

    else:
        mapping = {c.lower().strip(): c for c in df.columns}
        pick = mapping.get("close") or mapping.get("adj close")
        if pick is None:
            raise ValueError(f"Could not find Close/Adj Close in: {list(df.columns)}")
        out = df[[pick]].rename(columns={pick: "close"})

    return out


def get_fred_series(series_list, api_key=None, start="1960-01-01") -> pd.DataFrame:
    """Download FRED series and return a single DataFrame with tz-naive index."""
    fred = Fred(api_key=api_key or os.getenv("FRED_API_KEY"))
    frames = []
    for s in series_list:
        ser = fred.get_series(s)
        ser = pd.Series(ser, name=s)
        ser.index = pd.to_datetime(ser.index).tz_localize(None)
        frames.append(ser)
    df = pd.concat(frames, axis=1).sort_index()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df.index >= pd.to_datetime(start)]
    return df


# ---- Robust cached wrappers (retry + local cache fallback) ----
def _retry(fn, tries=3, backoff=1.7):
    for i in range(tries):
        try:
            return fn()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(backoff ** i)

def get_prices_cached(cache_path="data_cache/prices.csv", **kwargs):
    cache = Path(cache_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    try:
        df = _retry(lambda: get_prices(**kwargs))
        df.to_csv(cache)
        return df
    except Exception as e:
        if cache.exists():
            print(f"[WARN] get_prices failed ({e}); using cached {cache}")
            return pd.read_csv(cache, index_col=0, parse_dates=True)
        raise

def get_fred_series_cached(series_list, cache_path="data_cache/fred.csv", api_key=None, start="1960-01-01"):
    cache = Path(cache_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    try:
        df = _retry(lambda: get_fred_series(series_list, api_key=api_key, start=start))
        df.to_csv(cache)
        return df
    except Exception as e:
        if cache.exists():
            print(f"[WARN] get_fred_series failed ({e}); using cached {cache}")
            return pd.read_csv(cache, index_col=0, parse_dates=True)
        raise
