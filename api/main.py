# api/main.py
"""
FastAPI backend for the Macro-Regime Quant Terminal.
Wraps existing src/ Python ML modules as REST endpoints.
Deploy on Railway: uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations
import os
import sys
import json
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.data.fetch import get_prices_cached, get_fred_series_cached
from src.features.engineer import compute_weekly_returns, to_weekly_last, assemble_panel
from src.models.regimes import fit_hmm, align_regimes
from src.models.baseline import make_baseline, FEATURE_SET, fit_predict, evaluate_cls, get_feature_importance
from src.utils.cv import rolling_windows
from src.evaluate.backtest import backtest_directional, compute_performance_metrics, drawdown_series

# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Quant Terminal API", version="2.0.0")

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["*"],
)

DATA_CACHE = Path("data_cache")
DATA_CACHE.mkdir(exist_ok=True)

FRED_LIST = ["DGS2","DGS10","T10Y2Y","CPIAUCSL","UNRATE","INDPRO","FEDFUNDS","T5YIFR","BAA10Y","TB3MS"]
REGIME_LABELS = {0: "Bear / Stress", 1: "Bull / Risk-On", 2: "Transition", 3: "High-Vol"}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _load_panel(start: str) -> pd.DataFrame:
    prices = get_prices_cached(symbol="^GSPC", start=start, cache_path=str(DATA_CACHE/"prices.csv"))
    fred   = get_fred_series_cached(FRED_LIST, start="1960-01-01", cache_path=str(DATA_CACHE/"fred.csv"))
    pw     = compute_weekly_returns(prices)
    fw     = to_weekly_last(fred).interpolate(limit_direction="both")
    panel  = assemble_panel(pw, fw).dropna()
    return panel[panel.index >= pd.to_datetime(start)]


def _build_panel_with_regime(start: str, n_states: int) -> pd.DataFrame:
    panel = _load_panel(start)
    _, regimes = fit_hmm(panel, n_states=n_states, covariance_type="full", feature_cols=("ret_w","rv_w"))
    panel["regime"] = align_regimes(panel.index, regimes)
    return panel


def _safe(v):
    """Convert numpy scalars / NaN → JSON-safe Python types."""
    if v is None:
        return None
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def _df_to_records(df: pd.DataFrame, date_col: str = "date") -> list[dict]:
    """Convert a DataFrame with DatetimeIndex to list of JSON-safe dicts."""
    out = []
    for idx, row in df.iterrows():
        rec = {date_col: str(idx.date())}
        for col in df.columns:
            rec[col] = _safe(row[col])
        out.append(rec)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
def home():
    return {
        "message": "Quant Terminal API is running.",
        "version": "2.0.0",
        "docs": "/docs",
        "frontend": "http://localhost:3000"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/live_signal")
def live_signal(
    start: str = Query("1999-01-01"),
    model: str = Query("logistic"),
    n_states: int = Query(3),
    threshold: float = Query(0.50),
):
    """Live next-week signal and model probability."""
    try:
        panel = _build_panel_with_regime(start, n_states)
        feat_list = [c for c in FEATURE_SET if c in panel.columns]
        X = panel[feat_list].ffill().dropna()
        y = (panel["excess_ret_next"].reindex(X.index) > 0).astype(int)

        if len(X) <= 60:
            raise HTTPException(400, "Not enough history for a signal (need >60 weeks).")

        mdl = make_baseline(model)
        mdl.fit(X.iloc[:-1], y.iloc[:-1])
        p_up = float(mdl.predict_proba(X.iloc[[-1]])[:, 1][0])

        action  = "LONG" if p_up >= threshold else ("UNCERTAIN" if p_up >= 0.45 else "FLAT")
        kelly_f = float(min(max(2 * p_up - 1, 0.0), 0.5))
        conf    = float(min(abs(p_up - 0.5) * 200, 100.0))
        regime  = int(panel["regime"].iloc[-1])

        fi = get_feature_importance(mdl, feat_list)
        importance = [{"feature": k, "importance": _safe(v)} for k, v in fi.items()]

        # Trailing 52-week sparkline
        tail = panel["ret_w"].dropna().tail(52)
        eq_tail = (1 + tail).cumprod()
        sparkline = [{"date": str(d.date()), "equity": _safe(v)}
                     for d, v in zip(eq_tail.index, eq_tail.values)]

        return {
            "p_up": p_up,
            "action": action,
            "confidence": conf,
            "kelly_fraction": kelly_f,
            "threshold": threshold,
            "regime": regime,
            "regime_label": REGIME_LABELS.get(regime, f"Regime {regime}"),
            "last_date": str(panel.index[-1].date()),
            "n_weeks": len(panel),
            "feature_importance": importance,
            "sparkline": sparkline,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/regime_map")
def regime_map(
    start: str = Query("1999-01-01"),
    n_states: int = Query(3),
):
    """Full price series with regime labels, stats, and transition matrix."""
    try:
        panel = _build_panel_with_regime(start, n_states)
        regs  = panel["regime"].astype(int)

        # Price + regime timeseries (downsample weekly → already weekly)
        price_series = [
            {"date": str(d.date()), "close": _safe(c), "regime": int(r), "ret_w": _safe(rw)}
            for d, c, r, rw in zip(panel.index, panel["close"], regs, panel["ret_w"])
        ]

        # Per-regime statistics
        regime_stats = []
        for r in sorted(regs.unique()):
            mask = regs == r
            rret = panel.loc[mask, "ret_w"].dropna()
            regime_stats.append({
                "regime": int(r),
                "label": REGIME_LABELS.get(r, f"Regime {r}"),
                "n_weeks": int(mask.sum()),
                "pct_time": float(mask.mean()),
                "ann_return": float(rret.mean() * 52),
                "ann_vol": float(rret.std() * np.sqrt(52)),
                "sharpe": _safe(rret.mean() / rret.std() * np.sqrt(52)) if rret.std() > 0 else None,
                "win_rate": float((rret > 0).mean()),
            })

        # Transition matrix
        k = n_states
        trans = np.zeros((k, k), dtype=int)
        rv = regs.values
        for i in range(len(rv) - 1):
            trans[rv[i], rv[i+1]] += 1
        trans_prob = (trans / trans.sum(axis=1, keepdims=True).clip(1)).tolist()

        return {
            "price_series": price_series,
            "regime_stats": regime_stats,
            "transition_matrix": trans_prob,
            "n_states": k,
            "regime_labels": {str(k): v for k, v in REGIME_LABELS.items()},
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/backtest")
def backtest(
    start: str = Query("1999-01-01"),
    n_states: int = Query(3),
    model: str = Query("logistic"),
    n_splits: int = Query(6),
    test_size: int = Query(104),
    embargo: int = Query(4),
    cost_bps: int = Query(5),
    turnover: float = Query(1.0),
):
    """Walk-forward backtest returning equity curve, drawdown, fold metrics, aggregate stats."""
    try:
        panel = _build_panel_with_regime(start, n_states)
        feat_list = [c for c in FEATURE_SET if c in panel.columns]
        X = panel[feat_list].ffill().dropna()
        y = (panel["excess_ret_next"].reindex(X.index) > 0).astype(int)

        rows, eq_list, bnh_list, oof_p_list, oof_y_list = [], [], [], [], []

        for tr_idx, te_idx in rolling_windows(len(X), n_splits, test_size, embargo):
            if len(tr_idx) < 52 or len(te_idx) < 8:
                continue
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
            dates_te   = X_te.index

            mdl = make_baseline(model)
            _, proba = fit_predict(mdl, X_tr, y_tr, X_te)
            metrics  = evaluate_cls(y_te, proba)
            ex_ret   = panel["excess_ret_next"].reindex(dates_te)
            bt = backtest_directional(dates_te, proba, ex_ret, cost_bps, turnover)
            perf = compute_performance_metrics(bt["equity"], bt["strat_ret"], benchmark_ret=ex_ret)

            rows.append({
                "start": str(dates_te[0].date()), "end": str(dates_te[-1].date()),
                "auc": _safe(metrics["auc"]), "acc": _safe(metrics["acc"]),
                **{k: _safe(v) for k, v in perf.items()},
            })
            eq_list.append(bt["equity"])
            bnh_list.append(ex_ret)
            oof_p_list.append(pd.Series(proba, index=dates_te))
            oof_y_list.append(pd.Series(y_te.values, index=dates_te))

        if not eq_list:
            raise HTTPException(400, "Not enough data for walk-forward CV.")

        eq_full  = pd.concat(eq_list).sort_index()
        bnh_full = pd.concat(bnh_list).sort_index()
        bnh_eq   = (1 + bnh_full).cumprod()
        dd       = drawdown_series(eq_full)

        agg_perf = compute_performance_metrics(
            eq_full, eq_full.pct_change().dropna(), benchmark_ret=bnh_full
        )

        equity_series = [
            {"date": str(d.date()), "strategy": _safe(s), "bnh": _safe(b), "drawdown": _safe(dv)}
            for d, s, b, dv in zip(eq_full.index, eq_full.values, bnh_eq.reindex(eq_full.index).values, dd.values)
        ]

        # OOF probabilities (for ModelQA to consume)
        oof_p = pd.concat(oof_p_list).sort_index()
        oof_y = pd.concat(oof_y_list).sort_index().astype(int)
        oof_aligned = oof_y.reindex(oof_p.index).dropna()
        oof_series = [
            {"date": str(d.date()), "p_up": _safe(p), "y": int(yv)}
            for d, p, yv in zip(oof_aligned.index, oof_p.reindex(oof_aligned.index).values, oof_aligned.values)
        ]

        return {
            "equity_series": equity_series,
            "fold_metrics": rows,
            "aggregate_performance": {k: _safe(v) for k, v in agg_perf.items()},
            "oof_signals": oof_series,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/factor_analysis")
def factor_analysis(
    start: str = Query("1999-01-01"),
    n_states: int = Query(3),
    model: str = Query("logistic"),
):
    """Feature importance and macro feature correlation."""
    try:
        panel = _build_panel_with_regime(start, n_states)
        feat_list = [c for c in FEATURE_SET if c in panel.columns]
        X = panel[feat_list].ffill().dropna()
        y = (panel["excess_ret_next"].reindex(X.index) > 0).astype(int)

        mdl = make_baseline(model)
        mdl.fit(X.iloc[:-1], y.iloc[:-1])
        fi = get_feature_importance(mdl, feat_list)
        importance = [{"feature": k, "importance": _safe(v)} for k, v in fi.items()]

        # Correlation matrix (exclude 'regime')
        corr_cols = [c for c in feat_list if c != "regime"]
        corr = X[corr_cols].corr()
        correlation = {
            "features": corr_cols,
            "matrix": [[_safe(corr.iloc[i, j]) for j in range(len(corr_cols))]
                       for i in range(len(corr_cols))],
        }

        # Regime-conditional return distributions (bins + densities)
        regs   = panel["regime"].astype(int)
        dists  = {}
        for r in sorted(regs.unique()):
            rret = panel.loc[regs == r, "ret_w"].dropna()
            counts, bin_edges = np.histogram(rret, bins=60, density=True)
            dists[str(r)] = {
                "bin_centers": [_safe((bin_edges[i] + bin_edges[i+1]) / 2) for i in range(len(counts))],
                "density": [_safe(v) for v in counts],
                "label": REGIME_LABELS.get(r, f"Regime {r}"),
            }

        return {
            "feature_importance": importance,
            "correlation": correlation,
            "regime_distributions": dists,
        }
    except Exception as e:
        raise HTTPException(500, str(e))
