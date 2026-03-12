# src/evaluate/backtest.py
import numpy as np
import pandas as pd


def turnover(prev_w, new_w):
    return np.abs(new_w - prev_w)


def backtest_directional(dates, proba, excess_ret, trans_cost_bps=5, turnover_cap=1.0):
    """Long/flat on S&P 500 based on predicted probability of positive excess return.
    - Position w_t in [0,1] = proba_t (continuous, or threshold to 0/1).
    - Apply turnover cap and transaction costs.
    Returns a DataFrame with columns: w, strat_ret, equity.
    """
    proba = pd.Series(np.asarray(proba, dtype=float), index=dates).clip(0, 1)
    w = proba.copy()
    if turnover_cap < 1.0:
        w.iloc[0] = 0.0
        for t in range(1, len(w)):
            change = w.iloc[t] - w.iloc[t - 1]
            change = np.clip(change, -turnover_cap, turnover_cap)
            w.iloc[t] = w.iloc[t - 1] + change
    # Transaction costs on absolute changes
    tc = turnover(w.shift(1).fillna(0.0), w) * (trans_cost_bps / 10_000.0)
    strat_ret = w * excess_ret - tc
    equity = (1 + strat_ret).cumprod()
    return pd.DataFrame({"w": w, "strat_ret": strat_ret, "equity": equity})


def compute_performance_metrics(
    equity: pd.Series,
    strat_ret: pd.Series,
    benchmark_ret: pd.Series = None,
    periods_per_year: int = 52,
) -> dict:
    """
    Compute professional-grade performance metrics for a strategy.

    Parameters
    ----------
    equity : pd.Series
        Cumulative equity (starts near 1.0).
    strat_ret : pd.Series
        Period-level strategy returns (net of costs).
    benchmark_ret : pd.Series, optional
        Buy-and-hold period returns for IR / Alpha calculation.
    periods_per_year : int
        52 for weekly, 252 for daily, 12 for monthly.

    Returns
    -------
    dict of metrics.
    """
    ann = periods_per_year
    sr = strat_ret.dropna()

    # ── Core return / vol ──────────────────────────────────────────────────
    total_return = float(equity.iloc[-1] - 1.0) if len(equity) else np.nan
    n_periods = max(len(sr), 1)
    ann_return = float((1 + total_return) ** (ann / n_periods) - 1)
    ann_vol = float(sr.std() * np.sqrt(ann)) if n_periods > 1 else np.nan

    # ── Sharpe (excess return above 0; rf already netted via excess_ret) ───
    sharpe = float(ann_return / ann_vol) if ann_vol and ann_vol > 0 else np.nan

    # ── Sortino (downside vol only) ────────────────────────────────────────
    downside = sr[sr < 0]
    d_vol = float(downside.std() * np.sqrt(ann)) if len(downside) > 1 else np.nan
    sortino = float(ann_return / d_vol) if d_vol and d_vol > 0 else np.nan

    # ── Max Drawdown ───────────────────────────────────────────────────────
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    max_dd = float(drawdown.min()) if len(drawdown) else np.nan
    calmar = float(ann_return / abs(max_dd)) if max_dd and max_dd != 0 else np.nan

    # ── Win rate ───────────────────────────────────────────────────────────
    win_rate = float((sr > 0).mean()) if len(sr) else np.nan

    # ── Information Ratio vs Benchmark ────────────────────────────────────
    ir = np.nan
    alpha = np.nan
    if benchmark_ret is not None:
        aligned = sr.align(benchmark_ret, join="inner")
        active = aligned[0] - aligned[1]
        if len(active) > 1 and active.std() > 0:
            ir = float(active.mean() / active.std() * np.sqrt(ann))
        bnh_ann = float((1 + benchmark_ret.sum()) ** (ann / max(len(benchmark_ret), 1)) - 1)
        alpha = ann_return - bnh_ann

    return {
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "win_rate": win_rate,
        "information_ratio": ir,
        "alpha_vs_bnh": alpha,
    }


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Return drawdown series (0 to -1 scale)."""
    roll_max = equity.cummax()
    return (equity - roll_max) / roll_max
