import numpy as np
import pandas as pd

def turnover(prev_w, new_w):
    return np.abs(new_w - prev_w)

def backtest_directional(dates, proba, excess_ret, trans_cost_bps=5, turnover_cap=1.0):
    """Long/flat on S&P 500 based on predicted probability of positive excess return.
    - Position w_t in [0,1] = proba_t (can be thresholded to 0/1).
    - Apply turnover cap and costs.
    """
    proba = pd.Series(proba, index=dates).clip(0,1)
    w = proba.copy()
    if turnover_cap < 1.0:
        w.iloc[0] = 0.0
        for t in range(1, len(w)):
            change = w.iloc[t] - w.iloc[t-1]
            change = np.clip(change, -turnover_cap, turnover_cap)
            w.iloc[t] = w.iloc[t-1] + change
    # transaction costs on absolute changes
    tc = turnover(w.shift(1).fillna(0.0), w) * (trans_cost_bps/10000.0)
    strat_ret = w * excess_ret - tc
    equity = (1 + strat_ret).cumprod()
    return pd.DataFrame({"w": w, "strat_ret": strat_ret, "equity": equity})
