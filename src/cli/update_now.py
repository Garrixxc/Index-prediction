
"""
CLI tool to pull latest data, refit, and emit a one-line prediction.
Usage:
  python -m src.cli.update_now
"""
import os, pandas as pd
from datetime import datetime
from src.data.fetch import get_prices, get_fred_series
from src.features.engineer import compute_weekly_returns, to_weekly_last, assemble_panel
from src.models.regimes import fit_hmm, align_regimes
from src.models.baseline import make_baseline, FEATURE_SET, fit_predict

def main():
    prices = get_prices("^GSPC", start="1999-01-01")
    fred_list = ["DGS2","DGS10","T10Y2Y","CPIAUCSL","UNRATE","INDPRO","FEDFUNDS","T5YIFR","BAA10Y","TB3MS"]
    fred = get_fred_series(fred_list, start="1960-01-01")
    pw = compute_weekly_returns(prices)
    fw = to_weekly_last(fred).interpolate(limit_direction="both")
    panel = assemble_panel(pw, fw).dropna()
    hmm, reg = fit_hmm(panel, n_states=3, covariance_type="full", feature_cols=("ret_w","rv_w"))
    panel["regime"] = align_regimes(panel.index, reg)
    X = panel[[c for c in FEATURE_SET if c in panel.columns]].fillna(method="ffill").dropna()
    y = (panel["excess_ret_next"].reindex(X.index) > 0).astype(int)
    model = make_baseline()
    model.fit(X.iloc[:-1], y.iloc[:-1])
    prob = model.predict_proba(X.iloc[[-1]])[0,1]
    print(f"{datetime.utcnow().isoformat()}Z | P(up next week)={prob:.4f} | Regime={int(panel['regime'].iloc[-1])}")

if __name__ == "__main__":
    main()
