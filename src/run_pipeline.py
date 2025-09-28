# src/run_pipeline.py
import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt

from src.data.fetch import get_prices_cached, get_fred_series_cached
from src.features.engineer import compute_weekly_returns, to_weekly_last, assemble_panel
from src.models.regimes import fit_hmm, align_regimes
from src.models.baseline import make_baseline, FEATURE_SET, fit_predict, evaluate_cls
from src.evaluate.backtest import backtest_directional
from src.utils.cv import rolling_windows
from src.utils.plotting import regime_heatmap

CONFIG_PATH = "configs/config.yaml"
DATA_CACHE = "data_cache"


def load_config(path: str = CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    os.makedirs(DATA_CACHE, exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    cfg = load_config()

    # -----------------------------
    # 1) Data (robust: retry + cache)
    # -----------------------------
    prices = get_prices_cached(
        cache_path=os.path.join(DATA_CACHE, "prices.csv"),
        symbol=cfg["symbol"],
        start=cfg["backtest"]["start_date"],
    )

    fred_list = list(cfg["fred_series"].keys())
    fred = get_fred_series_cached(
        fred_list,
        cache_path=os.path.join(DATA_CACHE, "fred.csv"),
        start="1960-01-01",
    )

    # Weekly aggregates
    pw = compute_weekly_returns(prices)  # gives columns: close, ret_w, rv_w
    fw = to_weekly_last(fred).interpolate(limit_direction="both")

    # Assemble modeling panel
    panel = assemble_panel(pw, fw)

    # Backtest window trim
    if cfg["backtest"].get("start_date"):
        panel = panel[panel.index >= pd.to_datetime(cfg["backtest"]["start_date"])]
    if cfg["backtest"].get("end_date"):
        panel = panel[panel.index <= pd.to_datetime(cfg["backtest"]["end_date"])]

    # -----------------------------
    # 2) HMM regimes
    # -----------------------------
    hmm_features = cfg["hmm"].get("features", ["ret_w", "rv_w"])
    hmm_model, regimes = fit_hmm(
        panel,
        n_states=int(cfg["hmm"]["n_states"]),
        covariance_type=str(cfg["hmm"]["covariance_type"]).strip('"').strip("'"),
        feature_cols=hmm_features,
    )
    panel["regime"] = align_regimes(panel.index, regimes)

    # -----------------------------
    # 3) Features / Target
    # -----------------------------
    cols = [c for c in FEATURE_SET if c in panel.columns]
    X = panel[cols].ffill().dropna()
    y = (panel["excess_ret_next"].reindex(X.index) > 0).astype(int)

    # -----------------------------
    # 4) CV + Backtest (safe rolling splitter)
    # -----------------------------
    n_splits = int(cfg["cv"]["n_splits"])
    test_size = int(cfg["cv"]["test_size_weeks"])
    embargo = int(cfg["cv"]["embargo_weeks"])

    rows = []
    equity_curves = []

    for tr_idx, te_idx in rolling_windows(len(X), n_splits, test_size, embargo):
        # sanity: require some train/test length
        if len(tr_idx) < 52 or len(te_idx) < 8:
            continue

        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        dates_te = X_te.index

        model = make_baseline()
        _, proba = fit_predict(model, X_tr, y_tr, X_te)

        metrics = evaluate_cls(y_te, proba)
        ex_ret = panel["excess_ret_next"].reindex(dates_te)

        bt = backtest_directional(
            dates_te,
            proba,
            ex_ret,
            trans_cost_bps=int(cfg["backtest"]["trans_cost_bps"]),
            turnover_cap=float(cfg["backtest"]["turnover_cap"]),
        )

        rows.append(
            {
                "fold_start": str(dates_te[0].date()),
                "fold_end": str(dates_te[-1].date()),
                **metrics,
                "cumret": bt["equity"].iloc[-1] - 1,
            }
        )
        equity_curves.append(bt["equity"].rename(f"{dates_te[0].date()}→{dates_te[-1].date()}"))

    report = pd.DataFrame(rows)
    report.to_csv(os.path.join(DATA_CACHE, "cv_report.csv"), index=False)
    print(report)

    # -----------------------------
    # 5) Plots
    # -----------------------------
    if equity_curves:
        eq_concat = pd.concat(equity_curves).sort_index()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(eq_concat.index, eq_concat.values)
        ax.set_title("Strategy Equity (out-of-fold concatenated)")
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join("reports", "equity_oof.png"), dpi=160)

    fig2, ax2 = regime_heatmap(panel.index, panel["regime"].values, title="HMM Regimes")
    fig2.savefig(os.path.join("reports", "regime_heatmap.png"), dpi=160)


if __name__ == "__main__":
    main()
