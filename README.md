# Macro‑Regime Aware Index Forecasts (S&P 500, Weekly)

Predict weekly **excess returns** on the S&P 500 with **macro regimes**.
Pipeline: data ingestion (FRED + Yahoo) → feature engineering (rates, spreads, inflation, growth) → **HMM regime labeling** → models (baseline ML + placeholder for **Temporal Fusion Transformer**) → **purged walk‑forward** backtest → reports (attribution, regime heatmaps).

## Quick start
```bash
# 1) Create env (recommended)
python -m venv .venv && source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Set your FRED API key (get one free at https://fred.stlouisfed.org/)
export FRED_API_KEY="YOUR_KEY"

# 4) Run end‑to‑end (downloads data into ./data_cache)
python -m src.run_pipeline
```

## What you get
- **Regime detection:** Gaussian HMM on returns/vol to discover bull/bear/stress regimes.
- **Models:** Baseline Logistic / Gradient Boosting for next‑week excess‑return sign. `src/models/tft_placeholder.py` shows how to wire **Temporal Fusion Transformer** (via `pytorch-forecasting`).
- **Purged CV:** Custom time‑series splitter with **embargo** to avoid leakage.
- **Backtest:** Directional strategy on S&P 500 with turnover & **transaction costs**.
- **Reports:** Regime heatmap, confusion matrix, equity curve, factor attribution stubs.

## Data sources
- Prices: Yahoo Finance (`^GSPC`, risk‑free from FRED)  
- Macro (FRED): `DGS2, DGS10, T10Y2Y, CPIAUCSL, UNRATE, INDPRO, FEDFUNDS, T5YIFR, BAA10Y, TB3MS` etc.

## Notes
- This repo ships a **working baseline** (sklearn) + **TFT placeholder**. You can swap in N‑BEATS/TFT later with GPUs.
- All plotting code is matplotlib‑only to keep compatibility.

— Generated on 2025-09-28

---
## Streamlit App (website-like UI)
```bash
# Run locally
streamlit run src/app/streamlit_app.py
```
- Sidebar lets you choose regime count, CV params, costs, and refresh data.
- Main view shows regime heatmap, CV metrics, equity curve, and a **live next-week probability**.

## One-shot CLI (for automation / cron)
```bash
# Print a one-line live prediction (UTC timestamp)
python -m src.cli.update_now
```

### Example cron (runs every Friday 5pm PT)
```cron
0 17 * * FRI cd /path/to/macro_regime_sp500 && /usr/bin/env bash -lc 'source .venv/bin/activate && export FRED_API_KEY=YOUR_KEY && python -m src.cli.update_now >> logs/predictions.log 2>&1'
```
