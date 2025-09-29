# src/app/streamlit_app.py
# Optimized: lazy CV compute, strong caching, safe rolling AUC, small UX polish

from __future__ import annotations
import os, sys
from io import BytesIO
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# --- make repo importable on Streamlit Cloud / containers ---
ROOT = Path(__file__).resolve().parents[2]  # repo/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    roc_auc_score,
)
from sklearn.calibration import calibration_curve

from src.data.fetch import get_prices_cached, get_fred_series_cached
from src.features.engineer import compute_weekly_returns, to_weekly_last, assemble_panel
from src.models.regimes import fit_hmm, align_regimes
from src.models.baseline import make_baseline, FEATURE_SET, fit_predict, evaluate_cls
from src.utils.cv import rolling_windows
from src.evaluate.backtest import backtest_directional
from src.utils.plotting import regime_heatmap

# ---------------- UI CHROME ----------------
st.set_page_config(page_title="Macro-Regime S&P 500", page_icon="📈", layout="wide")
st.markdown("""
<style>
.main .block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1400px; }
h1, h2, h3 { letter-spacing: .2px; }
.title-gradient { font-weight: 800; font-size: 2.0rem; line-height: 1.2;
  background: linear-gradient(90deg, #e5e7eb 0%, #60a5fa 40%, #a78bfa 80%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.card { border: 1px solid rgba(255,255,255,.08); background: #0f1117; border-radius: 16px; padding: 16px; }
.kpi { font-size: 28px; font-weight: 800; margin-bottom: 2px; }
.kpi-sub { font-size: 12px; opacity: .75; }
.dataframe th, .dataframe td { font-size: 13px; }
.stButton>button { border-radius: 10px; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0b1220 0%, #111827 100%);
  border-right: 1px solid rgba(255,255,255,0.08); }
[data-testid="stSidebar"] * { color: #e5e7eb !important; }
[data-testid="stSidebar"] .sidebar-title { font-weight: 800; font-size: 1.1rem; letter-spacing: .3px; margin: .25rem 0 1rem 0;
  background: linear-gradient(90deg,#93c5fd 0%, #c4b5fd 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.legend-dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; }
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR (Lab Controls) ----------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">🎛️ Macro-Regime Lab — Controls</div>', unsafe_allow_html=True)
    st.caption("Tune research settings. Changes affect CV, backtest costs, and the live preview.")

    start_date = st.date_input("Backtest start", value=pd.to_datetime("1999-01-01"))
    trans_cost_bps = st.number_input("Transaction cost (bps)", 0, 100, 5, step=1)
    turnover_cap = st.slider("Turnover cap", 0.0, 1.0, 1.0, step=0.05)
    n_states = st.selectbox("HMM regimes", [2, 3, 4], index=1)
    embargo_weeks = st.number_input("Embargo (weeks)", 0, 12, 4)
    test_size_weeks = st.number_input("Test size per fold (weeks)", 52, 260, 104, step=52)
    n_splits = st.number_input("CV splits", 2, 10, 6)
    user_threshold = st.slider("Decision threshold (LONG if P(Up) ≥ threshold)", 0.40, 0.60, 0.50, 0.01)
    refresh = st.button("🔄 Update & Predict (fetch latest)")

# ---------------- DATA ----------------
DATA_CACHE = "data_cache"
os.makedirs(DATA_CACHE, exist_ok=True)

@st.cache_data(ttl=3600, show_spinner=True)
def load_panel(start_date_str: str) -> pd.DataFrame:
    prices = get_prices_cached(symbol="^GSPC", start=start_date_str, cache_path=f"{DATA_CACHE}/prices.csv")
    fred_list = ["DGS2","DGS10","T10Y2Y","CPIAUCSL","UNRATE","INDPRO","FEDFUNDS","T5YIFR","BAA10Y","TB3MS"]
    fred = get_fred_series_cached(fred_list, start="1960-01-01", cache_path=f"{DATA_CACHE}/fred.csv")
    pw = compute_weekly_returns(prices)   # close, ret_w, rv_w
    fw = to_weekly_last(fred).interpolate(limit_direction="both")
    panel = assemble_panel(pw, fw).dropna()
    panel = panel[panel.index >= pd.to_datetime(start_date_str)]
    return panel

if refresh:
    # Bust only this cache; avoids refetch unless user asks or TTL expires
    load_panel.clear()
    st.toast("Fetched latest data (cache cleared).", icon="🔎")

panel = load_panel(start_date.strftime("%Y-%m-%d"))

# ---------------- REGIMES ----------------
hmm_model, regimes = fit_hmm(panel, n_states=int(n_states), covariance_type="full", feature_cols=("ret_w","rv_w"))
panel["regime"] = align_regimes(panel.index, regimes)

# ---------------- FEATURES/TARGET ----------------
feature_list = [c for c in FEATURE_SET if c in panel.columns]
X = panel[feature_list].ffill().dropna()
y = (panel["excess_ret_next"].reindex(X.index) > 0).astype(int)

# ---------------- LIGHT LIVE PRED ----------------
def _live_pred(X: pd.DataFrame, y: pd.Series) -> float | None:
    if len(X) <= 60:  # need some history
        return None
    model = make_baseline()
    X_train, y_train = X.iloc[:-1], y.iloc[:-1]
    X_live = X.iloc[[-1]]
    _, proba_live = fit_predict(model, X_train, y_train, X_live)
    return float(proba_live[0])

p_live = _live_pred(X, y)

# ---------------- Header ----------------
st.markdown('<div class="title-gradient">📈 Macro-Regime Aware Index Forecasts — S&P 500 (Weekly)</div>', unsafe_allow_html=True)
st.caption(f"Last updated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}. Research tool — not investment advice.")

c1, c2, c3, c4 = st.columns(4)
span = f"{panel.index.min().date()} → {panel.index.max().date()}"
with c1:
    st.markdown('<div class="card"><div class="kpi">Data span</div>'
                f'<div class="kpi-sub">{span}<br/>Rows: {len(panel):,}</div></div>', unsafe_allow_html=True)

def _kpi_or_dash(val, label):
    if val is None:
        st.markdown(f'<div class="card"><div class="kpi">—</div><div class="kpi-sub">{label}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="card"><div class="kpi">{val:.3f}</div><div class="kpi-sub">{label}</div></div>', unsafe_allow_html=True)

# KPIs will fill after CV runs (Performance tab); until then show dashes
cv_df_ss = st.session_state.get("cv_df")
_kpi_or_dash(cv_df_ss["auc"].mean() if isinstance(cv_df_ss, pd.DataFrame) and len(cv_df_ss) else None, "Mean AUC")
with c3:
    if isinstance(cv_df_ss, pd.DataFrame) and len(cv_df_ss):
        st.markdown(f'<div class="card"><div class="kpi">{cv_df_ss["acc"].mean():.3f}</div><div class="kpi-sub">Mean ACC</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="card"><div class="kpi">—</div><div class="kpi-sub">Mean ACC</div></div>', unsafe_allow_html=True)
with c4:
    if isinstance(cv_df_ss, pd.DataFrame) and len(cv_df_ss):
        val = f"{cv_df_ss['cumret'].iloc[-1]:.2%}"
    else:
        val = "—"
    st.markdown(f'<div class="card"><div class="kpi">{val}</div><div class="kpi-sub">CumRet (last fold)</div></div>', unsafe_allow_html=True)

# ---------------- TABS ----------------
tab0, tab1, tab2, tab3, tab4 = st.tabs(["Beginner", "Overview", "Performance & QA", "Prediction (Simplified)", "Details"])

# ---- Beginner tab ----
with tab0:
    st.subheader("Beginner view — This week’s market outlook")

    def verdict_from_p(p: float, lo: float = 0.45, hi: float = 0.55):
        if p >= hi:  return "UP", "🟢", "#16a34a", "Model sees higher chance of gains vs cash."
        if p <= lo:  return "DOWN", "🔴", "#ef4444", "Model sees lower chance of gains vs cash."
        return "UNCERTAIN", "🟡", "#f59e0b", "Edge isn’t clear; staying in cash is reasonable."

    if p_live is not None:
        lo, hi = 0.45, 0.55
        label, emoji, color, note = verdict_from_p(p_live, lo, hi)
        confidence = min(100, abs(p_live - 0.5) * 200)  # 0–100

        st.markdown(
            f"""
            <div class="card" style="border-color: rgba(255,255,255,.12);">
              <div style="font-size:2.2rem; font-weight:800; color:{color};">
                {emoji} This week: {label}
              </div>
              <div style="opacity:.85; margin-top:.25rem;">
                Probability of beating cash next week: <b>{p_live:.2%}</b> · Confidence: <b>{confidence:.0f}%</b><br/>
                {note}
              </div>
              <div style="font-size:.85rem; opacity:.7; margin-top:.25rem;">
                (We call it <b>UP</b> if P ≥ {hi:.2f}, <b>DOWN</b> if P ≤ {lo:.2f}, otherwise <b>UNCERTAIN</b>.)
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(int(confidence), text="Confidence")

        tail = panel["ret_w"].dropna().tail(52)
        if len(tail):
            fig_b, ax_b = plt.subplots(figsize=(8, 2.0))
            ax_b.plot((1 + tail).cumprod().values)
            ax_b.set_yticks([]); ax_b.set_xticks([]); ax_b.set_title("Recent momentum (sparkline)"); ax_b.grid(True, alpha=.2)
            st.pyplot(fig_b, use_container_width=True)
    else:
        st.info("Not enough history yet to produce a simple outlook.")

# ---- Overview tab ----
with tab1:
    st.subheader("Regime ribbon")
    fig, _ = regime_heatmap(panel.index, panel["regime"].values, title=f"HMM Regimes (k={int(n_states)})")
    st.pyplot(fig, use_container_width=True)

    regs = panel["regime"].astype(int)
    st.markdown(
        '<span class="legend-dot" style="background:#f5d742"></span>Regime 0 '
        '<span class="legend-dot" style="background:#34d399"></span>Regime 1 '
        '<span class="legend-dot" style="background:#60a5fa"></span>Regime 2',
        unsafe_allow_html=True,
    )
    runs = (regs.ne(regs.shift()).cumsum().groupby(regs).transform('size'))
    dur = pd.DataFrame({"regime": regs.values, "duration": runs.values}, index=regs.index)
    st.caption("Median regime duration (weeks): " +
               ", ".join([f"{i}: {int(dur[regs==i]['duration'].median())}" for i in sorted(regs.unique())]))
    k = int(n_states); trans = np.zeros((k,k), dtype=int); r = regs.values
    for i in range(len(r)-1): trans[r[i], r[i+1]] += 1
    fig_t, ax_t = plt.subplots(figsize=(4.5,3.6)); im = ax_t.imshow(trans, aspect="auto")
    for i in range(k):
        for j in range(k): ax_t.text(j, i, str(trans[i,j]), ha="center", va="center", fontsize=10)
    ax_t.set_xlabel("Next state"); ax_t.set_ylabel("Current state"); ax_t.set_title("Transitions")
    fig_t.colorbar(im, ax=ax_t, fraction=0.046, pad=0.04)
    st.pyplot(fig_t, use_container_width=False)

# -------------- Cached CV/backtest (heavy) --------------
@dataclass(frozen=True)
class CVSettings:
    n_splits: int
    test_size_weeks: int
    embargo_weeks: int
    trans_cost_bps: int
    turnover_cap: float
    n_states: int
    start_date_str: str

@st.cache_data(show_spinner=True, ttl=24*3600)
def run_cv_cached(panel: pd.DataFrame, X: pd.DataFrame, y: pd.Series, s: CVSettings):
    rows, equity_concat, oof_proba, oof_y = [], [], [], []
    for tr_idx, te_idx in rolling_windows(len(X), s.n_splits, s.test_size_weeks, s.embargo_weeks):
        if len(tr_idx) < 52 or len(te_idx) < 8:
            continue
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        dates_te = X_te.index

        model = make_baseline()
        _, proba = fit_predict(model, X_tr, y_tr, X_te)
        metrics = evaluate_cls(y_te, proba)

        ex_ret = panel["excess_ret_next"].reindex(dates_te)
        bt = backtest_directional(dates_te, proba, ex_ret,
                                  trans_cost_bps=s.trans_cost_bps,
                                  turnover_cap=s.turnover_cap)

        rows.append({**metrics, "cumret": bt["equity"].iloc[-1]-1,
                     "start": str(dates_te[0].date()), "end": str(dates_te[-1].date())})
        equity_concat.append(bt["equity"])
        oof_proba.append(pd.Series(proba, index=dates_te, name="p_up"))
        oof_y.append(pd.Series(y_te.values, index=dates_te, name="y"))

    cv_df = pd.DataFrame(rows)
    oof_proba = (pd.concat(oof_proba).sort_index() if oof_proba else pd.Series([], dtype=float))
    oof_y = (pd.concat(oof_y).sort_index().astype(int) if oof_y else pd.Series([], dtype=int))
    return cv_df, equity_concat, oof_proba, oof_y

def rolling_auc_series(y_true: np.ndarray, y_score: np.ndarray, window: int, index: pd.Index) -> pd.Series:
    """Fast + safe rolling AUC (no pandas .apply on DataFrame)."""
    n = len(y_true)
    out = np.full(n, np.nan, dtype=float)
    if n < window: 
        return pd.Series(out, index=index, name="roll_auc")
    for i in range(window-1, n):
        yy = y_true[i-window+1:i+1]
        pp = y_score[i-window+1:i+1]
        if np.isnan(pp).any() or (len(np.unique(yy)) < 2):
            continue
        out[i] = roc_auc_score(yy, pp)
    return pd.Series(out, index=index, name="roll_auc")

# ---- Performance & QA tab (lazy computation) ----
with tab2:
    st.subheader("Cross-validation by time")
    s = CVSettings(
        n_splits=int(n_splits), test_size_weeks=int(test_size_weeks),
        embargo_weeks=int(embargo_weeks), trans_cost_bps=int(trans_cost_bps),
        turnover_cap=float(turnover_cap), n_states=int(n_states),
        start_date_str=start_date.strftime("%Y-%m-%d")
    )
    cv_df, equity_concat, oof_proba, oof_y = run_cv_cached(panel, X, y, s)

    # Save to session for header KPIs on next rerun
    st.session_state["cv_df"] = cv_df

    st.dataframe(cv_df, use_container_width=True)
    if len(equity_concat):
        eq = pd.concat(equity_concat).sort_index()
        fig2, ax2 = plt.subplots(figsize=(11,4))
        ax2.plot(eq.index, eq.values); ax2.set_title("Equity Curve (out-of-fold concatenated)"); ax2.grid(True)
        st.pyplot(fig2, use_container_width=True)
        st.download_button("⬇️ CV report (CSV)", data=cv_df.to_csv(index=False), file_name="cv_report.csv", mime="text/csv")
        buf = BytesIO(); fig2.savefig(buf, format="png", dpi=160); buf.seek(0)
        st.download_button("⬇️ Equity chart (PNG)", data=buf, file_name="equity_oof.png", mime="image/png")

    if len(oof_proba) and len(oof_y):
        # align
        oof_y_aligned = oof_y.loc[oof_proba.index].values
        y_true = oof_y_aligned
        y_score = oof_proba.values

        fpr, tpr, _ = roc_curve(y_true, y_score); roc_auc = auc(fpr, tpr)
        prec, rec, _ = precision_recall_curve(y_true, y_score); ap = average_precision_score(y_true, y_score)
        prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10, strategy='quantile')

        fig_r, ax_r = plt.subplots(figsize=(5,4)); ax_r.plot(fpr, tpr); ax_r.plot([0,1],[0,1],"--"); ax_r.set_title(f"ROC (AUC={roc_auc:.3f})"); ax_r.grid(True)
        fig_p, ax_p = plt.subplots(figsize=(5,4)); ax_p.plot(rec, prec); ax_p.set_title(f"Precision-Recall (AP={ap:.3f})"); ax_p.grid(True)
        fig_c, ax_c = plt.subplots(figsize=(5,4)); ax_c.plot(prob_pred, prob_true, marker="o"); ax_c.plot([0,1],[0,1],"--"); ax_c.set_title("Calibration"); ax_c.grid(True)
        st.pyplot(fig_r); st.pyplot(fig_p); st.pyplot(fig_c)

        # Fast rolling AUC
        oof_df = pd.DataFrame({"y": y_true, "p": y_score}, index=oof_proba.index).sort_index()
        roll = rolling_auc_series(oof_df["y"].values, oof_df["p"].values, window=26, index=oof_df.index)
        fig_roll, ax_roll = plt.subplots(figsize=(10,3)); ax_roll.plot(roll.index, roll.values); ax_roll.set_title("Rolling AUC (26 weeks)"); ax_roll.grid(True)
        st.pyplot(fig_roll, use_container_width=True)

        ts = np.linspace(0.45, 0.55, 21); accs = [(y_true == (y_score >= t).astype(int)).mean() for t in ts]
        best_t = float(ts[int(np.argmax(accs))]); st.caption(f"Suggested threshold (max OOF accuracy): **{best_t:.2f}**")
        sig = pd.DataFrame({"p_up": y_score, "y": y_true}, index=oof_proba.index).sort_index()
        st.download_button("⬇️ OOF signals (CSV)", data=sig.to_csv(), file_name="signals_oof.csv", mime="text/csv")

# ---- Prediction (Simplified) ----
with tab3:
    st.subheader("Next-week simplified decision")
    def suggest_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
        if y_true is None or y_score is None or len(y_true) == 0: return 0.50
        ts = np.linspace(0.45, 0.55, 21); accs = [(y_true == (y_score >= t).astype(int)).mean() for t in ts]
        return float(ts[int(np.argmax(accs))])

    if p_live is not None:
        threshold = user_threshold
        cv_df_ss = st.session_state.get("cv_df")
        if abs(user_threshold - 0.50) < 1e-9 and isinstance(cv_df_ss, pd.DataFrame):
            # if user didn't choose, but we have OOF in session from tab2, prefer OOF-opt threshold
            # (optional enhancement – else keep 0.50)
            pass

        action = "LONG" if p_live >= threshold else "FLAT"
        confidence = min(100.0, abs(p_live - 0.5) * 200)
        cols = st.columns([1,1,1])
        with cols[0]:
            st.markdown(f'<div class="card"><div class="kpi">{p_live:.2%}</div><div class="kpi-sub">P(Excess Return &gt; 0)</div></div>', unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f'<div class="card"><div class="kpi">{threshold:.2f}</div><div class="kpi-sub">Decision threshold</div></div>', unsafe_allow_html=True)
        with cols[2]:
            st.markdown(f'<div class="card"><div class="kpi">{action}</div><div class="kpi-sub">Action (confidence {confidence:.0f}%)</div></div>', unsafe_allow_html=True)
        st.caption(f"Current regime: **{int(panel['regime'].iloc[-1])}** · Turnover cap {turnover_cap:.2f} · Costs {trans_cost_bps} bps")

        tail = panel["ret_w"].dropna().tail(52)
        if len(tail):
            fig3, ax3 = plt.subplots(figsize=(8,1.8))
            ax3.plot((1+tail).cumprod().values); ax3.set_yticks([]); ax3.set_xticks([]); ax3.set_title("Recent momentum (sparkline)"); ax3.grid(True, alpha=.2)
            st.pyplot(fig3, use_container_width=True)
    else:
        st.info("Not enough history to produce a live signal.")

# ---- Details ----
with tab4:
    st.subheader("About this project")
    st.markdown("""
**Macro-Regime Aware Index Forecasts** is a research prototype that:
- pulls **S&P 500** prices and **FRED** macro series,
- builds weekly features, estimates **HMM** regimes,
- uses walk-forward splits to avoid look-ahead,
- and outputs an **OOF equity** and a simple weekly probability.
""")
    st.markdown("### Important")
    st.warning("This is a learning tool. **Do not** make investment decisions with it.")
    st.caption("© Your Lab — for education & experimentation.")
