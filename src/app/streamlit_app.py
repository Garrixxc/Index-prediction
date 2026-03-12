# src/app/streamlit_app.py
# Professional Macro-Regime Quantitative Research Terminal
from __future__ import annotations
import os, sys
from io import BytesIO
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score
from sklearn.calibration import calibration_curve

from src.data.fetch import get_prices_cached, get_fred_series_cached
from src.features.engineer import compute_weekly_returns, to_weekly_last, assemble_panel
from src.models.regimes import fit_hmm, align_regimes
from src.models.baseline import make_baseline, FEATURE_SET, fit_predict, evaluate_cls, get_feature_importance
from src.utils.cv import rolling_windows
from src.evaluate.backtest import backtest_directional, compute_performance_metrics, drawdown_series

# ─────────────────────────────────────────────────────────────────────────────
# THEME CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
BG       = "#050a0f"
CARD_BG  = "#0d1117"
BORDER   = "rgba(0,212,170,0.18)"
CYAN     = "#00d4aa"
PURPLE   = "#7c3aed"
AMBER    = "#f59e0b"
RED      = "#ef4444"
GREEN    = "#22c55e"
BLUE     = "#3b82f6"
TEXT     = "#e2e8f0"
MUTED    = "#64748b"

REGIME_COLORS = ["#f59e0b", "#22c55e", "#3b82f6", "#a855f7"]
REGIME_LABELS = {0: "Bear / Stress", 1: "Bull / Risk-On", 2: "Transition", 3: "High-Vol"}

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & CSS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Quant Terminal — S&P 500", page_icon="📈", layout="wide")
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {{
  font-family: 'Inter', sans-serif;
  background-color: {BG};
  color: {TEXT};
}}
.main .block-container {{ padding-top: 0.8rem; padding-bottom: 2rem; max-width: 1440px; }}
h1,h2,h3 {{ letter-spacing:.3px; }}

.title-block {{
  background: linear-gradient(135deg, {CARD_BG} 0%, #0f1e2e 100%);
  border: 1px solid {BORDER};
  border-radius: 16px; padding: 22px 28px 18px;
  margin-bottom: 1.2rem;
}}
.title-main {{
  font-size: 1.85rem; font-weight: 900; line-height: 1.2;
  background: linear-gradient(90deg, {CYAN} 0%, {BLUE} 50%, {PURPLE} 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}}
.title-sub {{ font-size: .85rem; color: {MUTED}; margin-top: 4px; font-family: 'JetBrains Mono', monospace; }}

.kpi-card {{
  background: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 14px;
  padding: 16px 20px; min-height: 88px;
  box-shadow: 0 0 18px rgba(0,212,170,0.06);
  transition: box-shadow .2s;
}}
.kpi-card:hover {{ box-shadow: 0 0 28px rgba(0,212,170,0.15); }}
.kpi-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 1.2px; color: {MUTED}; margin-bottom: 6px; }}
.kpi-value {{ font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; font-weight: 700; line-height:1; }}
.kpi-sub   {{ font-size: 11px; color: {MUTED}; margin-top:4px; }}

.signal-badge {{
  display: inline-block; padding: 6px 18px; border-radius: 8px;
  font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 1.1rem;
  letter-spacing: 1.5px;
}}
.badge-long  {{ background: rgba(34,197,94,.18);  color: {GREEN}; border: 1px solid {GREEN}; }}
.badge-flat  {{ background: rgba(239,68,68,.18);   color: {RED};   border: 1px solid {RED}; }}
.badge-unc   {{ background: rgba(245,158,11,.18);  color: {AMBER}; border: 1px solid {AMBER}; }}

.regime-pill {{
  display: inline-block; padding: 3px 12px; border-radius: 20px;
  font-size: 12px; font-weight: 600; margin-right: 6px;
}}

.info-card {{
  background: {CARD_BG}; border: 1px solid rgba(255,255,255,.07);
  border-radius: 12px; padding: 18px 22px; margin-bottom: 16px;
}}
.math-block {{
  background: #060d14; border-left: 3px solid {CYAN}; border-radius: 4px;
  padding: 12px 18px; font-family: 'JetBrains Mono', monospace; font-size: .82rem;
  color: {CYAN}; margin: 12px 0;
}}

[data-testid="stSidebar"] {{
  background: linear-gradient(180deg,#070e1a 0%,#0b1320 100%);
  border-right: 1px solid rgba(0,212,170,.12);
}}
[data-testid="stSidebar"] * {{ color: {TEXT} !important; }}
.sidebar-logo {{
  font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 1.05rem;
  background: linear-gradient(90deg,{CYAN} 0%,{PURPLE} 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  margin-bottom: 1rem;
}}
.stButton>button {{ border-radius: 8px; font-weight: 600; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DARK MATPLOTLIB THEME
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": CARD_BG, "axes.facecolor": "#0a1220",
    "axes.edgecolor": "#1e293b", "axes.labelcolor": TEXT,
    "text.color": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
    "grid.color": "#1e293b", "grid.linewidth": 0.6,
    "font.family": "sans-serif", "font.size": 10,
    "legend.facecolor": CARD_BG, "legend.edgecolor": "#1e293b",
})

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">⬡ QUANT TERMINAL v2.0</div>', unsafe_allow_html=True)
    st.caption("Macro-Regime S&P 500 Research System")
    st.divider()

    st.markdown("**📅 Data**")
    start_date = st.date_input("Backtest start", value=pd.to_datetime("1999-01-01"))

    st.markdown("**🏦 Model**")
    model_type = st.selectbox("Model", ["logistic", "xgboost"], index=0,
                              help="logistic = fast & interpretable; xgboost = requires pip install xgboost")
    n_states = st.selectbox("HMM regimes (k)", [2, 3, 4], index=1)

    st.markdown("**🔁 Walk-Forward CV**")
    n_splits = st.number_input("CV folds", 2, 10, 6)
    test_size_weeks = st.number_input("Test size (weeks/fold)", 52, 260, 104, step=52)
    embargo_weeks = st.number_input("Embargo (weeks)", 0, 12, 4)

    st.markdown("**💰 Execution**")
    trans_cost_bps = st.number_input("Transaction cost (bps)", 0, 100, 5)
    turnover_cap = st.slider("Turnover cap", 0.0, 1.0, 1.0, 0.05)
    user_threshold = st.slider("Decision threshold", 0.40, 0.60, 0.50, 0.01,
                               help="LONG if P(Up) ≥ threshold")

    st.divider()
    refresh = st.button("🔄 Refresh Data (bust cache)", use_container_width=True)
    st.caption("Data: Yahoo Finance + FRED. Not investment advice.")

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
DATA_CACHE = "data_cache"
os.makedirs(DATA_CACHE, exist_ok=True)

@st.cache_data(ttl=3600, show_spinner="⏳ Loading market data…")
def load_panel(start_date_str: str) -> pd.DataFrame:
    prices = get_prices_cached(symbol="^GSPC", start=start_date_str, cache_path=f"{DATA_CACHE}/prices.csv")
    fred_list = ["DGS2","DGS10","T10Y2Y","CPIAUCSL","UNRATE","INDPRO","FEDFUNDS","T5YIFR","BAA10Y","TB3MS"]
    fred = get_fred_series_cached(fred_list, start="1960-01-01", cache_path=f"{DATA_CACHE}/fred.csv")
    pw = compute_weekly_returns(prices)
    fw = to_weekly_last(fred).interpolate(limit_direction="both")
    panel = assemble_panel(pw, fw).dropna()
    return panel[panel.index >= pd.to_datetime(start_date_str)]

if refresh:
    load_panel.clear()
    st.toast("Cache cleared — fetching latest data.", icon="🔄")

panel = load_panel(start_date.strftime("%Y-%m-%d"))

# ─────────────────────────────────────────────────────────────────────────────
# REGIMES & FEATURES
# ─────────────────────────────────────────────────────────────────────────────
hmm_model, regimes = fit_hmm(panel, n_states=int(n_states), covariance_type="full", feature_cols=("ret_w","rv_w"))
panel["regime"] = align_regimes(panel.index, regimes)

feature_list = [c for c in FEATURE_SET if c in panel.columns]
X = panel[feature_list].ffill().dropna()
y = (panel["excess_ret_next"].reindex(X.index) > 0).astype(int)

# ─────────────────────────────────────────────────────────────────────────────
# LIVE PREDICTION (fast, fit on all-but-last)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def live_pred(start_date_str: str, mtype: str, n_states: int):
    panel = load_panel(start_date_str)
    hmm_model, regimes = fit_hmm(panel, n_states=int(n_states), covariance_type="full", feature_cols=("ret_w","rv_w"))
    panel["regime"] = align_regimes(panel.index, regimes)
    feature_list = [c for c in FEATURE_SET if c in panel.columns]
    X = panel[feature_list].ffill().dropna()
    y = (panel["excess_ret_next"].reindex(X.index) > 0).astype(int)

    if len(X) <= 60:
        return None, None
    mdl = make_baseline(mtype)
    mdl.fit(X.iloc[:-1], y.iloc[:-1])
    p = float(mdl.predict_proba(X.iloc[[-1]])[:, 1][0])
    fi = get_feature_importance(mdl, feature_list)
    return p, fi

p_live, feat_imp_live = live_pred(start_date.strftime("%Y-%m-%d"), model_type, n_states)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
cv_ss = st.session_state.get("cv_perf")
cur_regime = int(panel["regime"].iloc[-1])
regime_label = REGIME_LABELS.get(cur_regime, f"Regime {cur_regime}")
regime_color = REGIME_COLORS[cur_regime % len(REGIME_COLORS)]

st.markdown(f"""
<div class="title-block">
  <div class="title-main">📈 Macro-Regime Quantitative Research Terminal</div>
  <div class="title-sub">S&amp;P 500 · Walk-Forward · HMM Regimes · {len(panel):,} weeks · Updated {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC</div>
</div>
""", unsafe_allow_html=True)

# KPI row
def _kpi(label, value, sub="", color=TEXT):
    return f"""<div class="kpi-card">
  <div class="kpi-label">{label}</div>
  <div class="kpi-value" style="color:{color}">{value}</div>
  <div class="kpi-sub">{sub}</div>
</div>"""

def _fmt(v, fmt=".3f", fallback="—"):
    return format(v, fmt) if v is not None and not (isinstance(v, float) and np.isnan(v)) else fallback

cols_kpi = st.columns(5)
signal_color = GREEN if (p_live or 0) >= user_threshold else AMBER if (p_live or 0) >= 0.45 else RED
signal_txt   = "LONG" if (p_live or 0) >= user_threshold else ("UNCERT" if (p_live or 0) >= 0.45 else "FLAT")

with cols_kpi[0]:
    st.markdown(_kpi("Weekly Signal", signal_txt if p_live else "—", f"threshold {user_threshold:.2f}", signal_color), unsafe_allow_html=True)
with cols_kpi[1]:
    st.markdown(_kpi("P(Excess Ret > 0)", f"{p_live:.1%}" if p_live else "—", "live model output", CYAN), unsafe_allow_html=True)
with cols_kpi[2]:
    st.markdown(_kpi("Current Regime", f"R{cur_regime}", regime_label, regime_color), unsafe_allow_html=True)
with cols_kpi[3]:
    sh = cv_ss.get("sharpe") if cv_ss else None
    st.markdown(_kpi("OOF Sharpe", _fmt(sh), "walk-forward annualised", CYAN if sh and sh > 0 else RED), unsafe_allow_html=True)
with cols_kpi[4]:
    mdd = cv_ss.get("max_drawdown") if cv_ss else None
    st.markdown(_kpi("Max Drawdown", f"{mdd:.1%}" if mdd else "—", "out-of-fold equity", RED if mdd else MUTED), unsafe_allow_html=True)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_signal, tab_regime, tab_backtest, tab_qa, tab_factor, tab_method = st.tabs([
    "🎯 Live Signal",
    "📊 Regime Map",
    "📈 Backtest",
    "🔬 Model QA",
    "🧮 Factor Analysis",
    "📚 Methodology",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 0 — LIVE SIGNAL
# ═══════════════════════════════════════════════════════════════════════════
with tab_signal:
    if p_live is None:
        st.info("Not enough history to produce a live signal (need > 60 weeks).")
    else:
        conf = min(100.0, abs(p_live - 0.5) * 200)
        kelly_f = max(0.0, (p_live - (1 - p_live)))  # simplified Kelly: f* = p - q (b=1)
        kelly_f = min(kelly_f, 0.5)  # cap at 50%

        badge_cls = "badge-long" if signal_txt == "LONG" else ("badge-unc" if signal_txt == "UNCERT" else "badge-flat")
        st.markdown(f"""
<div class="info-card" style="border-color:{signal_color}33">
  <div style="display:flex; align-items:center; gap:18px; flex-wrap:wrap;">
    <span class="signal-badge {badge_cls}">{signal_txt}</span>
    <div>
      <div style="font-size:1.3rem; font-weight:700; color:{signal_color}">{p_live:.2%} probability of outperforming cash</div>
      <div style="color:{MUTED}; font-size:.85rem; margin-top:3px">
        Confidence: <b style="color:{TEXT}">{conf:.0f}%</b> &nbsp;|&nbsp;
        Current Regime: <span style="color:{regime_color}; font-weight:600">{regime_label}</span> &nbsp;|&nbsp;
        Decision threshold: <b style="color:{TEXT}">{user_threshold:.2f}</b>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(_kpi("P(Excess Ret > 0)", f"{p_live:.3f}", "raw model probability", CYAN), unsafe_allow_html=True)
        with c2:
            st.markdown(_kpi("Kelly Position Size", f"{kelly_f:.1%}", "f* = p − q, capped 50%", AMBER), unsafe_allow_html=True)
        with c3:
            st.markdown(_kpi("Model Confidence", f"{conf:.0f}%", "distance from 0.5 × 200", signal_color), unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.progress(int(conf), text=f"Signal Confidence: {conf:.0f}%")

        # Sparkline — last 52 weeks
        tail = panel["ret_w"].dropna().tail(52)
        eq_tail = (1 + tail).cumprod()
        fig_sp, ax_sp = plt.subplots(figsize=(10, 2.2))
        ax_sp.plot(eq_tail.index, eq_tail.values, color=CYAN, lw=1.6)
        ax_sp.fill_between(eq_tail.index, 1, eq_tail.values,
                           where=(eq_tail.values >= 1), alpha=0.15, color=GREEN)
        ax_sp.fill_between(eq_tail.index, 1, eq_tail.values,
                           where=(eq_tail.values < 1), alpha=0.15, color=RED)
        ax_sp.axhline(1, color=MUTED, lw=0.8, ls="--")
        ax_sp.set_title("S&P 500 — trailing 52-week equity (base=1)", fontsize=10)
        ax_sp.set_xlabel(""); ax_sp.grid(True, alpha=0.4)
        fig_sp.tight_layout(pad=0.4)
        st.pyplot(fig_sp, use_container_width=True)

        # Threshold sensitivity
        ts = np.linspace(0.40, 0.60, 41)
        actions = ["LONG" if p_live >= t else "FLAT" for t in ts]
        st.caption(f"**Threshold sweep:** LONG for thresholds ≤ {p_live:.3f}, FLAT for thresholds > {p_live:.3f}")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — REGIME MAP
# ═══════════════════════════════════════════════════════════════════════════
with tab_regime:
    st.subheader("Macro Regime History")

    # Price + regime band chart
    fig_rm, ax_rm = plt.subplots(figsize=(12, 4))
    close = panel["close"]
    ax_rm.plot(close.index, close.values, color=TEXT, lw=1.0, zorder=3, label="S&P 500")
    regs = panel["regime"].astype(int)
    for r in sorted(regs.unique()):
        mask = regs == r
        idx  = mask.index[mask]
        rc   = REGIME_COLORS[r % len(REGIME_COLORS)]
        # shade contiguous blocks
        in_block = False
        blk_start = None
        prev_date  = None
        for dt, val in zip(regs.index, regs.values):
            if val == r:
                if not in_block:
                    blk_start = dt; in_block = True
            else:
                if in_block:
                    ax_rm.axvspan(blk_start, prev_date, alpha=0.18, color=rc, lw=0)
                    in_block = False
            prev_date = dt
        if in_block:
            ax_rm.axvspan(blk_start, prev_date, alpha=0.18, color=rc, lw=0)

    patches = [mpatches.Patch(color=REGIME_COLORS[r % len(REGIME_COLORS)],
                               alpha=0.6, label=f"R{r}: {REGIME_LABELS.get(r, '')}")
               for r in sorted(regs.unique())]
    ax_rm.legend(handles=patches, loc="upper left", fontsize=9)
    ax_rm.set_title("S&P 500 Price with HMM Regime Bands", fontsize=11)
    ax_rm.set_ylabel("Price"); ax_rm.grid(True, alpha=0.3)
    fig_rm.tight_layout(pad=0.5)
    st.pyplot(fig_rm, use_container_width=True)

    # Regime stats table
    st.subheader("Regime Statistics")
    regime_stats = []
    for r in sorted(regs.unique()):
        mask = regs == r
        rret = panel.loc[mask, "ret_w"].dropna()
        regime_stats.append({
            "Regime": f"R{r} — {REGIME_LABELS.get(r, '')}",
            "# Weeks": int(mask.sum()),
            "% Time": f"{mask.mean():.1%}",
            "Ann. Return": f"{rret.mean() * 52:.1%}",
            "Ann. Vol": f"{rret.std() * np.sqrt(52):.1%}",
            "Sharpe": f"{(rret.mean() / rret.std() * np.sqrt(52)):.2f}" if rret.std() > 0 else "—",
            "Win Rate": f"{(rret > 0).mean():.1%}",
        })
    st.dataframe(pd.DataFrame(regime_stats).set_index("Regime"), use_container_width=True)

    # Transition matrix
    k = int(n_states)
    trans = np.zeros((k, k), dtype=int)
    rv = regs.values
    for i in range(len(rv) - 1):
        trans[rv[i], rv[i+1]] += 1
    trans_prob = trans / trans.sum(axis=1, keepdims=True).clip(1)

    st.subheader("HMM Transition Probability Matrix")
    fig_tm, ax_tm = plt.subplots(figsize=(5, 4))
    im = ax_tm.imshow(trans_prob, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
    for i in range(k):
        for j in range(k):
            ax_tm.text(j, i, f"{trans_prob[i,j]:.2f}", ha="center", va="center",
                      fontsize=11, fontweight="bold",
                      color="black" if trans_prob[i,j] > 0.5 else TEXT)
    ax_tm.set_xticks(range(k)); ax_tm.set_yticks(range(k))
    ax_tm.set_xticklabels([f"→ R{i}" for i in range(k)])
    ax_tm.set_yticklabels([f"R{i}" for i in range(k)])
    ax_tm.set_title("Transition Probabilities P(next | current)")
    fig_tm.colorbar(im, ax=ax_tm, fraction=0.046, pad=0.04)
    fig_tm.tight_layout(pad=0.5)
    col_tm, _ = st.columns([1, 1])
    with col_tm:
        st.pyplot(fig_tm)


# ═══════════════════════════════════════════════════════════════════════════
# CV / BACKTEST HELPER (cached)
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class CVSettings:
    n_splits: int; test_size_weeks: int; embargo_weeks: int
    trans_cost_bps: int; turnover_cap: float; n_states: int
    start_date_str: str; model_type: str

@st.cache_data(show_spinner="⏳ Running walk-forward cross-validation…", ttl=24*3600)
def run_cv_cached(s: CVSettings):
    panel = load_panel(s.start_date_str)
    hmm_model, regimes = fit_hmm(panel, n_states=int(s.n_states), covariance_type="full", feature_cols=("ret_w","rv_w"))
    panel["regime"] = align_regimes(panel.index, regimes)
    feature_list = [c for c in FEATURE_SET if c in panel.columns]
    X = panel[feature_list].ffill().dropna()
    y = (panel["excess_ret_next"].reindex(X.index) > 0).astype(int)

    rows, equity_list, oof_p, oof_y = [], [], [], []
    bnh_rets = []
    for tr_idx, te_idx in rolling_windows(len(X), s.n_splits, s.test_size_weeks, s.embargo_weeks):
        if len(tr_idx) < 52 or len(te_idx) < 8:
            continue
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        dates_te   = X_te.index
        mdl = make_baseline(s.model_type)
        _, proba = fit_predict(mdl, X_tr, y_tr, X_te)
        metrics  = evaluate_cls(y_te, proba)
        ex_ret   = panel["excess_ret_next"].reindex(dates_te)
        bt = backtest_directional(dates_te, proba, ex_ret, s.trans_cost_bps, s.turnover_cap)
        perf = compute_performance_metrics(bt["equity"], bt["strat_ret"],
                                           benchmark_ret=ex_ret)
        cumret = float(bt["equity"].iloc[-1] - 1)
        rows.append({**metrics, **{k: v for k,v in perf.items()},
                     "cumret": cumret,
                     "start": str(dates_te[0].date()), "end": str(dates_te[-1].date())})
        equity_list.append(bt["equity"])
        bnh_rets.append(ex_ret)
        oof_p.append(pd.Series(proba, index=dates_te, name="p_up"))
        oof_y.append(pd.Series(y_te.values, index=dates_te, name="y"))

    cv_df   = pd.DataFrame(rows)
    eq_full = pd.concat(equity_list).sort_index() if equity_list else pd.Series(dtype=float)
    bnh_full= pd.concat(bnh_rets).sort_index() if bnh_rets else pd.Series(dtype=float)
    oof_p_s = pd.concat(oof_p).sort_index() if oof_p else pd.Series(dtype=float)
    oof_y_s = pd.concat(oof_y).sort_index().astype(int) if oof_y else pd.Series(dtype=int)
    return cv_df, eq_full, bnh_full, oof_p_s, oof_y_s

# Run CV (loaded into all tabs that need it)
s = CVSettings(int(n_splits), int(test_size_weeks), int(embargo_weeks),
               int(trans_cost_bps), float(turnover_cap), int(n_states),
               start_date.strftime("%Y-%m-%d"), model_type)

cv_df, eq_full, bnh_full, oof_p_s, oof_y_s = run_cv_cached(s)

# Aggregate perf for header KPIs
if len(cv_df):
    agg_eq  = eq_full
    agg_ret = agg_eq.pct_change().dropna()
    agg_perf = compute_performance_metrics(agg_eq, agg_ret, benchmark_ret=bnh_full)
    st.session_state["cv_perf"] = agg_perf

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — BACKTEST
# ═══════════════════════════════════════════════════════════════════════════
with tab_backtest:
    st.subheader("Walk-Forward Backtest — Out-of-Fold Equity")
    if not len(cv_df):
        st.warning("Not enough data for walk-forward CV with current settings.")
    else:
        # Equity + B&H + Drawdown
        bnh_eq = (1 + bnh_full).cumprod()
        dd     = drawdown_series(eq_full)

        fig_bt, (ax_e, ax_d) = plt.subplots(2, 1, figsize=(12, 6),
                                             gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
        ax_e.plot(eq_full.index, eq_full.values, color=CYAN, lw=1.8, label="Strategy (OOF)")
        ax_e.plot(bnh_eq.index, bnh_eq.values, color=MUTED, lw=1.2, ls="--", label="Buy & Hold (excess)")
        ax_e.axhline(1, color=MUTED, lw=0.7, ls=":")
        ax_e.set_ylabel("Equity"); ax_e.legend(fontsize=9); ax_e.grid(True, alpha=0.35)
        ax_e.set_title("Out-of-Fold Equity Curve vs Buy & Hold", fontsize=11)

        ax_d.fill_between(dd.index, dd.values, 0, color=RED, alpha=0.55)
        ax_d.set_ylabel("Drawdown"); ax_d.grid(True, alpha=0.35)
        ax_d.set_title("Drawdown", fontsize=9)
        fig_bt.tight_layout(pad=0.5)
        st.pyplot(fig_bt, use_container_width=True)

        # Performance summary
        if agg_perf:
            pm = agg_perf
            perf_cols = st.columns(5)
            metrics_display = [
                ("Ann. Return",   f"{pm['ann_return']:.2%}",   pm['ann_return'] > 0),
                ("Ann. Vol",      f"{pm['ann_vol']:.2%}",      None),
                ("Sharpe Ratio",  f"{pm['sharpe']:.3f}",       pm['sharpe'] > 0 if pm['sharpe'] else False),
                ("Sortino",       f"{pm['sortino']:.3f}",      pm['sortino'] > 0 if pm['sortino'] else False),
                ("Max Drawdown",  f"{pm['max_drawdown']:.1%}", None),
            ]
            for col, (lbl, val, pos) in zip(perf_cols, metrics_display):
                color = (GREEN if pos else RED) if pos is not None else CYAN
                with col:
                    st.markdown(_kpi(lbl, val, "annualised (OOF)", color), unsafe_allow_html=True)

            more_cols = st.columns(5)
            metrics2 = [
                ("Calmar",        f"{_fmt(pm['calmar'])}",      pm['calmar'] > 0 if pm['calmar'] else False),
                ("Win Rate",      f"{pm['win_rate']:.1%}",      pm['win_rate'] > 0.5),
                ("Info. Ratio",   f"{_fmt(pm['information_ratio'])}",  None),
                ("Alpha vs B&H",  f"{_fmt(pm['alpha_vs_bnh'], '.2%')}", pm['alpha_vs_bnh'] > 0 if pm['alpha_vs_bnh'] else False),
                ("Total Return",  f"{pm['total_return']:.1%}",  pm['total_return'] > 0),
            ]
            for col, (lbl, val, pos) in zip(more_cols, metrics2):
                color = (GREEN if pos else RED) if pos is not None else CYAN
                with col:
                    st.markdown(_kpi(lbl, val, "OOF aggregate", color), unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)
        st.subheader("Per-Fold Summary")
        display_cols = ["start","end","auc","acc","sharpe","sortino","max_drawdown","win_rate","cumret"]
        display_cols = [c for c in display_cols if c in cv_df.columns]
        st.dataframe(cv_df[display_cols].style.format({
            "auc": "{:.3f}", "acc": "{:.3f}", "sharpe": "{:.3f}",
            "sortino": "{:.3f}", "max_drawdown": "{:.1%}",
            "win_rate": "{:.1%}", "cumret": "{:.1%}",
        }), use_container_width=True)

        buf = BytesIO(); eq_full.to_frame("equity").join(bnh_full.rename("bnh")).to_csv(buf)
        buf.seek(0)
        st.download_button("⬇️ Download equity CSV", buf, "equity_oof.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL QA
# ═══════════════════════════════════════════════════════════════════════════
with tab_qa:
    st.subheader("Model Quality Assurance — Out-of-Fold Diagnostics")
    if len(oof_p_s) and len(oof_y_s):
        oof_y_aligned = oof_y_s.reindex(oof_p_s.index).dropna().astype(int)
        oof_p_aligned = oof_p_s.reindex(oof_y_aligned.index)
        yt = oof_y_aligned.values; ys = oof_p_aligned.values

        fpr, tpr, _ = roc_curve(yt, ys); roc_auc_v = auc(fpr, tpr)
        prec, rec, _ = precision_recall_curve(yt, ys); ap = average_precision_score(yt, ys)
        prob_true, prob_pred = calibration_curve(yt, ys, n_bins=10, strategy="quantile")

        c1, c2, c3 = st.columns(3)
        with c1:
            fig_r, ax_r = plt.subplots(figsize=(5, 4))
            ax_r.plot(fpr, tpr, color=CYAN, lw=2, label=f"AUC = {roc_auc_v:.3f}")
            ax_r.plot([0,1],[0,1], color=MUTED, ls="--", lw=0.8)
            ax_r.fill_between(fpr, tpr, alpha=0.12, color=CYAN)
            ax_r.set_xlabel("FPR"); ax_r.set_ylabel("TPR")
            ax_r.set_title(f"ROC Curve (AUC = {roc_auc_v:.3f})", fontsize=10)
            ax_r.legend(fontsize=9); ax_r.grid(True, alpha=0.35)
            fig_r.tight_layout(pad=0.4); st.pyplot(fig_r)

        with c2:
            fig_p, ax_p = plt.subplots(figsize=(5, 4))
            ax_p.plot(rec, prec, color=AMBER, lw=2, label=f"AP = {ap:.3f}")
            ax_p.fill_between(rec, prec, alpha=0.12, color=AMBER)
            ax_p.set_xlabel("Recall"); ax_p.set_ylabel("Precision")
            ax_p.set_title(f"Precision-Recall (AP = {ap:.3f})", fontsize=10)
            ax_p.legend(fontsize=9); ax_p.grid(True, alpha=0.35)
            fig_p.tight_layout(pad=0.4); st.pyplot(fig_p)

        with c3:
            fig_c, ax_c = plt.subplots(figsize=(5, 4))
            ax_c.plot(prob_pred, prob_true, color=PURPLE, lw=2, marker="o", ms=5, label="Model")
            ax_c.plot([0,1],[0,1], color=MUTED, ls="--", lw=0.8, label="Perfect")
            ax_c.set_xlabel("Mean predicted probability")
            ax_c.set_ylabel("Fraction of positives")
            ax_c.set_title("Calibration Curve", fontsize=10)
            ax_c.legend(fontsize=9); ax_c.grid(True, alpha=0.35)
            fig_c.tight_layout(pad=0.4); st.pyplot(fig_c)

        # Rolling AUC
        st.markdown("<br/>", unsafe_allow_html=True)
        oof_df = pd.DataFrame({"y": yt, "p": ys}, index=oof_p_aligned.index).sort_index()
        n = len(oof_df); win = 26; out = np.full(n, np.nan)
        for i in range(win-1, n):
            yy = oof_df["y"].values[i-win+1:i+1]; pp = oof_df["p"].values[i-win+1:i+1]
            if not np.isnan(pp).any() and len(np.unique(yy)) == 2:
                out[i] = roc_auc_score(yy, pp)
        roll_auc = pd.Series(out, index=oof_df.index, name="roll_auc")

        fig_ra, ax_ra = plt.subplots(figsize=(12, 3))
        ax_ra.plot(roll_auc.index, roll_auc.values, color=CYAN, lw=1.5)
        ax_ra.axhline(0.5, color=MUTED, ls="--", lw=0.8)
        ax_ra.fill_between(roll_auc.index, 0.5, roll_auc.values,
                            where=(roll_auc.values >= 0.5), alpha=0.15, color=GREEN)
        ax_ra.fill_between(roll_auc.index, 0.5, roll_auc.values,
                            where=(roll_auc.values < 0.5), alpha=0.15, color=RED)
        ax_ra.set_title("Rolling AUC (26-week window)", fontsize=11)
        ax_ra.set_ylim(0.3, 0.8); ax_ra.grid(True, alpha=0.35)
        fig_ra.tight_layout(pad=0.4)
        st.pyplot(fig_ra, use_container_width=True)

        # Threshold sweep
        ts = np.linspace(0.40, 0.60, 41)
        accs = [(yt == (ys >= t).astype(int)).mean() for t in ts]
        best_t = float(ts[int(np.argmax(accs))])
        fig_th, ax_th = plt.subplots(figsize=(8, 2.5))
        ax_th.plot(ts, accs, color=PURPLE, lw=2)
        ax_th.axvline(best_t, color=AMBER, ls="--", lw=1.2, label=f"Best t={best_t:.2f}")
        ax_th.axvline(0.5, color=MUTED, ls=":", lw=0.8)
        ax_th.set_xlabel("Decision threshold"); ax_th.set_ylabel("OOF Accuracy")
        ax_th.set_title(f"Threshold Sensitivity (OOF Accuracy) — Best: {best_t:.2f}", fontsize=10)
        ax_th.legend(fontsize=9); ax_th.grid(True, alpha=0.35)
        fig_th.tight_layout(pad=0.4)
        st.pyplot(fig_th, use_container_width=True)

        sig = pd.DataFrame({"p_up": ys, "y": yt}, index=oof_p_aligned.index).sort_index()
        st.download_button("⬇️ Download OOF signals CSV", sig.to_csv(), "signals_oof.csv", "text/csv")
    else:
        st.warning("Run CV first (Performance tab) to see QA metrics.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — FACTOR ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
with tab_factor:
    st.subheader("Feature Importance & Factor Analysis")

    if feat_imp_live is not None and len(feat_imp_live):
        # Importance bar chart
        top_n = feat_imp_live.head(15)
        colors_bar = [CYAN if v > top_n.median() else PURPLE for v in top_n.values]

        fig_fi, ax_fi = plt.subplots(figsize=(9, 5))
        bars = ax_fi.barh(top_n.index[::-1], top_n.values[::-1], color=colors_bar[::-1], edgecolor="none")
        ax_fi.set_xlabel("Absolute Coefficient / Feature Importance")
        ax_fi.set_title("Top Feature Importance (Live Model — Trained on All-but-Last Week)", fontsize=10)
        ax_fi.grid(True, alpha=0.3, axis="x")
        for bar in bars:
            ax_fi.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height()/2,
                      f"{bar.get_width():.4f}", va="center", fontsize=8, color=MUTED)
        fig_fi.tight_layout(pad=0.5)
        st.pyplot(fig_fi, use_container_width=True)

    # Correlation heatmap of macro features
    st.subheader("Macro Feature Correlation Matrix")
    corr_cols = [c for c in feature_list if c != "regime"]
    if len(corr_cols) >= 2:
        corr = X[corr_cols].corr()
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        cax = ax_corr.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax_corr.set_xticks(range(len(corr_cols))); ax_corr.set_yticks(range(len(corr_cols)))
        ax_corr.set_xticklabels(corr_cols, rotation=45, ha="right", fontsize=8)
        ax_corr.set_yticklabels(corr_cols, fontsize=8)
        ax_corr.set_title("Pairwise Feature Correlations", fontsize=11)
        for i in range(len(corr_cols)):
            for j in range(len(corr_cols)):
                v = corr.values[i, j]
                ax_corr.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=6, color="white" if abs(v) > 0.6 else TEXT)
        fig_corr.colorbar(cax, ax=ax_corr, fraction=0.046, pad=0.04)
        fig_corr.tight_layout(pad=0.5)
        st.pyplot(fig_corr, use_container_width=True)

    # Regime-conditional return distributions
    st.subheader("Regime-Conditional Return Distribution")
    fig_rd, ax_rd = plt.subplots(figsize=(10, 3.5))
    for r in sorted(regs.unique()):
        mask = regs == r
        rret = panel.loc[mask, "ret_w"].dropna()
        ax_rd.hist(rret, bins=50, density=True, alpha=0.55,
                  color=REGIME_COLORS[r % len(REGIME_COLORS)],
                  label=f"R{r}: {REGIME_LABELS.get(r,'')}", edgecolor="none")
    ax_rd.axvline(0, color=MUTED, ls="--", lw=0.8)
    ax_rd.set_xlabel("Weekly Return"); ax_rd.set_ylabel("Density")
    ax_rd.set_title("Return Distributions by Regime")
    ax_rd.legend(fontsize=9); ax_rd.grid(True, alpha=0.3)
    fig_rd.tight_layout(pad=0.5)
    st.pyplot(fig_rd, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — METHODOLOGY
# ═══════════════════════════════════════════════════════════════════════════
with tab_method:
    st.subheader("Methodology & Mathematical Framework")

    st.markdown("""
<div class="info-card">
<h4 style="color:#00d4aa; margin-top:0">Overview</h4>
<p>This system is a <b>macro-regime-aware binary classifier</b> for weekly S&P 500 excess returns over the
3-Month T-Bill. It combines Hidden Markov Models (HMM) for latent market state detection with a regularised
logistic regression (or XGBoost) classifier trained on macroeconomic and technical features. Walk-forward
cross-validation with an embargo ensures <b>strictly no look-ahead bias</b>.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("### 1. Hidden Markov Model — Regime Detection")
    st.markdown("""
The HMM models a latent state sequence $z_t \\in \\{0, 1, \\ldots, K-1\\}$ from observed weekly returns and
realised volatility $\\mathbf{x}_t = (r_t^w,\\, \\text{RV}_t^w)$.

**Emission:**  $p(\\mathbf{x}_t \\mid z_t = k) = \\mathcal{N}(\\boldsymbol{\\mu}_k, \\boldsymbol{\\Sigma}_k)$

**Transition:** $P(z_t = j \\mid z_{t-1} = i) = A_{ij}$

Parameters are estimated via the **Baum-Welch (EM) algorithm**:
- **E-step:** Compute forward–backward probabilities (responsibilities)
- **M-step:** Update $\\boldsymbol{\\mu}_k$, $\\boldsymbol{\\Sigma}_k$, $A$ in closed form

Regimes are aligned by expected return: Regime 1 = highest mean return (Bull).
""")
    st.markdown('<div class="math-block">z* = argmax_k P(z_t = k | x_1,...,x_T)  [Viterbi decoding]</div>', unsafe_allow_html=True)

    st.markdown("### 2. Feature Engineering")
    st.markdown("""
| Feature | Formula | Economic Rationale |
|---|---|---|
| `term_spread` | DGS10 − DGS2 | Yield curve inversion predicts recessions |
| `t10y2y` | FRED T10Y2Y | 10Y−2Y spread (preferred recession indicator) |
| `fedfunds` | EFFR | Monetary policy stance |
| `cred_spread` | BAA − 10Y | Credit risk premium, risk-off indicator |
| `cpi_yoy` | CPI YoY % | Inflation regime |
| `indpro_yoy` | IP YoY % | Real economy momentum |
| `rsi_14` | Wilder RSI, 14w | Momentum / overbought-oversold |
| `mom_4w` | Σ r_{t-3:t} | Short-term price momentum |
| `mom_12w` | Σ r_{t-11:t} | Medium-term trend |
| `price_sma52` | P/SMA(52)−1 | Trend deviation signal |
| `rv_zscore` | (RV−μ)/σ (52w) | Volatility regime z-score |
| `regime` | HMM state | Latent market environment |
""")

    st.markdown("### 3. Walk-Forward Cross-Validation")
    st.markdown("""
**Expanding window** with **embargo** to prevent data leakage:
- Train on $[0, t_s - \\text{embargo})$ → Test on $[t_s, t_s + T_{\\text{test}})$
- Embargo weeks create a gap between train and test to absorb autocorrelation
- $n_{\\text{splits}}$ evenly-spaced test windows from $t_{\\text{min\_train}}$ to end of data
""")
    st.markdown('<div class="math-block">T_train(s) = [0, t_s - embargo) | T_test(s) = [t_s, t_s + T_test)</div>', unsafe_allow_html=True)

    st.markdown("### 4. Performance Metrics")
    st.markdown('<div class="math-block">Sharpe = (E[r_p] - r_f) / σ_p × √52\nSortino = (E[r_p] - r_f) / σ_downside × √52\nMax Drawdown = max( (peak - trough) / peak )\nCalmar = Ann.Return / |MDD|</div>', unsafe_allow_html=True)

    st.markdown("### 5. Kelly Criterion — Position Sizing")
    st.markdown("""
For a binary bet with win probability $p$, loss probability $q = 1-p$, and net odds $b$:
""")
    st.markdown('<div class="math-block">f* = (b·p - q) / b</div>', unsafe_allow_html=True)
    st.markdown("""
In this app we use the simplified unit-odds version ($b=1$): $f^* = p - q = 2p - 1$.
This is **capped at 50%** of capital for risk management and serves as a *guide*, not a mandate.
""")

    st.markdown("### 6. Backtest Execution Model")
    st.markdown("""
- **Position:** $w_t = P(r_{t+1}^{excess} > 0)$ — continuous long/flat (no shorting)
- **Turnover cap:** $\\Delta w_t = \\text{clip}(w_t^{raw} - w_{t-1}, -\\delta, +\\delta)$
- **Transaction cost:** $TC_t = |w_t - w_{t-1}| \\times \\frac{\\text{bps}}{10000}$
- **Net return:** $r_t^{net} = w_t \\cdot r_t^{excess} - TC_t$
""")

    st.markdown("---")
    st.warning("⚠️ **Disclaimer:** This is a research and educational prototype. It does not constitute investment advice. Past simulated performance does not guarantee future results. All results are hypothetical and subject to look-ahead bias if settings are misconfigured.")
    st.caption("© Quant Research Lab — For educational & experimental use only.")
