export default function Methodology() {
  const Block = ({ children }: { children: string }) => (
    <div className="math-block">{children}</div>
  );

  return (
    <div className="fade-in space-y-6 max-w-3xl">
      <div className="card p-6">
        <h2 className="text-base font-bold mb-2" style={{ color: "var(--cyan)" }}>Overview</h2>
        <p className="text-sm text-[var(--text)] leading-relaxed">
          This system is a <strong>macro-regime-aware binary classifier</strong> for weekly S&amp;P 500 excess returns
          over the 3-Month T-Bill. It combines Hidden Markov Models (HMM) for latent market state detection
          with a regularised logistic regression (or XGBoost) classifier trained on macroeconomic and technical
          features. Walk-forward cross-validation with an embargo ensures <strong>strictly no look-ahead bias</strong>.
        </p>
      </div>

      <div className="card p-6 space-y-4">
        <h3 className="font-bold text-sm text-[var(--text)]">1. Hidden Markov Model — Regime Detection</h3>
        <p className="text-sm text-[var(--muted)] leading-relaxed">
          The HMM models a latent state sequence z_t ∈ {"{0,...,K-1}"} from observed weekly returns and realised volatility.
          Parameters (means, covariances, transition matrix) are estimated via the <strong>Baum-Welch EM algorithm</strong>.
        </p>
        <Block>{"Emission:   p(x_t | z_t = k) = N(μ_k, Σ_k)\nTransition: P(z_t = j | z_{t-1} = i) = A_{ij}\nDecoding:   z* = argmax_k P(z_t = k | x_1,...,x_T)  [Viterbi]"}</Block>
      </div>

      <div className="card p-6 space-y-4">
        <h3 className="font-bold text-sm text-[var(--text)]">2. Feature Engineering</h3>
        <div className="overflow-x-auto">
          <table className="quant-table">
            <thead>
              <tr><th>Feature</th><th>Formula</th><th>Economic Rationale</th></tr>
            </thead>
            <tbody>
              {[
                ["term_spread", "DGS10 − DGS2", "Yield curve inversion predicts recessions"],
                ["t10y2y", "FRED T10Y2Y", "10Y−2Y spread (preferred recession indicator)"],
                ["fedfunds", "EFFR", "Monetary policy stance"],
                ["cred_spread", "BAA − 10Y", "Credit risk premium / risk-off indicator"],
                ["cpi_yoy", "CPI YoY %", "Inflation regime"],
                ["indpro_yoy", "IP YoY %", "Real economy momentum"],
                ["rsi_14", "Wilder RSI, 14w", "Price momentum / overbought-oversold"],
                ["mom_4w", "Σ r_{t-3:t}", "Short-term price momentum"],
                ["mom_12w", "Σ r_{t-11:t}", "Medium-term trend"],
                ["price_sma52", "P/SMA(52) − 1", "Trend deviation signal"],
                ["rv_zscore", "(RV − μ) / σ (52w)", "Volatility regime z-score"],
                ["regime", "HMM state", "Latent market environment"],
              ].map(([f, formula, rationale]) => (
                <tr key={f}>
                  <td className="font-mono text-[var(--cyan)]">{f}</td>
                  <td className="font-mono text-[11px]">{formula}</td>
                  <td className="text-[var(--muted)] text-[11px]">{rationale}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card p-6 space-y-4">
        <h3 className="font-bold text-sm text-[var(--text)]">3. Walk-Forward Cross-Validation</h3>
        <p className="text-sm text-[var(--muted)] leading-relaxed">
          Expanding train window with embargo gap to prevent autocorrelation leakage.
        </p>
        <Block>{"T_train(s) = [0, t_s − embargo)\nT_test(s)  = [t_s, t_s + T_test)"}</Block>
      </div>

      <div className="card p-6 space-y-4">
        <h3 className="font-bold text-sm text-[var(--text)]">4. Performance Metrics</h3>
        <Block>{"Sharpe  = (E[r_p] − r_f) / σ_p  ×  √52\nSortino = (E[r_p] − r_f) / σ_down  ×  √52\nMDD     = max((peak − trough) / peak)\nCalmar  = Ann. Return / |MDD|"}</Block>
      </div>

      <div className="card p-6 space-y-4">
        <h3 className="font-bold text-sm text-[var(--text)]">5. Kelly Criterion — Position Sizing</h3>
        <p className="text-sm text-[var(--muted)]">For unit odds (b = 1), win prob p, loss prob q = 1 − p:</p>
        <Block>{"f* = (b·p − q) / b  →  f* = p − q = 2p − 1   (capped at 50%)"}</Block>
      </div>

      <div className="card p-6 space-y-4">
        <h3 className="font-bold text-sm text-[var(--text)]">6. Execution Model</h3>
        <Block>{"w_t    = P(r_next > 0)          continuous position ∈ [0,1]\nΔw_t   = clip(w_raw − w_{t-1}, −δ, +δ)   turnover cap\nTC_t   = |w_t − w_{t-1}| × bps/10000     transaction cost\nr_net  = w_t · r_excess − TC_t"}</Block>
      </div>

      <div className="text-xs text-[var(--muted)] p-4 border border-[#1e293b] rounded-lg">
        ⚠️ <strong>Disclaimer:</strong> Research and educational prototype only. Not investment advice.
        Past simulated performance does not guarantee future results.
      </div>
    </div>
  );
}
