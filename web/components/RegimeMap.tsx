"use client";
import {
  ComposedChart, Line, Area, CartesianGrid, XAxis, YAxis, Tooltip, Legend,
  ResponsiveContainer, Cell
} from "recharts";
import type { RegimeMapData } from "@/lib/types";

const COLORS = ["#f59e0b", "#22c55e", "#3b82f6", "#a855f7"];

function fmt(v: number | null | undefined, digits = 1, suffix = "%") {
  if (v == null || isNaN(v)) return "—";
  return (v * 100).toFixed(digits) + suffix;
}
function fmtNum(v: number | null | undefined, digits = 2) {
  if (v == null || isNaN(v)) return "—";
  return v.toFixed(digits);
}

export default function RegimeMap({ data }: { data: RegimeMapData }) {
  const { price_series, regime_stats, transition_matrix, n_states, regime_labels } = data;

  // downsample for chart performance (max 500 pts)
  const step = Math.max(1, Math.floor(price_series.length / 500));
  const chartData = price_series.filter((_, i) => i % step === 0);

  return (
    <div className="fade-in space-y-6">
      {/* Price + Regime chart */}
      <div className="card p-4">
        <div className="text-xs text-[var(--muted)] mb-3 uppercase tracking-wider">S&P 500 Price with HMM Regime Background</div>
        <ResponsiveContainer width="100%" height={280}>
          <ComposedChart data={chartData} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
            <defs>
              {[0, 1, 2, 3].map(r => (
                <linearGradient key={r} id={`reg${r}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={COLORS[r]} stopOpacity={0.18} />
                  <stop offset="95%" stopColor={COLORS[r]} stopOpacity={0} />
                </linearGradient>
              ))}
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#64748b" }} tickLine={false}
              tickFormatter={d => d.slice(0, 7)} interval="preserveStartEnd" />
            <YAxis tick={{ fontSize: 10, fill: "#64748b" }} tickLine={false} />
            <Tooltip
              contentStyle={{ background: "#0d1117", border: "1px solid #1e293b", borderRadius: 8, fontSize: 12 }}
              formatter={(v: any) => [v.toFixed(3), "Equity"]}
            />
            <Line type="monotone" dataKey="close" stroke="#e2e8f0" strokeWidth={1.2} dot={false} />
          </ComposedChart>
        </ResponsiveContainer>
        {/* Legend */}
        <div className="flex gap-4 mt-3 flex-wrap">
          {Array.from({ length: n_states }).map((_, r) => (
            <span key={r} className="flex items-center gap-1.5 text-[11px] text-[var(--muted)]">
              <span className="w-3 h-3 rounded-full inline-block" style={{ background: COLORS[r] }} />
              R{r}: {regime_labels[String(r)]}
            </span>
          ))}
        </div>
      </div>

      {/* Regime stats table */}
      <div className="card p-4">
        <div className="text-xs text-[var(--muted)] mb-3 uppercase tracking-wider">Regime Statistics</div>
        <table className="quant-table">
          <thead>
            <tr>
              <th>Regime</th>
              <th>Weeks</th>
              <th>% Time</th>
              <th>Ann. Return</th>
              <th>Ann. Vol</th>
              <th>Sharpe</th>
              <th>Win Rate</th>
            </tr>
          </thead>
          <tbody>
            {regime_stats.map(r => (
              <tr key={r.regime}>
                <td>
                  <span className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full" style={{ background: COLORS[r.regime % 4] }} />
                    <span className="font-semibold">R{r.regime}</span>
                    <span className="text-[var(--muted)] text-[11px]">{r.label}</span>
                  </span>
                </td>
                <td className="font-mono">{r.n_weeks.toLocaleString()}</td>
                <td className="font-mono">{fmt(r.pct_time)}</td>
                <td className="font-mono" style={{ color: r.ann_return > 0 ? "var(--green)" : "var(--red)" }}>
                  {fmt(r.ann_return)}
                </td>
                <td className="font-mono">{fmt(r.ann_vol)}</td>
                <td className="font-mono" style={{ color: (r.sharpe ?? 0) > 0 ? "var(--cyan)" : "var(--red)" }}>
                  {fmtNum(r.sharpe)}
                </td>
                <td className="font-mono">{fmt(r.win_rate)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Transition matrix */}
      <div className="card p-4 max-w-sm">
        <div className="text-xs text-[var(--muted)] mb-3 uppercase tracking-wider">Transition Probability Matrix P(next | current)</div>
        <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${n_states + 1}, 1fr)` }}>
          {/* Header row */}
          <div />
          {Array.from({ length: n_states }).map((_, j) => (
            <div key={j} className="text-center text-[10px] text-[var(--muted)]">→ R{j}</div>
          ))}
          {transition_matrix.map((row, i) => (
            <>
              <div key={`h${i}`} className="text-[10px] text-[var(--muted)] flex items-center">R{i}</div>
              {row.map((v, j) => {
                const opacity = 0.1 + v * 0.85;
                return (
                  <div
                    key={`${i}-${j}`}
                    className="rounded text-center font-mono text-xs py-2"
                    style={{ background: `rgba(0,212,170,${opacity})`, color: v > 0.6 ? "#000" : "var(--text)" }}
                  >
                    {(v * 100).toFixed(0)}%
                  </div>
                );
              })}
            </>
          ))}
        </div>
      </div>
    </div>
  );
}
