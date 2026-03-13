"use client";
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar,
  CartesianGrid, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, ReferenceLine,
} from "recharts";
import KpiCard from "@/components/KpiCard";
import type { BacktestData, AggPerf } from "@/lib/types";

const pct = (v: number | null | undefined, d = 1) =>
  v == null || isNaN(v) ? "—" : (v * 100).toFixed(d) + "%";
const num = (v: number | null | undefined, d = 3) =>
  v == null || isNaN(v) ? "—" : v.toFixed(d);

export default function Backtest({ data }: { data: BacktestData }) {
  const { equity_series, fold_metrics, aggregate_performance: ap } = data;

  // downsample equity for chart
  const step = Math.max(1, Math.floor(equity_series.length / 600));
  const chartData = equity_series.filter((_, i) => i % step === 0);

  const kpis: [string, string | null, string, string][] = [
    ["Ann. Return", pct(ap.ann_return), "annualised OOF", ap.ann_return ?? 0 > 0 ? "var(--green)" : "var(--red)"],
    ["Ann. Vol", pct(ap.ann_vol), "annualised std", "var(--cyan)"],
    ["Sharpe Ratio", num(ap.sharpe), "annualised", ap.sharpe ?? 0 > 0 ? "var(--cyan)" : "var(--red)"],
    ["Sortino Ratio", num(ap.sortino), "downside-adj", ap.sortino ?? 0 > 0 ? "var(--cyan)" : "var(--red)"],
    ["Max Drawdown", pct(ap.max_drawdown), "peak-to-trough", "var(--red)"],
  ];
  const kpis2: [string, string | null, string, string][] = [
    ["Calmar", num(ap.calmar), "return / |MDD|", ap.calmar ?? 0 > 0 ? "var(--green)" : "var(--red)"],
    ["Win Rate", pct(ap.win_rate), "% positive weeks", ap.win_rate ?? 0 > 0.5 ? "var(--green)" : "var(--amber)"],
    ["Info. Ratio", num(ap.information_ratio), "vs buy & hold", "var(--cyan)"],
    ["Alpha vs B&H", pct(ap.alpha_vs_bnh), "excess ann. return", ap.alpha_vs_bnh ?? 0 > 0 ? "var(--green)" : "var(--red)"],
    ["Total Return", pct(ap.total_return, 0), "OOF aggregate", ap.total_return ?? 0 > 0 ? "var(--green)" : "var(--red)"],
  ];

  return (
    <div className="fade-in space-y-6">
      {/* KPI rows */}
      <div className="grid grid-cols-5 gap-3">
        {kpis.map(([label, value, sub, color]) => (
          <KpiCard key={label} label={label} value={value} sub={sub} color={color} />
        ))}
      </div>
      <div className="grid grid-cols-5 gap-3">
        {kpis2.map(([label, value, sub, color]) => (
          <KpiCard key={label} label={label} value={value} sub={sub} color={color} />
        ))}
      </div>

      {/* Equity + Drawdown chart */}
      <div className="card p-4">
        <div className="text-xs text-[var(--muted)] mb-3 uppercase tracking-wider">Out-of-Fold Equity Curve vs Buy &amp; Hold</div>
        <ResponsiveContainer width="100%" height={240}>
          <LineChart data={chartData} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#64748b" }} tickLine={false}
              tickFormatter={d => d.slice(0, 7)} interval="preserveStartEnd" />
            <YAxis tick={{ fontSize: 10, fill: "#64748b" }} tickLine={false} />
            <Tooltip contentStyle={{ background: "#0d1117", border: "1px solid #1e293b", borderRadius: 8, fontSize: 12 }}
              formatter={(v: number, n: string) => [v.toFixed(3), n === "strategy" ? "Strategy" : "Buy & Hold"]} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <ReferenceLine y={1} stroke="#64748b" strokeDasharray="4 4" strokeWidth={0.8} />
            <Line type="monotone" dataKey="strategy" stroke="#00d4aa" strokeWidth={2} dot={false} name="Strategy" />
            <Line type="monotone" dataKey="bnh" stroke="#64748b" strokeWidth={1.2} strokeDasharray="4 4" dot={false} name="Buy & Hold" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="card p-4">
        <div className="text-xs text-[var(--muted)] mb-3 uppercase tracking-wider">Drawdown</div>
        <ResponsiveContainer width="100%" height={100}>
          <AreaChart data={chartData} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
            <defs>
              <linearGradient id="ddGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#ef4444" stopOpacity={0.5} />
                <stop offset="95%" stopColor="#ef4444" stopOpacity={0.05} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#64748b" }} tickLine={false}
              tickFormatter={d => d.slice(0, 7)} interval="preserveStartEnd" />
            <YAxis tick={{ fontSize: 10, fill: "#64748b" }} tickLine={false}
              tickFormatter={v => pct(v, 0)} />
            <Tooltip contentStyle={{ background: "#0d1117", border: "1px solid #1e293b", borderRadius: 8, fontSize: 12 }}
              formatter={(v: number) => [pct(v), "Drawdown"]} />
            <Area type="monotone" dataKey="drawdown" stroke="#ef4444" fill="url(#ddGrad)" strokeWidth={1} dot={false} />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Fold table */}
      <div className="card p-4 overflow-x-auto">
        <div className="text-xs text-[var(--muted)] mb-3 uppercase tracking-wider">Per-Fold Summary</div>
        <table className="quant-table whitespace-nowrap">
          <thead>
            <tr>
              {["Start","End","AUC","Acc","Sharpe","Sortino","Max DD","Win Rate","Cum. Ret"].map(h => (
                <th key={h}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {fold_metrics.map((r, i) => (
              <tr key={i}>
                <td className="font-mono">{r.start}</td>
                <td className="font-mono">{r.end}</td>
                <td className="font-mono" style={{ color: (r.auc ?? 0) > 0.5 ? "var(--cyan)" : "var(--muted)" }}>{num(r.auc)}</td>
                <td className="font-mono">{num(r.acc)}</td>
                <td className="font-mono" style={{ color: (r.sharpe ?? 0) > 0 ? "var(--green)" : "var(--red)" }}>{num(r.sharpe)}</td>
                <td className="font-mono">{num(r.sortino)}</td>
                <td className="font-mono text-[var(--red)]">{pct(r.max_drawdown)}</td>
                <td className="font-mono">{pct(r.win_rate)}</td>
                <td className="font-mono" style={{ color: (r.cumret ?? 0) > 0 ? "var(--green)" : "var(--red)" }}>{pct(r.cumret)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
