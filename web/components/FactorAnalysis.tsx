"use client";
import { BarChart, Bar, CartesianGrid, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import type { FactorData } from "@/lib/types";

const COLORS = ["#f59e0b", "#22c55e", "#3b82f6", "#a855f7"];
const CYAN = "#00d4aa";
const PURPLE = "#7c3aed";

export default function FactorAnalysis({ data }: { data: FactorData }) {
  const { feature_importance, correlation, regime_distributions } = data;
  const topN = feature_importance.slice(0, 15);
  const maxImp = topN[0]?.importance ?? 1;

  return (
    <div className="fade-in space-y-6">
      {/* Feature Importance */}
      <div className="card p-4">
        <div className="text-xs text-[var(--muted)] mb-4 uppercase tracking-wider">
          Feature Importance — Live Model (all-but-last-week)
        </div>
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={[...topN].reverse()} layout="vertical" margin={{ left: 80, right: 40 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
            <XAxis type="number" tick={{ fontSize: 10, fill: "#64748b" }} tickLine={false} />
            <YAxis type="category" dataKey="feature" tick={{ fontSize: 11, fill: "#e2e8f0" }} tickLine={false} width={80} />
            <Tooltip
              contentStyle={{ background: "#0d1117", border: "1px solid #1e293b", borderRadius: 8, fontSize: 12 }}
              formatter={(v: any) => [v.toFixed(5), "Importance"]}
            />
            <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
              {[...topN].reverse().map((f, i) => (
                <Cell key={f.feature} fill={f.importance > maxImp * 0.6 ? CYAN : PURPLE} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Correlation heatmap */}
      <div className="card p-4 overflow-x-auto">
        <div className="text-xs text-[var(--muted)] mb-3 uppercase tracking-wider">Pairwise Feature Correlation Matrix</div>
        <div
          className="inline-grid gap-px"
          style={{ gridTemplateColumns: `80px repeat(${correlation.features.length}, 44px)` }}
        >
          {/* Header */}
          <div />
          {correlation.features.map(f => (
            <div key={f} className="text-[9px] text-[var(--muted)] text-center truncate px-0.5 pb-1"
              title={f} style={{ writingMode: "vertical-rl", height: 60 }}>{f}</div>
          ))}
          {/* Rows */}
          {correlation.features.map((rowF, i) => (
            <div key={`group-${i}`} className="contents">
              <div className="text-[10px] text-[var(--muted)] flex items-center pr-2 truncate">{rowF}</div>
              {correlation.matrix[i].map((v, j) => {
                const val = v ?? 0;
                const absV = Math.abs(val);
                const bg = val > 0
                  ? `rgba(0,212,170,${absV * 0.85})`
                  : `rgba(239,68,68,${absV * 0.85})`;
                const textColor = absV > 0.6 ? "#000" : "var(--text)";
                return (
                  <div key={`c${i}-${j}`} className="h-10 flex items-center justify-center rounded text-[9px] font-mono"
                    style={{ background: bg, color: textColor }}>
                    {val.toFixed(2)}
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>

      {/* Regime return distributions as bar histograms */}
      <div className="card p-4">
        <div className="text-xs text-[var(--muted)] mb-3 uppercase tracking-wider">Return Distribution by Regime</div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {Object.entries(regime_distributions).map(([r, dist]) => (
            <div key={r}>
              <div className="text-[11px] font-semibold mb-2" style={{ color: COLORS[Number(r) % 4] }}>
                R{r}: {dist.label}
              </div>
              <ResponsiveContainer width="100%" height={120}>
                <BarChart
                  data={dist.bin_centers.map((x, i) => ({ x, density: dist.density[i] }))}
                  margin={{ left: -20, right: 4 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="x" tick={{ fontSize: 9, fill: "#64748b" }} tickLine={false}
                    tickFormatter={(v: number) => (v * 100).toFixed(1) + "%"} interval="preserveStartEnd" />
                  <YAxis tick={{ fontSize: 9, fill: "#64748b" }} tickLine={false} />
                  <Tooltip contentStyle={{ background: "#0d1117", border: "1px solid #1e293b", borderRadius: 8, fontSize: 11 }}
                    formatter={(v: any, n: string) => [v.toFixed(2), "Density"]}
                    labelFormatter={(v: any) => `Return: ${(Number(v) * 100).toFixed(2)}%`} />
                  <Bar dataKey="density" fill={COLORS[Number(r) % 4]} opacity={0.75} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
