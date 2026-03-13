"use client";
import { AreaChart, Area, CartesianGrid, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts";
import KpiCard from "@/components/KpiCard";
import type { LiveSignalData } from "@/lib/types";

const pct = (v: number) => (v * 100).toFixed(1) + "%";
const fmt3 = (v: number) => v.toFixed(3);

export default function LiveSignal({ data, threshold }: { data: LiveSignalData; threshold: number }) {
  const { p_up, action, confidence, kelly_fraction, regime_label, last_date, n_weeks, sparkline, feature_importance } = data;

  const actionColor =
    action === "LONG" ? "var(--green)" : action === "UNCERTAIN" ? "var(--amber)" : "var(--red)";
  const badgeCls =
    action === "LONG" ? "badge-long" : action === "UNCERTAIN" ? "badge-uncertain" : "badge-flat";

  return (
    <div className="fade-in space-y-6">
      {/* Hero signal block */}
      <div className="card p-6 glow-border" style={{ borderColor: `${actionColor}55` }}>
        <div className="flex flex-wrap items-center gap-4 mb-4">
          <span className={`${badgeCls} px-4 py-1.5 rounded-lg font-mono text-lg font-bold tracking-wider`}>
            {action}
          </span>
          <div>
            <div className="text-xl font-bold" style={{ color: actionColor }}>
              {pct(p_up)} probability of outperforming cash next week
            </div>
            <div className="text-sm text-[var(--muted)] mt-1">
              Regime: <span className="font-semibold text-[var(--text)]">{regime_label}</span>
              {" · "}Threshold: <span className="font-mono text-[var(--text)]">{threshold.toFixed(2)}</span>
              {" · "}As of <span className="font-mono text-[var(--text)]">{last_date}</span>
              {" · "}{n_weeks.toLocaleString()} weeks of data
            </div>
          </div>
        </div>

        {/* Confidence bar */}
        <div className="mb-1 text-xs text-[var(--muted)]">Signal Confidence: <span className="text-[var(--text)] font-semibold">{confidence.toFixed(0)}%</span></div>
        <div className="w-full h-2 rounded-full bg-[#0a1220] overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-700"
            style={{ width: `${confidence}%`, background: actionColor }}
          />
        </div>
      </div>

      {/* KPIs */}
      <div className="grid grid-cols-3 gap-4">
        <KpiCard label="P(Excess Return > 0)" value={fmt3(p_up)} sub="raw model probability" color="var(--cyan)" pulse />
        <KpiCard label="Kelly Position Size" value={pct(kelly_fraction)} sub="f* = p − q, capped 50%" color="var(--amber)" />
        <KpiCard label="Model Confidence" value={`${confidence.toFixed(0)}%`} sub="|p − 0.5| × 200" color={actionColor} />
      </div>

      {/* Trailing 52-week sparkline */}
      <div className="card p-4">
        <div className="text-xs text-[var(--muted)] mb-3 uppercase tracking-wider">S&P 500 — trailing 52-week equity (base = 1)</div>
        <ResponsiveContainer width="100%" height={160}>
          <AreaChart data={sparkline} margin={{ top: 4, right: 8, bottom: 0, left: -24 }}>
            <defs>
              <linearGradient id="equityGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#00d4aa" stopOpacity={0.25} />
                <stop offset="95%" stopColor="#00d4aa" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#64748b" }} tickLine={false}
              tickFormatter={d => d.slice(0, 7)} interval="preserveStartEnd" />
            <YAxis tick={{ fontSize: 10, fill: "#64748b" }} tickLine={false} />
            <ReferenceLine y={1} stroke="#64748b" strokeDasharray="4 4" strokeWidth={0.8} />
            <Tooltip
              contentStyle={{ background: "#0d1117", border: "1px solid #1e293b", borderRadius: 8, fontSize: 12 }}
              formatter={(v: number) => [v.toFixed(3), "Equity"]}
            />
            <Area type="monotone" dataKey="equity" stroke="#00d4aa" fill="url(#equityGrad)" strokeWidth={1.8} dot={false} />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Feature importance top 10 */}
      <div className="card p-4">
        <div className="text-xs text-[var(--muted)] mb-3 uppercase tracking-wider">Top Feature Signals</div>
        <div className="space-y-2">
          {feature_importance.slice(0, 10).map((f, i) => {
            const maxVal = feature_importance[0]?.importance ?? 1;
            const pctWidth = ((f.importance / maxVal) * 100).toFixed(1);
            return (
              <div key={f.feature} className="flex items-center gap-3">
                <div className="w-28 text-right font-mono text-[11px] text-[var(--muted)] shrink-0">{f.feature}</div>
                <div className="flex-1 h-2 rounded-full bg-[#0a1220] overflow-hidden">
                  <div className="h-full rounded-full" style={{
                    width: `${pctWidth}%`,
                    background: i < 3 ? "var(--cyan)" : "var(--purple)",
                  }} />
                </div>
                <div className="w-16 font-mono text-[11px] text-[var(--muted)]">{f.importance.toFixed(4)}</div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
