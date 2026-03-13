"use client";
import { useMemo } from "react";
import {
  LineChart, Line, BarChart, Bar, CartesianGrid, XAxis, YAxis,
  Tooltip, ResponsiveContainer, ReferenceLine, Legend,
} from "recharts";
import type { BacktestData } from "@/lib/types";

function rocCurve(signals: BacktestData["oof_signals"]) {
  const thresholds = Array.from({ length: 101 }, (_, i) => i / 100);
  const total_p = signals.filter(s => s.y === 1).length;
  const total_n = signals.filter(s => s.y === 0).length;
  return thresholds.map(t => {
    const tp = signals.filter(s => s.y === 1 && s.p_up >= t).length;
    const fp = signals.filter(s => s.y === 0 && s.p_up >= t).length;
    return { fpr: total_n ? fp / total_n : 0, tpr: total_p ? tp / total_p : 0, t };
  }).reverse();
}

function rollingAUC(signals: BacktestData["oof_signals"], window = 26) {
  const result: { date: string; auc: number | null }[] = [];
  for (let i = 0; i < signals.length; i++) {
    if (i < window - 1) { result.push({ date: signals[i].date, auc: null }); continue; }
    const slice = signals.slice(i - window + 1, i + 1);
    const pos = slice.filter(s => s.y === 1).length;
    const neg = slice.length - pos;
    if (pos === 0 || neg === 0) { result.push({ date: signals[i].date, auc: null }); continue; }
    // Wilcoxon approx
    const posScores = slice.filter(s => s.y === 1).map(s => s.p_up).sort((a, b) => a - b);
    const negScores = slice.filter(s => s.y === 0).map(s => s.p_up);
    let concordant = 0;
    for (const pp of posScores) for (const np of negScores) if (pp > np) concordant++;
    result.push({ date: signals[i].date, auc: concordant / (pos * neg) });
  }
  return result;
}

export default function ModelQA({ data }: { data: BacktestData }) {
  const { oof_signals } = data;
  const roc = useMemo(() => rocCurve(oof_signals), [oof_signals]);
  const roll = useMemo(() => rollingAUC(oof_signals), [oof_signals]);

  // Threshold sweep (acc vs threshold)
  const sweep = useMemo(() => {
    return Array.from({ length: 41 }, (_, i) => {
      const t = 0.40 + i * 0.005;
      const acc = oof_signals.filter(s => (s.p_up >= t ? 1 : 0) === s.y).length / oof_signals.length;
      return { threshold: parseFloat(t.toFixed(3)), acc };
    });
  }, [oof_signals]);

  const auc_val = useMemo(() => {
    let a = 0;
    for (let i = 1; i < roc.length; i++)
      a += ((roc[i].fpr - roc[i-1].fpr) * (roc[i].tpr + roc[i-1].tpr)) / 2;
    return Math.abs(a);
  }, [roc]);

  const step = Math.max(1, Math.floor(roll.length / 400));
  const rollChart = roll.filter((_, i) => i % step === 0);

  return (
    <div className="fade-in space-y-6">
      {/* ROC Curve */}
      <div className="grid grid-cols-2 gap-4">
        <div className="card p-4">
          <div className="text-xs text-[var(--muted)] mb-1 uppercase tracking-wider">ROC Curve</div>
          <div className="text-[11px] text-[var(--cyan)] mb-3 font-mono">AUC = {auc_val.toFixed(3)}</div>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={roc}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="fpr" tick={{ fontSize: 10, fill: "#64748b" }} tickLine={false}
                tickFormatter={v => v.toFixed(1)} label={{ value: "FPR", position: "insideBottom", offset: -2, fill: "#64748b", fontSize: 11 }} />
              <YAxis tick={{ fontSize: 10, fill: "#64748b" }} tickLine={false}
                label={{ value: "TPR", angle: -90, position: "insideLeft", fill: "#64748b", fontSize: 11 }} />
              <Tooltip contentStyle={{ background: "#0d1117", border: "1px solid #1e293b", borderRadius: 8, fontSize: 12 }}
                formatter={(v: number) => [v.toFixed(3)]} />
              <ReferenceLine x={0} y={0} stroke="#64748b" strokeDasharray="4 4" />
              {/* diagonal */}
              <Line dataKey="fpr" stroke="none" legendType="none" />
              <Line type="monotone" dataKey="tpr" stroke="#00d4aa" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Threshold sweep */}
        <div className="card p-4">
          <div className="text-xs text-[var(--muted)] mb-1 uppercase tracking-wider">Threshold Sensitivity</div>
          <div className="text-[11px] text-[var(--muted)] mb-3">OOF Accuracy vs decision threshold</div>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={sweep}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="threshold" tick={{ fontSize: 10, fill: "#64748b" }} tickLine={false} />
              <YAxis tick={{ fontSize: 10, fill: "#64748b" }} tickLine={false}
                domain={["auto", "auto"]} tickFormatter={(v: number) => (v * 100).toFixed(0) + "%"} />
              <Tooltip contentStyle={{ background: "#0d1117", border: "1px solid #1e293b", borderRadius: 8, fontSize: 12 }}
                formatter={(v: number) => [(v * 100).toFixed(1) + "%", "Accuracy"]} />
              <ReferenceLine x={0.5} stroke="#64748b" strokeDasharray="4 4" />
              <Line type="monotone" dataKey="acc" stroke="#7c3aed" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Rolling AUC */}
      <div className="card p-4">
        <div className="text-xs text-[var(--muted)] mb-3 uppercase tracking-wider">Rolling AUC (26-week window)</div>
        <ResponsiveContainer width="100%" height={160}>
          <BarChart data={rollChart} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#64748b" }} tickLine={false}
              tickFormatter={d => d.slice(0, 7)} interval="preserveStartEnd" />
            <YAxis tick={{ fontSize: 10, fill: "#64748b" }} tickLine={false} domain={[0.3, 0.8]} />
            <Tooltip contentStyle={{ background: "#0d1117", border: "1px solid #1e293b", borderRadius: 8, fontSize: 12 }}
              formatter={(v: number) => [v?.toFixed(3), "AUC"]} />
            <ReferenceLine y={0.5} stroke="#64748b" strokeDasharray="4 4" />
            <Bar dataKey="auc" fill="#00d4aa" opacity={0.8} radius={[2,2,0,0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
