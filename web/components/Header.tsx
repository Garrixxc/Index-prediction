// web/components/Header.tsx
"use client";
import KpiCard from "./KpiCard";
import type { LiveSignalData, AggPerf } from "@/lib/types";

interface HeaderProps {
  liveData?: LiveSignalData;
  perf?: AggPerf;
  loading: boolean;
}

export default function Header({ liveData, perf, loading }: HeaderProps) {
  const curRegimeColor = liveData ? ["#f59e0b", "#22c55e", "#3b82f6", "#a855f7"][liveData.regime % 4] : "var(--muted)";

  return (
    <header className="mb-6 space-y-6">
      {/* Title block */}
      <div className="flex justify-between items-end">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-[var(--text)]">
            📊 Macro-Regime Quantitative Research Terminal
          </h1>
          <p className="text-xs text-[var(--muted)] mt-1 flex items-center gap-2">
            <span className="flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
              Real-time Market Feed
            </span>
            <span>•</span>
            <span>S&P 500 Walk-Forward CV Engine</span>
            <span>•</span>
            <span>HMM Signal Layer</span>
          </p>
        </div>
        <div className="text-[10px] font-mono text-[var(--muted)] text-right">
          LATENCY: <span className="text-[var(--cyan)]">24ms</span><br/>
          ENGINE_V: <span className="text-[var(--cyan)]">2.4.0-STABLE</span>
        </div>
      </div>

      {/* Main KPI Bar */}
      <div className="grid grid-cols-5 gap-4">
        <KpiCard
          label="Weekly Signal"
          value={loading ? "..." : (liveData?.action ?? "—")}
          sub={`threshold ${(liveData?.threshold ?? 0.5).toFixed(2)}`}
          color={
            liveData?.action === "LONG" ? "var(--green)" :
            liveData?.action === "UNCERTAIN" ? "var(--amber)" : "var(--red)"
          }
          pulse={liveData?.action === "LONG"}
        />
        <KpiCard
          label="P(Excess Ret > 0)"
          value={loading ? "..." : liveData ? `${(liveData.p_up * 100).toFixed(1)}%` : "—"}
          sub="live model output"
          color="var(--cyan)"
        />
        <KpiCard
          label="Current Regime"
          value={loading ? "..." : liveData ? `R${liveData.regime}` : "—"}
          sub={liveData?.regime_label ?? "detecting..."}
          color={curRegimeColor}
        />
        <KpiCard
          label="OOF Sharpe"
          value={loading ? "..." : perf?.sharpe ? perf.sharpe.toFixed(3) : "—"}
          sub="walk-forward ann."
          color={(perf?.sharpe ?? 0) > 0 ? "var(--cyan)" : "var(--red)"}
        />
        <KpiCard
          label="Max Drawdown"
          value={loading ? "..." : perf?.max_drawdown ? `${(perf.max_drawdown * 100).toFixed(1)}%` : "—"}
          sub="out-of-fold equity"
          color={(perf?.max_drawdown ?? 0) !== 0 ? "var(--red)" : "var(--muted)"}
        />
      </div>
    </header>
  );
}
