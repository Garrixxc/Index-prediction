// web/app/page.tsx
"use client";
import { useState, useMemo } from "react";
import useSWR from "swr";

import Header from "@/components/Header";
import Sidebar from "@/components/Sidebar";
import LiveSignal from "@/components/LiveSignal";
import RegimeMap from "@/components/RegimeMap";
import Backtest from "@/components/Backtest";
import ModelQA from "@/components/ModelQA";
import FactorAnalysis from "@/components/FactorAnalysis";
import Methodology from "@/components/Methodology";

import { fetchLiveSignal, fetchRegimeMap, fetchBacktest, fetchFactorAnalysis } from "@/lib/api";
import type { Settings } from "@/lib/types";

const TABS = [
  { id: "signal", label: "🎯 Live Signal" },
  { id: "regime", label: "📊 Regime Map" },
  { id: "backtest", label: "📈 Backtest" },
  { id: "qa", label: "🔬 Model QA" },
  { id: "factor", label: "🧮 Factor Analysis" },
  { id: "method", label: "📚 Methodology" },
];

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState("signal");
  const [settings, setSettings] = useState<Settings>({
    start: "1999-01-01",
    model: "logistic",
    n_states: 3,
    threshold: 0.50,
    n_splits: 6,
    test_size: 104,
    embargo: 4,
    cost_bps: 5,
    turnover: 1.0,
  });

  // Data fetching
  const { data: signal, error: signalError, isLoading: signalLoading, mutate: mutSignal } = useSWR(
    ["signal", settings.start, settings.model, settings.n_states, settings.threshold],
    () => fetchLiveSignal(settings)
  );

  const { data: regime, error: regimeError, isLoading: regimeLoading, mutate: mutRegime } = useSWR(
    ["regime", settings.start, settings.n_states],
    () => fetchRegimeMap(settings)
  );

  const { data: backtest, error: btError, isLoading: btLoading, mutate: mutBt } = useSWR(
    ["backtest", settings.start, settings.model, settings.n_states, settings.n_splits, settings.test_size, settings.embargo, settings.cost_bps, settings.turnover],
    () => fetchBacktest(settings)
  );

  const { data: factor, error: factorError, isLoading: factorLoading, mutate: mutFactor } = useSWR(
    ["factor", settings.start, settings.model, settings.n_states],
    () => fetchFactorAnalysis(settings)
  );

  const handleRefresh = () => {
    mutSignal(); mutRegime(); mutBt(); mutFactor();
  };

  const currentLoading =
    (activeTab === "signal" && signalLoading) ||
    (activeTab === "regime" && regimeLoading) ||
    (activeTab === "backtest" && btLoading) ||
    (activeTab === "qa" && btLoading) ||
    (activeTab === "factor" && factorLoading);

  const currentError =
    (activeTab === "signal" && signalError) ||
    (activeTab === "regime" && regimeError) ||
    (activeTab === "backtest" && btError) ||
    (activeTab === "qa" && btError) ||
    (activeTab === "factor" && factorError);

  return (
    <div className="flex bg-[var(--bg)] min-h-screen text-[var(--text)] overflow-hidden">
      <Sidebar
        settings={settings}
        onChange={setSettings}
        onRefresh={handleRefresh}
        loading={currentLoading}
      />

      <main className="flex-1 overflow-y-auto min-h-screen">
        <div className="max-w-[1280px] mx-auto p-8">
          <Header
            liveData={signal}
            perf={backtest?.aggregate_performance}
            loading={signalLoading}
          />

          {/* Navigation */}
          <nav className="flex items-center gap-1 mb-6 border-b border-[#1e293b] pb-px">
            {TABS.map(t => (
              <button
                key={t.id}
                onClick={() => setActiveTab(t.id)}
                className={`tab-btn relative px-5 pb-3 pt-2 ${activeTab === t.id ? "active text-[var(--cyan)]" : ""}`}
              >
                {t.label}
                {activeTab === t.id && (
                  <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-[var(--cyan)] shadow-[0_0_8px_var(--cyan)]" />
                )}
              </button>
            ))}
          </nav>

          {/* Content area */}
          <div className="min-h-[600px]">
            {currentError && (
              <div className="card p-12 border-red-500/20 bg-red-500/5 text-center fade-in">
                <div className="text-red-400 font-bold mb-2">⚠ API Error</div>
                <div className="text-sm text-[var(--muted)]">{currentError.message ?? "Unknown error occurred"}</div>
                <button onClick={handleRefresh} className="mt-4 px-4 py-2 bg-red-500/10 border border-red-500/20 rounded-lg text-xs hover:bg-red-500/20">
                  Try Again
                </button>
              </div>
            )}

            {!currentError && (
              <>
                {activeTab === "signal" && signal && <LiveSignal data={signal} threshold={settings.threshold} />}
                {activeTab === "regime" && regime && <RegimeMap data={regime} />}
                {activeTab === "backtest" && backtest && <Backtest data={backtest} />}
                {activeTab === "qa" && backtest && <ModelQA data={backtest} />}
                {activeTab === "factor" && factor && <FactorAnalysis data={factor} />}
                {activeTab === "method" && <Methodology />}

                {currentLoading && !currentError && (
                  <div className="fixed bottom-8 right-8 bg-[#0d1117] border border-[var(--border)] rounded-full px-6 py-3 flex items-center gap-3 shadow-2xl pulse-live z-50">
                    <span className="w-2 h-2 rounded-full bg-[var(--cyan)] animate-ping" />
                    <span className="text-xs font-mono font-bold text-[var(--cyan)]">SYNCING_TERMINAL_DATA...</span>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
