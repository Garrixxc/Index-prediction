"use client";
import type { Settings } from "@/lib/types";

const REGIME_COLORS = ["#f59e0b", "#22c55e", "#3b82f6", "#a855f7"];

interface SidebarProps {
  settings: Settings;
  onChange: (s: Settings) => void;
  onRefresh: () => void;
  loading: boolean;
}

function Label({ children }: { children: React.ReactNode }) {
  return <label className="text-[10px] uppercase tracking-widest text-[var(--muted)] mb-1 block">{children}</label>;
}
function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mb-5">
      <div className="text-[11px] font-semibold text-[var(--muted)] uppercase tracking-wider mb-2 border-b border-[#1e293b] pb-1">{title}</div>
      <div className="space-y-3">{children}</div>
    </div>
  );
}

export default function Sidebar({ settings, onChange, onRefresh, loading }: SidebarProps) {
  const set = (key: keyof Settings, val: string | number) =>
    onChange({ ...settings, [key]: val });

  return (
    <aside className="sidebar w-64 flex-shrink-0 h-screen overflow-y-auto p-5 flex flex-col">
      {/* Logo */}
      <div className="mb-6">
        <div className="font-mono font-bold text-sm gradient-text mb-0.5">⬡ QUANT TERMINAL v2</div>
        <div className="text-[10px] text-[var(--muted)]">Macro-Regime S&P 500</div>
      </div>

      <Section title="Data">
        <div>
          <Label>Backtest start</Label>
          <input type="date" value={settings.start} onChange={e => set("start", e.target.value)} />
        </div>
      </Section>

      <Section title="Model">
        <div>
          <Label>Classifier</Label>
          <select value={settings.model} onChange={e => set("model", e.target.value as Settings["model"])}>
            <option value="logistic">Logistic Regression</option>
            <option value="xgboost">XGBoost</option>
          </select>
        </div>
        <div>
          <Label>HMM regimes (k)</Label>
          <select value={settings.n_states} onChange={e => set("n_states", Number(e.target.value))}>
            <option value={2}>2</option>
            <option value={3}>3</option>
            <option value={4}>4</option>
          </select>
        </div>
        <div>
          <Label>Decision threshold: {settings.threshold.toFixed(2)}</Label>
          <input type="range" min={0.40} max={0.60} step={0.01} value={settings.threshold}
            onChange={e => set("threshold", Number(e.target.value))} className="w-full" />
        </div>
      </Section>

      <Section title="Walk-Forward CV">
        <div>
          <Label>CV folds</Label>
          <input type="number" value={settings.n_splits} min={2} max={10}
            onChange={e => set("n_splits", Number(e.target.value))} />
        </div>
        <div>
          <Label>Test size (weeks/fold)</Label>
          <input type="number" value={settings.test_size} min={52} max={260} step={52}
            onChange={e => set("test_size", Number(e.target.value))} />
        </div>
        <div>
          <Label>Embargo (weeks)</Label>
          <input type="number" value={settings.embargo} min={0} max={12}
            onChange={e => set("embargo", Number(e.target.value))} />
        </div>
      </Section>

      <Section title="Execution">
        <div>
          <Label>Transaction cost (bps)</Label>
          <input type="number" value={settings.cost_bps} min={0} max={100}
            onChange={e => set("cost_bps", Number(e.target.value))} />
        </div>
        <div>
          <Label>Turnover cap: {settings.turnover.toFixed(2)}</Label>
          <input type="range" min={0} max={1} step={0.05} value={settings.turnover}
            onChange={e => set("turnover", Number(e.target.value))} className="w-full" />
        </div>
      </Section>

      <div className="mt-auto">
        <button
          onClick={onRefresh}
          disabled={loading}
          className="w-full rounded-lg py-2.5 text-sm font-semibold transition-all"
          style={{ background: "rgba(0,212,170,0.12)", color: "var(--cyan)", border: "1px solid var(--border)" }}
        >
          {loading ? "⏳ Loading…" : "🔄 Refresh Data"}
        </button>
        <p className="text-[10px] text-[var(--muted)] mt-3 text-center">Not investment advice.</p>
      </div>
    </aside>
  );
}
