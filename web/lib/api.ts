// web/lib/api.ts
import type { LiveSignalData, RegimeMapData, BacktestData, FactorData, Settings } from "./types";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

function qs(params: Record<string, string | number>): string {
  return "?" + new URLSearchParams(Object.entries(params).map(([k, v]) => [k, String(v)])).toString();
}

async function get<T>(path: string, params: Record<string, string | number>): Promise<T> {
  const res = await fetch(`${BASE}${path}${qs(params)}`, { next: { revalidate: 300 } });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "API error");
  }
  return res.json();
}

export function fetchLiveSignal(s: Settings) {
  return get<LiveSignalData>("/api/live_signal", {
    start: s.start, model: s.model, n_states: s.n_states, threshold: s.threshold,
  });
}

export function fetchRegimeMap(s: Settings) {
  return get<RegimeMapData>("/api/regime_map", {
    start: s.start, n_states: s.n_states,
  });
}

export function fetchBacktest(s: Settings) {
  return get<BacktestData>("/api/backtest", {
    start: s.start, n_states: s.n_states, model: s.model,
    n_splits: s.n_splits, test_size: s.test_size, embargo: s.embargo,
    cost_bps: s.cost_bps, turnover: s.turnover,
  });
}

export function fetchFactorAnalysis(s: Settings) {
  return get<FactorData>("/api/factor_analysis", {
    start: s.start, n_states: s.n_states, model: s.model,
  });
}
