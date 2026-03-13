// web/lib/types.ts
export interface LiveSignalData {
  p_up: number;
  action: "LONG" | "FLAT" | "UNCERTAIN";
  confidence: number;
  kelly_fraction: number;
  threshold: number;
  regime: number;
  regime_label: string;
  last_date: string;
  n_weeks: number;
  feature_importance: { feature: string; importance: number }[];
  sparkline: { date: string; equity: number }[];
}

export interface RegimeStat {
  regime: number;
  label: string;
  n_weeks: number;
  pct_time: number;
  ann_return: number;
  ann_vol: number;
  sharpe: number | null;
  win_rate: number;
}

export interface RegimeMapData {
  price_series: { date: string; close: number; regime: number; ret_w: number }[];
  regime_stats: RegimeStat[];
  transition_matrix: number[][];
  n_states: number;
  regime_labels: Record<string, string>;
}

export interface AggPerf {
  total_return: number | null;
  ann_return: number | null;
  ann_vol: number | null;
  sharpe: number | null;
  sortino: number | null;
  max_drawdown: number | null;
  calmar: number | null;
  win_rate: number | null;
  information_ratio: number | null;
  alpha_vs_bnh: number | null;
}

export interface FoldMetric extends AggPerf {
  start: string;
  end: string;
  auc: number | null;
  acc: number | null;
}

export interface EquityPoint {
  date: string;
  strategy: number | null;
  bnh: number | null;
  drawdown: number | null;
}

export interface OofSignal {
  date: string;
  p_up: number;
  y: number;
}

export interface BacktestData {
  equity_series: EquityPoint[];
  fold_metrics: FoldMetric[];
  aggregate_performance: AggPerf;
  oof_signals: OofSignal[];
}

export interface CorrelationData {
  features: string[];
  matrix: (number | null)[][];
}

export interface FactorData {
  feature_importance: { feature: string; importance: number }[];
  correlation: CorrelationData;
  regime_distributions: Record<string, {
    bin_centers: number[];
    density: number[];
    label: string;
  }>;
}

export interface Settings {
  start: string;
  model: "logistic" | "xgboost";
  n_states: number;
  threshold: number;
  n_splits: number;
  test_size: number;
  embargo: number;
  cost_bps: number;
  turnover: number;
}
