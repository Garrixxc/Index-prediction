// web/components/KpiCard.tsx
"use client";

interface KpiCardProps {
  label: string;
  value: string | null;
  sub?: string;
  color?: string;
  pulse?: boolean;
}

export default function KpiCard({ label, value, sub, color = "#00d4aa", pulse }: KpiCardProps) {
  return (
    <div className={`card p-4 min-h-[88px] flex flex-col justify-center ${pulse ? "pulse-live" : ""} fade-in`}>
      <div className="text-[10px] uppercase tracking-widest text-[var(--muted)] mb-1">{label}</div>
      <div
        className="font-mono text-2xl font-bold leading-none"
        style={{ color: value ? color : "var(--muted)" }}
      >
        {value ?? "—"}
      </div>
      {sub && <div className="text-[11px] text-[var(--muted)] mt-1">{sub}</div>}
    </div>
  );
}
