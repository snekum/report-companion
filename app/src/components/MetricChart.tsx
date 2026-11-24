import { useState } from "react";

export interface MetricRow { label: string; help: string; values: Partial<Record<SystemKey, number | null>> }
export type SystemKey = "rules" | "original" | "finetuned";
export const SYSTEMS: { key: SystemKey; label: string; color: string }[] = [
  { key: "rules", label: "Rules", color: "var(--series-rules)" },
  { key: "original", label: "Original model", color: "var(--series-orig)" },
  { key: "finetuned", label: "Fine-tuned model", color: "var(--series-ft)" },
];

const W = 640, LEFT = 210, RIGHT = 20, ROW = 30, TOP = 6, AXIS = 24;

export function MetricChart({ rows }: { rows: MetricRow[] }) {
  const [tip, setTip] = useState<{ x: number; y: number; text: string } | null>(null);
  const x = (v: number) => LEFT + v * (W - LEFT - RIGHT);
  const H = TOP + rows.length * ROW + AXIS;
  return (
    <div className="chart" onMouseLeave={() => setTip(null)}>
      <div className="legend" style={{ marginBottom: 6 }}>
        {SYSTEMS.map((s) => <span key={s.key}><span className="sw" style={{ background: s.color }} />{s.label}</span>)}
        <span className="muted">· a missing dot means that method does not attempt the field</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Evaluation metrics by method; see the table for the numbers">
        <g className="grid">{[0, 0.25, 0.5, 0.75, 1].map((t) => <line key={t} x1={x(t)} x2={x(t)} y1={TOP} y2={H - AXIS} />)}</g>
        {rows.map((r, i) => {
          const cy = TOP + i * ROW + ROW / 2;
          const present = SYSTEMS.filter((s) => typeof r.values[s.key] === "number");
          return (
            <g key={r.label}>
              <text x={0} y={cy + 4} style={{ fill: "var(--text)" }}>{r.label}</text>
              {present.length > 1 && (
                <line x1={x(Math.min(...present.map((s) => r.values[s.key]!)))} x2={x(Math.max(...present.map((s) => r.values[s.key]!)))} y1={cy} y2={cy} style={{ stroke: "var(--border-strong)", strokeWidth: 2 }} />
              )}
              {present.map((s) => (
                <circle key={s.key} cx={x(r.values[s.key]!)} cy={cy} r={6} style={{ fill: s.color, stroke: "var(--surface)", strokeWidth: 2 }}
                  onMouseMove={(e) => setTip({ x: e.clientX, y: e.clientY, text: `${s.label}: ${(r.values[s.key]! * 100).toFixed(1)}% · ${r.label}` })} />
              ))}
            </g>
          );
        })}
        <g className="axis">
          <line x1={LEFT} x2={W - RIGHT} y1={H - AXIS} y2={H - AXIS} />
          {[0, 0.25, 0.5, 0.75, 1].map((t) => <text key={t} x={x(t)} y={H - AXIS + 16} textAnchor="middle">{t * 100}%</text>)}
        </g>
      </svg>
      {tip && <div className="tooltip" style={{ left: tip.x + 12, top: tip.y + 12 }}>{tip.text}</div>}
    </div>
  );
}

export function MetricTable({ rows }: { rows: MetricRow[] }) {
  return (
    <div style={{ overflowX: "auto" }}>
      <table className="data metric-table">
        <thead><tr><th>Measure</th>{SYSTEMS.map((s) => <th key={s.key}>{s.label}</th>)}</tr></thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.label}>
              <td title={r.help}>{r.label}</td>
              {SYSTEMS.map((s) => <td key={s.key} className="num">{typeof r.values[s.key] === "number" ? `${(r.values[s.key]! * 100).toFixed(1)}%` : "not attempted"}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
