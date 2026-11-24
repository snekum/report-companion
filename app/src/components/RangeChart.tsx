import { useEffect, useRef, useState } from "react";
import type { ComparisonRow, Measurement } from "../lib/compare";

// One small chart per test: a row per report, each with that report's own
// printed range as a grey band and the value as a dot. One x-axis, no unit
// conversion. Hollow dashed dots are facts that did not pass the text checks.

const ROW = 40, RIGHT = 24, TOP = 10, AXIS = 26;

function niceTicks(lo: number, hi: number, n = 5) {
  const span = hi - lo || 1;
  const step = Math.pow(10, Math.floor(Math.log10(span / n)));
  const err = (span / n) / step;
  const mult = err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1;
  const s = step * mult;
  const out: number[] = [];
  for (let v = Math.ceil(lo / s) * s; v <= hi + 1e-9; v += s) out.push(Number(v.toFixed(6)));
  return out;
}

export function RangeChart({ row }: { row: ComparisonRow }) {
  const [tip, setTip] = useState<{ x: number; y: number; m: Measurement } | null>(null);
  // Draw at the container's real width so text stays at its CSS size.
  const box = useRef<HTMLDivElement>(null);
  const [W, setW] = useState(640);
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(300, Math.round(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const LEFT = W < 480 ? 118 : 190;
  const vals = row.points.flatMap((p) => [p.value!, p.range?.low ?? null, p.range?.high ?? null]).filter((v): v is number => v !== null);
  let lo = Math.min(...vals), hi = Math.max(...vals);
  const pad = (hi - lo || Math.abs(hi) || 1) * 0.15;
  lo = Math.max(0, lo - pad); hi = hi + pad;
  const ticks = niceTicks(lo, hi);
  lo = Math.min(lo, ticks[0]); hi = Math.max(hi, ticks[ticks.length - 1]);
  const x = (v: number) => LEFT + ((v - lo) / (hi - lo)) * (W - LEFT - RIGHT);
  const H = TOP + row.points.length * ROW + AXIS;

  return (
    <div className="chart" ref={box} onMouseLeave={() => setTip(null)}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${row.label}: ${row.points.map((p) => `${p.dateText ?? p.docTitle} ${p.lab.value} ${p.lab.unit ?? ""}`).join("; ")}`}>
        <g className="grid">{ticks.map((t) => <line key={t} x1={x(t)} x2={x(t)} y1={TOP} y2={H - AXIS} />)}</g>
        {row.points.map((p, i) => {
          const cy = TOP + i * ROW + ROW / 2;
          const r = p.range;
          const bandLo = r ? x(r.low ?? lo) : 0, bandHi = r ? x(r.high ?? hi) : 0;
          return (
            <g key={i}>
              <text x={0} y={cy - 2} style={{ fill: "var(--text)", fontWeight: 600 }}>{p.dateText ?? "No date"}{p.dateFromDocument && p.dateText ? " *" : ""}</text>
              <text x={0} y={cy + 13} style={{ fontSize: 11.5 }}>{p.lab.reference_range ? `range ${p.lab.reference_range}` : "no range printed"}</text>
              {r && <rect className="band" x={bandLo} y={cy - 7} width={Math.max(2, bandHi - bandLo)} height={14} rx={4} />}
              <circle className={`dot${p.verified ? "" : " review"}`} cx={x(p.value!)} cy={cy} r={6} />
              <text x={x(p.value!)} y={cy - 10} textAnchor="middle" style={{ fill: "var(--text)", fontFamily: "var(--mono)", fontSize: 11.5 }}>{p.lab.value}</text>
              <rect className="hit" x={LEFT} y={cy - ROW / 2} width={W - LEFT - RIGHT} height={ROW}
                onMouseMove={(e) => setTip({ x: e.clientX, y: e.clientY, m: p })} />
            </g>
          );
        })}
        <g className="axis">
          <line x1={LEFT} x2={W - RIGHT} y1={H - AXIS} y2={H - AXIS} />
          {ticks.map((t) => <text key={t} x={x(t)} y={H - AXIS + 16} textAnchor="middle">{t}</text>)}
        </g>
      </svg>
      {tip && (
        <div className="tooltip" style={{ left: tip.x + 12, top: tip.y + 12 }}>
          <strong>{tip.m.lab.test}: {tip.m.lab.value} {tip.m.lab.unit}</strong><br />
          {tip.m.docTitle}{tip.m.dateText ? `, ${tip.m.dateText}${tip.m.dateFromDocument ? " (document date; no result date printed)" : ""}` : ""}<br />
          Range printed on this report: {tip.m.lab.reference_range ?? "none"}<br />
          {tip.m.verified ? "Found in the document text" : "Could not be fully matched to the text: check this"}
        </div>
      )}
    </div>
  );
}
