import { useEffect, useMemo, useRef } from "react";
import type { Span } from "../lib/validate";

interface Props {
  title: string;
  text: string;
  /** All evidence spans for the current extraction (light highlight). */
  spans: Span[];
  /** The span of the selected fact (strong highlight, scrolled into view). */
  active: Span | null;
  badge?: React.ReactNode;
  onSelectText?: (text: string) => void;
}

interface Piece { text: string; kind: "plain" | "hl" | "active" }

function pieces(text: string, spans: Span[], active: Span | null): Piece[] {
  const marks = new Array<number>(text.length).fill(0);
  for (const s of spans) for (let i = s.start; i < s.end && i < text.length; i++) marks[i] = Math.max(marks[i], 1);
  if (active) for (let i = active.start; i < active.end && i < text.length; i++) marks[i] = 2;
  const out: Piece[] = [];
  let i = 0;
  while (i < text.length) {
    const m = marks[i];
    let j = i;
    while (j < text.length && marks[j] === m) j++;
    out.push({ text: text.slice(i, j), kind: m === 2 ? "active" : m === 1 ? "hl" : "plain" });
    i = j;
  }
  return out;
}

export function SourcePane({ title, text, spans, active, badge, onSelectText }: Props) {
  const activeRef = useRef<HTMLElement | null>(null);
  const parts = useMemo(() => pieces(text, spans, active), [text, spans, active]);

  useEffect(() => {
    activeRef.current?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [active]);

  const onMouseUp = () => {
    const sel = window.getSelection()?.toString() ?? "";
    if (sel.trim().length > 2) onSelectText?.(sel.trim());
  };

  return (
    <div>
      <div className="source-head">
        <h2 style={{ margin: 0 }}>{title}</h2>
        {badge}
      </div>
      <div className="source-paper">
        <pre className="source-text" onMouseUp={onMouseUp} aria-label="Original document text">
          {parts.map((p, i) =>
            p.kind === "plain" ? (
              <span key={i}>{p.text}</span>
            ) : (
              <mark key={i} className={p.kind === "active" ? "active" : undefined} ref={p.kind === "active" && !parts.slice(0, i).some((q) => q.kind === "active") ? (el) => { activeRef.current = el; } : undefined}>
                {p.text}
              </mark>
            ),
          )}
        </pre>
      </div>
      <p className="tiny muted" style={{ maxWidth: 820, marginTop: 8 }}>
        Highlights show the exact lines each fact was taken from. They are found by searching this text, not by trusting positions reported by a model.
      </p>
    </div>
  );
}
