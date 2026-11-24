import { useState } from "react";
import type { Corrected, MarkedItem, SourceOption } from "../lib/data";
import type { FactCheck, Lab, Medication, FollowUp, Section, MedStatus } from "../lib/validate";

export type FactRef = { section: Section | "document_date"; index: number };
export const refKey = (r: FactRef) => `${r.section}:${r.index}`;

const STATUS_LABEL: Record<MedStatus, string> = { current: "Current", not_current: "Stopped or not taken", unclear: "Status not clear" };
const STATUS_HELP: Record<MedStatus, string> = {
  current: "The document says this is being taken now.",
  not_current: "The document says this was stopped or is not being taken.",
  unclear: "The document mentions it but does not say whether it is taken now (for example an old prescription).",
};

interface Props {
  source: SourceOption;
  corrected: Corrected | null;
  active: FactRef | null;
  onActivate: (r: FactRef | null) => void;
  onEdit: (r: FactRef, value: Record<string, string | null>) => void;
  onDelete: (r: FactRef) => void;
  onAdd: (section: Section, value: Record<string, string | null>) => void;
  onAsk: (about: string) => void;
  selectedText: string;
}

function CheckPill({ check, mark }: { check: FactCheck | undefined; mark?: string }) {
  return (
    <>
      {mark && <span className="pill edited">{mark === "added" ? "Added by you" : "Corrected by you"}</span>}
      {check?.status === "verified" ? (
        <span className="pill ok" title="The quote was found in the document and every value appears in it.">✓ Found in text</span>
      ) : (
        <span className="pill check" title="Something could not be matched to the document. See the reasons below.">! Check this</span>
      )}
    </>
  );
}

const FIELDS: Record<Section | "document_date", { key: string; label: string; required?: boolean; full?: boolean }[]> = {
  labs: [
    { key: "test", label: "Test", required: true }, { key: "value", label: "Result", required: true },
    { key: "unit", label: "Unit" }, { key: "reference_range", label: "Range printed on report" },
    { key: "date", label: "Result date (if printed)" }, { key: "evidence", label: "Supporting quote from the document", required: true, full: true },
  ],
  medications: [
    { key: "name", label: "Medicine", required: true }, { key: "status", label: "Status", required: true },
    { key: "dose", label: "Dose" }, { key: "frequency", label: "How often" },
    { key: "evidence", label: "Supporting quote from the document", required: true, full: true },
  ],
  follow_up: [
    { key: "instruction", label: "Instruction", required: true, full: true },
    { key: "evidence", label: "Supporting quote from the document", required: true, full: true },
  ],
  document_date: [
    { key: "value", label: "Date as printed", required: true },
    { key: "evidence", label: "Supporting quote from the document", required: true, full: true },
  ],
};

function Editor({ section, initial, onSave, onCancel, selectedText }: {
  section: Section | "document_date"; initial: Record<string, string | null>; selectedText: string;
  onSave: (v: Record<string, string | null>) => void; onCancel: () => void;
}) {
  const [v, setV] = useState<Record<string, string | null>>(initial);
  const missing = FIELDS[section].filter((f) => f.required && !(v[f.key] ?? "").trim());
  return (
    <form className="editor" onSubmit={(e) => { e.preventDefault(); if (!missing.length) onSave(v); }}>
      <div className="grid">
        {FIELDS[section].map((f) => (
          <label key={f.key} className={`field${f.full ? " full" : ""}`}>
            <span>{f.label}{f.required ? "" : " (optional)"}</span>
            {f.key === "status" ? (
              <select value={v.status ?? "unclear"} onChange={(e) => setV({ ...v, status: e.target.value })}>
                <option value="current">Current</option>
                <option value="not_current">Stopped or not taken</option>
                <option value="unclear">Status not clear</option>
              </select>
            ) : f.key === "evidence" ? (
              <textarea rows={2} value={v.evidence ?? ""} onChange={(e) => setV({ ...v, evidence: e.target.value })} />
            ) : (
              <input type="text" value={v[f.key] ?? ""} onChange={(e) => setV({ ...v, [f.key]: e.target.value || null })} />
            )}
          </label>
        ))}
      </div>
      {selectedText && (
        <button type="button" className="btn small ghost" style={{ marginTop: 6 }} onClick={() => setV({ ...v, evidence: selectedText })}>
          Use the text selected in the document as the quote
        </button>
      )}
      <p className="tiny muted" style={{ margin: "6px 0" }}>Copy values exactly as printed. The quote is checked against the document after you save.</p>
      <div className="row">
        <button className="btn primary small" type="submit" disabled={missing.length > 0}>Save</button>
        <button className="btn small" type="button" onClick={onCancel}>Cancel</button>
        {missing.length > 0 && <span className="tiny muted">Needs: {missing.map((m) => m.label.toLowerCase()).join(", ")}</span>}
      </div>
    </form>
  );
}

export function FactsPanel({ source, corrected, active, onActivate, onEdit, onDelete, onAdd, onAsk, selectedText }: Props) {
  const [editing, setEditing] = useState<string | null>(null);
  const [adding, setAdding] = useState<Section | null>(null);

  if (source.parseError) {
    return (
      <div className="notice bad">
        <strong>This result could not be read.</strong> {source.parseError}
        <details style={{ marginTop: 8 }}><summary className="small">Show the raw output</summary><div className="raw">{source.raw}</div></details>
      </div>
    );
  }
  const c = source.checked!;
  const ex = c.extraction;
  const marked = corrected && source.kind === "yours" ? corrected.extraction : null;
  const markOf = (section: Section | "document_date", i: number) =>
    marked ? (section === "document_date" ? marked.document_date?._mark : (marked[section] as MarkedItem<object>[])[i]?._mark) : undefined;

  const factShell = (ref: FactRef, check: FactCheck | undefined, body: React.ReactNode, initial: Record<string, string | null>, askAbout: string) => {
    const k = refKey(ref);
    if (editing === k) {
      return <Editor key={k} section={ref.section} initial={initial} selectedText={selectedText} onCancel={() => setEditing(null)} onSave={(v) => { onEdit(ref, v); setEditing(null); }} />;
    }
    const isActive = active && refKey(active) === k;
    return (
      <div key={k} className={`fact${isActive ? " active" : ""}${check?.status === "needs_review" ? " review" : ""}`}>
        <button type="button" className="fact-select" aria-pressed={!!isActive} onClick={() => onActivate(isActive ? null : ref)}>
          {body}
          <span className="row" style={{ marginTop: 4 }}><CheckPill check={check} mark={markOf(ref.section, ref.index)} /></span>
        </button>
        {isActive && check?.reasons.length ? <ul className="reasons">{check.reasons.map((r) => <li key={r}>{r}</li>)}</ul> : null}
        {isActive && (
          <div className="fact-actions">
            <button className="btn ghost small" onClick={() => setEditing(k)}>Correct</button>
            <button className="btn ghost small danger" onClick={() => onDelete(ref)}>Remove</button>
            <button className="btn ghost small" onClick={() => onAsk(askAbout)}>Add a question about this</button>
          </div>
        )}
      </div>
    );
  };

  const statusGroups: MedStatus[] = ["current", "not_current", "unclear"];

  return (
    <div>
      <div className="section-h"><h3>Document date</h3></div>
      {ex.document_date
        ? factShell({ section: "document_date", index: 0 }, c.date ?? undefined,
          <span className="fact-main"><span className="fact-value">{ex.document_date.value}</span></span>,
          { value: ex.document_date.value, evidence: ex.document_date.evidence }, `the date ${ex.document_date.value}`)
        : <p className="small muted">No date found.</p>}

      <div className="section-h">
        <h3>Lab results</h3><span className="count">{ex.labs.length}</span><span className="spacer" />
        <button className="btn ghost small" onClick={() => setAdding("labs")}>+ Add</button>
      </div>
      {adding === "labs" && <Editor section="labs" initial={{ evidence: selectedText || "" }} selectedText={selectedText} onCancel={() => setAdding(null)} onSave={(v) => { onAdd("labs", v); setAdding(null); }} />}
      {ex.labs.length === 0 && adding !== "labs" && <p className="small muted">No lab results found.</p>}
      {ex.labs.map((l: Lab, i) => factShell({ section: "labs", index: i }, c.labs[i],
        <>
          <span className="fact-main"><span className="fact-name">{l.test}</span><span className="fact-value">{l.value}{l.unit ? ` ${l.unit}` : ""}</span></span>
          <span className="fact-sub">
            {l.reference_range ? <>Range printed on this report: <span className="mono">{l.reference_range}</span></> : "No range printed"}
            {l.date ? <> · Result date: {l.date}</> : null}
          </span>
        </>, { ...l }, `${l.test} ${l.value}${l.unit ? " " + l.unit : ""}`))}

      <div className="section-h">
        <h3>Medicines</h3><span className="count">{ex.medications.length}</span><span className="spacer" />
        <button className="btn ghost small" onClick={() => setAdding("medications")}>+ Add</button>
      </div>
      {adding === "medications" && <Editor section="medications" initial={{ status: "unclear", evidence: selectedText || "" }} selectedText={selectedText} onCancel={() => setAdding(null)} onSave={(v) => { onAdd("medications", v); setAdding(null); }} />}
      {ex.medications.length === 0 && adding !== "medications" && <p className="small muted">{source.kind === "rules" ? "The rule-based method does not read medicines." : "No medicines found."}</p>}
      {statusGroups.map((st) => {
        const items = ex.medications.map((m, i) => [m, i] as const).filter(([m]) => m.status === st);
        if (!items.length) return null;
        return (
          <div key={st}>
            <div className="status-group" title={STATUS_HELP[st]}>{STATUS_LABEL[st]} <span className="muted" style={{ fontWeight: 400 }}>· {STATUS_HELP[st]}</span></div>
            {items.map(([m, i]: readonly [Medication, number]) => factShell({ section: "medications", index: i }, c.medications[i],
              <>
                <span className="fact-main"><span className="fact-name">{m.name}</span>{m.dose && <span className="fact-value">{m.dose}</span>}</span>
                <span className="fact-sub">{m.frequency ? `How often: ${m.frequency}` : "How often: not stated"}</span>
              </>, { ...m }, m.name))}
          </div>
        );
      })}

      <div className="section-h">
        <h3>Follow-up written in the document</h3><span className="count">{ex.follow_up.length}</span><span className="spacer" />
        <button className="btn ghost small" onClick={() => setAdding("follow_up")}>+ Add</button>
      </div>
      {adding === "follow_up" && <Editor section="follow_up" initial={{ evidence: selectedText || "" }} selectedText={selectedText} onCancel={() => setAdding(null)} onSave={(v) => { onAdd("follow_up", v); setAdding(null); }} />}
      {ex.follow_up.length === 0 && adding !== "follow_up" && <p className="small muted">{source.kind === "rules" ? "The rule-based method does not read follow-up instructions." : "No follow-up instructions found."}</p>}
      {ex.follow_up.map((f: FollowUp, i) => factShell({ section: "follow_up", index: i }, c.follow_up[i],
        <span className="fact-main"><span>{f.instruction}</span></span>, { ...f }, f.instruction))}
    </div>
  );
}
