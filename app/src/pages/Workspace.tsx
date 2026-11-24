import { useEffect, useMemo, useState } from "react";
import { SourcePane } from "../components/SourcePane";
import { FactsPanel, type FactRef } from "../components/FactsPanel";
import { ExplainPanel } from "../components/ExplainPanel";
import { QuestionsPanel } from "../components/QuestionsPanel";
import { LivePanel } from "../components/LivePanel";
import { journeyById, newId, sourcesFor, workDoc, type Corrected, type LiveOutput, type SourceOption, type UserDoc } from "../lib/data";
import { useStored } from "../lib/store";
import { go, href } from "../lib/router";
import type { Question } from "../lib/brief";
import type { Extraction, Section, Span } from "../lib/validate";
import { MODEL_CONFIG } from "../config";

const PREFERENCE = ["yours", "finetuned", "live-finetuned", "original", "live-original", "rules"];

export function pickDefault(sources: SourceOption[], wanted?: string | null): SourceOption {
  if (wanted) { const w = sources.find((s) => s.key === wanted); if (w) return w; }
  for (const k of PREFERENCE) { const s = sources.find((x) => x.key === k); if (s) return s; }
  return sources[0];
}

function toCorrected(base: SourceOption, existing: Corrected | null): Corrected {
  if (existing) return existing;
  const ex = base.checked!.extraction;
  return { basedOn: base.label, updatedAt: new Date().toISOString(), removed: 0, extraction: JSON.parse(JSON.stringify(ex)) as Corrected["extraction"] };
}

function AddDocument({ onAdd, onCancel }: { onAdd: (d: UserDoc) => void; onCancel: () => void }) {
  const [title, setTitle] = useState("");
  const [text, setText] = useState("");
  const tooLong = text.length > MODEL_CONFIG.maxInputChars;
  return (
    <div className="source-paper stack">
      <h2 style={{ margin: 0 }}>Add a document</h2>
      <div className="notice info small">
        Paste the text of a short document (a lab report, visit summary or letter). It stays in this browser: it is saved to this device's storage
        so it's here next time, and is never uploaded. You can delete it at any time. Please use fictional or your own documents only.
      </div>
      <label className="field"><span>Name (for your list)</span><input type="text" value={title} onChange={(e) => setTitle(e.target.value)} placeholder="e.g. Blood test, March" /></label>
      <label className="field"><span>Document text</span><textarea rows={14} className="mono" value={text} onChange={(e) => setText(e.target.value)} placeholder="Paste the text here…" /></label>
      <div className="row">
        <span className={`tiny ${tooLong ? "" : "muted"}`} style={tooLong ? { color: "var(--bad)" } : undefined}>{text.length.toLocaleString()} / {MODEL_CONFIG.maxInputChars.toLocaleString()} characters</span>
        <span className="spacer" />
        <button className="btn" onClick={onCancel}>Cancel</button>
        <button className="btn primary" disabled={!text.trim() || tooLong} onClick={() => onAdd({ id: newId("mine"), title: title.trim() || "My document", text: text.replace(/\r\n/g, "\n"), createdAt: new Date().toISOString() })}>Add document</button>
      </div>
      <p className="tiny muted">Scanned PDFs and photos are not supported in this prototype; it reads plain text only.</p>
    </div>
  );
}

export function Workspace({ journeyId, docParam, adding }: { journeyId: string; docParam: string | null; adding: boolean }) {
  const journey = journeyById(journeyId);
  const [userDocs, setUserDocs] = useStored<UserDoc[]>("userDocs", []);
  const docIds = journey ? journey.docIds : [];
  const currentId = docParam ?? docIds[0] ?? userDocs[0]?.id ?? null;
  const doc = currentId ? workDoc(currentId, userDocs) : null;

  const [corrected, setCorrected] = useStored<Corrected | null>(`corr.${currentId}`, null);
  const [live, setLive] = useStored<LiveOutput[]>(`live.${currentId}`, []);
  const [questions, setQuestions] = useStored<Question[]>(`questions.${journeyId}`, []);
  const [wantedSource, setWantedSource] = useState<string | null>(null);
  const [active, setActive] = useState<FactRef | null>(null);
  const [tab, setTab] = useState<"facts" | "explain" | "questions">("facts");
  const [mobilePane, setMobilePane] = useState<"docs" | "source" | "facts">("source");
  const [showLive, setShowLive] = useState(false);
  const [selectedText, setSelectedText] = useState("");
  const [draftAbout, setDraftAbout] = useState<string | null>(null);

  useEffect(() => { setActive(null); setWantedSource(null); setShowLive(false); setSelectedText(""); }, [currentId]);

  const sources = useMemo(() => (doc ? sourcesFor(doc, corrected, live) : []), [doc?.id, doc?.text, corrected, live]);
  const source = sources.length ? pickDefault(sources, wantedSource) : null;

  const allSpans: Span[] = useMemo(() => {
    const c = source?.checked;
    if (!c) return [];
    return [c.date, ...c.labs, ...c.medications, ...c.follow_up].flatMap((x) => (x?.span ? [x.span] : []));
  }, [source]);
  const activeSpan = useMemo(() => {
    if (!active || !source?.checked) return null;
    const c = source.checked;
    const check = active.section === "document_date" ? c.date : c[active.section][active.index];
    return check?.span ?? null;
  }, [active, source]);

  const edit = (mutate: (ex: Corrected["extraction"]) => void) => {
    if (!source?.checked) return;
    const base = toCorrected(source, source.kind === "yours" ? corrected : null);
    const next: Corrected = JSON.parse(JSON.stringify(base));
    mutate(next.extraction);
    next.updatedAt = new Date().toISOString();
    setCorrected(next);
    setWantedSource("yours");
  };

  const clean = (v: Record<string, string | null>) => Object.fromEntries(Object.entries(v).map(([k, x]) => [k, typeof x === "string" && x.trim() === "" ? null : x]));

  const onEdit = (r: FactRef, v: Record<string, string | null>) => edit((ex) => {
    const val = { ...clean(v), _mark: "edited" } as never;
    if (r.section === "document_date") ex.document_date = { ...(ex.document_date ?? {}), ...(val as object), _mark: "edited" } as never;
    else (ex[r.section] as unknown[])[r.index] = { ...((ex[r.section] as unknown[])[r.index] as object), ...(val as object), _mark: (((ex[r.section] as { _mark?: string }[])[r.index])?._mark === "added" ? "added" : "edited") };
  });
  const onDelete = (r: FactRef) => { edit((ex) => {
    if (r.section === "document_date") ex.document_date = null;
    else (ex[r.section] as unknown[]).splice(r.index, 1);
  }); setActive(null); };
  const onAdd = (section: Section, v: Record<string, string | null>) => edit((ex) => {
    const c = clean(v);
    const item = section === "labs"
      ? { test: c.test ?? "", value: c.value ?? "", unit: c.unit ?? null, reference_range: c.reference_range ?? null, date: c.date ?? null, evidence: c.evidence ?? "", _mark: "added" }
      : section === "medications"
        ? { name: c.name ?? "", dose: c.dose ?? null, frequency: c.frequency ?? null, status: (c.status ?? "unclear") as Extraction["medications"][number]["status"], evidence: c.evidence ?? "", _mark: "added" }
        : { instruction: c.instruction ?? "", evidence: c.evidence ?? "", _mark: "added" };
    (ex[section] as unknown[]).push(item);
  });

  const userList = (
    <>
      <div className="doclist-h">Your documents</div>
      {userDocs.length === 0 && <p className="tiny muted" style={{ margin: "0 10px" }}>None yet. Text you add stays on this device.</p>}
      <ul className="doclist">
        {userDocs.map((d) => (
          <li key={d.id}><a href={href(`journey/${journeyId}`, { doc: d.id })} aria-current={d.id === currentId}>{d.title}<div className="meta">Added {new Date(d.createdAt).toLocaleDateString()}</div></a></li>
        ))}
      </ul>
      <div style={{ margin: "8px 10px" }}><a className="btn small" href={href(`journey/${journeyId}`, { add: "1" })}>+ Add a document</a></div>
    </>
  );

  return (
    <>
      <div className="ws-tabs-mobile tabs" role="tablist" aria-label="Workspace panes">
        {(["docs", "source", "facts"] as const).map((p) => (
          <button key={p} role="tab" aria-selected={mobilePane === p} onClick={() => setMobilePane(p)}>{p === "docs" ? "Documents" : p === "source" ? "Original" : "Facts"}</button>
        ))}
      </div>
      <div className="workspace">
        <nav className={`ws-col ws-docs${mobilePane === "docs" ? "" : " hidden-mobile"}`} aria-label="Documents">
          {journey ? (
            <>
              <div className="doclist-h" style={{ marginTop: 0 }}>Example journey</div>
              <p className="small" style={{ margin: "0 10px 6px", fontWeight: 600 }}>{journey.title}</p>
              <p className="tiny muted" style={{ margin: "0 10px 8px" }}>{journey.who} Fictional patient.</p>
              <ul className="doclist">
                {docIds.map((id) => { const d = workDoc(id, userDocs)!; return (
                  <li key={id}><a href={href(`journey/${journeyId}`, { doc: id })} aria-current={id === currentId} onClick={() => setMobilePane("source")}>{d.title}<div className="meta">{d.sample?.kind === "lab_report" ? "Lab report" : "Visit document"}</div></a></li>
                ); })}
              </ul>
              {journey.compare && <div style={{ margin: "8px 10px" }}><a className="btn small" href={href(`compare/${journeyId}`)}>Compare two reports</a></div>}
            </>
          ) : (
            <p className="small muted" style={{ margin: "0 10px" }}>Your own documents. Nothing here leaves this browser.</p>
          )}
          {userList}
        </nav>

        <main className={`ws-col ws-source${mobilePane === "source" ? "" : " hidden-mobile"}`}>
          {adding ? (
            <AddDocument onCancel={() => go(`journey/${journeyId}`, { doc: currentId ?? undefined })} onAdd={(d) => { setUserDocs([...userDocs, d]); go(`journey/${journeyId}`, { doc: d.id }); }} />
          ) : !doc ? (
            <div className="empty card pad">
              <h2>No document selected</h2>
              <p>Add the text of a document, or open one of the example journeys from the home page.</p>
              <a className="btn primary" href={href(`journey/${journeyId}`, { add: "1" })}>Add a document</a>
            </div>
          ) : (
            <>
              <SourcePane title={doc.title} text={doc.text} spans={allSpans} active={activeSpan}
                onSelectText={setSelectedText}
                badge={doc.sample ? <span className="pill prepared">Fictional example</span> : <span className="pill live">On this device only</span>} />
              <div className="row" style={{ marginTop: 10, maxWidth: 820 }}>
                <button className="btn" onClick={() => setShowLive((s) => !s)}>{showLive ? "Hide on-device model" : "Run a model on this device"}</button>
                {doc.user && <button className="btn ghost danger" onClick={() => {
                  if (!window.confirm("Delete this document and everything you added to it from this browser?")) return;
                  setUserDocs(userDocs.filter((d) => d.id !== doc.id)); setCorrected(null); setLive([]); go(`journey/${journeyId}`);
                }}>Delete this document</button>}
              </div>
              {showLive && <div style={{ maxWidth: 820 }}><LivePanel text={doc.text} onClose={() => setShowLive(false)} onResult={(o) => { setLive([...live, o]); setWantedSource(`live-${o.key}`); setTab("facts"); }} /></div>}
            </>
          )}
        </main>

        <aside className={`ws-col ws-facts${mobilePane === "facts" ? "" : " hidden-mobile"}`} aria-label="Extracted information">
          {doc && source && !adding ? (
            <>
              <div className="facts-head">
                <div className="small muted">Facts from</div>
                <div className="source-select" role="group" aria-label="Which result to show">
                  {sources.map((s) => (
                    <button key={s.key} aria-pressed={s.key === source.key} onClick={() => { setWantedSource(s.key); setActive(null); }}>{s.label}</button>
                  ))}
                </div>
                <div className="row" style={{ gap: 6 }}>
                  <span className={`pill ${source.kind}`}>{source.kind === "prepared" ? "Prepared, not live" : source.kind === "live" ? "Run on this device" : source.kind === "rules" ? "Rules" : "Your version"}</span>
                  <span className="tiny muted" style={{ flex: 1 }}>{source.note}</span>
                </div>
                {source.checked && (
                  <div className="summary-line" style={{ marginTop: 6 }}>
                    {source.checked.total} facts · {source.checked.total - source.checked.needsReview} found in the text · {source.checked.needsReview} to check
                    {source.schemaErrors.length > 0 && <> · <span title={source.schemaErrors.join("\n")}>output didn't fully follow the format</span></>}
                  </div>
                )}
                {source.kind === "yours" && <button className="btn ghost small" onClick={() => { if (window.confirm("Discard your corrections for this document?")) { setCorrected(null); setWantedSource(null); } }}>Discard my corrections</button>}
              </div>
              <div className="tabs" role="tablist">
                <button role="tab" aria-selected={tab === "facts"} onClick={() => setTab("facts")}>Facts</button>
                <button role="tab" aria-selected={tab === "explain"} onClick={() => setTab("explain")}>Explain</button>
                <button role="tab" aria-selected={tab === "questions"} onClick={() => setTab("questions")}>Questions{questions.length ? ` (${questions.length})` : ""}</button>
              </div>
              <div className="facts-body">
                {tab === "facts" && (
                  <FactsPanel source={source} corrected={corrected} active={active} selectedText={selectedText}
                    onActivate={(r) => { setActive(r); if (r) setMobilePane((p) => p); }}
                    onEdit={onEdit} onDelete={onDelete} onAdd={onAdd}
                    onAsk={(about) => { setDraftAbout(about); setTab("questions"); }} />
                )}
                {tab === "explain" && (source.checked ? <ExplainPanel checked={source.checked} sourceLabel={source.label} /> : <p className="small muted">Nothing to explain: this result could not be read.</p>)}
                {tab === "questions" && (
                  <QuestionsPanel journeyId={journeyId} questions={questions} setQuestions={setQuestions} starters={journey?.starterQuestions ?? []}
                    draftAbout={draftAbout} clearDraftAbout={() => setDraftAbout(null)} docId={doc.id} />
                )}
              </div>
            </>
          ) : (
            <div className="empty">Facts appear here once a document is open.</div>
          )}
        </aside>
      </div>
    </>
  );
}
