import { useState } from "react";
import { MetricChart, MetricTable, SYSTEMS, type MetricRow, type SystemKey } from "../components/MetricChart";
import { PREPARED, SAMPLE_DOCS, sourcesFor } from "../lib/data";
import { href } from "../lib/router";

type Metrics = Partial<Record<string, number | null>>;
interface Summary {
  generatedOn?: string;
  testSet?: { docs: number; patients: number; heldOutFamilies: string[] };
  challengeSet?: { docs: number };
  groups?: Record<string, Partial<Record<SystemKey, Metrics>>>;
  training?: Record<string, unknown>;
  browser?: Record<string, unknown>;
  notes?: string[];
}

const ROWS: { key: string; label: string; help: string }[] = [
  { key: "all_f1", label: "All facts (F1)", help: "Balance of finding every fact (recall) and not adding wrong ones (precision), across all fields the method attempts." },
  { key: "labs_f1", label: "Lab results (F1)", help: "A lab result counts as found when the test name and the exact printed value match." },
  { key: "unit_exact", label: "Units copied exactly", help: "Of the lab results found, how many have exactly the printed unit." },
  { key: "range_exact", label: "Ranges copied exactly", help: "Of the lab results found that have a printed range, how many have exactly that range." },
  { key: "meds_f1", label: "Medicines (F1)", help: "A medicine counts as found when its name matches." },
  { key: "meds_status_acc", label: "Medicine status correct", help: "Of the medicines found, how many were labelled current / not current / unclear correctly." },
  { key: "follow_up_f1", label: "Follow-up (F1)", help: "Follow-up instructions matched by wording overlap." },
  { key: "date_acc", label: "Document date correct", help: "Exact printed date, or correctly reporting that there is none." },
  { key: "schema_valid", label: "Output follows the format", help: "The output was valid JSON with exactly the expected fields." },
  { key: "evidence_valid", label: "Quotes found in the document", help: "Share of extracted facts whose supporting quote appears in the document." },
];

function rowsFor(g: Partial<Record<SystemKey, Metrics>> | undefined): MetricRow[] {
  if (!g) return [];
  return ROWS.map((r) => ({ label: r.label, help: r.help, values: Object.fromEntries(SYSTEMS.map((s) => [s.key, g[s.key]?.[r.key] ?? null])) as MetricRow["values"] }))
    .filter((r) => Object.values(r.values).some((v) => typeof v === "number"));
}

const GROUPS: { key: string; label: string }[] = [
  { key: "test", label: "Test set (all)" },
  { key: "test_seen", label: "Seen layouts" },
  { key: "test_held_out", label: "Unseen layouts" },
  { key: "challenge", label: "Challenge set" },
];

export function ModelPage() {
  const ev = PREPARED.evaluation as Summary;
  const [group, setGroup] = useState("test");
  const [view, setView] = useState<"chart" | "table">("chart");
  const [docId, setDocId] = useState(SAMPLE_DOCS[0]?.id ?? "");
  const doc = SAMPLE_DOCS.find((d) => d.id === docId);
  const rows = rowsFor(ev.groups?.[group]);
  const srcs = doc ? sourcesFor({ text: doc.text, sample: doc }, null, []) : [];
  const orig = srcs.find((s) => s.key === "original");
  const ft = srcs.find((s) => s.key === "finetuned");
  const unsupported = (k: SystemKey) => ev.groups?.[group]?.[k]?.unsupported_items;

  return (
    <div className="page">
      <h1>Original model vs fine-tuned model</h1>
      <p className="muted" style={{ maxWidth: 780 }}>
        Both are Qwen3-1.7B. The fine-tuned one has a small add-on (a LoRA adapter) trained on synthetic documents to return facts with
        exact quotes. Both get the same instructions, the same settings and the same length limit. Fine-tuning here is about following
        the format and copying text faithfully; it is not medical validation.
      </p>

      <h2 style={{ marginTop: 28 }}>Evaluation results</h2>
      {!ev.groups ? (
        <div className="notice">The evaluation has not been run yet, so there are no numbers to show. The prepared examples below still work.</div>
      ) : (
        <div className="card pad stack">
          <div className="row">
            <div className="source-select" role="group" aria-label="Which documents">
              {GROUPS.filter((g) => ev.groups?.[g.key]).map((g) => <button key={g.key} aria-pressed={group === g.key} onClick={() => setGroup(g.key)}>{g.label}</button>)}
            </div>
            <span className="spacer" />
            <div className="source-select" role="group" aria-label="View">
              <button aria-pressed={view === "chart"} onClick={() => setView("chart")}>Chart</button>
              <button aria-pressed={view === "table"} onClick={() => setView("table")}>Table</button>
            </div>
          </div>
          {view === "chart" ? <MetricChart rows={rows} /> : <MetricTable rows={rows} />}
          <p className="small">
            Facts that do not appear in the document at all (made up or altered values):
            {" "}{SYSTEMS.map((s) => `${s.label} ${unsupported(s.key) ?? "–"}`).join(" · ")}.
          </p>
          <p className="tiny muted">
            {ev.testSet && <>Test set: {ev.testSet.docs} synthetic documents from {ev.testSet.patients} patients never used in training; layouts {ev.testSet.heldOutFamilies.join(" and ")} were never seen in training. </>}
            {ev.challengeSet && <>Challenge set: {ev.challengeSet.docs} hand-written documents with tricky cases. </>}
            Results on synthetic documents do not show how well this would work on real clinical records. Evaluated on {ev.generatedOn}.
          </p>
          {ev.notes?.length ? <ul className="small">{ev.notes.map((n) => <li key={n}>{n}</li>)}</ul> : null}
        </div>
      )}

      <h2 style={{ marginTop: 28 }}>Side by side on the example documents</h2>
      <p className="small muted">Prepared results made before publishing this page, not live. To run a model yourself, open a document and choose "Run a model on this device".</p>
      <div className="row" style={{ marginBottom: 10 }}>
        <label className="field" style={{ minWidth: 280 }}><span>Document</span>
          <select value={docId} onChange={(e) => setDocId(e.target.value)}>
            {PREPARED.journeys.map((j) => <optgroup key={j.id} label={j.title}>{j.docIds.map((id) => <option key={id} value={id}>{SAMPLE_DOCS.find((d) => d.id === id)?.title}</option>)}</optgroup>)}
          </select>
        </label>
        {doc && <a className="btn small" href={href(`journey/${doc.journey}`, { doc: doc.id })}>Open in the workspace</a>}
      </div>
      <div className="side-by-side">
        {[["Original model", orig], ["Fine-tuned model", ft]].map(([label, s]) => {
          const src = s as typeof orig;
          return (
            <div key={label as string} className="card pad">
              <h3>{label as string}</h3>
              {!src ? <p className="small muted">No prepared result yet.</p> : (
                <>
                  {src.parseError ? <div className="notice bad small">{src.parseError}</div> : (
                    <p className="small">
                      {src.checked!.total} facts · {src.checked!.total - src.checked!.needsReview} found in the text · {src.checked!.needsReview} to check
                      {src.schemaErrors.length ? <> · <span className="pill check">format problems: {src.schemaErrors.length}</span></> : <> · <span className="pill ok">format OK</span></>}
                    </p>
                  )}
                  {src.checked && (
                    <ul className="small" style={{ paddingLeft: 18 }}>
                      {src.checked.extraction.labs.map((l, i) => <li key={"l" + i}>{l.test} <span className="mono">{l.value} {l.unit}</span>{src.checked!.labs[i].status === "verified" ? "" : " ⚠"}</li>)}
                      {src.checked.extraction.medications.map((m, i) => <li key={"m" + i}>{m.name} · {m.status.replace("_", " ")}{src.checked!.medications[i].status === "verified" ? "" : " ⚠"}</li>)}
                      {src.checked.extraction.follow_up.map((f, i) => <li key={"f" + i}>{f.instruction}{src.checked!.follow_up[i].status === "verified" ? "" : " ⚠"}</li>)}
                    </ul>
                  )}
                  <details><summary className="small">Raw output</summary><div className="raw">{src.raw}</div></details>
                  {typeof src.latencyMs === "number" && <p className="tiny muted">Generation time when prepared: {(src.latencyMs / 1000).toFixed(1)} s (batched, GTX 1650).</p>}
                </>
              )}
            </div>
          );
        })}
      </div>
      <p className="tiny muted" style={{ marginTop: 8 }}>⚠ marks a fact whose quote or values could not be found in the document.</p>
    </div>
  );
}
