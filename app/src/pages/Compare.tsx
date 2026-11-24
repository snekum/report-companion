import { useMemo, useState } from "react";
import { RangeChart } from "../components/RangeChart";
import { compareMeasurements, parseRange, parseValue, type Measurement } from "../lib/compare";
import { journeyById, sourcesFor, workDoc, type Corrected, type LiveOutput, type UserDoc } from "../lib/data";
import { getJSON, useStored } from "../lib/store";
import { href } from "../lib/router";
import { pickDefault } from "./Workspace";

export function Compare({ journeyId, a, b }: { journeyId: string; a: string | null; b: string | null }) {
  const journey = journeyById(journeyId);
  const [userDocs] = useStored<UserDoc[]>("userDocs", []);
  const candidates = [...(journey?.docIds ?? []), ...userDocs.map((d) => d.id)];
  const [left, setLeft] = useState(a ?? journey?.compare?.[0] ?? candidates[0] ?? "");
  const [right, setRight] = useState(b ?? journey?.compare?.[1] ?? candidates[1] ?? "");
  const [view, setView] = useState<"charts" | "table">("charts");

  const prepared = useMemo(() => [left, right].map((id) => {
    const d = workDoc(id, userDocs);
    if (!d) return null;
    const all = sourcesFor(d, getJSON<Corrected | null>(`corr.${id}`, null), getJSON<LiveOutput[]>(`live.${id}`, []));
    // Your corrections always win. Otherwise use the result with the most lab
    // values that passed the text checks, and say which one it was.
    const yours = all.find((s) => s.kind === "yours");
    const verifiedLabs = (s: (typeof all)[number]) => s.checked?.labs.filter((c) => c.status === "verified").length ?? -1;
    const src = yours ?? [...all].sort((x, y) => verifiedLabs(y) - verifiedLabs(x))[0] ?? pickDefault(all);
    return { d, src };
  }), [left, right, userDocs]);

  const ms: Measurement[] = prepared.flatMap((p) => {
    if (!p?.src.checked) return [];
    const c = p.src.checked;
    return c.extraction.labs.map((lab, i) => ({
      docId: p.d.id, docTitle: p.d.title, lab, dateText: lab.date ?? c.extraction.document_date?.value ?? null, dateFromDocument: !lab.date,
      value: parseValue(lab.value), range: parseRange(lab.reference_range), verified: c.labs[i]?.status === "verified",
    }));
  });
  const cmp = compareMeasurements(ms);
  const same = left === right;

  return (
    <div className="page">
      <p className="small"><a href={href(`journey/${journeyId}`)}>← Back to the documents</a></p>
      <h1>Compare two reports</h1>
      <p className="muted" style={{ maxWidth: 760 }}>
        Only results for the same test, printed in the same unit, are put side by side. Each result is shown against the range printed
        on <em>its own</em> report, because labs can use different ranges. Nothing is converted.
      </p>
      <div className="card pad row" style={{ gap: 14, alignItems: "flex-end" }}>
        {[["First report", left, setLeft], ["Second report", right, setRight]].map(([label, val, set]) => (
          <label key={label as string} className="field" style={{ minWidth: 220, flex: 1 }}>
            <span>{label as string}</span>
            <select value={val as string} onChange={(e) => (set as (v: string) => void)(e.target.value)}>
              {candidates.map((id) => <option key={id} value={id}>{workDoc(id, userDocs)?.title}</option>)}
            </select>
          </label>
        ))}
        <div className="row" role="group" aria-label="View">
          <button className="btn small" aria-pressed={view === "charts"} onClick={() => setView("charts")} style={view === "charts" ? { borderColor: "var(--accent)" } : undefined}>Charts</button>
          <button className="btn small" aria-pressed={view === "table"} onClick={() => setView("table")} style={view === "table" ? { borderColor: "var(--accent)" } : undefined}>Table</button>
        </div>
      </div>
      <p className="tiny muted" style={{ marginTop: 6 }}>
        Facts used: {prepared.map((p, i) => p ? `${i ? "second" : "first"} report from ${p.src.label.toLowerCase()}` : "").filter(Boolean).join("; ")}. For each report this uses your corrected version if you made one, otherwise the result with the most lab values found in the text.
      </p>

      {same ? <div className="notice warn">Pick two different reports to compare.</div> : (
        <>
          <p style={{ marginTop: 16 }}>
            <strong>{cmp.rows.length}</strong> test{cmp.rows.length === 1 ? "" : "s"} can be compared.
            {cmp.incompatible.length > 0 && <> {cmp.incompatible.length} appear in both but are not compared (see below).</>}
            {cmp.onlyInOne.length > 0 && <> {cmp.onlyInOne.length} result{cmp.onlyInOne.length === 1 ? " appears" : "s appear"} in only one report.</>}
          </p>
          {cmp.rows.length === 0 && <div className="empty card pad">These two reports have no results for the same test in the same unit.</div>}
          {view === "charts" && cmp.rows.length > 0 && (
            <>
              <div className="legend" style={{ margin: "8px 0 10px" }}>
                <span><span className="sw" style={{ background: "var(--series-ft)" }} />Result</span>
                <span><span className="sw" style={{ background: "var(--surface)", border: "2px dashed var(--series-ft)" }} />Result that needs checking</span>
                <span><span className="sw band" />Range printed on that report</span>
                <span>* date of the document, because no date was printed next to the result</span>
              </div>
              <div className="cmp-grid">
                {cmp.rows.map((r) => (
                  <div key={r.key} className="card cmp-card">
                    <h3>{r.label} <span className="muted small" style={{ fontWeight: 400 }}>{r.unit}</span></h3>
                    <RangeChart row={r} />
                  </div>
                ))}
              </div>
            </>
          )}
          {view === "table" && cmp.rows.length > 0 && (
            <div className="card pad" style={{ overflowX: "auto" }}>
              <table className="data">
                <thead><tr><th>Test</th><th>Report</th><th>Date</th><th>Result</th><th>Unit</th><th>Range on that report</th><th>Checked</th></tr></thead>
                <tbody>
                  {cmp.rows.flatMap((r) => r.points.map((p, i) => (
                    <tr key={r.key + i}>
                      <td>{i === 0 ? r.label : ""}</td><td>{p.docTitle}</td><td>{p.dateText ?? ""}{p.dateFromDocument && p.dateText ? " (document date)" : ""}</td>
                      <td className="num">{p.lab.value}</td><td>{p.lab.unit}</td><td className="num">{p.lab.reference_range ?? "none"}</td>
                      <td>{p.verified ? "Found in text" : "Check this"}</td>
                    </tr>
                  )))}
                </tbody>
              </table>
            </div>
          )}
          {cmp.incompatible.length > 0 && (
            <>
              <h2 style={{ marginTop: 24 }}>In both reports but not compared</h2>
              <ul>{cmp.incompatible.map((x) => <li key={x.key}><strong>{x.label}</strong>: {x.reason}</li>)}</ul>
            </>
          )}
          <p className="small muted" style={{ marginTop: 24 }}>A change between two numbers is not a diagnosis. Labs, methods and timing all affect results; ask the clinician what any difference means.</p>
        </>
      )}
    </div>
  );
}
