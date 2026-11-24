import { buildBrief, download, type BriefDoc, type Question } from "../lib/brief";
import { journeyById, sourcesFor, workDoc, type Corrected, type LiveOutput, type UserDoc } from "../lib/data";
import { getJSON, useStored } from "../lib/store";
import { href } from "../lib/router";
import { pickDefault } from "./Workspace";

export function Brief({ journeyId }: { journeyId: string }) {
  const journey = journeyById(journeyId);
  const [userDocs] = useStored<UserDoc[]>("userDocs", []);
  const [questions] = useStored<Question[]>(`questions.${journeyId}`, []);
  const ids = [...(journey?.docIds ?? []), ...(journeyId === "mine" || !journey ? userDocs.map((d) => d.id) : [])];
  const docs: BriefDoc[] = ids.flatMap((id) => {
    const d = workDoc(id, userDocs);
    if (!d) return [];
    const corrected = getJSON<Corrected | null>(`corr.${id}`, null);
    const src = pickDefault(sourcesFor(d, corrected, getJSON<LiveOutput[]>(`live.${id}`, [])));
    return src.checked ? [{ title: d.title, sourceLabel: src.label, checked: src.checked, corrected: src.kind === "yours" }] : [];
  });
  const title = journey ? `${journey.person} (fictional example)` : "My documents";
  const md = buildBrief(title, docs, questions);
  return (
    <div className="page">
      <div className="row no-print" style={{ marginBottom: 12 }}>
        <a href={href(`journey/${journeyId}`)}>← Back to the documents</a>
        <span className="spacer" />
        <button className="btn" onClick={() => window.print()}>Print or save as PDF</button>
        <button className="btn primary" onClick={() => download(`appointment-brief-${journeyId}.md`, md)}>Download (.md)</button>
      </div>
      {docs.length === 0 && <div className="notice warn no-print">There are no documents in this brief yet.</div>}
      <article className="brief-sheet"><pre>{md}</pre></article>
    </div>
  );
}
