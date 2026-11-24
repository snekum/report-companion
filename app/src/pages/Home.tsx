import { JOURNEYS } from "../lib/data";
import { href } from "../lib/router";

const KICKERS: Record<string, string> = { checkup: "Example 1", newdoctor: "Example 2", parent: "Example 3" };

export function Home() {
  const first = JOURNEYS[0];
  return (
    <div className="page">
      <section className="hero">
        <h1>Report Companion</h1>
        <p className="lede">Turn a confusing medical document into an organised, understandable record—with every extracted fact linked to its source.</p>
        <div className="row" style={{ marginTop: 16 }}>
          {first && <a className="btn primary" href={href(`journey/${first.id}`, { doc: first.docIds[0] })}>Open a sample document</a>}
          <a className="btn" href={href("journey/mine", { add: "1" })}>Paste your own text</a>
        </div>
        <p className="small muted" style={{ marginTop: 12 }}>
          A prototype for reading and organising documents. It does not diagnose, interpret results or suggest treatment.
          All examples are fictional, made from synthetic records.
        </p>
      </section>

      <h2>Try an example journey</h2>
      <div className="journeys">
        {JOURNEYS.map((j) => (
          <a key={j.id} className="card journey" href={href(`journey/${j.id}`, { doc: j.docIds[0] })}>
            <span className="kicker">{KICKERS[j.id] ?? "Example"}</span>
            <h3>{j.title}</h3>
            <span className="small muted">{j.who}</span>
            <span className="small">{j.goal}</span>
            <span className="go">Open {j.docIds.length} documents →</span>
          </a>
        ))}
      </div>

      <h2>What you can do</h2>
      <div className="how">
        <div className="card"><h3>Read facts next to the original</h3><p className="small muted">Click a fact and the exact line it came from lights up in the document. Facts that can't be matched to the text are marked "check this".</p></div>
        <div className="card"><h3>Fix mistakes</h3><p className="small muted">Correct, remove or add facts. Your version is checked against the text again and kept in this browser.</p></div>
        <div className="card"><h3>Compare two reports</h3><p className="small muted">See the same test from two reports side by side, each against the range printed on its own report. Different units are never mixed.</p></div>
        <div className="card"><h3>Bring a brief</h3><p className="small muted">Save your questions and download a one-page appointment brief with the facts and where each came from.</p></div>
      </div>

      <div className="notice info" style={{ marginTop: 24 }}>
        <strong>Where your text goes:</strong> nowhere. Pasted documents, corrections and questions stay in this browser. If you choose to run a
        model, it runs on this device after a one-time download of the model files; the document is never uploaded. <a href={href("about")}>How it works</a>
      </div>
    </div>
  );
}
