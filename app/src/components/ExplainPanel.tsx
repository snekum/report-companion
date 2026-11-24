import { aboutDocument, GLOSSARY_ACCESSED, termsInDocument } from "../lib/explain";
import type { CheckedExtraction } from "../lib/validate";

export function ExplainPanel({ checked, sourceLabel }: { checked: CheckedExtraction; sourceLabel: string }) {
  const about = aboutDocument(checked);
  const terms = termsInDocument(checked);
  return (
    <div>
      <h3 style={{ marginTop: 12 }}>About this document</h3>
      <p className="tiny muted">Worked out by code from the facts shown in the Facts tab ({sourceLabel}). If a fact is wrong, correct it there and this updates.</p>
      <ul className="about-list">
        {about.map((s, i) => <li key={i}>{s.text}{s.detail && <div className="small muted">{s.detail}</div>}</li>)}
      </ul>

      <h3 style={{ marginTop: 22 }}>General definitions</h3>
      <p className="tiny muted">
        What these terms mean in general, written in plain words and based on the linked MedlinePlus pages (checked {GLOSSARY_ACCESSED}).
        They say nothing about this person's results.
      </p>
      {terms.length === 0 && <p className="small muted">No supported terms were found in this document yet.</p>}
      {terms.map((t) => (
        <div className="def" key={t.id}>
          <h4>{t.title}</h4>
          <p>{t.definition}</p>
          <a className="tiny" href={t.source.url} target="_blank" rel="noreferrer">{t.source.title}</a>
        </div>
      ))}
      <p className="tiny muted" style={{ marginTop: 16 }}>Report Companion organises documents. It does not diagnose, interpret results or suggest treatment.</p>
    </div>
  );
}
