import { useState } from "react";
import type { Question } from "../lib/brief";
import { href } from "../lib/router";
import { newId } from "../lib/data";

interface Props {
  journeyId: string;
  questions: Question[];
  setQuestions: (q: Question[]) => void;
  starters: string[];
  draftAbout: string | null;
  clearDraftAbout: () => void;
  docId: string;
}

export function QuestionsPanel({ journeyId, questions, setQuestions, starters, draftAbout, clearDraftAbout, docId }: Props) {
  const [text, setText] = useState("");
  const add = (t: string, about?: string | null) => {
    if (!t.trim()) return;
    setQuestions([...questions, { id: newId("q"), text: t.trim(), about: about ?? undefined, docId, createdAt: new Date().toISOString() }]);
    setText("");
    clearDraftAbout();
  };
  return (
    <div>
      <h3 style={{ marginTop: 12 }}>Questions for the appointment</h3>
      <p className="tiny muted">Saved only in this browser. They go into the brief you can download.</p>
      <form onSubmit={(e) => { e.preventDefault(); add(text, draftAbout); }} className="stack">
        {draftAbout && (
          <div className="notice info small">
            About: <strong>{draftAbout}</strong> <button type="button" className="btn ghost small" onClick={clearDraftAbout}>clear</button>
          </div>
        )}
        <textarea rows={2} placeholder="Type a question to ask…" value={text} onChange={(e) => setText(e.target.value)} aria-label="New question" />
        <div className="row"><button className="btn primary small" type="submit" disabled={!text.trim()}>Add question</button></div>
      </form>
      {questions.length === 0 ? (
        <div className="empty small">
          No questions yet.
          {starters.length > 0 && (
            <div style={{ marginTop: 8 }}>
              {starters.map((s) => <button key={s} className="btn small" onClick={() => add(s)}>+ {s}</button>)}
            </div>
          )}
        </div>
      ) : (
        <ul className="q-list" style={{ marginTop: 12 }}>
          {questions.map((q, i) => (
            <li key={q.id}>
              <span className="muted">{i + 1}.</span>
              <div style={{ flex: 1 }}>{q.text}{q.about && <div className="about">About: {q.about}</div>}</div>
              <button className="btn ghost small" aria-label={`Remove question ${i + 1}`} onClick={() => setQuestions(questions.filter((x) => x.id !== q.id))}>Remove</button>
            </li>
          ))}
        </ul>
      )}
      <div className="row" style={{ marginTop: 16 }}>
        <a className="btn" href={href(`brief/${journeyId}`)}>Preview and export the appointment brief</a>
      </div>
    </div>
  );
}
