// Appointment brief: the visitor's organised facts and questions as a
// Markdown file they can print, email to themselves or bring along.
import type { CheckedExtraction } from "./validate";

export interface BriefDoc { title: string; sourceLabel: string; checked: CheckedExtraction; corrected: boolean }
export interface Question { id: string; text: string; docId?: string; about?: string; createdAt: string }

const STATUS_TEXT = { current: "Current", not_current: "Stopped / not taken", unclear: "Status not clear" } as const;

export function buildBrief(title: string, docs: BriefDoc[], questions: Question[], now = new Date()): string {
  const L: string[] = [];
  L.push(`# Appointment brief: ${title}`, "");
  L.push(`Prepared ${now.toISOString().slice(0, 10)} with Report Companion, a prototype. Facts were extracted automatically and may contain mistakes. Check each one against the original document. This is not medical advice.`, "");
  if (questions.length) {
    L.push("## My questions", "");
    questions.forEach((q, i) => L.push(`${i + 1}. ${q.text}${q.about ? ` _(about: ${q.about})_` : ""}`));
    L.push("");
  }
  const meds = docs.flatMap((d) => d.checked.extraction.medications.map((m, i) => ({ m, d, ok: d.checked.medications[i]?.status === "verified" })));
  if (meds.length) {
    L.push("## Medicines mentioned", "");
    for (const status of ["current", "not_current", "unclear"] as const) {
      const rows = meds.filter((x) => x.m.status === status);
      if (!rows.length) continue;
      L.push(`**${STATUS_TEXT[status]}**`, "");
      for (const { m, d, ok } of rows) {
        const detail = [m.dose, m.frequency].filter(Boolean).join(", ");
        L.push(`- ${m.name}${detail ? ` (${detail})` : ""}. From "${d.title}": "${m.evidence}"${ok ? "" : " [check this]"}`);
      }
      L.push("");
    }
  }
  for (const d of docs) {
    const ex = d.checked.extraction;
    L.push(`## ${d.title}${ex.document_date ? ` (${ex.document_date.value})` : ""}`, "");
    L.push(`_Facts from: ${d.sourceLabel}${d.corrected ? ", with my corrections" : ""}._`, "");
    if (ex.labs.length) {
      L.push("| Test | Result | Unit | Range printed on this report | Date | Checked |", "|---|---|---|---|---|---|");
      ex.labs.forEach((l, i) => L.push(`| ${l.test} | ${l.value} | ${l.unit ?? ""} | ${l.reference_range ?? ""} | ${l.date ?? ""} | ${d.checked.labs[i]?.status === "verified" ? "yes" : "check this"} |`));
      L.push("");
    }
    if (ex.follow_up.length) {
      L.push("Follow-up written in this document:", "");
      ex.follow_up.forEach((f) => L.push(`- ${f.instruction}`));
      L.push("");
    }
    if (!ex.labs.length && !ex.follow_up.length && !ex.medications.length) L.push("No facts were extracted from this document.", "");
  }
  return L.join("\n");
}

export function download(filename: string, text: string, type = "text/markdown") {
  const blob = new Blob([text], { type: `${type};charset=utf-8` });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
