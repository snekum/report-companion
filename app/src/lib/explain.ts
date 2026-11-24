// Plain-language layer. Two kinds of text, always shown apart:
//   1. statements about THIS document, computed from the checked facts
//      (counts, dates, what was printed) - deterministic, no model involved;
//   2. general definitions of terms from shared/glossary.json, each linked
//      to its MedlinePlus source - they say nothing about this person.
import glossary from "../../../shared/glossary.json";
import { parseRange, parseValue } from "./compare";
import type { CheckedExtraction } from "./validate";

export interface GlossaryEntry { id: string; kind: string; title: string; terms: string[]; definition: string; source: { title: string; url: string } }
export const GLOSSARY = (glossary as { entries: GlossaryEntry[]; accessed: string }).entries;
export const GLOSSARY_ACCESSED = (glossary as { accessed: string }).accessed;

const norm = (s: string) => s.toLowerCase().replace(/\s+/g, " ").trim();

export function glossaryFor(testName: string): GlossaryEntry | null {
  const n = norm(testName);
  return GLOSSARY.find((g) => g.kind !== "structure" && g.terms.includes(n)) ?? null;
}

export function structureEntry(id: string): GlossaryEntry | undefined {
  return GLOSSARY.find((g) => g.id === id);
}

export type Position = "below" | "within" | "above" | "unknown";

/** Where a printed value sits relative to the range printed next to it. Pure arithmetic on the page. */
export function positionInRange(value: string, range: string | null): Position {
  const v = parseValue(value);
  const r = parseRange(range);
  if (v === null || !r) return "unknown";
  if (r.low !== null && v < r.low) return "below";
  if (r.high !== null && v > r.high) return "above";
  return "within";
}

export interface DocStatement { text: string; detail?: string }

export function aboutDocument(c: CheckedExtraction): DocStatement[] {
  const ex = c.extraction;
  const out: DocStatement[] = [];
  if (ex.document_date) out.push({ text: `The document is dated ${ex.document_date.value}.`, detail: `From the line "${ex.document_date.evidence}".` });
  else out.push({ text: "No document date was found. Check the original for a date before relying on the order of events." });
  if (ex.labs.length) {
    const tests = new Set(ex.labs.map((l) => norm(l.test)));
    const withRange = ex.labs.filter((l) => l.reference_range);
    const dated = ex.labs.filter((l) => l.date);
    let s = `It lists ${ex.labs.length} lab result${ex.labs.length === 1 ? "" : "s"}`;
    if (tests.size < ex.labs.length) s += ` for ${tests.size} different test${tests.size === 1 ? "" : "s"}, so some tests appear more than once (repeated measurements)`;
    out.push({ text: s + "." });
    if (dated.length) out.push({ text: `${dated.length} result${dated.length === 1 ? " has" : "s have"} its own date printed beside it, which can differ from the document date.` });
    out.push({
      text: withRange.length
        ? `${withRange.length} of ${ex.labs.length} results have a reference range printed next to them. The range belongs to the lab that wrote this report.`
        : "No reference ranges are printed next to these results.",
    });
    const outside = withRange.filter((l) => ["above", "below"].includes(positionInRange(l.value, l.reference_range)));
    if (outside.length) {
      out.push({
        text: `For ${outside.length} result${outside.length === 1 ? "" : "s"} (${outside.map((l) => l.test).join(", ")}), the printed number falls outside the printed range.`,
        detail: "This is only a comparison of two numbers on the page. It is not a judgement about health; ask the clinician what it means.",
      });
    }
  }
  if (ex.medications.length) {
    const by = (s: string) => ex.medications.filter((m) => m.status === s).length;
    const parts = [
      by("current") && `${by("current")} described as current`,
      by("not_current") && `${by("not_current")} described as stopped or not taken`,
      by("unclear") && `${by("unclear")} whose status the document does not make clear (for example an old prescription)`,
    ].filter(Boolean);
    out.push({ text: `It mentions ${ex.medications.length} medicine${ex.medications.length === 1 ? "" : "s"}: ${parts.join("; ")}.` });
  }
  if (ex.follow_up.length) out.push({ text: `It contains ${ex.follow_up.length} follow-up instruction${ex.follow_up.length === 1 ? "" : "s"}.` });
  if (c.needsReview) out.push({ text: `${c.needsReview} of ${c.total} extracted facts could not be fully matched to the text and are marked "check this".` });
  return out;
}

export function termsInDocument(c: CheckedExtraction): GlossaryEntry[] {
  const found = new Map<string, GlossaryEntry>();
  for (const l of c.extraction.labs) {
    const g = glossaryFor(l.test);
    if (g) found.set(g.id, g);
  }
  const extra: GlossaryEntry[] = [];
  if (c.extraction.labs.some((l) => l.reference_range)) extra.push(structureEntry("reference_range")!);
  if (c.extraction.labs.length) extra.push(structureEntry("units")!);
  return [...extra.filter(Boolean), ...found.values()];
}
