// Code-side checks on any extraction, whatever produced it (model, rules or a
// person's correction). Character offsets for highlighting are computed here
// from the source text; offsets suggested by a model are never used.
//
// Mirrors model/common.py (parse_output, schema_errors, item_support), and the
// shared cases in src/lib/validate.test.ts keep the two in step.

export type Section = "labs" | "medications" | "follow_up";
export type MedStatus = "current" | "not_current" | "unclear";

export interface DocDate { value: string; evidence: string }
export interface Lab { test: string; value: string; unit: string | null; reference_range: string | null; date: string | null; evidence: string }
export interface Medication { name: string; dose: string | null; frequency: string | null; status: MedStatus; evidence: string }
export interface FollowUp { instruction: string; evidence: string }
export interface Extraction { document_date: DocDate | null; labs: Lab[]; medications: Medication[]; follow_up: FollowUp[] }

export const EMPTY_EXTRACTION: Extraction = { document_date: null, labs: [], medications: [], follow_up: [] };

// ------------------------------------------------------------------ parsing
export interface ParseResult { value: Extraction | null; raw: unknown; error: string | null; schemaErrors: string[] }

export function parseModelOutput(raw: string): ParseResult {
  let s = raw.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
  s = s.replace(/^```(?:json)?\s*/, "").replace(/\s*```$/, "").trim();
  const start = s.indexOf("{");
  if (start < 0) return { value: null, raw: null, error: "The model did not return a JSON object.", schemaErrors: [] };
  let depth = 0, end = -1, inStr = false, esc = false;
  for (let i = start; i < s.length; i++) {
    const c = s[i];
    if (inStr) {
      if (esc) esc = false;
      else if (c === "\\") esc = true;
      else if (c === '"') inStr = false;
      continue;
    }
    if (c === '"') inStr = true;
    else if (c === "{") depth++;
    else if (c === "}") { depth--; if (depth === 0) { end = i; break; } }
  }
  if (end < 0) return { value: null, raw: null, error: "The output stopped before the JSON was complete (it probably hit the length limit).", schemaErrors: [] };
  let obj: unknown;
  try { obj = JSON.parse(s.slice(start, end + 1)); } catch (e) {
    return { value: null, raw: null, error: `The output was not valid JSON (${(e as Error).message}).`, schemaErrors: [] };
  }
  const errs = schemaErrors(obj);
  return { value: coerce(obj), raw: obj, error: null, schemaErrors: errs };
}

const isStrOrNull = (v: unknown) => v === null || typeof v === "string";
const isNonEmpty = (v: unknown) => typeof v === "string" && v.trim().length > 0;
const sameKeys = (o: object, keys: string[]) => {
  const k = Object.keys(o).sort();
  return k.length === keys.length && [...keys].sort().every((x, i) => x === k[i]);
};

export function schemaErrors(obj: unknown): string[] {
  const errs: string[] = [];
  if (!obj || typeof obj !== "object" || Array.isArray(obj)) return ["top level is not an object"];
  const o = obj as Record<string, unknown>;
  if (!sameKeys(o, ["document_date", "labs", "medications", "follow_up"])) errs.push("unexpected or missing top-level keys");
  const dd = o.document_date;
  if (dd !== null && dd !== undefined) {
    if (typeof dd !== "object" || !sameKeys(dd as object, ["value", "evidence"]) || !isNonEmpty((dd as DocDate).value) || !isNonEmpty((dd as DocDate).evidence)) {
      errs.push("document_date malformed");
    }
  }
  const spec: Record<Section, { fields: string[]; required: string[] }> = {
    labs: { fields: ["test", "value", "unit", "reference_range", "date", "evidence"], required: ["test", "value", "evidence"] },
    medications: { fields: ["name", "dose", "frequency", "status", "evidence"], required: ["name", "evidence"] },
    follow_up: { fields: ["instruction", "evidence"], required: ["instruction", "evidence"] },
  };
  for (const sec of Object.keys(spec) as Section[]) {
    const arr = o[sec];
    if (!Array.isArray(arr)) { errs.push(`${sec} is not a list`); continue; }
    arr.forEach((it, i) => {
      if (!it || typeof it !== "object" || !sameKeys(it, spec[sec].fields)) { errs.push(`${sec}[${i}] has wrong fields`); return; }
      const r = it as Record<string, unknown>;
      for (const f of spec[sec].fields) {
        if (spec[sec].required.includes(f)) { if (!isNonEmpty(r[f])) errs.push(`${sec}[${i}].${f} must be a non-empty string`); }
        else if (f === "status") { if (!["current", "not_current", "unclear"].includes(r[f] as string)) errs.push(`${sec}[${i}].status invalid`); }
        else if (!isStrOrNull(r[f])) errs.push(`${sec}[${i}].${f} must be text or null`);
      }
    });
  }
  return errs;
}

const str = (v: unknown): string | null => (typeof v === "string" ? v : null);

/** Best-effort view of partially invalid output; items that are not objects are dropped. */
export function coerce(obj: unknown): Extraction {
  const out: Extraction = { document_date: null, labs: [], medications: [], follow_up: [] };
  if (!obj || typeof obj !== "object") return out;
  const o = obj as Record<string, unknown>;
  const dd = o.document_date as Record<string, unknown> | null;
  if (dd && typeof dd === "object" && typeof dd.value === "string") out.document_date = { value: dd.value, evidence: str(dd.evidence) ?? "" };
  const items = (k: string) => (Array.isArray(o[k]) ? (o[k] as unknown[]).filter((x) => x && typeof x === "object") as Record<string, unknown>[] : []);
  out.labs = items("labs").map((r) => ({ test: str(r.test) ?? "", value: str(r.value) ?? "", unit: str(r.unit), reference_range: str(r.reference_range), date: str(r.date), evidence: str(r.evidence) ?? "" }));
  out.medications = items("medications").map((r) => ({
    name: str(r.name) ?? "", dose: str(r.dose), frequency: str(r.frequency),
    status: (["current", "not_current", "unclear"].includes(r.status as string) ? r.status : "unclear") as MedStatus,
    evidence: str(r.evidence) ?? "",
  }));
  out.follow_up = items("follow_up").map((r) => ({ instruction: str(r.instruction) ?? "", evidence: str(r.evidence) ?? "" }));
  return out;
}

// ------------------------------------------------------------------ locating evidence
export interface Span { start: number; end: number }

/**
 * Find `quote` in `source`, ignoring differences in whitespace (a model often
 * collapses the padding of table columns). Returns every match as offsets into
 * the original source text.
 */
export function findSpans(source: string, quote: string): Span[] {
  const q = quote.trim().split(/\s+/).filter(Boolean);
  if (!q.length) return [];
  const pattern = q.map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("\\s+");
  const re = new RegExp(pattern, "g");
  const spans: Span[] = [];
  let m: RegExpExecArray | null;
  while ((m = re.exec(source)) !== null) {
    spans.push({ start: m.index, end: m.index + m[0].length });
    if (m[0].length === 0) re.lastIndex++;
  }
  return spans;
}

const normWs = (s: string) => s.replace(/\s+/g, " ").trim();
const escapeRe = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/** `needle` occurs in `haystack` and is not part of a longer number. */
export function tokenIn(haystack: string, needle: string, caseSensitive = true): boolean {
  let h = normWs(haystack), n = normWs(needle);
  if (!n) return false;
  if (!caseSensitive) { h = h.toLowerCase(); n = n.toLowerCase(); }
  let pat = escapeRe(n);
  if (/^[\d<>.]/.test(n)) pat = "(?<![\\d.])" + pat;
  if (/\d$/.test(n)) pat = pat + "(?!\\d)";
  return new RegExp(pat).test(h);
}

// ------------------------------------------------------------------ checking items
export type Check = "verified" | "needs_review";

export interface FactCheck {
  status: Check;
  /** Offsets of the evidence in the source; null when the quote was not found. */
  span: Span | null;
  /** Plain-language reasons shown to the reader when something needs review. */
  reasons: string[];
}

const FIELD_LABEL: Record<string, string> = {
  test: "test name", value: "value", unit: "unit", reference_range: "reference range", date: "date",
  name: "medicine name", dose: "dose", frequency: "frequency", instruction: "instruction",
};

const KEY_FIELDS: Record<Section | "document_date", string[]> = {
  labs: ["test", "value", "unit", "reference_range", "date"],
  medications: ["name", "dose", "frequency"],
  follow_up: ["instruction"],
  document_date: ["value"],
};

export function checkFact(source: string, section: Section | "document_date", item: Record<string, unknown>): FactCheck {
  const reasons: string[] = [];
  const evidence = typeof item.evidence === "string" ? item.evidence : "";
  const spans = evidence ? findSpans(source, evidence) : [];
  if (!evidence) reasons.push("No supporting quote was given.");
  else if (!spans.length) reasons.push("The supporting quote does not appear in the document.");
  else if (spans.length > 1) reasons.push(`The supporting quote appears ${spans.length} times, so the highlighted one may not be the right one.`);
  for (const f of KEY_FIELDS[section]) {
    const v = item[f];
    if (typeof v !== "string" || !v) continue;
    const caseSensitive = !["test", "name", "instruction"].includes(f);
    if (!tokenIn(source, v, caseSensitive)) reasons.push(`The ${FIELD_LABEL[f]} "${v}" does not appear in the document.`);
    else if (evidence && spans.length && !tokenIn(evidence, v, caseSensitive)) reasons.push(`The ${FIELD_LABEL[f]} "${v}" is not inside the supporting quote.`);
  }
  if (section === "labs" && typeof item.value === "string" && !/\d/.test(item.value)) {
    reasons.push("The value has no number in it, so it cannot be charted.");
  }
  return { status: reasons.length ? "needs_review" : "verified", span: spans[0] ?? null, reasons };
}

export interface CheckedExtraction {
  extraction: Extraction;
  date: FactCheck | null;
  labs: FactCheck[];
  medications: FactCheck[];
  follow_up: FactCheck[];
  needsReview: number;
  total: number;
}

export function checkExtraction(source: string, ex: Extraction): CheckedExtraction {
  const date = ex.document_date ? checkFact(source, "document_date", ex.document_date as unknown as Record<string, unknown>) : null;
  const labs = ex.labs.map((x) => checkFact(source, "labs", x as unknown as Record<string, unknown>));
  const medications = ex.medications.map((x) => checkFact(source, "medications", x as unknown as Record<string, unknown>));
  const follow_up = ex.follow_up.map((x) => checkFact(source, "follow_up", x as unknown as Record<string, unknown>));
  const all = [date, ...labs, ...medications, ...follow_up].filter(Boolean) as FactCheck[];
  return { extraction: ex, date, labs, medications, follow_up, total: all.length, needsReview: all.filter((c) => c.status === "needs_review").length };
}
