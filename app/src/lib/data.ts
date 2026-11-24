// The documents a visitor works with, and the extraction "sources" for each:
// prepared results (made earlier, clearly labelled), live results from this
// device, rule-based results, and the visitor's own corrected version.
import prepared from "../data/prepared.json";
import { checkExtraction, coerce, parseModelOutput, type CheckedExtraction, type Extraction } from "./validate";
import { extractWithRules } from "./rules";

export interface PreparedOutput { label: string; description: string; raw: string; latencyMs?: number | null; generatedOn: string; runtime: string }
export interface SampleDoc {
  id: string; journey: string; title: string; kind: string; family: string; text: string;
  reference: Extraction; outputs: Partial<Record<"finetuned" | "original" | "rules", PreparedOutput>>;
  provenance: Record<string, unknown>;
}
export interface Journey {
  id: string; title: string; person: string; who: string; goal: string;
  docIds: string[]; compare: [string, string] | null; starterQuestions: string[];
}
export interface LiveOutput { key: "original" | "finetuned"; raw: string; latencyMs: number; ranAt: string; modelId: string }
export interface UserDoc { id: string; title: string; text: string; createdAt: string }

export interface EvalData { [k: string]: unknown }

export const PREPARED = prepared as unknown as { builtOn: string; journeys: Journey[]; documents: SampleDoc[]; evaluation: EvalData };
export const JOURNEYS = PREPARED.journeys;
export const SAMPLE_DOCS = PREPARED.documents;
export const sampleById = (id: string) => SAMPLE_DOCS.find((d) => d.id === id);
export const journeyById = (id: string) => JOURNEYS.find((j) => j.id === id);

export type MarkedItem<T> = T & { _mark?: "edited" | "added" };
export interface Corrected {
  basedOn: string;
  updatedAt: string;
  extraction: {
    document_date: MarkedItem<NonNullable<Extraction["document_date"]>> | null;
    labs: MarkedItem<Extraction["labs"][number]>[];
    medications: MarkedItem<Extraction["medications"][number]>[];
    follow_up: MarkedItem<Extraction["follow_up"][number]>[];
  };
  removed: number;
}

export type SourceKind = "prepared" | "live" | "rules" | "yours";

export interface SourceOption {
  key: string;
  kind: SourceKind;
  label: string;
  /** One line saying exactly where this result came from. */
  note: string;
  raw: string | null;
  parseError: string | null;
  schemaErrors: string[];
  checked: CheckedExtraction | null;
  latencyMs?: number | null;
}

export function workDoc(id: string, userDocs: UserDoc[]): { id: string; title: string; text: string; sample: SampleDoc | null; user: UserDoc | null } | null {
  const s = sampleById(id);
  if (s) return { id, title: s.title, text: s.text, sample: s, user: null };
  const u = userDocs.find((d) => d.id === id);
  if (u) return { id, title: u.title, text: u.text, sample: null, user: u };
  return null;
}

function fromRaw(key: string, kind: SourceKind, label: string, note: string, text: string, raw: string, latencyMs?: number | null): SourceOption {
  const p = parseModelOutput(raw);
  return {
    key, kind, label, note, raw, parseError: p.error, schemaErrors: p.schemaErrors,
    checked: p.value ? checkExtraction(text, p.value) : null, latencyMs,
  };
}

export function sourcesFor(doc: { text: string; sample: SampleDoc | null }, corrected: Corrected | null, live: LiveOutput[]): SourceOption[] {
  const out: SourceOption[] = [];
  if (corrected) {
    out.push({ key: "yours", kind: "yours", label: "Your corrected version", note: `Your edits on top of: ${corrected.basedOn}. Saved only in this browser.`, raw: null, parseError: null, schemaErrors: [], checked: checkExtraction(doc.text, coerce(corrected.extraction)) });
  }
  if (doc.sample) {
    const o = doc.sample.outputs;
    if (o.finetuned) out.push(fromRaw("finetuned", "prepared", o.finetuned.label.includes("pilot") ? "Fine-tuned model (pilot)" : "Fine-tuned model", `Prepared result, generated ${o.finetuned.generatedOn} (${o.finetuned.runtime}). Not run live in this browser.`, doc.text, o.finetuned.raw, o.finetuned.latencyMs));
    if (o.original) out.push(fromRaw("original", "prepared", "Original model", `Prepared result, generated ${o.original.generatedOn} (${o.original.runtime}). Not run live in this browser.`, doc.text, o.original.raw, o.original.latencyMs));
  }
  for (const l of [...live].reverse()) {
    if (out.some((s) => s.key === `live-${l.key}`)) continue;
    out.push(fromRaw(`live-${l.key}`, "live", l.key === "finetuned" ? "Fine-tuned model (run here)" : "Original model (run here)", `Run on this device on ${new Date(l.ranAt).toLocaleString()} with ${l.modelId}.`, doc.text, l.raw, l.latencyMs));
  }
  const rules = extractWithRules(doc.text);
  out.push({ key: "rules", kind: "rules", label: "Rules (lab rows only)", note: "Pattern matching run instantly in this browser. Finds lab rows and the document date; it does not read medicines or follow-up.", raw: null, parseError: null, schemaErrors: [], checked: checkExtraction(doc.text, rules) });
  return out;
}

export function newId(prefix: string): string {
  return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`;
}
