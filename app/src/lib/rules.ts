// Deterministic lab-row extraction. Port of model/baseline_regex.py: same
// dictionary, same patterns, so in-app rule results match the evaluated
// baseline. It finds lab rows and the document date only.
import catalog from "../../../pipeline/fixtures/lab_catalog.json";
import type { Extraction, Lab } from "./validate";

const tests = (catalog as { tests: Record<string, { names: string[]; abbrev: string }> }).tests;
const NAMES = [...new Set(Object.values(tests).flatMap((t) => [...t.names, t.abbrev]))].sort((a, b) => b.length - a.length);
const esc = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const NAME_RE = new RegExp(`(?<![A-Za-z])(${NAMES.map(esc).join("|")})(?![A-Za-z])`, "gi");
const NUM = String.raw`[<>]?\d+(?:\.\d+)?`;
const DATE = String.raw`(?:\d{4}-\d{2}-\d{2}|\d{1,2} [A-Z][a-z]{2} \d{4}|\d{2}/\d{2}/\d{4}|[A-Z][a-z]+ \d{1,2}, \d{4})`;
const RANGE = String.raw`[<>]\s?\d+(?:\.\d+)?|\d+(?:\.\d+)?\s?(?:-| - | to )\s?\d+(?:\.\d+)?`;
const RANGE_RE = new RegExp(String.raw`(?:ref(?:erence)?|normal range|lab range|range is|range|\|)\s*(${RANGE})`);
const BARE_RANGE_RE = new RegExp(String.raw`\s(${RANGE})\s*$`);
const DRUG_DOSE_RE = /\d+(?:\.\d+)? (?:MG|mg|MCG|mcg)(?:\/(?:ML|mL|ACTUAT|HR))?(?![A-Za-z/])/;
const DOC_DATE_RES = [
  String.raw`Reported:\s*(${DATE})`, String.raw`report issued\s*(${DATE})`, String.raw`Printed\s*(${DATE})`,
  String.raw`printed\s*(${DATE})\)`, String.raw`Visit date:\s*(${DATE})`, String.raw`^Date:\s*(${DATE})`,
  String.raw`care team - (${DATE})`, String.raw`COLL\s*(${DATE})`, String.raw`Specimen collected:?\s*(${DATE})`, String.raw`Collected:\s*(${DATE})`,
].map((s) => new RegExp(s, "m"));

export function extractWithRules(text: string): Extraction {
  const out: Extraction = { document_date: null, labs: [], medications: [], follow_up: [] };
  for (const rx of DOC_DATE_RES) {
    const m = rx.exec(text);
    if (m) {
      const line = text.split("\n").find((l) => l.includes(m[0])) ?? m[0];
      out.document_date = { value: m[1], evidence: line.trim() };
      break;
    }
  }
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line || line.startsWith("TEST") || line.startsWith("CODE|") || DRUG_DOSE_RE.test(line)) continue;
    NAME_RE.lastIndex = 0;
    let m: RegExpExecArray | null;
    while ((m = NAME_RE.exec(line)) !== null) {
      const rest = line.slice(m.index + m[0].length);
      const vm = new RegExp(String.raw`^\s*(?:\||:|was|came back at|)\s*(${NUM})(%?)\s*\|?\s*([^\s|(;,]+)?`).exec(rest);
      if (!vm) continue;
      let unit: string | null = vm[2] ? "%" : vm[3] ?? null;
      if (unit && (new RegExp(`^${NUM}$`).test(unit) || ["and", "mmol", "the"].includes(unit.toLowerCase()))) unit = null;
      if (unit) unit = unit.replace(/[.;]+$/, "");
      const tail = rest.slice(vm[0].length);
      const rm = RANGE_RE.exec(tail) ?? BARE_RANGE_RE.exec(tail);
      const dm = new RegExp(String.raw`^\s*(${DATE})`).exec(line);
      const lab: Lab = { test: m[1], value: vm[1], unit: unit || null, reference_range: rm ? rm[1] : null, date: dm ? dm[1] : null, evidence: line };
      out.labs.push(lab);
    }
  }
  return out;
}
