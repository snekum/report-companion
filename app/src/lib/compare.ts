// Deciding which measurements from two reports can be compared, and turning
// printed text into numbers for charts. No unit conversion is ever done:
// different units means "not compared", and each report keeps its own range.
import type { Lab } from "./validate";

const ALIASES: Record<string, string[]> = {
  glucose: ["glucose", "blood glucose", "glucose, fasting", "fasting glucose", "glu"],
  hba1c: ["hemoglobin a1c", "haemoglobin a1c", "hba1c", "a1c"],
  cholesterol_total: ["total cholesterol", "cholesterol, total", "cholesterol", "chol"],
  ldl: ["ldl cholesterol", "ldl-c (direct)", "ldl-c", "ldl"],
  hdl: ["hdl cholesterol", "hdl-c", "hdl"],
  triglycerides: ["triglycerides", "triglyceride", "trig"],
  creatinine: ["creatinine", "creatinine, serum", "creat"],
  egfr: ["egfr", "estimated gfr"],
  bun: ["urea nitrogen (bun)", "blood urea nitrogen", "urea nitrogen", "bun"],
  sodium: ["sodium", "na"],
  potassium: ["potassium", "k"],
  chloride: ["chloride", "cl"],
  co2: ["carbon dioxide (co2)", "bicarbonate (total co2)", "carbon dioxide", "bicarbonate", "co2"],
  calcium: ["calcium", "calcium, total", "ca"],
  hemoglobin: ["hemoglobin", "haemoglobin", "hgb"],
  wbc: ["white blood cell count", "wbc", "leukocytes"],
  rbc: ["red blood cell count", "rbc"],
  hematocrit: ["hematocrit", "haematocrit", "hct"],
  mcv: ["mcv"],
  mch: ["mch"],
  mchc: ["mchc"],
  platelets: ["platelet count", "platelets", "plt"],
  protein: ["total protein", "protein, total", "tp"],
  albumin: ["albumin", "alb"],
  bilirubin: ["bilirubin, total", "total bilirubin", "bilirubin", "tbil"],
  alp: ["alkaline phosphatase", "alp"],
  alt: ["alt", "alanine aminotransferase (alt)", "alanine aminotransferase"],
  ast: ["ast", "aspartate aminotransferase (ast)", "aspartate aminotransferase"],
};

const LOOKUP = new Map<string, string>();
for (const [key, names] of Object.entries(ALIASES)) for (const n of names) LOOKUP.set(n, key);

const norm = (s: string) => s.toLowerCase().replace(/\s+/g, " ").trim();

/** Canonical test key for a printed test name; unknown names fall back to the name itself. */
export function testKey(printed: string): string {
  const n = norm(printed);
  return LOOKUP.get(n) ?? `name:${n}`;
}

export function unitKey(unit: string | null): string {
  if (!unit) return "";
  return unit.toLowerCase().replace(/µ/g, "u").replace(/\s+/g, "").replace(/^x(?=10)/, "");
}

/** Parse a printed value to a number. "<0.5", "Negative" etc. return null (not chartable). */
export function parseValue(v: string): number | null {
  const s = v.trim();
  if (/^[<>≤≥]/.test(s)) return null;
  const m = s.match(/^-?\d+(?:[.,]\d+)?$/);
  if (!m) return null;
  return Number(s.replace(",", "."));
}

export interface ParsedRange { low: number | null; high: number | null }

/** "70-99", "4.8 - 5.6", "70 to 100", "3.9–5.5", "<200", ">40", "12,0-16,0". */
export function parseRange(r: string | null): ParsedRange | null {
  if (!r) return null;
  const s = r.trim().replace(/(\d),(\d)/g, "$1.$2");
  let m = s.match(/^(\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(\d+(?:\.\d+)?)$/);
  if (m) return { low: Number(m[1]), high: Number(m[2]) };
  m = s.match(/^(<|≤|<=)\s*(\d+(?:\.\d+)?)$/);
  if (m) return { low: null, high: Number(m[2]) };
  m = s.match(/^(>|≥|>=)\s*(\d+(?:\.\d+)?)$/);
  if (m) return { low: Number(m[2]), high: null };
  return null;
}

const MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"];

/**
 * Best-effort parse of a printed date, only used for ordering points.
 * Slash dates are read month-first (US style) and flagged as ambiguous when
 * both parts could be a month. The printed text is always what is shown.
 */
export function parseDate(s: string | null | undefined): { time: number; ambiguous: boolean } | null {
  if (!s) return null;
  const t = s.trim();
  let m = t.match(/^(\d{4})-(\d{2})-(\d{2})/);
  if (m) return { time: Date.UTC(+m[1], +m[2] - 1, +m[3]), ambiguous: false };
  m = t.match(/^(\d{1,2})\/(\d{1,2})\/(\d{4})$/);
  if (m) return { time: Date.UTC(+m[3], +m[1] - 1, +m[2]), ambiguous: +m[1] <= 12 && +m[2] <= 12 && m[1] !== m[2] };
  m = t.match(/^(\d{1,2})\.(\d{1,2})\.(\d{4})$/);
  if (m) return { time: Date.UTC(+m[3], +m[2] - 1, +m[1]), ambiguous: false };
  m = t.match(/^(\d{1,2}) ([A-Za-z]{3,})\.? (\d{4})$/);
  if (m && MONTHS.indexOf(m[2].slice(0, 3).toLowerCase()) >= 0) return { time: Date.UTC(+m[3], MONTHS.indexOf(m[2].slice(0, 3).toLowerCase()), +m[1]), ambiguous: false };
  m = t.match(/^([A-Za-z]{3,})\.? (\d{1,2}),? (\d{4})$/);
  if (m && MONTHS.indexOf(m[1].slice(0, 3).toLowerCase()) >= 0) return { time: Date.UTC(+m[3], MONTHS.indexOf(m[1].slice(0, 3).toLowerCase()), +m[2]), ambiguous: false };
  m = t.match(/^([A-Za-z]{3,}) (\d{4})$/);
  if (m && MONTHS.indexOf(m[1].slice(0, 3).toLowerCase()) >= 0) return { time: Date.UTC(+m[2], MONTHS.indexOf(m[1].slice(0, 3).toLowerCase()), 1), ambiguous: false };
  return null;
}

export interface Measurement {
  docId: string;
  docTitle: string;
  lab: Lab;
  /** Date shown for this point: the result's own date, else the document date. */
  dateText: string | null;
  /** True when the result had no date of its own and the document's date is used. */
  dateFromDocument?: boolean;
  value: number | null;
  range: ParsedRange | null;
  verified: boolean;
}

export interface ComparisonRow {
  key: string;
  label: string;
  unit: string | null;
  points: Measurement[];
}

export interface Incompatible { key: string; label: string; reason: string; points: Measurement[] }

export interface Comparison { rows: ComparisonRow[]; incompatible: Incompatible[]; onlyInOne: Measurement[] }

/** Group measurements from several documents by test; only same-unit groups are compared. */
export function compareMeasurements(ms: Measurement[]): Comparison {
  const byKey = new Map<string, Measurement[]>();
  for (const m of ms) {
    const k = testKey(m.lab.test);
    byKey.set(k, [...(byKey.get(k) ?? []), m]);
  }
  const rows: ComparisonRow[] = [], incompatible: Incompatible[] = [], onlyInOne: Measurement[] = [];
  for (const [key, group] of byKey) {
    const docs = new Set(group.map((m) => m.docId));
    if (docs.size < 2) { onlyInOne.push(...group); continue; }
    const label = group[0].lab.test;
    const units = new Set(group.map((m) => unitKey(m.lab.unit)));
    if (units.size > 1) {
      incompatible.push({ key, label, reason: `Printed in different units (${[...new Set(group.map((m) => m.lab.unit ?? "no unit"))].join(" vs ")}), so the numbers are not compared.`, points: group });
      continue;
    }
    if (group.some((m) => m.value === null)) {
      incompatible.push({ key, label, reason: "At least one value is not a plain number (for example \"<0.5\"), so it is listed but not charted.", points: group });
      continue;
    }
    const sorted = [...group].sort((a, b) => (parseDate(a.dateText)?.time ?? 0) - (parseDate(b.dateText)?.time ?? 0));
    rows.push({ key, label, unit: group[0].lab.unit, points: sorted });
  }
  rows.sort((a, b) => a.label.localeCompare(b.label));
  return { rows, incompatible, onlyInOne };
}
