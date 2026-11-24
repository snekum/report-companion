import { describe, expect, it } from "vitest";
import cases from "../../../shared/validation_cases.json";
import parity from "../../../shared/rules_parity.json";
import { checkFact, findSpans, parseModelOutput, schemaErrors, type Section } from "./validate";
import { extractWithRules } from "./rules";
import { compareMeasurements, parseDate, parseRange, parseValue, testKey, unitKey } from "./compare";
import { positionInRange } from "./explain";
import { buildBrief } from "./brief";

describe("shared validation cases (same file as model/test_common.py)", () => {
  for (const c of cases.cases) {
    it(c.name, () => {
      const r = checkFact(cases.source, c.section as Section, c.item as Record<string, unknown>);
      expect(r.span !== null).toBe(c.evidence_found);
      expect(r.status === "verified").toBe(c.verified);
    });
  }
});

describe("parsing model output", () => {
  it("strips the empty think block WebLLM returns, and code fences", () => {
    const r = parseModelOutput('<think>\n\n</think>\n\n```json\n{"document_date": null, "labs": [], "medications": [], "follow_up": []}\n```');
    expect(r.error).toBeNull();
    expect(r.schemaErrors).toEqual([]);
  });
  it("explains a truncated answer", () => {
    expect(parseModelOutput('{"labs": [{"test": "Glu').error).toMatch(/length limit/);
  });
  it("flags bare strings where objects are required, but still shows what it can", () => {
    const r = parseModelOutput('{"document_date": null, "labs": [], "medications": [], "follow_up": ["Repeat in 3 months"]}');
    expect(schemaErrors(r.raw).length).toBeGreaterThan(0);
    expect(r.value?.follow_up).toEqual([]);
  });
});

describe("evidence offsets come from code", () => {
  it("finds a quote whose spacing differs and returns offsets into the original", () => {
    const src = "A\nGlucose      104   mg/dL\nB";
    const [s] = findSpans(src, "Glucose 104 mg/dL");
    expect(src.slice(s.start, s.end)).toBe("Glucose      104   mg/dL");
  });
  it("reports every occurrence so ambiguity can be flagged", () => {
    const r = checkFact("Sodium 140 mmol/L\nSodium 140 mmol/L\n", "labs", { test: "Sodium", value: "140", unit: "mmol/L", reference_range: null, date: null, evidence: "Sodium 140 mmol/L" });
    expect(r.status).toBe("needs_review");
    expect(r.reasons.join(" ")).toMatch(/2 times/);
  });
});

describe("rules match the evaluated Python baseline exactly", () => {
  for (const d of parity as { doc_id: string; text: string; expected: unknown }[]) {
    it(d.doc_id, () => expect(extractWithRules(d.text)).toEqual(d.expected));
  }
});

describe("comparison helpers", () => {
  it("parses printed ranges in several styles", () => {
    expect(parseRange("70-99")).toEqual({ low: 70, high: 99 });
    expect(parseRange("4.8 - 5.6")).toEqual({ low: 4.8, high: 5.6 });
    expect(parseRange("70 to 100")).toEqual({ low: 70, high: 100 });
    expect(parseRange("3.9–5.5")).toEqual({ low: 3.9, high: 5.5 });
    expect(parseRange("<200")).toEqual({ low: null, high: 200 });
    expect(parseRange(">40")).toEqual({ low: 40, high: null });
    expect(parseRange("12,0-16,0")).toEqual({ low: 12, high: 16 });
    expect(parseRange("see note")).toBeNull();
  });
  it("does not chart censored or non-numeric values", () => {
    expect(parseValue("<0.5")).toBeNull();
    expect(parseValue("Negative")).toBeNull();
    expect(parseValue("13,2")).toBe(13.2);
  });
  it("maps spelling variants to one test, keeps unknown names apart", () => {
    expect(testKey("HbA1c")).toBe(testKey("Hemoglobin A1c"));
    expect(testKey("Urea nitrogen (BUN)")).toBe(testKey("Blood urea nitrogen"));
    expect(testKey("TSH")).not.toBe(testKey("T4"));
  });
  it("treats unit spelling variants as the same, never converts", () => {
    expect(unitKey("x10^3/uL")).toBe(unitKey("10^3/uL"));
    expect(unitKey("mg/dL")).not.toBe(unitKey("mmol/L"));
  });
  it("refuses to compare different units", () => {
    const lab = (unit: string, value: string) => ({ test: "Glucose", value, unit, reference_range: null, date: null, evidence: "" });
    const c = compareMeasurements([
      { docId: "a", docTitle: "A", lab: lab("mg/dL", "104"), dateText: "2024-01-01", value: 104, range: null, verified: true },
      { docId: "b", docTitle: "B", lab: lab("mmol/L", "5.8"), dateText: "2025-01-01", value: 5.8, range: null, verified: true },
    ]);
    expect(c.rows).toHaveLength(0);
    expect(c.incompatible).toHaveLength(1);
  });
  it("orders printed dates in different formats", () => {
    expect(parseDate("March 8, 2026")!.time).toBeGreaterThan(parseDate("2025-10-05")!.time);
    expect(parseDate("14 Nov 2023")!.time).toBe(parseDate("2023-11-14")!.time);
    expect(parseDate("02/12/2024")!.ambiguous).toBe(true);
  });
  it("positions a value against its own printed range only", () => {
    expect(positionInRange("104", "70-99")).toBe("above");
    expect(positionInRange("50", ">40")).toBe("within");
    expect(positionInRange("<0.5", "<5.0")).toBe("unknown");
  });
});

describe("appointment brief", () => {
  it("includes questions, statuses and the not-advice notice", () => {
    const md = buildBrief("Test", [{
      title: "Visit", sourceLabel: "Fine-tuned model", corrected: true,
      checked: {
        extraction: { document_date: null, labs: [], medications: [{ name: "aspirin", dose: null, frequency: null, status: "not_current", evidence: "Denies aspirin." }], follow_up: [] },
        date: null, labs: [], medications: [{ status: "verified", span: null, reasons: [] }], follow_up: [], needsReview: 0, total: 1,
      },
    }], [{ id: "q1", text: "Is this right?", createdAt: "" }]);
    expect(md).toMatch(/not medical advice/);
    expect(md).toMatch(/Stopped \/ not taken/);
    expect(md).toMatch(/1\. Is this right\?/);
    expect(md).toMatch(/with my corrections/);
  });
});
