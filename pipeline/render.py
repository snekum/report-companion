"""Render short fictional documents from Synthea records, with gold targets.

Every gold item is created at the moment its line is written, from the same
structured value that produced the text, and records that exact line as its
evidence. `check_gold` then confirms each value, unit, range and date really
occurs in its evidence line and the evidence occurs in the document.

Template families
  lab reports:   L1 table, L2 key-value, L3 compact codes (TEST ONLY),
                 L4 trend list (repeated measurements), L5 portal message
  visit notes:   V1 after-visit summary, V2 clinic letter,
                 V3 medication list printout (TEST ONLY)
"""
from __future__ import annotations

import json
import os
import random
import re
from datetime import date, datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "fixtures")

with open(os.path.join(FIX, "lab_catalog.json"), encoding="utf-8") as f:
    CATALOG = json.load(f)
with open(os.path.join(FIX, "fictional_lab_ranges.json"), encoding="utf-8") as f:
    LABS = json.load(f)["labs"]

LOINC_TO_TEST = {code: key for key, t in CATALOG["tests"].items() for code in t["loinc"]}
TEST_ORDER = list(CATALOG["tests"].keys())

HELD_OUT_FAMILIES = {"L3", "V3"}
LAB_FAMILIES = ["L1", "L2", "L3", "L4", "L5"]
VISIT_FAMILIES = ["V1", "V2", "V3"]

# Authored fictional names for clinics and clinicians (no real organisations).
CLINICS = ["Maple Street Family Practice", "Eastgate Health Centre", "Lakeside Primary Care", "Cedar Hill Medical Group", "Old Mill Community Clinic"]
CLINICIANS = ["Dr. A. Okafor", "Dr. M. Lindqvist", "Dr. R. Castellano", "Dr. J. Whitcombe", "Dr. S. Nakamura", "Dr. P. Adeyemi", "Dr. L. Fournier"]

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
DATE_STYLES = ["iso", "dmy_short", "us_slash", "long"]


def fmt_date(d: date, style: str) -> str:
    if style == "iso":
        return d.isoformat()
    if style == "dmy_short":
        return f"{d.day} {MONTHS[d.month - 1][:3]} {d.year}"
    if style == "us_slash":
        return f"{d.month:02d}/{d.day:02d}/{d.year}"
    if style == "long":
        return f"{MONTHS[d.month - 1]} {d.day}, {d.year}"
    raise ValueError(style)


def to_date(ts: str) -> date:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).date()


def strip_tag(s: str) -> str:
    """'Diabetic diet (finding)' -> 'Diabetic diet'."""
    return re.sub(r"\s*\([^()]*\)\s*$", "", s).strip()


# ---------------------------------------------------------------- medications
MED_RE = re.compile(
    r"^(?:(?P<er>\d+ HR) )?(?P<name>[A-Za-z][A-Za-z\-]*(?: [A-Za-z][A-Za-z\-]*)*?) "
    r"(?P<dose>\d+(?:\.\d+)? (?:MG|MCG|MG/ML|UNT/ML|MG/HR))"
    r" (?P<form>[A-Za-z][A-Za-z ]*?)(?: \[(?P<brand>[^\]]+)\])?$"
)


def parse_med(description: str):
    m = MED_RE.match(description)
    if not m:
        return None
    return {"name": m["name"], "dose": m["dose"], "form": m["form"], "extended": bool(m["er"])}


def frequency_phrases(dosing) -> list[str] | None:
    if not dosing:
        return None
    if dosing.get("as_needed"):
        return ["as needed", "when needed", "PRN (as needed)"]
    f, p, u = dosing.get("frequency"), dosing.get("period"), dosing.get("period_unit")
    if f is None or p is None:
        return None
    p = int(p) if float(p).is_integer() else p
    words = {1: "once", 2: "twice", 3: "three times", 4: "four times"}
    if u == "d" and p == 1 and f in words:
        base = [f"{words[f]} daily", f"{words[f]} a day"]
        if f == 1:
            base.append("every day")
        if f == 2:
            base.append("BID")
        return base
    if f == 1 and u == "d":
        return [f"every {p} days"]
    if f == 1 and u == "h":
        return [f"every {p} hours"]
    return None


# ---------------------------------------------------------------- document builder
class Doc:
    def __init__(self):
        self.lines: list[str] = []
        self.gold = {"document_date": None, "labs": [], "medications": [], "follow_up": []}
        self.flags: list[str] = []

    def add(self, line: str = "") -> str:
        self.lines.append(line)
        return line.strip()

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip() + "\n"

    def set_date(self, value: str, evidence: str):
        self.gold["document_date"] = {"value": value, "evidence": evidence}

    def lab(self, test, value, unit, rng_text, rdate, evidence):
        self.gold["labs"].append({"test": test, "value": value, "unit": unit, "reference_range": rng_text, "date": rdate, "evidence": evidence})

    def med(self, name, dose, freq, status, evidence):
        self.gold["medications"].append({"name": name, "dose": dose, "frequency": freq, "status": status, "evidence": evidence})

    def follow(self, instruction, evidence):
        self.gold["follow_up"].append({"instruction": instruction, "evidence": evidence})


def pad(name: str, width: int = 26) -> str:
    return name.ljust(max(width, len(name) + 2))


def sentence_case(name: str) -> str:
    """'Hemoglobin A1c' -> 'hemoglobin A1c' mid-sentence; keeps 'HbA1c', 'ALT', 'eGFR'."""
    if len(name) > 1 and name[0].isupper() and name[1].islower() and not re.match(r"^[A-Z][a-z][A-Z]", name):
        return name[0].lower() + name[1:]
    return name


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def check_gold(doc: Doc) -> list[str]:
    """Return problems; an empty list means every gold field is supported by its evidence."""
    text = norm_ws(doc.text)
    problems = []

    def ev_ok(item, fields):
        ev = norm_ws(item["evidence"])
        if ev not in text:
            problems.append(f"evidence not in text: {ev!r}")
        for f in fields:
            v = item.get(f)
            if v is not None and norm_ws(v) not in ev:
                problems.append(f"{f}={v!r} not in evidence {ev!r}")

    if doc.gold["document_date"]:
        ev_ok(doc.gold["document_date"], ["value"])
    for x in doc.gold["labs"]:
        ev_ok(x, ["test", "value", "unit", "reference_range", "date"])
    for x in doc.gold["medications"]:
        ev_ok(x, ["name", "dose", "frequency"])
    for x in doc.gold["follow_up"]:
        ev_ok(x, ["instruction"])
    return problems


# ---------------------------------------------------------------- context
class Ctx:
    """Everything a template needs about one patient at one moment."""

    def __init__(self, patient, anchor_encounter, rng: random.Random):
        self.p = patient
        self.enc = anchor_encounter
        self.rng = rng
        self.d = to_date(anchor_encounter["start"])
        self.date_style = rng.choice(DATE_STYLES)
        self.lab = rng.choice(LABS)
        self.unit_style = self.lab["unit_style"]

    def fdate(self, d: date) -> str:
        return fmt_date(d, self.date_style)

    @property
    def name(self) -> str:
        return f"{self.p['first']} {self.p['last']}"

    @property
    def dob(self) -> date:
        return date.fromisoformat(self.p["birthdate"])

    def unit(self, ucum: str) -> str:
        opts = CATALOG["unit_display"].get(ucum)
        return opts[self.unit_style] if opts else ucum

    def ref_range(self, test_key: str) -> str | None:
        r = self.lab["ranges"].get(test_key)
        if r is None:
            return None
        if self.lab["range_dash"] == " - " and re.fullmatch(r"[\d.]+-[\d.]+", r):
            r = r.replace("-", " - ")
        return r

    def labs_for(self, encounter_id: str, limit: int) -> list[dict]:
        seen, out = set(), []
        rows = [r for r in self.p["labs"] if r["encounter"] == encounter_id and r["code"] in LOINC_TO_TEST]
        rows.sort(key=lambda r: TEST_ORDER.index(LOINC_TO_TEST[r["code"]]))
        for r in rows:
            key = LOINC_TO_TEST[r["code"]]
            if key in seen:
                continue
            seen.add(key)
            out.append({**r, "test_key": key})
        if len(out) > limit:
            keep = sorted(self.rng.sample(range(len(out)), limit))
            out = [out[i] for i in keep]
        return out

    def missing_tests(self, present: set[str], k: int) -> list[str]:
        common = ["hba1c", "cholesterol", "hemoglobin", "glucose", "creatinine", "alt", "ldl", "platelets"]
        cands = [t for t in common if t not in present]
        return self.rng.sample(cands, min(k, len(cands)))

    def test_name(self, key: str, compact=False) -> str:
        t = CATALOG["tests"][key]
        if compact:
            return t["abbrev"]
        return t["names"][0] if self.rng.random() < 0.6 else self.rng.choice(t["names"])

    # medications at date d
    def meds_at(self):
        d = self.d
        by_code: dict[str, list[dict]] = {}
        for m in self.p["medications"]:
            if parse_med(m["description"]) is None:
                continue
            by_code.setdefault(m["code"], []).append(m)
        current, stopped, history = [], [], []
        for code, rows in by_code.items():
            rows.sort(key=lambda r: r["start"])
            started = [r for r in rows if to_date(r["start"]) <= d]
            if not started:
                continue
            last = started[-1]
            stop = to_date(last["stop"]) if last["stop"] else None
            if stop is None or stop > d:
                current.append(last)
            elif stop >= d - timedelta(days=730):
                stopped.append(last)
            elif to_date(last["start"]) >= d - timedelta(days=365 * 6):
                history.append(last)
        rng = self.rng
        rng.shuffle(current); rng.shuffle(stopped); rng.shuffle(history)
        return current[:5], stopped[:3], history[:2]

    def care_activities(self) -> list[str]:
        out = []
        for cp in self.p["careplans"]:
            s = to_date(cp["start"])
            e = to_date(cp["stop"]) if cp["stop"] else None
            if s <= self.d and (e is None or e > self.d):
                for text, status in cp["activities"]:
                    if status == "in-progress":
                        t = strip_tag(text)
                        if t and t not in out:
                            out.append(t)
        return out[:3]

    def next_visit(self):
        for e in self.p["encounters"]:
            ed = to_date(e["start"])
            if ed > self.d + timedelta(days=1) and e["class"] in ("wellness", "ambulatory", "outpatient") and ed <= self.d + timedelta(days=400):
                return ed, strip_tag(e["description"])
        return None

    def vitals_line(self):
        vs = {v["kind"]: v for v in self.p["vitals"] if v["encounter"] == self.enc["id"]}
        parts = []
        if "systolic" in vs and "diastolic" in vs:
            parts.append(f"BP {float(vs['systolic']['value']):.0f}/{float(vs['diastolic']['value']):.0f} mmHg")
        if "heart_rate" in vs:
            parts.append(f"Pulse {float(vs['heart_rate']['value']):.0f} /min")
        if "weight" in vs:
            parts.append(f"Weight {vs['weight']['value']} kg")
        if "bmi" in vs:
            parts.append(f"BMI {vs['bmi']['value']} kg/m2")
        return ", ".join(parts) if parts else None


def med_print(ctx: Ctx, m: dict, lower=False, with_form=None) -> tuple[str, str, str]:
    """Returns (printed text, printed name, printed dose)."""
    pm = parse_med(m["description"])
    name = pm["name"]
    dose = pm["dose"].lower() if lower else pm["dose"]
    if lower and name[:1].isupper() and name[1:2].islower():
        name = name[0].lower() + name[1:]
    if with_form is None:
        with_form = ctx.rng.random() < 0.5
    form = (" " + pm["form"].lower() if lower else " " + pm["form"]) if with_form else ""
    if pm["extended"] and not with_form:
        form = " ER"
    return f"{name} {dose}{form}", name, dose


def maybe_freq(ctx: Ctx, m: dict, p_print=0.8) -> str | None:
    phrases = frequency_phrases(m.get("dosing"))
    if not phrases or ctx.rng.random() > p_print:
        return None
    return ctx.rng.choice(phrases)


# ---------------------------------------------------------------- lab families
def header_lab(ctx: Ctx, doc: Doc, collected: date, reported: date | None):
    doc.add(ctx.lab["name"].upper() if ctx.rng.random() < 0.5 else ctx.lab["name"])
    doc.add(f"Patient: {ctx.name}    DOB: {ctx.fdate(ctx.dob)}")
    if reported:
        line = doc.add(f"Collected: {ctx.fdate(collected)}    Reported: {ctx.fdate(reported)}")
        doc.set_date(ctx.fdate(reported), line)
    else:
        line = doc.add(f"Specimen collected: {ctx.fdate(collected)}")
        doc.set_date(ctx.fdate(collected), line)
    if ctx.rng.random() < 0.5:
        doc.add(f"Ordering clinician: {ctx.rng.choice(CLINICIANS)}")


def negation_lines(ctx: Ctx, doc: Doc, present: set[str], fmt):
    if ctx.rng.random() < 0.35:
        for key in ctx.missing_tests(present, ctx.rng.choice([1, 1, 2])):
            status = ctx.rng.choice(["not performed", "pending", "test cancelled", "result to follow"])
            doc.add(fmt(key, status))
            doc.flags.append("negated_lab")


def L1(ctx: Ctx) -> Doc:
    doc = Doc()
    labs = ctx.labs_for(ctx.enc["id"], 9)
    reported = ctx.d + timedelta(days=ctx.rng.choice([0, 1, 1, 2]))
    header_lab(ctx, doc, ctx.d, reported if ctx.rng.random() < 0.8 else None)
    show_range = ctx.rng.random() < 0.85
    doc.add()
    doc.add(f"{'TEST':<26}{'RESULT':<9}{'UNITS':<15}{'REFERENCE RANGE' if show_range else ''}".rstrip())
    for r in labs:
        name, unit = ctx.test_name(r["test_key"]), ctx.unit(r["unit"])
        rr = ctx.ref_range(r["test_key"]) if show_range else None
        line = doc.add(f"{pad(name)}{r['value']:<9}{unit:<15}{(rr or ('--' if show_range else ''))}".rstrip())
        doc.lab(name, r["value"], unit, rr, None, line)
    negation_lines(ctx, doc, {r["test_key"] for r in labs}, lambda k, s: pad(ctx.test_name(k)) + s)
    if ctx.rng.random() < 0.3:
        doc.add()
        doc.add("Results reviewed electronically. Ranges are those of this laboratory.")
    return doc


def L2(ctx: Ctx) -> Doc:
    doc = Doc()
    labs = ctx.labs_for(ctx.enc["id"], 9)
    doc.add(f"{ctx.lab['name']} - Laboratory Report")
    doc.add(f"Name: {ctx.p['last']}, {ctx.p['first']}   Date of birth: {ctx.fdate(ctx.dob)}")
    issued = ctx.d + timedelta(days=ctx.rng.choice([0, 1, 3]))
    line = doc.add(f"Specimen collected {ctx.fdate(ctx.d)}; report issued {ctx.fdate(issued)}")
    doc.set_date(ctx.fdate(issued), line)
    doc.add()
    word = ctx.rng.choice(["reference", "ref", "normal range"])
    for r in labs:
        name, unit = ctx.test_name(r["test_key"]), ctx.unit(r["unit"])
        rr = ctx.ref_range(r["test_key"]) if ctx.rng.random() < 0.85 else None
        tail = f" ({word} {rr})" if rr else ""
        line = doc.add(f"  {name}: {r['value']} {unit}{tail}")
        doc.lab(name, r["value"], unit, rr, None, line)
    negation_lines(ctx, doc, {r["test_key"] for r in labs}, lambda k, s: f"  {ctx.test_name(k)}: {s}")
    return doc


def L3(ctx: Ctx) -> Doc:  # held out: compact codes, pipes
    doc = Doc()
    labs = ctx.labs_for(ctx.enc["id"], 9)
    doc.add(f"{ctx.lab['name'].upper()} | RESULTS")
    doc.add(f"PT {ctx.p['last'].upper()}, {ctx.p['first'].upper()} | DOB {ctx.fdate(ctx.dob)}")
    line = doc.add(f"COLL {ctx.fdate(ctx.d)} | FINAL")
    doc.set_date(ctx.fdate(ctx.d), line)
    doc.add("CODE|RESULT|UNIT|REF")
    for r in labs:
        code, unit = CATALOG["tests"][r["test_key"]]["abbrev"], ctx.unit(r["unit"])
        rr = ctx.ref_range(r["test_key"])
        line = doc.add(f"{code}|{r['value']}|{unit}|{rr or ''}")
        doc.lab(code, r["value"], unit, rr, None, line)
    negation_lines(ctx, doc, {r["test_key"] for r in labs}, lambda k, s: f"{CATALOG['tests'][k]['abbrev']}|{s.upper()}||")
    return doc


def L4(ctx: Ctx) -> Doc:
    """Trend list: the same tests on several dates (repeated measurements)."""
    doc = Doc()
    p = ctx.p
    encs = [e for e in p["encounters"] if to_date(e["start"]) <= ctx.d]
    lab_encs = [e for e in encs if any(r["encounter"] == e["id"] and r["code"] in LOINC_TO_TEST for r in p["labs"])][-3:]
    keys_by_enc = [{LOINC_TO_TEST[r["code"]] for r in p["labs"] if r["encounter"] == e["id"] and r["code"] in LOINC_TO_TEST} for e in lab_encs]
    common = set.intersection(*keys_by_enc) if keys_by_enc else set()
    keys = [k for k in TEST_ORDER if k in common]
    ctx.rng.shuffle(keys)
    keys = sorted(keys[: ctx.rng.choice([2, 3])], key=TEST_ORDER.index)
    printed = ctx.d + timedelta(days=ctx.rng.choice([0, 2, 7]))
    doc.add(f"{ctx.lab['name']} - results trend")
    doc.add(f"Patient: {ctx.name} (DOB {ctx.fdate(ctx.dob)})")
    line = doc.add(f"Printed {ctx.fdate(printed)}")
    doc.set_date(ctx.fdate(printed), line)
    show_range = ctx.rng.random() < 0.6
    for k in keys:
        doc.add()
        name = ctx.test_name(k)
        rr = ctx.ref_range(k) if show_range else None
        for e in lab_encs:
            r = next(r for r in p["labs"] if r["encounter"] == e["id"] and LOINC_TO_TEST.get(r["code"]) == k)
            ed = ctx.fdate(to_date(e["start"]))
            unit = ctx.unit(r["unit"])
            line = doc.add(f"  {ed}   {name}   {r['value']} {unit}" + (f"   (range {rr})" if rr else ""))
            doc.lab(name, r["value"], unit, rr, ed, line)
    if len(lab_encs) < 2 or not keys:
        return None
    doc.flags.append("repeated_measurements")
    return doc


def L5(ctx: Ctx) -> Doc:
    """Patient portal message: results in prose."""
    doc = Doc()
    labs = ctx.labs_for(ctx.enc["id"], 5)
    sent = ctx.d + timedelta(days=ctx.rng.choice([1, 2, 4]))
    line = doc.add(f"Message from your care team - {ctx.fdate(sent)}")
    doc.set_date(ctx.fdate(sent), line)
    doc.add()
    doc.add(f"Hi {ctx.p['first']},")
    doc.add()
    doc.add(f"Your lab results from {ctx.fdate(ctx.d)} are now available.")
    for r in labs:
        name, unit = ctx.test_name(r["test_key"]), ctx.unit(r["unit"])
        rr = ctx.ref_range(r["test_key"]) if ctx.rng.random() < 0.7 else None
        nm = sentence_case(name)
        joiner = "" if unit == "%" else " "
        if rr:
            sent_txt = ctx.rng.choice([
                f"Your {nm} was {r['value']}{joiner}{unit}; the lab's range is {rr}.",
                f"{name}: {r['value']}{joiner}{unit} (lab range {rr}).",
            ])
        else:
            sent_txt = ctx.rng.choice([f"Your {nm} was {r['value']}{joiner}{unit}.", f"{name} came back at {r['value']}{joiner}{unit}."])
        line = doc.add(sent_txt)
        doc.lab(nm if sent_txt.startswith("Your") else name, r["value"], unit, rr, None, line)
    nv = ctx.next_visit()
    doc.add()
    if nv and ctx.rng.random() < 0.6:
        txt = f"Please keep your appointment on {ctx.fdate(nv[0])}."
        line = doc.add(txt)
        doc.follow(txt.rstrip("."), line)
    doc.add("If you have questions, reply to this message.")
    doc.add(ctx.rng.choice(CLINICIANS))
    return doc


# ---------------------------------------------------------------- visit families
def visit_header(ctx: Ctx, doc: Doc, title: str):
    doc.add(title)
    doc.add(ctx.rng.choice(CLINICS))
    doc.add(f"Patient: {ctx.name}   DOB: {ctx.fdate(ctx.dob)}")
    reason = strip_tag(ctx.enc["description"])
    if ctx.rng.random() < 0.92:
        line = doc.add(f"Visit date: {ctx.fdate(ctx.d)}   Reason: {reason}")
        doc.set_date(ctx.fdate(ctx.d), line)
    else:
        doc.add(f"Reason for visit: {reason}")
        doc.flags.append("no_document_date")


def V1(ctx: Ctx) -> Doc:
    doc = Doc()
    visit_header(ctx, doc, "AFTER VISIT SUMMARY")
    vit = ctx.vitals_line()
    if vit and ctx.rng.random() < 0.8:
        doc.add()
        doc.add(f"Vital signs: {vit}")
        doc.flags.append("vitals_distractor")
    labs = ctx.labs_for(ctx.enc["id"], 4)
    if labs and ctx.rng.random() < 0.7:
        doc.add()
        doc.add("Results from this visit:")
        for r in labs:
            name, unit = ctx.test_name(r["test_key"]), ctx.unit(r["unit"])
            rr = ctx.ref_range(r["test_key"]) if ctx.rng.random() < 0.5 else None
            line = doc.add(f"  {name}: {r['value']} {unit}" + (f" (ref {rr})" if rr else ""))
            doc.lab(name, r["value"], unit, rr, None, line)
    current, stopped, history = ctx.meds_at()
    if current:
        doc.add()
        doc.add(ctx.rng.choice(["Your current medications:", "Active medications:", "Current medicines:"]))
        for m in current:
            txt, name, dose = med_print(ctx, m)
            freq = maybe_freq(ctx, m)
            line = doc.add(f"  - {txt}" + (f", {freq}" if freq else ""))
            doc.med(name, dose, freq, "current", line)
    if stopped:
        doc.add()
        doc.add(ctx.rng.choice(["Medications stopped:", "Discontinued medications:", "No longer taking:"]))
        for m in stopped:
            txt, name, dose = med_print(ctx, m)
            line = doc.add(f"  - {txt} (stopped {ctx.fdate(to_date(m['stop']))})")
            doc.med(name, dose, None, "not_current", line)
    if history and ctx.rng.random() < 0.6:
        doc.add()
        doc.add(ctx.rng.choice(["Previous prescriptions on file:", "Prescription history:"]))
        for m in history:
            txt, name, dose = med_print(ctx, m)
            line = doc.add(f"  - {ctx.fdate(to_date(m['start']))}: {txt}")
            doc.med(name, dose, None, "unclear", line)
        doc.flags.append("old_prescription")
    acts = ctx.care_activities()
    nv = ctx.next_visit()
    if acts or nv:
        doc.add()
        doc.add(ctx.rng.choice(["Care plan and next steps:", "Plan:", "What happens next:"]))
        for a in acts:
            line = doc.add(f"  - {a}")
            doc.follow(a, line)
        if nv:
            txt = f"Next appointment: {ctx.fdate(nv[0])}" + (f" ({nv[1]})" if ctx.rng.random() < 0.5 else "")
            line = doc.add(f"  - {txt}")
            doc.follow(txt, line)
    elif ctx.rng.random() < 0.5:
        doc.add()
        doc.add("No follow-up appointment has been scheduled.")
        doc.flags.append("negated_follow_up")
    if not (doc.gold["labs"] or doc.gold["medications"] or doc.gold["follow_up"]):
        return None
    return doc


def pronoun(ctx: Ctx):
    return ("She", "her") if ctx.p["gender"] == "F" else ("He", "him")


def V2(ctx: Ctx) -> Doc:
    """Clinic letter: facts in prose."""
    doc = Doc()
    clinic = ctx.rng.choice(CLINICS)
    doc.add(clinic)
    written = ctx.d + timedelta(days=ctx.rng.choice([0, 1, 3, 5]))
    line = doc.add(f"Date: {ctx.fdate(written)}")
    doc.set_date(ctx.fdate(written), line)
    doc.add()
    doc.add(f"Re: {ctx.name}, DOB {ctx.fdate(ctx.dob)}")
    doc.add()
    doc.add("Dear colleague,")
    doc.add()
    she, her = pronoun(ctx)
    doc.add(f"I saw {ctx.p['first']} on {ctx.fdate(ctx.d)} for {strip_tag(ctx.enc['description']).lower()}.")
    current, stopped, history = ctx.meds_at()
    for m in current[:4]:
        txt, name, dose = med_print(ctx, m, lower=True, with_form=False)
        freq = maybe_freq(ctx, m, 0.7)
        s = ctx.rng.choice([f"{she} is currently taking {txt}" + (f" {freq}" if freq else "") + ".",
                            f"{she} continues on {txt}" + (f" {freq}" if freq else "") + "."])
        line = doc.add(s)
        doc.med(name, dose, freq, "current", line)
    for m in stopped[:2]:
        txt, name, dose = med_print(ctx, m, lower=True, with_form=False)
        when = to_date(m["stop"])
        s = ctx.rng.choice([f"{txt[0].upper() + txt[1:]} was stopped in {MONTHS[when.month - 1]} {when.year}.",
                            f"{she} is no longer taking {txt}."])
        line = doc.add(s)
        nm = name[0].upper() + name[1:] if s.startswith(txt[0].upper() + txt[1:]) else name
        doc.med(nm, dose, None, "not_current", line)
    if history and ctx.rng.random() < 0.5:
        m = history[0]
        txt, name, dose = med_print(ctx, m, lower=True, with_form=False)
        y = to_date(m["start"]).year
        line = doc.add(f"There is an older prescription for {txt} from {y} in the record.")
        doc.med(name, dose, None, "unclear", line)
        doc.flags.append("old_prescription")
    labs = ctx.labs_for(ctx.enc["id"], 3)
    if labs:
        parts = []
        for r in labs:
            name, unit = ctx.test_name(r["test_key"]), ctx.unit(r["unit"])
            nm = sentence_case(name)
            parts.append((nm, r["value"], unit))
        s = "Results today: " + ", ".join(f"{n} {v} {u}" for n, v, u in parts) + "."
        line = doc.add(s)
        for n, v, u in parts:
            doc.lab(n, v, u, None, None, line)
    vit = ctx.vitals_line()
    if vit and ctx.rng.random() < 0.5:
        doc.add(f"Observations: {vit}.")
        doc.flags.append("vitals_distractor")
    acts = ctx.care_activities()
    nv = ctx.next_visit()
    for a in acts[:2]:
        s = f"We discussed the care plan: {a.lower()}."
        line = doc.add(s)
        doc.follow(a.lower(), line)
    if nv:
        s = f"We will see {her} again on {ctx.fdate(nv[0])}."
        line = doc.add(s)
        doc.follow(s.rstrip("."), line)
    doc.add()
    doc.add("Kind regards,")
    doc.add(ctx.rng.choice(CLINICIANS))
    if not (doc.gold["labs"] or doc.gold["medications"] or doc.gold["follow_up"]):
        return None
    return doc


def V3(ctx: Ctx) -> Doc:  # held out: portal medication list printout
    doc = Doc()
    line = doc.add(f"MY MEDICATIONS (printed {ctx.fdate(ctx.d)})")
    doc.set_date(ctx.fdate(ctx.d), line)
    doc.add(f"{ctx.name}")
    current, stopped, history = ctx.meds_at()
    if not (current or stopped or history):
        return None
    if current:
        doc.add()
        doc.add("ACTIVE")
        for m in current:
            txt, name, dose = med_print(ctx, m, with_form=True)
            freq = maybe_freq(ctx, m, 0.9)
            line = doc.add(f"* {txt}" + (f" -- take {freq}" if freq else ""))
            doc.med(name, dose, freq, "current", line)
    if stopped:
        doc.add()
        doc.add("INACTIVE")
        for m in stopped:
            txt, name, dose = med_print(ctx, m, with_form=True)
            line = doc.add(f"* {txt} -- ended {ctx.fdate(to_date(m['stop']))}")
            doc.med(name, dose, None, "not_current", line)
    if history:
        doc.add()
        doc.add("OTHER RECORDS (status not confirmed)")
        for m in history:
            txt, name, dose = med_print(ctx, m, with_form=True)
            line = doc.add(f"* {txt} -- prescribed {ctx.fdate(to_date(m['start']))}")
            doc.med(name, dose, None, "unclear", line)
        doc.flags.append("old_prescription")
    nv = ctx.next_visit()
    if nv:
        doc.add()
        txt = f"Next visit: {ctx.fdate(nv[0])}"
        line = doc.add(txt)
        doc.follow(txt, line)
    return doc


FAMILIES = {"L1": L1, "L2": L2, "L3": L3, "L4": L4, "L5": L5, "V1": V1, "V2": V2, "V3": V3}
