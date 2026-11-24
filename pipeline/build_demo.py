"""Pick demo patients and render the three example journeys for the app.

Only patients in the `demo` split are used, so no journey document (or its
patient) is ever seen in training or testing. Documents are rendered with
the same templates and gold checks as the dataset.

  python pipeline/build_demo.py  -> data/processed/demo_journeys.jsonl, data/processed/journeys.json
"""
from __future__ import annotations

import json
import os
import random
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_dataset import RECORDS, lab_encounters, split_for, stable_hash, visit_encounters  # noqa: E402
from render import FAMILIES, LABS, LOINC_TO_TEST, Ctx, check_gold, to_date  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "processed")


def age_at(p, d: date) -> int:
    b = date.fromisoformat(p["birthdate"])
    return d.year - b.year - ((d.month, d.day) < (b.month, b.day))


def tests_at(p, enc_id):
    return {LOINC_TO_TEST[r["code"]] for r in p["labs"] if r["encounter"] == enc_id and r["code"] in LOINC_TO_TEST}


def render(p, enc, family, lab_id=None, salt="demo", date_style=None):
    key = stable_hash(p["id"], enc["id"], family, salt)
    ctx = Ctx(p, enc, random.Random(key))
    if lab_id:
        ctx.lab = next(l for l in LABS if l["id"] == lab_id)
        ctx.unit_style = ctx.lab["unit_style"]
    if date_style:
        ctx.date_style = date_style
    doc = FAMILIES[family](ctx)
    assert doc is not None, (family, p["id"])
    assert not check_gold(doc), check_gold(doc)
    return doc, ctx, key


def row(doc, ctx, key, p, enc, family, title, journey):
    return {
        "doc_id": f"demo-{journey}-{family}-{key[:8]}", "patient_id": "p-" + stable_hash("patient", p["id"])[:10], "split": "demo",
        "family": family, "held_out_family": False, "kind": "lab_report" if family.startswith("L") else "visit_summary",
        "title": title, "journey": journey, "text": doc.text, "target": doc.gold, "flags": sorted(set(doc.flags)),
        "provenance": {"source": "Synthea v4.0.0 synthetic record", "synthea_patient_uuid": p["id"], "anchor_encounter": enc["id"],
                       "anchor_date": to_date(enc["start"]).isoformat(), "fictional_lab": ctx.lab["id"], "generator": "build_demo-v1"},
    }


def main():
    patients = [p for p in json.load(open(RECORDS, encoding="utf-8")) if split_for(p["id"]) == "demo"]
    patients.sort(key=lambda p: p["id"])
    rows, journeys, used = [], [], set()

    def two_lab_encounters(p, min_overlap=4):
        le = lab_encounters(p)
        for i in range(len(le) - 1, 0, -1):
            for j in range(i - 1, -1, -1):
                a, b = le[j], le[i]
                gap = (to_date(b["start"]) - to_date(a["start"])).days
                if 240 <= gap <= 800 and len(tests_at(p, a["id"]) & tests_at(p, b["id"])) >= min_overlap:
                    return a, b
        return None

    def meds_profile(p, enc):
        ctx = Ctx(p, enc, random.Random(0))
        cur, stopped, hist = ctx.meds_at()
        return len(cur), len(stopped), len(hist)

    # 1. Routine check-up: adult 30-65 with two comparable lab reports ~1 year apart.
    for p in patients:
        pair = two_lab_encounters(p)
        if not pair or not (30 <= age_at(p, to_date(pair[1]["start"])) <= 65):
            continue
        a, b = pair
        d1, c1, k1 = render(p, a, "L1", "northbrook", "checkup-1")
        d2, c2, k2 = render(p, b, "L2", "riverside", "checkup-2", date_style=c1.date_style)
        ids = [row(d1, c1, k1, p, a, "L1", "Lab report, last year", "checkup"), row(d2, c2, k2, p, b, "L2", "Lab report, this year", "checkup")]
        rows += ids
        journeys.append({"id": "checkup", "title": "Preparing for a routine check-up", "person": f"{p['first']} {p['last']}",
                         "who": f"{p['first']}, {age_at(p, to_date(b['start']))}, has two lab reports from different labs a year apart.",
                         "goal": "See what changed between the two reports, and write down questions before the check-up.",
                         "docIds": [r["doc_id"] for r in ids], "compare": [ids[0]["doc_id"], ids[1]["doc_id"]],
                         "starterQuestions": ["Which of these results would you like to recheck, and when?"]})
        used.add(p["id"])
        break

    # 2. New doctor: adult with current and stopped medications plus a lab report.
    best = None
    for p in patients:
        if p["id"] in used:
            continue
        for enc in visit_encounters(p)[-6:]:
            cur, stopped, hist = meds_profile(p, enc)
            le = [e for e in lab_encounters(p) if e["start"] <= enc["start"]]
            if cur >= 2 and stopped >= 1 and le and age_at(p, to_date(enc["start"])) >= 40:
                score = cur + 2 * stopped + hist
                if not best or score > best[0]:
                    best = (score, p, enc, le[-1])
    _, p, enc, lenc = best
    d1, c1, k1 = render(p, enc, "V1", None, "newdoc-1")
    d2, c2, k2 = render(p, enc, "V2", None, "newdoc-2", date_style=c1.date_style)
    d3, c3, k3 = render(p, lenc, "L2", "harborview", "newdoc-3", date_style=c1.date_style)
    ids = [row(d1, c1, k1, p, enc, "V1", "After-visit summary", "newdoctor"), row(d2, c2, k2, p, enc, "V2", "Letter from the clinic", "newdoctor"),
           row(d3, c3, k3, p, lenc, "L2", "Most recent lab report", "newdoctor")]
    rows += ids
    journeys.append({"id": "newdoctor", "title": "Organising records before seeing a new doctor", "person": f"{p['first']} {p['last']}",
                     "who": f"{p['first']}, {age_at(p, to_date(enc['start']))}, is moving to a new practice and has a summary, a letter and a lab report.",
                     "goal": "Build one tidy list of medicines (current, stopped, unclear) and recent results to bring along.",
                     "docIds": [r["doc_id"] for r in ids], "compare": None,
                     "starterQuestions": ["Should I keep taking all of the medicines on the current list?"]})
    used.add(p["id"])

    # 3. Helping a parent: patient aged 68+, trend report + portal message + visit summary.
    for p in patients:
        if p["id"] in used:
            continue
        le = lab_encounters(p)
        if not le or age_at(p, to_date(le[-1]["start"])) < 68:
            continue
        ve = [e for e in visit_encounters(p) if e["start"] >= le[-1]["start"]] or visit_encounters(p)[-1:]
        cur, stopped, hist = meds_profile(p, ve[0])
        if cur < 1:
            continue
        try:
            d1, c1, k1 = render(p, le[-1], "L4", "northbrook", "parent-1")
        except AssertionError:
            continue
        d2, c2, k2 = render(p, le[-1], "L5", "northbrook", "parent-2", date_style=c1.date_style)
        d3, c3, k3 = render(p, ve[0], "V1", None, "parent-3", date_style=c1.date_style)
        ids = [row(d1, c1, k1, p, le[-1], "L4", "Results trend printout", "parent"), row(d2, c2, k2, p, le[-1], "L5", "Message from the care team", "parent"),
               row(d3, c3, k3, p, ve[0], "V1", "After-visit summary", "parent")]
        rows += ids
        journeys.append({"id": "parent", "title": "Helping a parent prepare for an appointment", "person": f"{p['first']} {p['last']}",
                         "who": f"An adult child is helping {p['first']}, {age_at(p, to_date(ve[0]['start']))}, make sense of a results printout, a portal message and a visit summary.",
                         "goal": "Understand what each paper says, spot repeated tests, and bring a one-page brief to the appointment.",
                         "docIds": [r["doc_id"] for r in ids], "compare": [ids[0]["doc_id"], ids[1]["doc_id"]],
                         "starterQuestions": ["Is there anything we should bring or do before the next appointment?"]})
        break

    with open(os.path.join(OUT, "demo_journeys.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "journeys.json"), "w", encoding="utf-8") as f:
        json.dump(journeys, f, indent=2)
    for j in journeys:
        print(j["id"], j["person"], j["docIds"])


if __name__ == "__main__":
    main()
