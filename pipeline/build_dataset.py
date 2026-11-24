"""Build the extraction dataset from data/interim/records.json.

1. Assign every patient to a split (train/val/test/demo) by a salted hash of
   the Synthea patient ID, before any document is generated.
2. For each patient, pick one lab encounter and one visit encounter and render
   one document each with a template family allowed for that split.
   Families L3 and V3 are only ever used for test patients.
3. Validate every gold item against its evidence line; drop documents that
   fail (there should be none) and exact-duplicate texts.
4. Write data/processed/{train,val,test,demo}.jsonl, stats.json and
   provenance.json, plus a few rendered samples for inspection.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render import (FAMILIES, HELD_OUT_FAMILIES, LOINC_TO_TEST, Ctx, check_gold, to_date)  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORDS = os.path.join(ROOT, "data", "interim", "records.json")
OUT = os.path.join(ROOT, "data", "processed")
SPLIT_SALT = "report-companion-split-v1"
GENERATOR_VERSION = "render-v1"


def stable_hash(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def split_for(patient_id: str) -> str:
    u = int(stable_hash(SPLIT_SALT, patient_id)[:8], 16) / 0xFFFFFFFF
    if u < 0.70:
        return "train"
    if u < 0.78:
        return "val"
    if u < 0.94:
        return "test"
    return "demo"


LAB_WEIGHTS = {
    "train": {"L1": 3, "L2": 3, "L4": 2, "L5": 2},
    "val": {"L1": 3, "L2": 3, "L4": 2, "L5": 2},
    "test": {"L1": 2, "L2": 2, "L3": 3, "L4": 1, "L5": 2},
    "demo": {"L1": 3, "L2": 2, "L4": 2, "L5": 1},
}
VISIT_WEIGHTS = {
    "train": {"V1": 3, "V2": 2},
    "val": {"V1": 3, "V2": 2},
    "test": {"V1": 2, "V2": 2, "V3": 3},
    "demo": {"V1": 3, "V2": 1},
}


def pick(rng: random.Random, weights: dict[str, int]) -> str:
    keys = list(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys])[0]


def lab_encounters(p):
    counts = Counter(r["encounter"] for r in p["labs"] if r["code"] in LOINC_TO_TEST)
    return [e for e in p["encounters"] if counts.get(e["id"], 0) >= 3]


def visit_encounters(p):
    return [e for e in p["encounters"] if e["class"] in ("wellness", "ambulatory", "outpatient")]


def render_one(p, enc, family, split, purpose):
    doc_key = stable_hash(p["id"], enc["id"], family, purpose)
    rng = random.Random(doc_key)
    ctx = Ctx(p, enc, rng)
    doc = FAMILIES[family](ctx)
    if doc is None:
        return None
    problems = check_gold(doc)
    if problems:
        return {"error": problems, "family": family}
    patient_id = "p-" + stable_hash("patient", p["id"])[:10]
    return {
        "doc_id": f"{split}-{family}-{doc_key[:10]}",
        "patient_id": patient_id,
        "split": split,
        "family": family,
        "held_out_family": family in HELD_OUT_FAMILIES,
        "kind": "lab_report" if family.startswith("L") else "visit_summary",
        "text": doc.text,
        "target": doc.gold,
        "flags": sorted(set(doc.flags)),
        "provenance": {
            "source": "Synthea v4.0.0 synthetic record",
            "synthea_patient_uuid": p["id"],
            "anchor_encounter": enc["id"],
            "anchor_date": to_date(enc["start"]).isoformat(),
            "fictional_lab": ctx.lab["id"],
            "reference_ranges": "authored fictional fixture" if doc.gold["labs"] and any(x["reference_range"] for x in doc.gold["labs"]) else None,
            "date_style": ctx.date_style,
            "generator": GENERATOR_VERSION,
        },
    }


def main():
    with open(RECORDS, encoding="utf-8") as f:
        patients = json.load(f)
    patients.sort(key=lambda p: p["id"])
    out = {"train": [], "val": [], "test": [], "demo": []}
    errors, seen_text = [], set()
    dup = 0
    for p in patients:
        split = split_for(p["id"])
        rng = random.Random(stable_hash("plan", p["id"]))
        plans = []
        le = lab_encounters(p)
        if le:
            enc = rng.choice(le[-4:])
            plans.append((enc, pick(rng, LAB_WEIGHTS[split]), "lab"))
            if split == "demo":
                plans.append((le[-1], "L4", "trend"))
        ve = visit_encounters(p)
        if ve:
            enc = rng.choice(ve[-4:])
            plans.append((enc, pick(rng, VISIT_WEIGHTS[split]), "visit"))
        for enc, family, purpose in plans:
            rec = render_one(p, enc, family, split, purpose)
            if rec is None and family == "L4":
                rec = render_one(p, enc, "L1" if split != "test" else "L3", split, purpose)
            if rec is None:
                continue
            if "error" in rec:
                errors.append(rec)
                continue
            h = stable_hash(" ".join(rec["text"].split()).lower())
            if h in seen_text:
                dup += 1
                continue
            seen_text.add(h)
            out[split].append(rec)
    os.makedirs(OUT, exist_ok=True)
    for split, rows in out.items():
        with open(os.path.join(OUT, f"{split}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # leakage checks
    pid = {s: {r["patient_id"] for r in rows} for s, rows in out.items()}
    for a in pid:
        for b in pid:
            if a < b:
                assert not (pid[a] & pid[b]), f"patient leak between {a} and {b}"
    for s in ("train", "val", "demo"):
        assert not any(r["held_out_family"] for r in out[s]), f"held-out family in {s}"

    def stats(rows):
        c = Counter()
        for r in rows:
            t = r["target"]
            c["docs"] += 1
            c[f"family_{r['family']}"] += 1
            c["labs"] += len(t["labs"])
            c["labs_with_range"] += sum(1 for x in t["labs"] if x["reference_range"])
            c["labs_with_date"] += sum(1 for x in t["labs"] if x["date"])
            c["medications"] += len(t["medications"])
            for x in t["medications"]:
                c[f"med_status_{x['status']}"] += 1
            c["meds_with_frequency"] += sum(1 for x in t["medications"] if x["frequency"])
            c["follow_up"] += len(t["follow_up"])
            c["no_document_date"] += t["document_date"] is None
            for fl in r["flags"]:
                c[f"flag_{fl}"] += 1
        return dict(sorted(c.items()))

    st = {s: stats(rows) for s, rows in out.items()}
    st["_patients"] = {s: len(v) for s, v in pid.items()}
    st["_dropped_exact_duplicates"] = dup
    st["_gold_check_failures"] = len(errors)
    with open(os.path.join(OUT, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2)
    jar = os.path.join(ROOT, "tools", "synthea-v4.0.0.jar")
    prov = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "synthea": {
            "version": "v4.0.0",
            "release": "https://github.com/synthetichealth/synthea/releases/tag/v4.0.0",
            "jar_sha256": hashlib.sha256(open(jar, "rb").read()).hexdigest() if os.path.exists(jar) else None,
            "command": "pipeline/run_synthea.sh (seed 20261123, clinician seed 20261123, reference date 20250601, population 700, Massachusetts)",
            "code_licence": "Apache-2.0",
            "data_note": "Synthea's FAQ describes its generated synthetic data as public domain. All patients are synthetic; no real patient data is used.",
        },
        "fixtures": {
            "pipeline/fixtures/fictional_lab_ranges.json": "Authored fictional labs and illustrative reference ranges (Synthea has none). Not clinical reference intervals.",
            "pipeline/fixtures/lab_catalog.json": "Authored mapping of Synthea LOINC codes to display names.",
        },
        "split": {"method": "sha256(salt + Synthea patient UUID) before rendering", "salt": SPLIT_SALT, "fractions": {"train": 0.70, "val": 0.08, "test": 0.16, "demo": 0.06}},
        "held_out_template_families": sorted(HELD_OUT_FAMILIES),
        "generator": GENERATOR_VERSION,
    }
    with open(os.path.join(OUT, "provenance.json"), "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2)
    print(json.dumps({k: (v.get("docs") if isinstance(v, dict) and "docs" in v else v) for k, v in st.items()}, indent=1))
    if errors:
        print("GOLD CHECK FAILURES (first 5):")
        for e in errors[:5]:
            print(e)


if __name__ == "__main__":
    main()
