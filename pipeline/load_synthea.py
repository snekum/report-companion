"""Load the Synthea export into one compact JSON file per run.

Reads labs, vitals, encounters, medications and care plans from the CSV
export (consistent UTC timestamps), and adds two things only FHIR has:
medication dosing (frequency/period, "as needed") and care plan activities.
FHIR rows are joined to CSV rows by encounter ID + RxNorm code (medications)
and by care plan ID (activities).

Output: data/interim/records.json
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "synthea")
OUT = os.path.join(ROOT, "data", "interim", "records.json")

VITAL_CODES = {
    "8302-2": "height",
    "29463-7": "weight",
    "39156-5": "bmi",
    "8867-4": "heart_rate",
    "8480-6": "systolic",
    "8462-4": "diastolic",
    "8310-5": "temperature",
}


def read_csv(name):
    with open(os.path.join(RAW, "csv", name), encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def strip_digits(name: str) -> str:
    """Synthea appends digits to generated names (e.g. 'Aaron697')."""
    return re.sub(r"\d+", "", name or "").strip()


def load_fhir_extras():
    dosing = {}  # (encounter_id, rxnorm) -> dosage dict
    activities = {}  # careplan id -> [(text, status)]
    files = [f for f in glob.glob(os.path.join(RAW, "fhir", "*.json")) if "Information" not in os.path.basename(f)]
    for i, fn in enumerate(sorted(files)):
        with open(fn, encoding="utf-8") as f:
            bundle = json.load(f)
        for entry in bundle.get("entry", []):
            r = entry["resource"]
            rt = r["resourceType"]
            if rt == "MedicationRequest":
                enc = r.get("encounter", {}).get("reference", "").replace("urn:uuid:", "")
                codings = r.get("medicationCodeableConcept", {}).get("coding", [])
                if not enc or not codings:
                    continue
                for di in r.get("dosageInstruction", [])[:1]:
                    rep = di.get("timing", {}).get("repeat", {})
                    dq = (di.get("doseAndRate") or [{}])[0].get("doseQuantity", {})
                    dosing[(enc, codings[0]["code"])] = {
                        "frequency": rep.get("frequency"),
                        "period": rep.get("period"),
                        "period_unit": rep.get("periodUnit"),
                        "as_needed": bool(di.get("asNeededBoolean")),
                        "units_per_dose": dq.get("value"),
                    }
            elif rt == "CarePlan":
                acts = []
                for a in r.get("activity", []):
                    d = a.get("detail", {})
                    text = d.get("code", {}).get("text")
                    if text:
                        acts.append((text, d.get("status")))
                activities[r["id"]] = acts
        if i % 100 == 0:
            print(f"  fhir {i}/{len(files)}")
    return dosing, activities


def main():
    patients = {}
    for r in read_csv("patients.csv"):
        patients[r["Id"]] = {
            "id": r["Id"],
            "first": strip_digits(r["FIRST"]),
            "last": strip_digits(r["LAST"]),
            "prefix": r["PREFIX"],
            "birthdate": r["BIRTHDATE"],
            "gender": r["GENDER"],
            "city": r["CITY"],
            "encounters": {},
            "labs": [],
            "vitals": [],
            "medications": [],
            "careplans": [],
        }
    for r in read_csv("encounters.csv"):
        p = patients.get(r["PATIENT"])
        if p:
            p["encounters"][r["Id"]] = {
                "id": r["Id"],
                "start": r["START"],
                "class": r["ENCOUNTERCLASS"],
                "description": r["DESCRIPTION"],
                "reason": r["REASONDESCRIPTION"],
            }
    for r in read_csv("observations.csv"):
        p = patients.get(r["PATIENT"])
        if not p or r["TYPE"] != "numeric":
            continue
        row = {"date": r["DATE"], "encounter": r["ENCOUNTER"], "code": r["CODE"], "description": r["DESCRIPTION"], "value": r["VALUE"], "unit": r["UNITS"]}
        if r["CATEGORY"] == "laboratory":
            p["labs"].append(row)
        elif r["CATEGORY"] == "vital-signs" and r["CODE"] in VITAL_CODES:
            row["kind"] = VITAL_CODES[r["CODE"]]
            p["vitals"].append(row)
    print("loading FHIR extras (dosing, care plan activities)...")
    dosing, activities = load_fhir_extras()
    matched = 0
    for r in read_csv("medications.csv"):
        p = patients.get(r["PATIENT"])
        if not p:
            continue
        dose = dosing.get((r["ENCOUNTER"], r["CODE"]))
        matched += dose is not None
        p["medications"].append({"start": r["START"], "stop": r["STOP"], "encounter": r["ENCOUNTER"], "code": r["CODE"], "description": r["DESCRIPTION"], "dosing": dose})
    for r in read_csv("careplans.csv"):
        p = patients.get(r["PATIENT"])
        if p:
            p["careplans"].append({"id": r["Id"], "start": r["START"], "stop": r["STOP"], "description": r["DESCRIPTION"], "activities": activities.get(r["Id"], [])})
    for p in patients.values():
        p["encounters"] = sorted(p["encounters"].values(), key=lambda e: e["start"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(list(patients.values()), f)
    n_meds = sum(len(p["medications"]) for p in patients.values())
    print(f"patients={len(patients)} medications={n_meds} with FHIR dosing={matched} careplans_with_activities={sum(1 for a in activities.values() if a)}")


if __name__ == "__main__":
    main()
