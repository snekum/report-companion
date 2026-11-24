"""Deterministic baseline: dictionary + regex extraction of lab rows and the document date.

It does not attempt medications or follow-up. Its test-name dictionary is the
same authored catalogue the document templates draw from, so on this
synthetic data it is an optimistic baseline: it never meets an unknown
spelling except through prose phrasing.

  python model/baseline_regex.py --split test
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, ROOT, load_jsonl  # noqa: E402

CATALOG = json.load(open(os.path.join(ROOT, "pipeline", "fixtures", "lab_catalog.json"), encoding="utf-8"))
NAMES = sorted({n for t in CATALOG["tests"].values() for n in t["names"] + [t["abbrev"]]}, key=len, reverse=True)
NAME_RE = re.compile(r"(?<![A-Za-z])(" + "|".join(re.escape(n) for n in NAMES) + r")(?![A-Za-z])", re.I)
NUM = r"[<>]?\d+(?:\.\d+)?"
DATE = r"(?:\d{4}-\d{2}-\d{2}|\d{1,2} [A-Z][a-z]{2} \d{4}|\d{2}/\d{2}/\d{4}|[A-Z][a-z]+ \d{1,2}, \d{4})"
RANGE_RE = re.compile(r"(?:ref(?:erence)?|normal range|lab range|range is|range|\|)\s*(" + r"[<>]\s?\d+(?:\.\d+)?|\d+(?:\.\d+)?\s?(?:-| - | to )\s?\d+(?:\.\d+)?" + r")")
# a range printed as a bare table column at the end of the row (no keyword)
BARE_RANGE_RE = re.compile(r"\s(" + r"[<>]\s?\d+(?:\.\d+)?|\d+(?:\.\d+)?\s?(?:-| - | to )\s?\d+(?:\.\d+)?" + r")\s*$")
# medication lines ("Penicillin V Potassium 500 MG ...") would otherwise match salt names
DRUG_DOSE_RE = re.compile(r"\d+(?:\.\d+)? (?:MG|mg|MCG|mcg)(?:/(?:ML|mL|ACTUAT|HR))?(?![A-Za-z/])")
DOC_DATE_RES = [
    re.compile(r"Reported:\s*(" + DATE + ")"),
    re.compile(r"report issued\s*(" + DATE + ")"),
    re.compile(r"Printed\s*(" + DATE + ")"),
    re.compile(r"printed\s*(" + DATE + r")\)"),
    re.compile(r"Visit date:\s*(" + DATE + ")"),
    re.compile(r"^Date:\s*(" + DATE + ")", re.M),
    re.compile(r"care team - (" + DATE + ")"),
    re.compile(r"COLL\s*(" + DATE + ")"),
    re.compile(r"Specimen collected:?\s*(" + DATE + ")"),
    re.compile(r"Collected:\s*(" + DATE + ")"),
]


def extract(text: str) -> dict:
    out = {"document_date": None, "labs": [], "medications": [], "follow_up": []}
    for rx in DOC_DATE_RES:
        m = rx.search(text)
        if m:
            line = next(l for l in text.splitlines() if m.group(0) in l)
            out["document_date"] = {"value": m.group(1), "evidence": line.strip()}
            break
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("TEST", "CODE|")) or DRUG_DOSE_RE.search(line):
            continue
        for m in NAME_RE.finditer(line):
            rest = line[m.end():]
            vm = re.match(r"\s*(?:\||:|was|came back at|)\s*(" + NUM + r")(%?)\s*\|?\s*([^\s|(;,]+)?", rest)
            if not vm:
                continue
            value = vm.group(1)
            unit = "%" if vm.group(2) else vm.group(3)
            if unit and (re.fullmatch(NUM, unit) or unit.lower() in ("and", "mmol", "the")):
                unit = None
            if unit:
                unit = unit.rstrip(".;")
            tail = rest[vm.end():]
            rm = RANGE_RE.search(tail) or BARE_RANGE_RE.search(tail)
            dm = re.match(r"\s*(" + DATE + ")", line)
            out["labs"].append({
                "test": m.group(1),
                "value": value,
                "unit": unit or None,
                "reference_range": rm.group(1) if rm else None,
                "date": dm.group(1) if dm else None,
                "evidence": line,
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    path = a.split if a.split.endswith(".jsonl") else os.path.join(DATA, f"{a.split}.jsonl")
    name = os.path.splitext(os.path.basename(path))[0]
    out = a.out or os.path.join(ROOT, "results", "predictions", f"regex_{name}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for d in load_jsonl(path):
            t0 = time.perf_counter()
            r = extract(d["text"])
            f.write(json.dumps({"doc_id": d["doc_id"], "raw": json.dumps(r), "latency_ms": round((time.perf_counter() - t0) * 1000, 3)}) + "\n")
    print(out)


if __name__ == "__main__":
    main()
