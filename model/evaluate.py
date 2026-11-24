"""Score a predictions file against gold targets.

  python model/evaluate.py --pred results/predictions/base_test.jsonl --split test
  python model/evaluate.py --pred ... --split test --sections labs,document_date   (regex baseline)

A prediction row is {"doc_id", "raw", "latency_ms"?, "new_tokens"?, "peak_mem_mb"?}.
Writes <pred>.metrics.json next to the predictions and prints a summary.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DATA, MATCHERS, SECTIONS, coerce, greedy_pairs, item_support, load_jsonl,  # noqa: E402
                    norm_key, norm_ws, parse_output, schema_errors)


def score_doc(doc: dict, raw: str, sections: list[str]) -> dict:
    gold = doc["target"]
    obj, perr = parse_output(raw)
    serr = schema_errors(obj) if obj is not None else ["unparseable"]
    pred = coerce(obj) if obj is not None else coerce(None)
    c = Counter()
    c["docs"] = 1
    c["parse_ok"] = obj is not None
    c["schema_valid"] = obj is not None and not serr
    detail = {"parse_error": perr, "schema_errors": serr[:5]}
    for sec in SECTIONS:
        if sec not in sections:
            continue
        P, G = pred[sec], gold[sec]
        pairs = greedy_pairs(P, G, MATCHERS[sec])
        c[f"{sec}_pred"] += len(P)
        c[f"{sec}_gold"] += len(G)
        c[f"{sec}_tp"] += len(pairs)
        for i, j in pairs:
            p, g = P[i], G[j]
            if sec == "labs":
                c["labs_test_exact"] += norm_ws(p["test"]) == norm_ws(g["test"])
                c["labs_unit_exact"] += norm_ws(p["unit"] or "") == norm_ws(g["unit"] or "")
                c["labs_range_exact"] += norm_ws(p["reference_range"] or "") == norm_ws(g["reference_range"] or "")
                c["labs_date_exact"] += norm_key(p["date"]) == norm_key(g["date"])
                c["labs_range_gold_present"] += g["reference_range"] is not None
                c["labs_range_present_exact"] += g["reference_range"] is not None and norm_ws(p["reference_range"] or "") == norm_ws(g["reference_range"])
            elif sec == "medications":
                c["meds_dose_exact"] += norm_key(p["dose"]) == norm_key(g["dose"])
                c["meds_freq_exact"] += norm_key(p["frequency"]) == norm_key(g["frequency"])
                c["meds_status_correct"] += p["status"] == g["status"]
                c[f"meds_status_gold_{g['status']}"] += 1
                c[f"meds_status_gold_{g['status']}_correct"] += p["status"] == g["status"]
                c["meds_false_current"] += p["status"] == "current" and g["status"] != "current"
        if sec == "labs":
            matched_pred = {i for i, _ in pairs}
            for i, p in enumerate(P):
                if i in matched_pred:
                    continue
                # a lab whose test name matches a gold lab but whose value matches none of them
                same = [g for g in G if p["test"] and norm_key(g["test"]) and (norm_key(p["test"]) in norm_key(g["test"]) or norm_key(g["test"]) in norm_key(p["test"]))]
                if same and not any(norm_ws(g["value"]) == norm_ws(p["value"] or "") for g in same):
                    c["labs_value_altered"] += 1
        for p in P:
            sup = item_support(doc["text"], sec, {k: v for k, v in p.items()})
            c["items_pred"] += 1
            c["items_evidence_ok"] += sup["evidence_ok"]
            c["items_supported"] += sup["supported"]
            c["items_in_evidence"] += sup["in_evidence"]
    if "document_date" in sections:
        g, p = gold["document_date"], pred["document_date"]
        c["date_docs"] += 1
        c["date_correct"] += (g is None and p is None) or (g is not None and p is not None and norm_ws(p["value"]) == norm_ws(g["value"]))
        if p is not None:
            c["items_pred"] += 1
            sup = item_support(doc["text"], "labs", {"test": None, "value": p["value"], "unit": None, "reference_range": None, "date": None, "evidence": p.get("evidence")})
            c["items_evidence_ok"] += sup["evidence_ok"]
            c["items_supported"] += sup["supported"]
            c["items_in_evidence"] += sup["in_evidence"]
    return {"counts": c, "detail": detail}


def ratio(a, b):
    return round(a / b, 4) if b else None


def summarise(c: Counter, sections: list[str]) -> dict:
    s = {"docs": c["docs"], "parse_rate": ratio(c["parse_ok"], c["docs"]), "schema_valid_rate": ratio(c["schema_valid"], c["docs"])}
    tp = pr = gd = 0
    for sec in SECTIONS:
        if sec not in sections:
            continue
        p, r = ratio(c[f"{sec}_tp"], c[f"{sec}_pred"]), ratio(c[f"{sec}_tp"], c[f"{sec}_gold"])
        f1 = round(2 * p * r / (p + r), 4) if p and r else 0.0
        s[sec] = {"precision": p, "recall": r, "f1": f1, "tp": c[f"{sec}_tp"], "pred": c[f"{sec}_pred"], "gold": c[f"{sec}_gold"]}
        tp, pr, gd = tp + c[f"{sec}_tp"], pr + c[f"{sec}_pred"], gd + c[f"{sec}_gold"]
    p, r = ratio(tp, pr), ratio(tp, gd)
    s["all_items"] = {"precision": p, "recall": r, "f1": round(2 * p * r / (p + r), 4) if p and r else 0.0}
    if "labs" in sections:
        t = c["labs_tp"]
        s["labs"].update({
            "test_name_exact": ratio(c["labs_test_exact"], t),
            "unit_exact": ratio(c["labs_unit_exact"], t),
            "reference_range_exact": ratio(c["labs_range_exact"], t),
            "reference_range_exact_when_printed": ratio(c["labs_range_present_exact"], c["labs_range_gold_present"]),
            "date_exact": ratio(c["labs_date_exact"], t),
            "value_altered": c["labs_value_altered"],
        })
    if "medications" in sections:
        t = c["medications_tp"]
        s["medications"].update({
            "dose_exact": ratio(c["meds_dose_exact"], t),
            "frequency_exact": ratio(c["meds_freq_exact"], t),
            "status_accuracy": ratio(c["meds_status_correct"], t),
            "status_accuracy_by_gold": {k: ratio(c[f"meds_status_gold_{k}_correct"], c[f"meds_status_gold_{k}"]) for k in ("current", "not_current", "unclear")},
            "false_current": c["meds_false_current"],
        })
    if "document_date" in sections:
        s["document_date_accuracy"] = ratio(c["date_correct"], c["date_docs"])
    s["evidence_valid_rate"] = ratio(c["items_evidence_ok"], c["items_pred"])
    s["fields_in_evidence_rate"] = ratio(c["items_in_evidence"], c["items_pred"])
    s["unsupported_item_rate"] = ratio(c["items_pred"] - c["items_supported"], c["items_pred"])
    s["unsupported_items"] = c["items_pred"] - c["items_supported"]
    return s


def bootstrap_f1(per_doc: list[Counter], sections, n=1000, seed=0):
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        tp = pr = gd = 0
        for _ in range(len(per_doc)):
            c = per_doc[rng.randrange(len(per_doc))]
            for sec in SECTIONS:
                if sec in sections:
                    tp, pr, gd = tp + c[f"{sec}_tp"], pr + c[f"{sec}_pred"], gd + c[f"{sec}_gold"]
        p, r = (tp / pr if pr else 0), (tp / gd if gd else 0)
        vals.append(2 * p * r / (p + r) if p and r else 0)
    vals.sort()
    return [round(vals[int(0.025 * n)], 4), round(vals[int(0.975 * n)], 4)]


def evaluate(pred_path: str, split_path: str, sections: list[str]) -> dict:
    gold = {d["doc_id"]: d for d in load_jsonl(split_path)}
    preds = {p["doc_id"]: p for p in load_jsonl(pred_path)}
    missing = [k for k in gold if k not in preds]
    groups = defaultdict(Counter)
    per_doc, per_doc_ids, failures = [], [], []
    for doc_id, doc in gold.items():
        p = preds.get(doc_id)
        if p is None:
            continue
        r = score_doc(doc, p["raw"], sections)
        per_doc.append(r["counts"])
        per_doc_ids.append(doc_id)
        groups["all"].update(r["counts"])
        groups[f"kind:{doc['kind']}"].update(r["counts"])
        groups["family:held_out" if doc.get("held_out_family") else "family:seen"].update(r["counts"])
        groups[f"family:{doc['family']}"].update(r["counts"])
        if not r["counts"]["schema_valid"]:
            failures.append({"doc_id": doc_id, **r["detail"]})
    lat = [p["latency_ms"] for p in preds.values() if p.get("latency_ms")]
    toks = [p["new_tokens"] for p in preds.values() if p.get("new_tokens")]
    mem = [p["peak_mem_mb"] for p in preds.values() if p.get("peak_mem_mb")]
    out = {
        "predictions": os.path.relpath(pred_path),
        "split": os.path.relpath(split_path),
        "sections": sections,
        "missing_predictions": len(missing),
        "overall": summarise(groups["all"], sections),
        "all_items_f1_95ci": bootstrap_f1(per_doc, sections) if per_doc else None,
        "groups": {k: summarise(v, sections) for k, v in sorted(groups.items()) if k != "all"},
        "latency_ms": {"mean": round(statistics.mean(lat)), "median": round(statistics.median(lat)), "p90": round(sorted(lat)[int(0.9 * (len(lat) - 1))])} if lat else None,
        "new_tokens_mean": round(statistics.mean(toks), 1) if toks else None,
        "peak_mem_mb": max(mem) if mem else None,
        "schema_failures_sample": failures[:10],
        "per_doc": {i: dict(c) for i, c in zip(per_doc_ids, per_doc)},
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--split", default="test", help="split name or path to a jsonl with gold targets")
    ap.add_argument("--sections", default="document_date,labs,medications,follow_up")
    a = ap.parse_args()
    split_path = a.split if a.split.endswith(".jsonl") else os.path.join(DATA, f"{a.split}.jsonl")
    res = evaluate(a.pred, split_path, a.sections.split(","))
    out = a.pred.replace(".jsonl", ".metrics.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    o = res["overall"]
    print(f"{a.pred}: docs={o['docs']} schema_valid={o['schema_valid_rate']} all_items_F1={o['all_items']['f1']} CI={res['all_items_f1_95ci']}")
    for sec in SECTIONS:
        if sec in o:
            print(f"  {sec}: P={o[sec]['precision']} R={o[sec]['recall']} F1={o[sec]['f1']}")
    print(f"  date_acc={o.get('document_date_accuracy')} evidence_valid={o['evidence_valid_rate']} unsupported_rate={o['unsupported_item_rate']}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
