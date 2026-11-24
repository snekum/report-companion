"""Collect evaluation metrics into results/summary.json (read by the app) and
results/RESULTS.md (read by people).

  python model/summarize.py --ft-tag ft-final [--base-tag base] [--browser results/deployment/browser_eval.json]

Expects <tag>_<split>.metrics.json files written by model/evaluate.py.
Missing files are reported, never filled in.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, ROOT, SECTIONS  # noqa: E402

PRED = os.path.join(ROOT, "results", "predictions")


def load(tag, split):
    p = os.path.join(PRED, f"{tag}_{split}.metrics.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def flat(s: dict | None, sections) -> dict | None:
    if not s:
        return None
    g = lambda *k: _get(s, k)  # noqa: E731
    out = {
        "all_f1": g("all_items", "f1") if {"medications", "follow_up"} <= set(sections) else None,
        "labs_f1": g("labs", "f1"), "labs_precision": g("labs", "precision"), "labs_recall": g("labs", "recall"),
        "unit_exact": g("labs", "unit_exact"), "range_exact": g("labs", "reference_range_exact_when_printed"),
        "lab_date_exact": g("labs", "date_exact"), "value_altered": g("labs", "value_altered"),
        "meds_f1": g("medications", "f1"), "meds_precision": g("medications", "precision"), "meds_recall": g("medications", "recall"),
        "meds_status_acc": g("medications", "status_accuracy"), "false_current": g("medications", "false_current"),
        "dose_exact": g("medications", "dose_exact"), "frequency_exact": g("medications", "frequency_exact"),
        "follow_up_f1": g("follow_up", "f1"), "date_acc": s.get("document_date_accuracy"),
        "schema_valid": s.get("schema_valid_rate"), "evidence_valid": s.get("evidence_valid_rate"),
        "fields_in_evidence": s.get("fields_in_evidence_rate"),
        "unsupported_rate": s.get("unsupported_item_rate"), "unsupported_items": s.get("unsupported_items"), "docs": s.get("docs"),
    }
    return out


def _get(d, keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def f1_from(counts_list, sections):
    tp = pr = gd = 0
    for c in counts_list:
        for sec in SECTIONS:
            if sec in sections:
                tp += c.get(f"{sec}_tp", 0); pr += c.get(f"{sec}_pred", 0); gd += c.get(f"{sec}_gold", 0)
    p, r = (tp / pr if pr else 0), (tp / gd if gd else 0)
    return 2 * p * r / (p + r) if p and r else 0.0


def paired_bootstrap(a: dict, b: dict, sections, n=2000, seed=0):
    """95% interval for F1(b) - F1(a), resampling the same documents for both."""
    ids = sorted(set(a) & set(b))
    rng = random.Random(seed)
    diffs = []
    for _ in range(n):
        sample = [ids[rng.randrange(len(ids))] for _ in ids]
        diffs.append(f1_from([b[i] for i in sample], sections) - f1_from([a[i] for i in sample], sections))
    diffs.sort()
    point = f1_from([b[i] for i in ids], sections) - f1_from([a[i] for i in ids], sections)
    return {"difference": round(point, 4), "ci95": [round(diffs[int(0.025 * n)], 4), round(diffs[int(0.975 * n)], 4)], "docs": len(ids)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft-tag", required=True)
    ap.add_argument("--base-tag", default="base")
    ap.add_argument("--rules-tag", default="regex")
    ap.add_argument("--train-run", default=None, help="model/outputs/<run> to read training config and summary from")
    ap.add_argument("--browser", default=None)
    a = ap.parse_args()
    full = ["document_date", "labs", "medications", "follow_up"]
    systems = {"rules": a.rules_tag, "original": a.base_tag, "finetuned": a.ft_tag}
    groups, missing, raw = {}, [], {}
    for split in ["test", "challenge"]:
        for key, tag in systems.items():
            m = load(tag, split)
            raw[(key, split)] = m
            if m is None:
                missing.append(f"{tag}_{split}")
                continue
            secs = m["sections"]
            groups.setdefault(split, {})[key] = flat(m["overall"], secs)
            if split == "test":
                groups.setdefault("test_seen", {})[key] = flat(m["groups"].get("family:seen"), secs)
                groups.setdefault("test_held_out", {})[key] = flat(m["groups"].get("family:held_out"), secs)
            groups[split][key]["latency_median_ms"] = (m.get("latency_ms") or {}).get("median")
            groups[split][key]["peak_mem_mb"] = m.get("peak_mem_mb")
            groups[split][key]["new_tokens_mean"] = m.get("new_tokens_mean")
    comparisons = {}
    for split in ["test", "challenge"]:
        o, f = raw.get(("original", split)), raw.get(("finetuned", split))
        if o and f:
            comparisons[f"{split}: fine-tuned minus original, all-facts F1"] = paired_bootstrap(o["per_doc"], f["per_doc"], full)
            for sec in ["labs", "medications", "follow_up"]:
                comparisons[f"{split}: fine-tuned minus original, {sec} F1"] = paired_bootstrap(o["per_doc"], f["per_doc"], [sec])
        r = raw.get(("rules", split))
        if r and f:
            comparisons[f"{split}: fine-tuned minus rules, labs F1"] = paired_bootstrap(r["per_doc"], f["per_doc"], ["labs"])
    stats = json.load(open(os.path.join(DATA, "stats.json")))
    training = None
    if a.train_run:
        d = os.path.join(ROOT, "model", "outputs", a.train_run)
        training = {"config": json.load(open(os.path.join(d, "train_config.json"))),
                    "summary": json.load(open(os.path.join(d, "train_summary.json"))) if os.path.exists(os.path.join(d, "train_summary.json")) else None}
    browser = json.load(open(a.browser)) if a.browser and os.path.exists(a.browser) else None
    summary = {
        "generatedOn": dt.date.today().isoformat(),
        "testSet": {"docs": stats["test"]["docs"], "patients": stats["_patients"]["test"], "heldOutFamilies": ["L3 (compact codes)", "V3 (medication list printout)"]},
        "challengeSet": {"docs": 20},
        "groups": groups,
        "comparisons": comparisons,
        "training": training,
        "browser": browser,
        "missing": missing,
        "notes": [
            "Rules only attempt lab rows and the document date, so they have no all-facts, medicine or follow-up score.",
            "The rules' test-name dictionary is the same list the synthetic templates use, which flatters them on this data.",
            "Original and fine-tuned models were both run in 4-bit (nf4) with the same prompt, greedy decoding and a 1,024-token limit.",
        ],
    }
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(summary, open(os.path.join(ROOT, "results", "summary.json"), "w"), indent=2)
    # human-readable table
    rows = [("All facts F1", "all_f1"), ("Lab results F1", "labs_f1"), ("  units exact", "unit_exact"), ("  ranges exact (when printed)", "range_exact"),
            ("  result dates exact", "lab_date_exact"), ("Medicines F1", "meds_f1"), ("  status correct", "meds_status_acc"),
            ("  wrongly called current (count)", "false_current"), ("  dose exact", "dose_exact"), ("  frequency exact", "frequency_exact"),
            ("Follow-up F1", "follow_up_f1"), ("Document date correct", "date_acc"), ("Valid format (schema)", "schema_valid"),
            ("Quotes found in document", "evidence_valid"), ("Unsupported facts (count)", "unsupported_items"),
            ("Median latency per doc (ms)", "latency_median_ms"), ("Peak GPU memory (MB)", "peak_mem_mb"), ("Mean output tokens", "new_tokens_mean")]
    L = [f"# Results ({summary['generatedOn']})", "", "Generated by `model/summarize.py` from the metrics files in `results/predictions/`.", ""]
    for g, title in [("test", f"Test set ({stats['test']['docs']} documents, {stats['_patients']['test']} unseen patients)"), ("test_seen", "Test set, layouts seen in training"),
                     ("test_held_out", "Test set, layouts never seen in training"), ("challenge", "Challenge set (20 hand-written documents)")]:
        if g not in groups:
            continue
        L += [f"## {title}", "", "| Measure | Rules | Original | Fine-tuned |", "|---|---|---|---|"]
        for label, k in rows:
            vals = []
            for sys_ in ["rules", "original", "finetuned"]:
                v = (groups[g].get(sys_) or {}).get(k)
                vals.append("–" if v is None else (f"{v:.3f}" if isinstance(v, float) and v <= 1 and "count" not in label and "ms" not in label and "MB" not in label and "tokens" not in label else f"{v}"))
            L.append(f"| {label} | " + " | ".join(vals) + " |")
        L.append("")
    if comparisons:
        L += ["## Paired differences (bootstrap over documents, 2,000 resamples)", "", "| Comparison | Difference | 95% interval |", "|---|---|---|"]
        for k, v in comparisons.items():
            L.append(f"| {k} | {v['difference']:+.3f} | {v['ci95'][0]:+.3f} to {v['ci95'][1]:+.3f} |")
        L.append("")
    if missing:
        L += ["Missing (not run): " + ", ".join(missing), ""]
    open(os.path.join(ROOT, "results", "RESULTS.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
