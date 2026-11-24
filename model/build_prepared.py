"""Assemble the app's prepared data: journeys, documents and labelled outputs.

Every output carries where it came from (which model, which quantisation,
when it was generated), so the app can say "prepared result" and never pass
it off as live inference. Missing outputs are simply left out.

  python model/build_prepared.py   -> app/src/data/prepared.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, ROOT, load_jsonl  # noqa: E402

PRED = os.path.join(ROOT, "results", "predictions")
OUT = os.path.join(ROOT, "app", "src", "data", "prepared.json")

SOURCES = {
    # key: (file tag, label, description)
    "finetuned": ("ft-q4grid", "Fine-tuned Qwen3-1.7B", "LoRA adapter trained for this task on weights rounded to 4 bits the way the browser rounds them; same prompt and greedy decoding as the original. This is the model the browser build is made from."),
    "original": ("base", "Original Qwen3-1.7B", "The original model with the extraction prompt; 4-bit (nf4), greedy decoding."),
    "rules": ("regex", "Rule-based lab rows", "Deterministic dictionary + pattern matching. Lab rows and the document date only."),
}


# Where the prepared outputs were generated, by the quantisation recorded in each row.
RUNTIME = {
    "bnb-nf4-fp16": "Python, transformers + bitsandbytes (4-bit nf4) on a GTX 1650",
    "q4f16_1-grid-fp16": "Python, transformers on a Kaggle T4, weights rounded to the browser's 4-bit grid",
}


def mtime(path):
    return dt.datetime.fromtimestamp(os.path.getmtime(path), dt.timezone.utc).date().isoformat()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft-tag", default="ft-q4grid", help="predictions tag of the fine-tuned model to show")
    a = ap.parse_args()
    ft_label = "Fine-tuned (pilot run)" if "pilot" in a.ft_tag else SOURCES["finetuned"][1]
    ft_desc = SOURCES["finetuned"][2] + (" Pilot adapter: 160 training documents, 1 epoch; known to cut some lab lists short." if "pilot" in a.ft_tag else "")
    SOURCES["finetuned"] = (a.ft_tag, ft_label, ft_desc)
    journeys = json.load(open(os.path.join(DATA, "journeys.json"), encoding="utf-8"))
    docs = load_jsonl(os.path.join(DATA, "demo_journeys.jsonl"))
    outputs = {}
    for key, (tag, label, desc) in SOURCES.items():
        path = os.path.join(PRED, f"{tag}_demo_journeys.jsonl")
        if not os.path.exists(path):
            print(f"  (no {key} outputs yet: {os.path.relpath(path, ROOT)})")
            continue
        for r in load_jsonl(path):
            outputs.setdefault(r["doc_id"], {})[key] = {
                "label": label, "description": desc, "raw": r["raw"],
                "latencyMs": r.get("latency_ms"), "generatedOn": mtime(path),
                "runtime": RUNTIME.get(r.get("quant"), "Python reference implementation"),
            }
    out_docs = []
    for d in docs:
        out_docs.append({
            "id": d["doc_id"], "journey": d["journey"], "title": d["title"], "kind": d["kind"], "family": d["family"],
            "text": d["text"], "reference": d["target"], "outputs": outputs.get(d["doc_id"], {}),
            "provenance": d["provenance"],
        })
    evals = {}
    summary = os.path.join(ROOT, "results", "summary.json")
    if os.path.exists(summary):
        evals = json.load(open(summary, encoding="utf-8"))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"builtOn": dt.date.today().isoformat(), "journeys": journeys, "documents": out_docs, "evaluation": evals}, f, ensure_ascii=False, indent=1)
    print(f"{len(out_docs)} documents, outputs for {len(outputs)} -> {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
