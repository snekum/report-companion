#!/usr/bin/env bash
# Packs exactly what the cloud notebook needs (code + synthetic data) into
# dist/report-companion-cloud.zip. Contains no model weights, no secrets and
# only fictional/synthetic documents.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p dist
rm -f dist/report-companion-cloud.zip
python - <<'PY'
import zipfile
files = ["model/common.py", "model/train.py", "model/generate.py", "model/evaluate.py", "model/q4grid.py", "model/requirements.txt",
         "shared/extraction_prompt.txt", "shared/extraction.schema.json",
         "data/processed/train.jsonl", "data/processed/val.jsonl", "data/processed/val40.jsonl",
         "data/processed/test.jsonl", "data/processed/challenge.jsonl", "data/processed/demo_journeys.jsonl",
         "data/processed/provenance.json", "data/processed/stats.json"]
with zipfile.ZipFile("dist/report-companion-cloud.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f, f)
print("wrote dist/report-companion-cloud.zip with", len(files), "files")
PY
