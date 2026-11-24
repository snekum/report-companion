#!/usr/bin/env bash
# Waits for the pilot adapter, then runs the pilot comparison on 40 validation
# documents (never the test set) and the 8 demo documents, one job at a time.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv-train/Scripts/python.exe
until [ -f model/outputs/pilot/adapter/adapter_config.json ] && [ -f model/outputs/pilot/train_summary.json ]; do sleep 30; done
echo "pilot finished: $(cat model/outputs/pilot/train_summary.json | tr -d '\n ')"
$PY model/generate.py --split val40 --tag base --batch-size 8 2>&1 | grep -E "to do|eta|Error|Traceback" | tail -3
$PY model/generate.py --split val40 --tag ft-pilot --adapter model/outputs/pilot/adapter --batch-size 8 2>&1 | grep -E "to do|eta|Error|Traceback" | tail -3
python -I model/evaluate.py --pred results/predictions/base_val40.jsonl --split val40
python -I model/evaluate.py --pred results/predictions/ft-pilot_val40.jsonl --split val40
$PY model/generate.py --split data/processed/demo_journeys.jsonl --tag base --batch-size 8 2>&1 | grep -E "to do|Error|Traceback" | tail -2
$PY model/generate.py --split data/processed/demo_journeys.jsonl --tag ft-pilot --adapter model/outputs/pilot/adapter --batch-size 8 2>&1 | grep -E "to do|Error|Traceback" | tail -2
echo "QUEUE DONE"
