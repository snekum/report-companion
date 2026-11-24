#!/usr/bin/env bash
# Rebuilds results/RESULTS.md and results/summary.json from the metrics files
# (run model/evaluate.py first). The shipped model is q4grid; see PROGRESS.md.
set -euo pipefail
cd "$(dirname "$0")/.."
python -I model/summarize.py --ft-tag ft-q4grid --train-run q4grid \
  --extra "Original, browser rounding=base-q4grid" --extra "Fine-tuned, nf4 (final)=ft-final" \
  --browser results/deployment/browser_eval_q4grid.json \
  --note "The fine-tuned model is the browser-ready version: it was trained on Qwen3-1.7B weights rounded to 4 bits exactly the way the browser rounds them, and these numbers come from that rounded model run in Python. In Chrome it gives the same answers on 36 of 40 test documents and the same scores." \
  --note "The original model is shown at its stronger 4-bit setting (nf4). With the browser's rounding, which is what the app runs live, it does worse: all-facts F1 0.634 on the test set instead of 0.820." \
  --note "An earlier fine-tune trained on the nf4 weights (final) scored about the same on the test set (0.992) and better on the challenge set (0.930 vs 0.857, though with 20 documents the 95% interval for the difference crosses zero), but it broke when converted for the browser, so it is not the one shown." \
  --note "All models get the same prompt, greedy decoding and a 1,024-token limit." \
  --note "Latency and memory are from a Kaggle T4 at batch size 16, not from a browser. The browser-rounded models use more memory here only because Python keeps the rounded weights in 16-bit; the browser stores them in 4-bit (about 0.9 GB). In Chrome on a GTX 1650 the fine-tuned model takes about 32 s per document." \
  --note "Rules only attempt lab rows and the document date, so they have no all-facts, medicine or follow-up score. Their test-name dictionary is the same list the synthetic templates use, which flatters them on this data."
