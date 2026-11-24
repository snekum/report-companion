# Report Companion: progress log

Newest entries at the bottom of each section. "Done" means it actually ran.

## Current state (2026-11-24)

- The fine-tuned model (`final`: QLoRA, 951 docs x 1 epoch on a Kaggle T4) works in Python: test all-facts F1 0.992 vs 0.820 for the original model, challenge set 0.930 vs 0.613. Tables in `results/RESULTS.md`.
- Its browser build (WebLLM, q4f16_1) was broken: the adapter didn't survive the browser's 4-bit rounding. Diagnosis under "Done" and in `results/deployment/diag_browser/`.
- Fixed by retraining on a base rounded exactly like the browser's (`q4grid`, `--base-grid q4f16_1`). Its browser build works: test40 in Chrome 92.5% valid format, all-facts F1 0.996, matching the same model in Python. This is the shipped model. Weights not yet hosted.
- The app shows `q4grid`'s prepared outputs and runs the original model live.

**Gotchas:**
- Windows MLC wheels lack `SPIRV-Tools.dll`, so conversion runs in Docker (`model/Dockerfile.mlc`, with apache-tvm-ffi 0.1.13 and numpy 2.2.6 pinned).
- Vite must ignore `app/.chrome-profile`.
- Chrome uses the Intel iGPU unless `--force_high_performance_gpu` is passed.
- WebLLM returns the empty `<think></think>` block inside its reply; the parser strips it.
- On Windows under memory pressure, safetensors' memory-mapped reads crash (access violation / "paging file too small"); the diagnostic scripts and `model/q4grid.py` read tensors with plain file reads instead.
- transformers 5 writes `rope_parameters` instead of `rope_theta`, which mlc-llm 0.26.dev6 can't read (`merge_adapter.py` writes the original config).

## Environment (checked 2026-11-23)

- Windows 11, i7-9750H (12 threads), 16 GB RAM, D: has ~200 GB free.
- GPU: GTX 1650, 4 GB VRAM, Turing (sm_75). Has no tensor cores and no bf16, so training uses fp16 compute with 4-bit weights.
- Node 24.14, npm 11.11, Python 3.12 (via uv) for the project venvs, Chrome 157.
- No Java was installed, so a portable Temurin 21 JRE is downloaded into `tools/` (gitignored).

## Decisions

| # | Decision | Why |
|---|---|---|
| D1 | Train locally on the GTX 1650 (QLoRA, 4-bit, fp16 compute) | The owner already ran LoRA SFT on this GPU (`../SFT`). A 1.7B model in 4-bit is ~1.2 GB, which leaves room for LoRA plus activations at 1.5k tokens with gradient checkpointing. No paid compute needed unless the pilot shows it doesn't fit. |
| D2 | Browser inference through WebLLM 0.2.89 | Its prebuilt list includes `Qwen3-1.7B-q4f16_1-MLC` (~2 GB VRAM) and a `q4f32_1` fallback for GPUs without 16-bit shaders. A fine-tune with the same architecture and quantisation can reuse the prebuilt compiled library (`.wasm`); only the weights change. |
| D3 | Weight conversion with `mlc-llm-nightly-cpu 0.26.dev20` + `mlc-ai-nightly-cpu 0.26.dev529` | Both have Windows wheels, and conversion needs no GPU. |
| D4 | Synthea v4.0.0 (tagged 2026-03-05), run locally with a fixed seed | A pinned version plus seed makes the data reproducible. Code is Apache-2.0; the project FAQ says generated data is public domain. |
| D5 | Training stack: transformers 5.24.0, peft 0.22.0, trl 1.18.2, bitsandbytes 0.50.2, accelerate 1.16.0, datasets 5.2.0 | Latest on PyPI on 2026-11-23. Pinned in `model/requirements.txt`. |
| D6 | Evaluate base and fine-tuned models in the same 4-bit (nf4) setup | fp16 1.7B weights (3.4 GB) don't fit next to the KV cache in 4 GB. Same quantisation, prompt, greedy decoding and token limit for both. The browser (q4f16_1) model is evaluated separately after conversion. |
| D7 | Reference ranges come only from an authored fixture of fictional labs | Synthea has no reference ranges. The spec forbids inventing them except in an explicitly authored fictional fixture, so `pipeline/fixtures/fictional_lab_ranges.json` is labelled as such and the app says so. |
| D8 | Medication status is `current`, `not_current` or `unclear` | An old prescription line is `unclear`, never `current`. |
| D9 | Everything runs in the browser; there is no server path | Pasted text never leaves the device. The only network traffic is the model download. That removes the logging/persistence risk entirely. |

## Plan

1. Environment + deployment check: load the original Qwen3-1.7B in Chrome via WebLLM and run one extraction.
2. Data pipeline: Synthea, patient split, rendered documents, gold targets, provenance, challenge set. Inspect sample pairs.
3. App with labelled prepared outputs: workspace, highlighting, corrections, compare view, questions, brief export.
4. Baselines: regex lab-row extractor and original Qwen3 with the extraction prompt.
5. QLoRA pilot, then a full (still small) run if the pilot shows signal. Evaluate.
6. Merge, convert to MLC, test the fine-tuned model in the browser, re-evaluate it.
7. End-to-end check, README, model card, walkthrough.

## Done

- **Deployment check, original model (2026-11-23).** `app/scripts/verify-webgpu.mjs` loads `Qwen3-1.7B-q4f16_1-MLC` through WebLLM 0.2.89 in Chrome 157 and runs one extraction twice. Results in `results/deployment/`.
  - Works, deterministic at temperature 0 (both runs identical).
  - Chrome picked the Intel UHD 630 by default: 154 s for 217 output tokens. With `--force_high_performance_gpu` it used the GTX 1650: 16 s warm (46 s first call, including shader compilation). Same output on both GPUs.
  - First load took ~6 min, including the ~984 MB download; a cached load took 9 s.
  - WebLLM returns the empty `<think>\n\n</think>` block inside the reply, so the parser strips it.
  - Original model on the sample: no evidence quotes, report date missed, follow-up returned as bare strings (schema-invalid). A useful baseline signal.
- **Synthea run:** 700 patients in 2.5 min. `pipeline/load_synthea.py` joins CSV (labs, encounters, medication dates) with FHIR (dosing, care plan activities).
- **Dataset v1** (`pipeline/build_dataset.py`): train 951 docs / 486 patients, val 121 / 62, test 204 / 107, demo 132 / 45. 0 gold-check failures, 0 exact duplicates. The checker caught one real template bug (test name only on a heading line in the trend family), which was fixed.

- **Baselines.** `model/baseline_regex.py`: on validation lab rows it reaches F1 1.0 after fixing two bugs found on validation (range printed as a bare column; drug salt names like "Penicillin V Potassium" matching "Potassium"). On the 20-case challenge set it drops to lab F1 0.64 and date accuracy 0.15. Its dictionary is the templates' own list of test names, so it is an optimistic baseline on synthetic data.
- **Challenge set:** 20 hand-written fictional documents (`pipeline/challenge/challenge_set.py`), each aimed at one failure mode. Labels checked by hand and by code (evidence contains every value). Not clinically reviewed.
- **Prompt change before any test evaluation:** medications now exclude allergies and other people's medicines (found while writing challenge cases).
- **Training feasibility (local GTX 1650):** works, peak 2.9 GB, ~32 s per example (33 tokens/s). Uses a short explicit loop (transformers + peft) with `logits_to_keep` so only the answer positions produce logits; TRL's SFTTrainer would materialise full-sequence logits that do not fit in 4 GB. Smoke test: val loss 0.148 -> 0.111 after 3 steps.
- **4-bit generation speed locally:** ~6 tok/s at batch 1, ~10 tok/s total at batch 4. Base model on the 204-doc test set is roughly 3 h.
- **App (React 19 + TS + Vite 8):** workspace with highlight-to-source, corrections ("your version", re-checked by code), explain tab (document statements kept apart from MedlinePlus-sourced definitions), questions + brief export, compare view with per-report ranges, model comparison page, on-device WebLLM panel, rules for pasted text. 83 unit tests (incl. Python/TS parity on 60 docs) and 17 browser checks pass, also from a `/report-companion/` subpath build.
- **Cloud notebook** `model/cloud_train_eval.ipynb` + `tools/make_cloud_bundle.sh` prepared for a free Kaggle/Colab T4, after the owner offered on 2026-11-23.
- **MLC on Windows:** the Windows nightly wheels are missing `SPIRV-Tools.dll` (tvm_compiler.dll fails to load). Conversion runs in a Linux container instead (`model/convert_mlc.sh`, Docker Desktop).

- **Docs drafted:** README (reproducible commands), model card, technical walkthrough. Results sections are filled in after the final evaluation.
- **Test runner quirk (Windows):** vitest must be started from a `D:/` path, not `d:/`, or it loads itself twice and every suite fails with "failed to find the current suite". Not a code issue.
- **Compare view:** a result with no date of its own is drawn at the document's date and now marked with "*", after a dark-mode check showed the same glucose value appearing as two dates.

- **Pilot run (local, 160 examples, 1 epoch, 20 steps):** val loss 0.123 -> 0.0075; 128 min wall time including a ~20 min battery sleep (17:03-17:23, laptop not restarted, training resumed). Peak 3.2 GB.
- **Browser export route verified:** self-converted original weights load with the prebuilt WebGPU library; 2 of 3 demo outputs identical to the official weights, 1 differs by one token. Toolchain fixes: Linux container, apache-tvm-ffi 0.1.13 (0.1.16 segfaults), numpy 2.2.6 (2.5 rejects TVM dtypes). Details in `results/deployment/conversion_check.json`.
- **Dev-server fixes:** Vite now ignores `app/.chrome-profile` and pre-bundles WebLLM, after a mid-run page reload killed a browser evaluation.

- **Pilot vs original, 40 validation docs (val40), 2026-11-23 20:00.** Pilot: format valid 20% -> 87.5%, document date 30% -> 100%, follow-up F1 0.31 -> 0.99, medicines F1 0.76 -> 0.89, quotes found 71% -> 99.6%, unsupported facts 3.4% -> 1.8%. **But lab F1 0.98 -> 0.72 (recall 0.57)**, which fails criterion (b).
  - Cause, from inspecting the raw outputs: list formatting, not reading. In 4 docs it repeats the `"labs":` key for each result (JSON keeps only the last). In 7 key-value (L2) or letter (V2) docs it closes the list after the first result. Table layouts: 47/47. Labs it does write: precision 0.98. Also copies wording ("continues") into the medicine status field 4 times.
  - Diagnostic only (not the official score): merging the duplicate keys gives lab recall 0.73.
  - Same pattern on the 8 demo docs (pilot lab recall 0.39 vs original 0.80; everything else better).
  - Read: under-training (160 examples x 1 epoch; ~20 key-value lab docs seen). Owner asked to decide: full run (951 docs x 1 epoch on a free T4, then re-check val40 before any test use) or stop and report the pilot.
- **App now shows the pilot's prepared outputs**, labelled "Fine-tuned model (pilot)". The compare view now picks, per report, the owner's corrections or else the result with the most verified lab values, and says which. Before this, it defaulted to the pilot and had no shared tests to chart.

- **Cloud notebook reworked for the full run (2026-11-23, owner chose the full run on Kaggle).** Run name `final`, 951 docs x 1 epoch (the rule written before the pilot). Fixes: Kaggle unzips uploaded zips, so the notebook now accepts the extracted folder; the packaging cell had a line break inside a string (would have crashed before writing `outputs.zip`). With "GPU T4 x2" the original model runs on GPU 1 while training uses GPU 0. Only val40 is scored in the cloud; test/challenge predictions are generated but not scored. The cloud original-model val40 file is tagged `base-t4` so it doesn't overwrite the local `base_val40`. Bundle now also has `evaluate.py`, `val40.jsonl`, `demo_journeys.jsonl`; checked by scoring the pilot's val40 predictions from an unpacked bundle (same numbers).

- **Full run on Kaggle (2026-11-24).** 951 docs x 1 epoch, 119 steps, 26.7 min on one T4 (0.60 examples/s, ~27x the GTX 1650), peak 3.4 GB, val loss 0.123 -> 0.0001. All prediction files complete (val40, test, challenge, demo, b1 latency; one row per doc, no empty outputs). torch 2.11.0+cu130, Python 3.13 on Kaggle.
- **Val40 check: PASSES all three criteria.** Full run vs original on the same T4 (`base-t4`): format valid 20% -> 100%, document date 30% -> 100%, follow-up F1 0.31 -> 1.0, medicines F1 0.77 -> 1.0, **lab F1 0.98 -> 0.997** (one CO2 row dropped in one L5 doc), quotes found 71% -> 100%, unsupported facts 3.4% -> 0%. The pilot's failure modes are gone: no repeated `"labs"` keys, status values only current/not_current/unclear. `base-t4` matches the local `base_val40` closely (labs identical), so the hardware change didn't move the baseline.
  - Caveat: val40 has only layouts seen in training (V1, L2, V2, L1, L5, L4) and no doc text is copied from train, but the templates are shared. So this shows the pilot's formatting problem is fixed, not that the model generalises. The held-out layouts (L3, V3) in test and the challenge set answer that.
- **Test + challenge scored ONCE (2026-11-24, owner's go).** Tables in `results/RESULTS.md` / `results/summary.json` (`model/summarize.py --ft-tag ft-final --train-run final`). Do not re-tune on these numbers.
  - Test (204 docs): all-facts F1 original 0.820 -> fine-tuned 0.992; labs 0.900 -> 0.992; medicines 0.814 -> 0.987; follow-up 0.367 -> 1.000; format 22% -> 99.5%; unsupported facts 52 -> 0. Rules get lab F1 1.000 on test (their dictionary is the templates' own test names: optimistic, as noted).
  - Unseen layouts (L3, V3): all-facts F1 0.983. Main weakness: **document date 66% (L3 only, 26 misses)**. 24 of 26 found the right line but copied extra text into the value ("COLL October 19, 2022 | FINAL"), 1 null, 1 unparseable output (the only schema failure on test).
  - Challenge (20 docs): all-facts F1 original 0.613 -> fine-tuned 0.930, rules labs 0.643. Fine-tuned misses: negated aspirin (should be not_current, left out), uncertain salbutamol (should be unclear, left out), a pending Vitamin D written as a lab with value null (schema failure), one unsupported item in the no-unit case, a follow-up made up from a negated one ("no follow-up needed"), and follow-up/date in the appointment-without-date case.
  - Latency/memory in these tables are from the T4 at batch 16, not the laptop or the browser.
- **Merge + MLC conversion of the final adapter (2026-11-24).** `model/merged/final` (fp16, 3.4 GB; LoRA changed the attention/MLP weights, embeddings untouched, no leftover LoRA tensors). Converted to `model/mlc/ReportCompanion-Qwen3-1.7B-q4f16_1-MLC` (939 MB, 30 shards, `SHA256SUMS` inside): same 339 tensor names/shapes/dtypes as the verified base conversion, 29 of 30 shards changed; config + tokenizer byte-identical to the official prebuilt. Four fixes on the way:
  - The `rc-mlc` image had been built before the tvm-ffi pin and still had 0.1.16. Rebuilt from the Dockerfile (now 0.1.13); `convert_mlc.sh` now refuses an image with any other version. Dockerfile got a longer pip timeout after a pytorch.org download timed out.
  - MLC writes `model.safetensors.index.json` next to the weights, which fails on the read-only mount: `merge_adapter.py` now writes the index.
  - transformers 5 saves `rope_theta` inside `rope_parameters`, which mlc-llm 0.26.dev6 can't read: `merge_adapter.py` now writes the original Qwen `config.json` (dtype float16). Checked it still loads in transformers 5.
  - `convert_mlc.sh` printed "note" and exited 0 when the config/tokenizer downloads failed; now it fails.
- **Browser model is broken (2026-11-24).** WebLLM in Chrome (GTX 1650) on `data/processed/test40.jsonl` (first 5 docs of each layout): valid format 10%, all-facts F1 0.22, lab recall 0.14, vs 100% / 0.996 for the Python adapter run on the same 40 docs. Signature: closes the answer after the date and starts a new object per field (`{...}}, {"labs": []}, ...`); also `"status": "inactive"`, two loops to the token cap. Investigation in `results/deployment/diag_browser/`, on 8 failed docs:
  - Prompt ruled out: browser prompt token counts equal the training prompt's (incl. the empty think block) on every doc.
  - Merge ruled out: merged model loaded in nf4 in Python = 8/8 perfect, identical to the adapter.
  - **Runtime ruled out, conversion is the cause:** the q4f16_1 weights unpacked back into Python (`mlc_weight_check.py`; unpacking validated on the original model, 10% relative error = normal 4-bit rounding) fail identically to the browser: same scores, 6/8 outputs character-identical, the other 2 differ only in line breaks.
  - Loss on the gold answer (4 docs): merged 0.008, unpacked browser weights 0.040, original model 0.367. The fine-tune is mostly there, but the LoRA change is tiny (0.4% of a weight matrix) next to 4-bit rounding (10%), and one wrong structural token (`}}` vs `,`) after the date derails the whole answer.
  - 4-bit embedding/output layer is not the main cause: browser weights with an fp16 embedding = loss 0.037, 1/8 valid (`mlc_embed_check.py`).
  - **Adapter doesn't transfer to the q4f16_1 grid** (`grid_mismatch_check.py`): adapter on the nf4 base (training setup) loss 0.011; same adapter, unmerged, on the q4f16_1-rounded base (unpacked from the converted original model) loss 0.044, 2/8 valid, 3/8 drift back to chatty original-model output. So merging and the converter are not at fault; the q4f16_1 base is different enough from what the adapter was trained on.
  - WebLLM ships Qwen3-1.7B libraries only for q4f16_1 / q4f32_1 (same uniform 4-bit grid); q0f16 would need a self-compiled WebGPU library.
- **Option A chosen by the owner (2026-11-24): retrain on the browser's grid.** Other deployments considered and set aside for now: 8-bit in another browser runtime (Transformers.js / wllama; fallback if A fails), llama.cpp K-quants, server (breaks D9), local app, or ship the original model live + fine-tuned as prepared outputs.
  - `model/q4grid.py` reproduces q4f16_1 exactly (fp16 division; scale = max|w|/7, code = clamp(round(w/scale + 7), 0, 14)). `--verify` against `model/mlc/check-base-q4f16_1-MLC`: all 113 quantised tensors, every code and scale identical (incl. the 311M-value embedding).
  - `train.py` / `generate.py` take `--base-grid q4f16_1` (default stays nf4). Smoke tests: generate on 2 diag docs = character-identical to the unpacked-browser-base diagnostic; one training step runs (197 tensors rounded, peak 4.1 GB locally). Val loss of the untrained rounded base 0.309 vs 0.123 on nf4: the browser grid hurts this task much more than nf4.
  - Notebook generator moved to `tools/make_cloud_notebook.py`; notebook set to run `q4grid` (951 x 1 epoch, `--base-grid q4f16_1`), fine-tuned and original predictions both on the rounded base (`ft-q4grid`, `base-q4grid`). Bundle rebuilt with `q4grid.py`.
  - After the run: merge into the fp16 base as before, convert, unpack-and-generate check (diag tools) before Chrome. Gate on val40; disclose second use of the test set.
- **`q4grid` run on Kaggle (2026-11-24).** 951 docs x 1 epoch on the q4f16_1-rounded fp16 base, 119 steps, 27.0 min on a T4, peak 5.4 GB, val loss 0.241 -> 0.0002. All prediction files complete (val40, test, challenge, demo; one row per doc, no empty outputs, quant `q4f16_1-grid-fp16`).
- **Val40 check: PASSES all three criteria.** Both models on the rounded base: format valid 2.5% -> 100%, document date 0% -> 100%, follow-up F1 0.14 -> 1.0, lab F1 0.34 -> 0.997, unsupported facts 6.2% -> 0%. Fine-tuned numbers identical to `final`. Note the original model on the rounded base is much weaker than on nf4 (lab F1 0.34 vs 0.98): the "original model" running live in the app is this weaker one.
- **Merge + convert (2026-11-24).** Merged into the unrounded fp16 weights (`model/merged/q4grid`; merging into the rounded weights would let the converter snap small LoRA changes back to the old grid points). Converted to `model/mlc/ReportCompanion-q4grid-Qwen3-1.7B-q4f16_1-MLC` (939 MB, 30 shards). The config/tokenizer download failed once on the network; refetched, all five byte-identical to the earlier copies.
- **Unpack-and-generate check passes** (`diag_browser/mlc_q4grid_check.py`, the 8 docs the broken build failed): loss on the gold answer, training setup 0.0078, merged fp16 0.0068, unpacked browser weights 0.0153 (first 4 docs: 0.029 vs 0.040 for `final`). Generation: 6/8 valid, every lab/medicine/follow-up item right (F1 1.0, `final` build: 0/8, 0.25). The 2 failures are `"status": "inactive"` on V3 medicine lists; the 1 date miss is the known L3 "COLL ..." case.
- **Browser build works (2026-11-24).** WebLLM in Chrome (GTX 1650) on test40: valid format 92.5%, all-facts F1 0.996, lab recall 0.989, unsupported facts 0, vs the broken `final` build 10% / 0.223 / 0.137. Matches `q4grid` in Python on the same 40 docs (92.5%, 0.994); 36/40 parsed answers identical, the other 4 differ only in which stretch of text is quoted as evidence. Mean 32 s per document in Chrome. Remaining errors, the same in Python: `"inactive"` status on 2 V3 docs, one malformed null date (V1), 5 L3 date misses (`final` in Python: 0 format failures, 3 L3 date misses, so the rounded base costs a little).
- **Test + challenge scored for `q4grid` (2026-11-24, owner approved; second use of the test set, after `final`).** Python predictions from Kaggle, rounded base.
  - Test (204): valid format 95.6% (`final` 99.5%), all-facts F1 0.994 (0.992), labs 0.995, medicines 0.984, follow-up 1.0, document date 81% (87%), unsupported facts 0. The 9 format failures: 6 `"inactive"` status (V3, 81% valid there vs 100% for `final`), 3 malformed null date (V1). Dates: L3 18% (`final` 42%), the "COLL ..." prefix copied into the value.
  - Original model on the rounded base (`base-q4grid`, what the app runs live): test all-facts F1 0.634 vs 0.820 on nf4, valid format 8.8%.
  - Challenge (20): all-facts F1 0.857 (`final` 0.930), valid format 100% (95%). Lab F1 0.80 vs 0.97, from 4 docs: blood pressure listed as a lab, both HbA1c values in prose missed, unit pasted into a value ("14 pmol/L"), "Repeat lipid panel / 3 months" listed as a lab. So the rounded base mainly costs robustness on the hand-written tricky cases.

- **Shipped `q4grid` (owner's call, 2026-11-24: ship as is rather than retrain for the challenge loss).** `tools/make_results.sh` rebuilds RESULTS.md / summary.json: app columns rules / original (nf4, the stronger baseline) / fine-tuned (`q4grid`); RESULTS.md adds "original, browser rounding" and `final`, plus plain-language notes and the browser check (`results/deployment/browser_eval_q4grid.json`). `summarize.py` got `--extra` and `--note`. `build_prepared.py` defaults to `ft-q4grid` and labels each prepared output with its real runtime. README results section, MODEL_CARD results, WALKTHROUGH sections 6 (the browser failure and fix) and 8 filled in. App: 83 tests, 17/17 UI checks, build OK.
- **Framing (owner, 2026-11-24):** "Independent prototype exploring an approach for MedARC's AMFV project." AMFV = Agentic Medical Fact Verifier (github.com/MedARC-AI/amfv: claim decomposer, fact verifier, search agent; claims scored on a five-level scale against a curated corpus). Its README doesn't mention medical records, so the README links the two by the shared idea (split into individual facts, check each against evidence) and says it isn't an official MedARC project. No logos or branding.

## Blocked / needs the owner

- **Hosting the converted fine-tuned weights** (~1 GB, e.g. a Hugging Face model repo) still needs the owner's approval, and only once a browser build works. Until then the app shows the fine-tuned model as prepared results only.
- Pushing the repo and GitHub Pages were approved on 2026-11-24 and done: https://github.com/snekum/report-companion (public, commits as snekum with the GitHub no-reply email, set in this repo only), app at https://snekum.github.io/report-companion/ (workflow `pages.yml`, first deploy passed).

## Next steps

1. Owner to decide: commit and push (updates the live site).
2. Owner to decide: hosting the converted `q4grid` weights (~1 GB) so the app can run the fine-tuned model live.
