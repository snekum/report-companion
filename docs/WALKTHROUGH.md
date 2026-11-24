# Technical walkthrough

A tour of how Report Companion works, in the order the pieces were built. Each section says what the piece does, why it is built that way, and where to look in the code.

## 1. The task, and why it's this narrow

The model gets one short document and has to fill in a fixed form: document date, lab results, medicines, follow-up instructions. Every item must include a quote copied from the document.

Keeping it to a fixed form (a "schema") is what makes the project testable. You can count exactly how many lab results the model found, how many it invented, and whether it copied "mg/dL" or quietly changed it to "mmol/L". A free-text summary can't be scored like that.

The rules the model follows are in one file, [shared/extraction_prompt.txt](../shared/extraction_prompt.txt), used by both the Python evaluation and the browser app. The schema is [shared/extraction.schema.json](../shared/extraction.schema.json).

Two rules that matter more than they look:

- **Copy, don't convert.** Values, units, dates and ranges stay exactly as printed. "02/12/2024" is not rewritten as "2024-02-12", because nobody can tell whether it means February or December.
- **Medicine status has three answers, not two.** "Current", "not current", or "unclear". A 2019 prescription in a records summary doesn't tell you whether the person still takes it, so the honest answer is "unclear".

## 2. Never trust the model's word: the checking layer

The model's output is treated as a claim. Plain code then checks it ([app/src/lib/validate.ts](../app/src/lib/validate.ts), mirrored in [model/common.py](../model/common.py)):

1. Is it valid JSON with exactly the expected fields?
2. Does the quote actually appear in the document? Whitespace differences are ignored, because models often squash the padding in table columns.
3. Does every value, unit, range and date appear inside that quote? A number only counts if it isn't part of a longer number, so "4" is not supported by "14".

Anything that fails is shown as "check this", with the reason in plain words. Highlight positions are always found by searching the document text, never taken from the model. Small models are bad at counting characters, and a wrong offset would highlight the wrong line, which is worse than no highlight.

The Python and TypeScript versions are kept identical by a shared file of test cases ([shared/validation_cases.json](../shared/validation_cases.json)) that both test suites run.

## 3. Data: synthetic patients, rendered into documents

There is no public set of real lab reports with labels that can be used freely, and scraping real records is off the table. So the data is built:

- **Synthea** generates fictional patients with full histories: lab results with units, medications with start and stop dates and dosing, care plans and visits ([pipeline/run_synthea.sh](../pipeline/run_synthea.sh), pinned to v4.0.0 with fixed seeds).
- **The loader** ([pipeline/load_synthea.py](../pipeline/load_synthea.py)) reads labs, visits and medication dates from Synthea's CSV export, and dosing and care plan activities from its FHIR export, which has things the CSV doesn't.
- **The renderer** ([pipeline/render.py](../pipeline/render.py)) writes short documents in eight layouts: a lab table, key-value lab lines, compact pipe-separated codes, a trend printout, a patient-portal message, an after-visit summary, a clinic letter, and a medication list.

The important design choice: **each label is written at the same moment as the line of text it describes**, from the same structured value, and the line is saved as the label's evidence. A checker then confirms every value appears in its line. On the first full run it caught a real bug (the trend layout printed the test name only on a heading line, so the result lines didn't support it). After the fix there were zero failures.

Things the spec forbids are handled explicitly:

- **No invented reference ranges.** Synthea has none, so ranges come from a small file of three openly fictional labs ([pipeline/fixtures/fictional_lab_ranges.json](../pipeline/fixtures/fictional_lab_ranges.json)). The labs deliberately disagree on some tests, which is why the app always keeps each report's own range.
- **No invented follow-up.** Follow-up items are either care plan activities active at the visit date or the patient's actual next visit in the record.

### Splitting fairly

Patients are assigned to train, validation, test or demo by a hash of their ID **before** any document is made. A patient's documents never straddle splits, so the test set measures new patients, not memorised ones. Two layouts are only ever used for test patients. That gives a "seen layouts" versus "unseen layouts" comparison: the gap shows how much the model learned the templates rather than the task.

A separate **challenge set** of 20 hand-written documents ([pipeline/challenge/challenge_set.py](../pipeline/challenge/challenge_set.py)) targets specific traps: "Patient denies taking aspirin", decimal commas, an H flag between value and unit, a document with allergies and a relative's medicine, and a letter whose only facts are an appointment.

## 4. Three systems compared

- **Rules** ([model/baseline_regex.py](../model/baseline_regex.py)): a list of test names plus patterns. It's fast and predictable, and on the tidy synthetic layouts it is very hard to beat for lab rows. That's partly because its list is the same list the templates use, which the results state openly.
- **Original Qwen3-1.7B** with the extraction prompt.
- **Fine-tuned Qwen3-1.7B**: the same model plus a LoRA adapter.

Both models run in the same 4-bit setup, with the same prompt, greedy decoding (always pick the most likely next word, so runs are repeatable) and the same 1,024-token limit ([model/generate.py](../model/generate.py)).

## 5. Fine-tuning on a 4 GB laptop GPU

**LoRA** freezes the original weights and learns a small correction for each layer, stored as two thin matrices. Here that is 17.4 million trainable numbers, about 1% of the model. **QLoRA** additionally stores the frozen weights in 4-bit, so a 1.7B model takes about 1.2 GB instead of 3.4 GB.

The loss is computed **only on the answer** (the JSON), not on the instructions or the document. The model is learning what to write, not how to predict the document.

The training loop ([model/train.py](../model/train.py)) is about 150 lines of plain PyTorch around transformers and peft, rather than TRL's trainer. The reason is memory. For every position in a 1,500-token example, the model scores all 151,936 words in its vocabulary. Done for every position and converted to 32-bit for the loss, that alone is about 1 GB, and the backward pass doubles it. Because only the answer positions are trained, the loop asks the model for scores at those positions only (`logits_to_keep`), which brings peak memory down to about 3 GB and makes training fit on the GTX 1650.

Training runs at about 32 seconds per example on this GPU. A small pilot ran locally first; see the progress log for how the full run was done.

## 6. Running in the browser

**WebLLM** runs models in the browser on the visitor's GPU through WebGPU. Its prebuilt list already includes Qwen3-1.7B in a 4-bit format called `q4f16_1`, with a compiled GPU program (a `.wasm` file) for that exact architecture.

A fine-tuned model has the same architecture, so it can reuse that compiled program. Only the weights change. The export path is:

1. Merge the adapter into the full-precision weights ([model/merge_adapter.py](../model/merge_adapter.py)).
2. Convert them to `q4f16_1` with MLC ([model/convert_mlc.sh](../model/convert_mlc.sh)). The Windows MLC packages are missing a DLL, so this runs in a Linux Docker container.
3. Copy the official prebuilt config and tokenizer files, so everything matches the compiled program.
4. Re-evaluate the converted model in Chrome ([app/scripts/browser-eval.mjs](../app/scripts/browser-eval.mjs)), because merging and re-quantising change the numbers slightly.

The conversion was first tested on the original model, by comparing the result with the official prebuilt files, before any fine-tuned weights existed.

Browsers without WebGPU still get the full app with clearly labelled prepared results and the rule-based extraction. There is no silent fallback to an online AI service.

## 7. The app

- [app/src/pages/Workspace.tsx](../app/src/pages/Workspace.tsx): the three-pane workspace. Facts come from a chosen "source": the fine-tuned or original model's prepared result, a live result from this device, the rules, or your corrected version. Each source is labelled with where it came from.
- Corrections create "your version" of the facts, saved only in this browser. Corrected facts are checked against the text again, so a typo in a correction is caught too.
- [app/src/lib/explain.ts](../app/src/lib/explain.ts): the plain-language tab. Statements about the document ("3 of 9 results have a range printed next to them") are computed by code and kept separate from general definitions, which link to MedlinePlus.
- [app/src/lib/compare.ts](../app/src/lib/compare.ts): compares two reports only for the same test in the same unit, with no conversions, each against its own printed range.

## 8. Results

_Added after the final evaluation run._
