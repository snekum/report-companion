# Model card: Report Companion extractor (Qwen3-1.7B + QLoRA)

## What it does

Reads one short medical document (a lab report, visit summary or clinic letter, as plain text) and returns JSON with four fields:

- `document_date`: the date the document was issued, as printed.
- `labs`: test, value, unit, the reference range printed on that report, and a result date if one is printed.
- `medications`: name, dose, frequency, and `status`: `current`, `not_current` or `unclear` (for example an old prescription).
- `follow_up`: follow-up instructions written in the document.

Every item carries `evidence`, a quote copied from the document. Missing fields are `null`. The model is told not to diagnose, interpret, convert units or reformat dates.

Code, not the model, decides what is trusted. It checks that each quote occurs in the document and that each value, unit, range and date occurs in the quote, and it computes highlight positions itself. Facts that fail are shown as "check this".

## What it is not

- Not a diagnostic or clinical decision tool. It does not say whether a result is normal or what it means.
- Not medically validated. Fine-tuning taught the model a format and how to copy text faithfully. No clinician reviewed its outputs.
- Not tested on real patient documents. All training and test data is synthetic or hand-written fiction.

## Model

| | |
|---|---|
| Base model | [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) (Apache-2.0), thinking mode off |
| Method | QLoRA: base weights frozen in 4-bit nf4 (double quantisation, fp16 compute), LoRA rank 16, alpha 32, dropout 0.05, on all attention and MLP projections (17.4M trainable parameters, about 1% of the model) |
| Loss | Cross-entropy on the answer tokens only (the JSON plus the end-of-turn token); the prompt and document are not trained on |
| Optimiser | AdamW, learning rate 2e-4, cosine schedule with 5 warm-up steps, gradient clipping at 1.0, effective batch 8 (batch 1 × 8 accumulation), seed 1234 |
| Sequence length | up to 2,048 tokens (training examples: median about 1,100, longest about 1,650) |
| Software | transformers 5.24.0, peft 0.22.0, bitsandbytes 0.50.2, torch 2.9.1 (pinned in `model/requirements.txt`) |
| Browser version | adapter merged into fp16 weights, converted with MLC to `q4f16_1`, run with WebLLM 0.2.89 using the prebuilt Qwen3-1.7B WebGPU library |

The exact run configuration is saved next to each adapter in `model/outputs/<run>/train_config.json`, with a per-step log in `train_log.jsonl`.

## Training data

Synthetic documents rendered from [Synthea](https://github.com/synthetichealth/synthea) v4.0.0 records (700 patients, fixed seeds; see `data/processed/provenance.json`).

- Patients were split before any document was rendered: 70% train, 8% validation, 16% test, 6% demo (salted hash of the patient ID). No patient appears in more than one split.
- Eight document layouts ("template families"). Two of them (a compact pipe-separated lab format and a patient-portal medication list) appear only in the test set, to measure behaviour on unseen layouts.
- Labels are built from the same structured values that produced each line of text, and each label stores that line as its evidence. A script checks every label against its line; 0 failures.
- Variation: four date formats, three fictional labs with different ranges, missing ranges, missing dates, repeated measurements, vital signs as distractors, "not performed" and "pending" lines, stopped and old medications.
- Reference ranges come from an authored fixture of three **fictional** labs, because Synthea has none. They are illustrative and not clinical reference intervals.
- Medication frequency and care plan activities come from Synthea's FHIR export. Some combinations are clinically odd (for example a statin "as needed") because Synthea's dosing is simulated.

## Evaluation

Three systems with the same prompt, greedy decoding and a 1,024-token limit (the two models), compared on:

- **Test set:** synthetic documents from unseen patients, split into seen and unseen layouts.
- **Challenge set:** 20 hand-written fictional documents, each aimed at one failure mode (negation, uncertainty, old prescriptions, unconverted units, decimal commas, flags between value and unit, distractor dates, allergies, other people's medicines). Labels were written and checked by hand against the rules; not clinically reviewed.
- **Rules baseline:** a dictionary-and-pattern extractor for lab rows and dates. Its dictionary is the templates' own test-name list, which flatters it on synthetic data.

Measures: precision, recall and F1 per field type; exact copying of units, ranges and dates; medicine status accuracy and how often something was wrongly called current; follow-up F1; document-date accuracy; schema validity; how often quotes are found in the document; unsupported facts (values that do not appear in the document); latency and memory.

Full tables: [results/RESULTS.md](../results/RESULTS.md). Headline results: see the section below.

## Results

_Filled in from `results/RESULTS.md` after the final evaluation run._

## Limitations

- Synthetic training data has a small vocabulary and tidy layouts. Real documents have scanning noise, abbreviations, tables split across pages, handwriting and far more variety.
- Test-set scores measure how well the model learned these templates and a few unseen ones. They do not establish real-world accuracy.
- The challenge set is small (20 documents), so its numbers are noisy.
- A quote can be found in the document and still be attached to the wrong field. The checks catch made-up text, not every misreading.
- Medication status is only as good as the document's wording. "Unclear" is the correct answer more often than people expect.
- The browser version is quantised differently from the evaluation setup (q4f16_1 vs nf4) and was re-evaluated separately; see the results.

## Intended use

Demonstrating document organisation with traceable extraction in a portfolio. Fictional or the user's own documents, on the user's own device.
