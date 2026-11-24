// Dev-only page: proves the original model loads and runs in this browser.
// Driven by scripts/verify-webgpu.mjs. ?model=original|finetuned
import { checkBrowser, modelOptions, loadModel, runExtraction } from "../src/lib/llm";

const SAMPLE = `NORTHBROOK COMMUNITY LAB (fictional)
Patient: Test Patient (fictional)   Collected: 2024-03-14   Reported: 2024-03-15

TEST                 RESULT   UNITS    REFERENCE RANGE
Glucose              104      mg/dL    70-99
Hemoglobin A1c       5.9      %        4.0-5.6
Total cholesterol    212      mg/dL    <200

Comment: Repeat fasting glucose in 3 months.`;

const log = document.getElementById("log")!;
const out: Record<string, unknown> = {};
const say = (s: string) => { log.textContent += "\n" + s; };

(async () => {
  try {
    const which = new URLSearchParams(location.search).get("model") ?? "original";
    const cap = await checkBrowser();
    out.capability = cap;
    say(JSON.stringify(cap));
    const option = modelOptions(cap).find((o) => o.key === which)!;
    out.option = option;
    if (!option.available) throw new Error(option.unavailableReason ?? "unavailable");
    const t0 = performance.now();
    await loadModel(option, (r) => say(`${r.progress.toFixed(2)} ${r.text}`));
    out.loadMs = performance.now() - t0;
    const r1 = await runExtraction(SAMPLE);
    out.first = r1;
    const r2 = await runExtraction(SAMPLE);
    out.second = r2;
    out.deterministic = r1.raw === r2.raw;
    say(r1.raw);
    out.ok = true;
  } catch (e) {
    out.ok = false;
    out.error = String(e);
    say("ERROR " + String(e));
  }
  (window as unknown as { __result: unknown }).__result = out;
})();
