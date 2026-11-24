// Dev-only: exposes load/run so scripts/browser-eval.mjs can push documents
// through the deployed (WebLLM, q4f16_1) model one at a time.
import { checkBrowser, loadModel, modelOptions, runExtraction } from "../src/lib/llm";
import { MODEL_CONFIG } from "../src/config";

const log = document.getElementById("log")!;
(window as unknown as { rc: unknown }).rc = {
  async load(which: "original" | "finetuned", url?: string) {
    if (url) MODEL_CONFIG.finetuned.url = url;
    const cap = await checkBrowser();
    const opt = modelOptions(cap).find((o) => o.key === which)!;
    if (!opt.available) throw new Error(opt.unavailableReason ?? "unavailable");
    const t0 = performance.now();
    await loadModel(opt, (r) => { log.textContent = r.text; });
    return { cap, modelId: opt.modelId, loadMs: performance.now() - t0 };
  },
  run: (text: string) => runExtraction(text),
};
