// Runs a JSONL of documents through a model inside Chrome (WebGPU) and writes
// predictions in the format model/evaluate.py reads. Resumable.
//   node scripts/browser-eval.mjs <original|finetuned> <docs.jsonl> <out.jsonl> [modelUrl] [limit]
// Needs `npx vite` (port 5199) and, for a local fine-tuned model, scripts/serve-model.mjs.
import { chromium } from "playwright-core";
import { readFileSync, existsSync, appendFileSync } from "node:fs";

const [which, docsPath, outPath, modelUrl, limit] = process.argv.slice(2);
const docs = readFileSync(docsPath, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l)).slice(0, limit ? Number(limit) : undefined);
const done = new Set(existsSync(outPath) ? readFileSync(outPath, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l).doc_id) : []);
const todo = docs.filter((d) => !done.has(d.doc_id));
console.log(`${todo.length} to do, ${done.size} done`);
if (!todo.length) process.exit(0);
const profile = new URL("../.chrome-profile/", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");
const ctx = await chromium.launchPersistentContext(profile, {
  channel: "chrome", headless: false,
  args: ["--enable-unsafe-webgpu", "--window-size=700,400", ...(process.env.IGPU === "1" ? [] : ["--force_high_performance_gpu"])],
});
const page = ctx.pages()[0] ?? (await ctx.newPage());
page.setDefaultTimeout(30 * 60 * 1000);
await page.goto("http://localhost:5199/dev/browser-eval.html");
await page.waitForFunction(() => window.rc);
const info = await page.evaluate(([w, u]) => window.rc.load(w, u || undefined), [which, modelUrl ?? ""]);
console.log("loaded", info.modelId, info.cap.adapterDescription, Math.round(info.loadMs), "ms");
let i = 0;
for (const d of todo) {
  const r = await page.evaluate((t) => window.rc.run(t), d.text);
  appendFileSync(outPath, JSON.stringify({ doc_id: d.doc_id, raw: r.raw, latency_ms: Math.round(r.latencyMs), new_tokens: r.completionTokens, prompt_tokens: r.promptTokens,
    model: info.modelId, runtime: `WebLLM 0.2.89, Chrome, ${info.cap.adapterDescription}`, quant: "q4f16_1" }) + "\n");
  i++;
  if (i % 5 === 0) console.log(`  ${i}/${todo.length}  last ${Math.round(r.latencyMs)} ms`);
}
await ctx.close();
