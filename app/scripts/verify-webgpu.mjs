// Loads a model through WebLLM in the locally installed Chrome and runs one
// extraction twice. Usage: node scripts/verify-webgpu.mjs [original|finetuned] [baseUrl]
// Expects `npm run dev` to be running. Headed by default: WebGPU in headless
// Chrome on Windows is unreliable. Set HEADLESS=1 to try headless.
import { chromium } from "playwright-core";
import { mkdirSync, writeFileSync } from "node:fs";

const which = process.argv[2] ?? "original";
const base = process.argv[3] ?? "http://localhost:5199/";
const profile = new URL("../.chrome-profile/", import.meta.url);
const ctx = await chromium.launchPersistentContext(profile.pathname.replace(/^\/([A-Za-z]:)/, "$1"), {
  channel: "chrome",
  headless: process.env.HEADLESS === "1",
  args: ["--enable-unsafe-webgpu", "--enable-features=WebGPU", "--window-size=900,700", ...(process.env.FORCE_DGPU === "1" ? ["--force_high_performance_gpu"] : [])],
});
const page = ctx.pages()[0] ?? (await ctx.newPage());
page.on("console", (m) => { if (m.type() === "error") console.log("console:", m.text()); });
await page.goto(`${base}dev/webgpu-check.html?model=${which}`);
const handle = await page.waitForFunction(() => window.__result, null, { timeout: 30 * 60 * 1000, polling: 1000 });
const result = await handle.jsonValue();
mkdirSync("../results/deployment", { recursive: true });
const file = `../results/deployment/webgpu-${which}${process.env.FORCE_DGPU === "1" ? "-dgpu" : ""}.json`;
writeFileSync(file, JSON.stringify({ checkedAt: new Date().toISOString(), ...result }, null, 2));
console.log(JSON.stringify({ ok: result.ok, error: result.error, loadMs: result.loadMs, latencyMs: result.first?.latencyMs, deterministic: result.deterministic }, null, 2));
console.log(result.first?.raw);
await ctx.close();
process.exit(result.ok ? 0 : 1);
