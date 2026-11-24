// Drives the main user flow in the installed Chrome and saves screenshots.
//   node scripts/verify-ui.mjs [baseUrl]   (default http://localhost:5199/)
// Works against `npm run dev` or `npm run preview`. No model download: it
// checks the prepared-results flow, which must work without WebGPU.
import { chromium } from "playwright-core";
import { mkdirSync } from "node:fs";

const BASE = process.argv[2] ?? "http://localhost:5199/";
const OUT = "../results/ui-check";
mkdirSync(OUT, { recursive: true });
const results = [];
const check = (name, ok, detail = "") => { results.push({ name, ok }); console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? `  (${detail})` : ""}`); };

const browser = await chromium.launch({ channel: "chrome", headless: true });
const errors = [];
async function page(viewport) {
  const ctx = await browser.newContext({ viewport, acceptDownloads: true });
  const p = await ctx.newPage();
  p.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  p.on("pageerror", (e) => errors.push(String(e)));
  return p;
}

const p = await page({ width: 1400, height: 900 });
await p.goto(BASE);
await p.waitForSelector(".hero");
check("home shows the one-sentence pitch", await p.getByText("Turn a confusing medical document into an organised, understandable record").isVisible());
check("three example journeys", (await p.locator(".journey").count()) === 3);
await p.screenshot({ path: `${OUT}/01-home.png`, fullPage: true });

await p.getByRole("link", { name: "Open a sample document" }).click();
await p.waitForSelector(".source-text");
check("sample document opens", await p.locator(".source-text").isVisible());
const facts = p.locator(".fact");
check("facts are listed", (await facts.count()) > 3, `${await facts.count()} facts`);
await facts.nth(2).locator(".fact-select").click();
const activeMark = p.locator(".source-text mark.active");
check("clicking a fact highlights its source text", (await activeMark.count()) > 0, (await activeMark.first().textContent())?.slice(0, 60));
await p.screenshot({ path: `${OUT}/02-workspace-highlight.png` });

// correction flow
await p.getByRole("button", { name: "Correct", exact: true }).click();
const valueInput = p.locator(".editor input").nth(1);
await valueInput.fill("999");
await p.getByRole("button", { name: "Save" }).click();
check("a correction creates 'your version'", await p.getByRole("button", { name: "Your corrected version" }).getAttribute("aria-pressed") === "true");
check("corrected value that is not in the text is flagged", (await p.locator(".fact.review").count()) > 0);
await p.screenshot({ path: `${OUT}/03-correction-flagged.png` });

// explain tab
await p.getByRole("tab", { name: "Explain" }).click();
check("explain tab separates document facts from definitions", await p.getByRole("heading", { name: "About this document" }).isVisible() && await p.getByRole("heading", { name: "General definitions" }).isVisible());
await p.screenshot({ path: `${OUT}/04-explain.png` });

// questions + brief
await p.getByRole("tab", { name: /Questions/ }).click();
await p.getByLabel("New question").fill("Should this be rechecked?");
await p.getByRole("button", { name: "Add question" }).click();
check("question saved", await p.getByText("Should this be rechecked?").isVisible());
await p.getByRole("link", { name: /appointment brief/ }).click();
await p.waitForSelector(".brief-sheet");
const brief = await p.locator(".brief-sheet").textContent();
check("brief includes the question and the corrected flag", brief.includes("Should this be rechecked?") && brief.includes("with my corrections"));
const [dl] = await Promise.all([p.waitForEvent("download"), p.getByRole("button", { name: "Download (.md)" }).click()]);
check("brief downloads as markdown", dl.suggestedFilename().endsWith(".md"), dl.suggestedFilename());
await p.screenshot({ path: `${OUT}/05-brief.png`, fullPage: true });

// compare
await p.goto(`${BASE}#/compare/checkup`);
await p.waitForSelector("h1");
const charts = await p.locator(".cmp-card").count();
check("compare view draws charts for shared tests", charts > 0, `${charts} charts`);
await p.screenshot({ path: `${OUT}/06-compare.png`, fullPage: true });
await p.getByRole("button", { name: "Table" }).click();
check("compare has a table view", await p.locator("table.data").isVisible());

// model page
await p.goto(`${BASE}#/model`);
await p.waitForSelector("h1");
check("model page renders", await p.getByRole("heading", { name: /Original model vs fine-tuned/ }).isVisible());
await p.screenshot({ path: `${OUT}/07-model.png`, fullPage: true });

// paste flow
await p.goto(`${BASE}#/journey/mine?add=1`);
await p.getByLabel("Document text").fill("Lab report\nReported: 2025-01-02\nGlucose: 101 mg/dL (ref 70-99)\nSodium: 140 mmol/L\n");
await p.getByRole("button", { name: "Add document" }).click();
await p.waitForSelector(".source-text");
check("pasted text gets rule-based facts immediately", (await p.locator(".fact").count()) >= 3);
await p.screenshot({ path: `${OUT}/08-pasted.png` });

// mobile
const m = await page({ width: 390, height: 844 });
await m.goto(`${BASE}#/journey/parent`);
await m.waitForSelector(".source-text");
const overflow = await m.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1);
check("no horizontal page scroll on a phone", !overflow);
await m.screenshot({ path: `${OUT}/09-mobile-source.png` });
await m.getByRole("tab", { name: "Facts" }).click();
await m.screenshot({ path: `${OUT}/10-mobile-facts.png` });
await m.goto(BASE);
await m.waitForSelector(".hero");
await m.screenshot({ path: `${OUT}/11-mobile-home.png`, fullPage: true });

check("no console errors", errors.length === 0, errors.slice(0, 3).join(" | "));
await browser.close();
const failed = results.filter((r) => !r.ok).length;
console.log(`\n${results.length - failed}/${results.length} checks passed`);
process.exit(failed ? 1 : 0);
