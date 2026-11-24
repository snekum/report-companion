// On-device inference through WebLLM. Nothing here talks to a server except
// to download model files; the document text stays inside this browser tab.
import type { MLCEngineInterface, AppConfig, ModelRecord, InitProgressReport } from "@mlc-ai/web-llm";
import PROMPT from "../../../shared/extraction_prompt.txt?raw";
import { MODEL_CONFIG } from "../config";

export const EXTRACTION_PROMPT = PROMPT.trim();

export type ModelKey = "original" | "finetuned";

export interface BrowserCapability {
  webgpu: boolean;
  shaderF16: boolean;
  adapterDescription: string | null;
  problem: string | null;
}

export async function checkBrowser(): Promise<BrowserCapability> {
  const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
  if (!gpu) {
    return {
      webgpu: false,
      shaderF16: false,
      adapterDescription: null,
      problem: "This browser does not expose WebGPU. Recent Chrome or Edge on a laptop or desktop usually does.",
    };
  }
  try {
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) {
      return { webgpu: false, shaderF16: false, adapterDescription: null, problem: "WebGPU is present but no graphics adapter was available." };
    }
    const info = (adapter as GPUAdapter & { info?: GPUAdapterInfo }).info;
    const description = info ? [info.vendor, info.architecture, info.description].filter(Boolean).join(" ") : null;
    return { webgpu: true, shaderF16: adapter.features.has("shader-f16"), adapterDescription: description || null, problem: null };
  } catch (err) {
    return { webgpu: false, shaderF16: false, adapterDescription: null, problem: `WebGPU check failed: ${String(err)}` };
  }
}

export interface ModelOption {
  key: ModelKey;
  label: string;
  modelId: string;
  downloadMB: number;
  available: boolean;
  unavailableReason: string | null;
}

/** Which model files to use for each option, given what the browser supports. */
export function modelOptions(cap: BrowserCapability): ModelOption[] {
  const f16 = cap.shaderF16;
  const original: ModelOption = {
    key: "original",
    label: "Original Qwen3-1.7B",
    modelId: f16 ? MODEL_CONFIG.original.f16Id : MODEL_CONFIG.original.f32Id,
    downloadMB: f16 ? MODEL_CONFIG.original.f16DownloadMB : MODEL_CONFIG.original.f32DownloadMB,
    available: cap.webgpu,
    unavailableReason: cap.webgpu ? null : cap.problem,
  };
  const ft = MODEL_CONFIG.finetuned;
  let ftReason: string | null = null;
  if (!cap.webgpu) ftReason = cap.problem;
  else if (!ft.url) ftReason = "The fine-tuned model files are not hosted yet, so only prepared results are shown for it.";
  else if (!f16) ftReason = "The fine-tuned model is only converted for GPUs with 16-bit shader support (shader-f16).";
  const finetuned: ModelOption = {
    key: "finetuned",
    label: "Fine-tuned Qwen3-1.7B (Report Companion)",
    modelId: ft.modelId,
    downloadMB: ft.downloadMB,
    available: ftReason === null,
    unavailableReason: ftReason,
  };
  return [original, finetuned];
}

function appConfig(prebuilt: ModelRecord[]): AppConfig {
  const records = [...prebuilt];
  const ft = MODEL_CONFIG.finetuned;
  if (ft.url) {
    const base = prebuilt.find((r) => r.model_id === MODEL_CONFIG.original.f16Id);
    records.push({
      model: ft.url,
      model_id: ft.modelId,
      // Same architecture and quantisation as the prebuilt Qwen3-1.7B q4f16_1,
      // so the compiled WebGPU library is shared; only the weights differ.
      model_lib: ft.modelLib ?? base?.model_lib ?? "",
      required_features: ["shader-f16"],
      overrides: { context_window_size: 4096 },
    });
  }
  return { model_list: records };
}

let engine: MLCEngineInterface | null = null;
let loadedId: string | null = null;

export async function loadModel(option: ModelOption, onProgress: (r: InitProgressReport) => void): Promise<void> {
  if (engine && loadedId === option.modelId) return;
  const webllm = await import("@mlc-ai/web-llm");
  if (engine) {
    await engine.unload();
    engine = null;
    loadedId = null;
  }
  engine = await webllm.CreateMLCEngine(option.modelId, {
    appConfig: appConfig(webllm.prebuiltAppConfig.model_list),
    initProgressCallback: onProgress,
  });
  loadedId = option.modelId;
}

export interface RawExtraction {
  raw: string;
  latencyMs: number;
  promptTokens: number | null;
  completionTokens: number | null;
}

/** Same prompt, greedy decoding and token cap as the Python evaluation. */
export async function runExtraction(documentText: string, maxTokens = MODEL_CONFIG.maxNewTokens): Promise<RawExtraction> {
  if (!engine) throw new Error("No model is loaded.");
  if (documentText.length > MODEL_CONFIG.maxInputChars) {
    throw new Error(`Documents are limited to ${MODEL_CONFIG.maxInputChars.toLocaleString()} characters in this prototype.`);
  }
  await engine.resetChat();
  const started = performance.now();
  const reply = await engine.chat.completions.create({
    messages: [
      { role: "system", content: EXTRACTION_PROMPT },
      { role: "user", content: documentText },
    ],
    temperature: 0,
    max_tokens: maxTokens,
    extra_body: { enable_thinking: false },
  });
  const latencyMs = performance.now() - started;
  return {
    raw: reply.choices[0]?.message?.content ?? "",
    latencyMs,
    promptTokens: reply.usage?.prompt_tokens ?? null,
    completionTokens: reply.usage?.completion_tokens ?? null,
  };
}

export async function unloadModel(): Promise<void> {
  if (engine) await engine.unload();
  engine = null;
  loadedId = null;
}
