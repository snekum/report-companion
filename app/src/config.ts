// Model hosting settings. The fine-tuned model URL stays null until the owner
// approves publishing the converted weights; until then the app only shows
// prepared (clearly labelled) results for it. For a local test, set
// VITE_FINETUNED_MODEL_URL when running `npm run dev`.
const env = import.meta.env as Record<string, string | undefined>;

export const MODEL_CONFIG = {
  original: {
    f16Id: "Qwen3-1.7B-q4f16_1-MLC",
    f32Id: "Qwen3-1.7B-q4f32_1-MLC",
    f16DownloadMB: 984,
    f32DownloadMB: 984,
  },
  finetuned: {
    modelId: "ReportCompanion-Qwen3-1.7B-q4f16_1-MLC",
    url: env.VITE_FINETUNED_MODEL_URL ?? null,
    modelLib: env.VITE_FINETUNED_MODEL_LIB ?? null,
    downloadMB: 0,
  },
  maxInputChars: 6000,
  maxNewTokens: 1024,
};
