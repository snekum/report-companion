// Model hosting settings. The fine-tuned model URL comes from
// VITE_FINETUNED_MODEL_URL: the Pages build sets it from the repository variable
// FINETUNED_MODEL_URL (the Hugging Face repo). Without it the app only shows
// prepared (clearly labelled) results for the fine-tuned model. For a local
// test, set it when running `npm run dev`.
const env = import.meta.env as Record<string, string | undefined>;

export const MODEL_CONFIG = {
  original: {
    f16Id: "Qwen3-1.7B-q4f16_1-MLC",
    f32Id: "Qwen3-1.7B-q4f32_1-MLC",
    f16DownloadMB: 984,
    f32DownloadMB: 984,
  },
  finetuned: {
    modelId: "ReportCompanion-q4grid-Qwen3-1.7B-q4f16_1-MLC",
    url: env.VITE_FINETUNED_MODEL_URL || null,
    modelLib: env.VITE_FINETUNED_MODEL_LIB || null,
    downloadMB: 984,
  },
  maxInputChars: 6000,
  maxNewTokens: 1024,
};
