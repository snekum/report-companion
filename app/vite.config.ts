/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "node:path";

// base "./" keeps every asset URL relative, so the build works from any
// GitHub Pages subpath (https://user.github.io/repo/) with hash routing.
export default defineConfig({
  base: "./",
  plugins: [react()],
  build: {
    rollupOptions: {
      input: {
        main: resolve(import.meta.dirname, "index.html"),
      },
    },
  },
  server: {
    port: 5199,
    strictPort: true,
    // the extraction prompt and schema live in ../shared
    fs: { allow: [".."] },
    // Chrome's test profile lives in app/.chrome-profile and changes constantly
    watch: { ignored: ["**/.chrome-profile/**"] },
  },
  // pre-bundle WebLLM so the first model load doesn't trigger a dependency
  // re-optimisation and a full page reload mid-run
  optimizeDeps: { include: ["@mlc-ai/web-llm"] },
  test: {
    environment: "jsdom",
    include: ["src/**/*.test.ts", "src/**/*.test.tsx"],
  },
});
