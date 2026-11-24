import { useEffect, useState } from "react";
import { checkBrowser, loadModel, modelOptions, runExtraction, type BrowserCapability, type ModelOption } from "../lib/llm";
import type { LiveOutput } from "../lib/data";
import { MODEL_CONFIG } from "../config";

type Phase = "idle" | "checking" | "loading" | "running" | "done" | "error";

export function LivePanel({ text, onResult, onClose }: { text: string; onResult: (o: LiveOutput) => void; onClose: () => void }) {
  const [cap, setCap] = useState<BrowserCapability | null>(null);
  const [phase, setPhase] = useState<Phase>("checking");
  const [choice, setChoice] = useState<ModelOption["key"]>("finetuned");
  const [progress, setProgress] = useState<{ p: number; text: string } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    checkBrowser().then((c) => { setCap(c); setPhase("idle"); });
  }, []);

  const options = cap ? modelOptions(cap) : [];
  const selected = options.find((o) => o.key === choice);
  useEffect(() => {
    if (selected && !selected.available) {
      const other = options.find((o) => o.available);
      if (other) setChoice(other.key);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cap]);

  const tooLong = text.length > MODEL_CONFIG.maxInputChars;

  const run = async () => {
    if (!selected?.available || phase === "loading" || phase === "running") return;
    setError(null);
    try {
      setPhase("loading");
      await loadModel(selected, (r) => setProgress({ p: r.progress, text: r.text }));
      setPhase("running");
      const r = await runExtraction(text);
      onResult({ key: selected.key, raw: r.raw, latencyMs: Math.round(r.latencyMs), ranAt: new Date().toISOString(), modelId: selected.modelId });
      setPhase("done");
    } catch (e) {
      // No automatic retry: the visitor decides whether to try again.
      setError(String(e instanceof Error ? e.message : e));
      setPhase("error");
    }
  };

  return (
    <div className="card pad stack" style={{ marginTop: 12 }}>
      <div className="row"><h3 style={{ margin: 0 }}>Run a model on this device</h3><span className="spacer" /><button className="btn ghost small" onClick={onClose}>Close</button></div>
      <div className="notice info small">
        <strong>Where this runs:</strong> entirely inside this browser tab, using your computer's graphics chip (WebGPU).
        The document text is not sent anywhere. The only network traffic is downloading the model files
        (about {Math.round((selected?.downloadMB || 984) / 10) * 10} MB the first time; your browser keeps them for next time).
      </div>
      {phase === "checking" && <p className="small"><span className="spinner" /> Checking what this browser supports…</p>}
      {cap && !cap.webgpu && (
        <div className="notice warn small">
          <strong>This browser can't run the model.</strong> {cap.problem} You can still explore the prepared examples and use the rule-based extraction, which needs no download.
        </div>
      )}
      {cap?.webgpu && (
        <>
          <p className="tiny muted">Graphics adapter: {cap.adapterDescription ?? "unknown"}{cap.shaderF16 ? " · supports 16-bit shaders" : " · no 16-bit shader support (slower 32-bit version used)"}.
            On laptops with two graphics chips the browser may pick the slower built-in one.</p>
          <fieldset style={{ border: "none", padding: 0, margin: 0 }}>
            <legend className="small" style={{ marginBottom: 4 }}>Model</legend>
            {options.map((o) => (
              <label key={o.key} className="row small" style={{ alignItems: "flex-start", marginBottom: 6, opacity: o.available ? 1 : 0.6 }}>
                <input type="radio" name="model" disabled={!o.available} checked={choice === o.key} onChange={() => setChoice(o.key)} />
                <span>{o.label}{o.available ? "" : <span className="muted"> · {o.unavailableReason}</span>}</span>
              </label>
            ))}
          </fieldset>
          {tooLong && <div className="notice warn small">This document is longer than the {MODEL_CONFIG.maxInputChars.toLocaleString()} characters this prototype accepts. Try a shorter part of it.</div>}
          <div className="row">
            <button className="btn primary" onClick={run} disabled={!selected?.available || tooLong || phase === "loading" || phase === "running"}>
              {phase === "loading" ? "Loading model…" : phase === "running" ? "Reading the document…" : "Extract facts on this device"}
            </button>
          </div>
          {phase === "loading" && progress && (
            <div className="stack">
              <div className="progress" aria-label="Model download progress"><div style={{ width: `${Math.round(progress.p * 100)}%` }} /></div>
              <p className="tiny muted">{progress.text}</p>
            </div>
          )}
          {phase === "running" && <p className="small"><span className="spinner" /> Reading the document. On a laptop GPU this usually takes 15 to 60 seconds.</p>}
          {phase === "done" && <div className="notice small">Done. The result is now selected in the Facts tab and labelled "run here".</div>}
          {error && <div className="notice bad small"><strong>That didn't work.</strong> {error}</div>}
        </>
      )}
    </div>
  );
}
