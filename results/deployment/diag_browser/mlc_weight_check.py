"""Diagnostic: are the converted (q4f16_1, MLC) weights themselves good?

Unpacks MLC q4f16_1 weights back into a Hugging Face Qwen3 model on the CPU
and measures teacher-forced loss on the gold answer (the training loss) for a
few documents. Compares:
  merged fp16 (reference) vs unpacked final MLC weights, and
  original HF weights vs unpacked original-model MLC weights (unpacking check).
If the unpacked final weights' loss is close to the merged reference, the
conversion is fine and the problem is in the browser runtime; if it is far
worse, the conversion/quantisation damaged the fine-tune.

Tensors are read one at a time. Runs in fp16 on the GPU by default (3.4 GB,
the browser also computes in fp16); DEVICE=cpu uses bf16 on the CPU instead.
  .venv-train/Scripts/python.exe results/deployment/diag_browser/mlc_weight_check.py [n_docs]
"""
import glob
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "model"))
os.chdir(ROOT)
from common import load_jsonl  # noqa: E402
from train import build_example, completion_loss  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from safetensors import safe_open  # noqa: E402

H, KV, HD, I = 16, 8, 128, 6144
OUT = "results/deployment/diag_browser"


def hf_source(d):
    # Files are opened per tensor and closed again: on Windows each open safetensors
    # mapping is charged in full against the commit limit (3.4 GB per file here).
    where = {}
    for f in glob.glob(os.path.join(d, "*.safetensors")):
        with safe_open(f, "pt") as h:
            where.update({k: f for k in h.keys()})

    def get(name):
        if name == "lm_head.weight" and name not in where:
            name = "model.embed_tokens.weight"
        with safe_open(where[name], "pt") as h:
            return h.get_tensor(name).float()
    return get


def mlc_source(d):
    cache = json.load(open(os.path.join(d, "tensor-cache.json")))
    index = {r["name"]: (os.path.join(d, s["dataPath"]), r) for s in cache["records"] for r in s["records"]}

    def raw(name):
        path, r = index[name]
        dt = {"uint32": np.uint32, "float16": np.float16}[r["dtype"]]
        a = np.fromfile(path, dtype=dt, count=int(np.prod(r["shape"])), offset=r["byteOffset"]).reshape(r["shape"])
        return torch.from_numpy(a.astype(np.int64) if dt == np.uint32 else a.astype(np.float32))

    last = {}

    def deq(prefix):
        if prefix not in last:
            last.clear()
            q, s = raw(prefix + ".q_weight"), raw(prefix + ".q_scale")
            shifts = torch.arange(8, dtype=torch.int64) * 4
            out = torch.empty(q.shape[0], q.shape[1] * 8)
            for r in range(0, q.shape[0], 8192):  # in row chunks: the embedding would otherwise need ~5 GB here
                v = ((q[r:r + 8192].unsqueeze(-1) >> shifts) & 0xF).reshape(-1, q.shape[1] * 8).float()  # 8 x 4-bit per uint32, low bits first
                out[r:r + 8192] = (v - 7.0) * s[r:r + 8192].repeat_interleave(32, dim=1)
            last[prefix] = out
        return last[prefix]

    def get(name):
        if name in ("model.embed_tokens.weight", "lm_head.weight"):
            return deq("model.embed_tokens")
        p, _, leaf = name.rpartition(".")
        if leaf != "weight" or name in index:
            return raw(name)
        layer, _, proj = p.rpartition(".")
        if proj in ("q_proj", "k_proj", "v_proj"):
            w = deq(layer + ".c_attn")
            return {"q_proj": w[: H * HD], "k_proj": w[H * HD:(H + KV) * HD], "v_proj": w[(H + KV) * HD:]}[proj]
        if proj in ("gate_proj", "up_proj"):
            w = deq(layer + ".gate_up_proj")
            return w[:I] if proj == "gate_proj" else w[I:]
        return deq(p)
    return get


def rel_err(a, b):
    return ((a - b).norm() / b.norm()).item()


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    torch.set_num_threads(os.cpu_count())
    device = os.environ.get("DEVICE", "cuda")
    dtype = torch.float16 if device == "cuda" else torch.bfloat16
    base = hf_source(snapshot_download("Qwen/Qwen3-1.7B", allow_patterns=["*.safetensors", "*.json"]))
    base_mlc = mlc_source("model/mlc/check-base-q4f16_1-MLC")
    merged = hf_source("model/merged/final")
    final_mlc = mlc_source("model/mlc/ReportCompanion-Qwen3-1.7B-q4f16_1-MLC")

    model = AutoModelForCausalLM.from_pretrained("model/merged/final", dtype=dtype, device_map={"": 0 if device == "cuda" else "cpu"})
    model.eval()
    names = [n for n, _ in model.named_parameters()]

    # 1) unpacking check on the original model: should show only 4-bit rounding error
    errs = {n: rel_err(base_mlc(n), base(n)) for n in names}
    worst = max(errs.values())
    print("unpack check, relative error: median", round(float(np.median(list(errs.values()))), 4), "worst", round(worst, 4), max(errs, key=errs.get), flush=True)
    assert worst < 0.25, "unpacking is wrong; don't trust anything below"
    n0 = "model.layers.0.self_attn.q_proj.weight"
    print(f"size of the LoRA change in {n0}: {rel_err(merged(n0), base(n0)):.5f}; 4-bit rounding error there: {rel_err(final_mlc(n0), merged(n0)):.4f}", flush=True)
    # does the rounded final model keep the LoRA change? project (final_mlc - base_mlc) onto (merged - base)
    d_true = merged(n0) - base(n0)
    d_mlc = final_mlc(n0) - base_mlc(n0)
    print(f"LoRA change kept after rounding (cosine, {n0}): {torch.nn.functional.cosine_similarity(d_true.flatten(), d_mlc.flatten(), dim=0).item():.3f}", flush=True)

    tok = AutoTokenizer.from_pretrained("model/merged/final")
    diag = load_jsonl(f"{OUT}/diag8.jsonl")[:N]
    examples = [build_example(tok, d) for d in diag]
    print("docs:", [d["doc_id"] for d in diag], flush=True)
    results = {}
    # The loaded model already holds the merged weights, so the reference needs no copy.
    # (Re-reading the 3.4 GB safetensors files here crashed with an access violation
    # under memory pressure; the MLC shards are read with np.fromfile instead.)
    for label, src in [("merged fp16 (reference)", None), ("final MLC unpacked", final_mlc), ("original MLC unpacked", base_mlc)]:
        with torch.no_grad():
            if src is not None:
                for n, p in model.named_parameters():
                    p.data.copy_(src(n))
            results[label] = [round(completion_loss(model, ex, device).item(), 5) for ex in examples]
        print(f"{label:26} loss per doc {results[label]}  mean {np.mean(results[label]):.5f}", flush=True)
    # Greedy generation with the unpacked browser weights, same settings as model/generate.py,
    # on all 8 docs the browser failed. Good answers here => the weights are fine, the runtime isn't.
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.data.copy_(final_mlc(n))
    sys.path.insert(0, os.path.join(ROOT, "model"))
    from common import messages_for
    with open(f"{OUT}/mlc-unpacked-python_diag8.jsonl", "w", encoding="utf-8") as f:
        for d in load_jsonl(f"{OUT}/diag8.jsonl"):
            p = tok.apply_chat_template(messages_for(d["text"]), tokenize=False, add_generation_prompt=True, enable_thinking=False)
            enc = tok(p, return_tensors="pt", add_special_tokens=False).to(model.device)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=1024, do_sample=False, temperature=None, top_p=None, top_k=None)
            new = g[0, enc["input_ids"].shape[1]:]
            f.write(json.dumps({"doc_id": d["doc_id"], "raw": tok.decode(new, skip_special_tokens=True), "new_tokens": len(new)}, ensure_ascii=False) + "\n")
            f.flush()
            print("generated", d["doc_id"], len(new), "tokens", flush=True)
    json.dump({"docs": [d["doc_id"] for d in diag], "loss": results, "unpack_rel_err_worst": worst, "compute": f"{device} {dtype}"},
              open(f"{OUT}/mlc_weight_check.json", "w"), indent=2)


if __name__ == "__main__":
    main()
