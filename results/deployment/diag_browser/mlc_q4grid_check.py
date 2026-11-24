"""Check the q4grid browser build before testing it in Chrome.

The `final` adapter broke in the browser because it was trained on an nf4 base
and the browser rounds weights to the q4f16_1 grid instead. The `q4grid`
adapter was trained on a base rounded exactly like the browser (model/q4grid.py).
This script unpacks the converted q4grid browser weights back into PyTorch and,
on the same 8 documents the broken browser model got wrong:
  1. measures teacher-forced loss (the training loss) for
       - the training setup: rounded base + LoRA adapter (PEFT, unmerged),
       - the merged fp16 model (input to the conversion),
       - the unpacked browser weights (merged model re-rounded by the converter);
  2. checks how much of the LoRA change survives the converter's rounding
     (cosine between the browser-weight change and the true change, one layer);
  3. generates greedily with the unpacked browser weights, same settings as
     model/generate.py -> q4grid-mlc-unpacked-python_diag8.jsonl.

  .venv-train/Scripts/python.exe results/deployment/diag_browser/mlc_q4grid_check.py
"""
import gc
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "model"))
sys.path.insert(0, HERE)
os.chdir(ROOT)
from common import load_jsonl, messages_for  # noqa: E402
from train import build_example, completion_loss  # noqa: E402
from q4grid import _read_hf  # noqa: E402
from mlc_weight_check import mlc_source  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

OUT = "results/deployment/diag_browser"
ADAPTER = "model/outputs/q4grid/adapter"
MERGED = "model/merged/q4grid"
MLC = "model/mlc/ReportCompanion-q4grid-Qwen3-1.7B-q4f16_1-MLC"
BASE_MLC = "model/mlc/check-base-q4f16_1-MLC"


def losses(model, examples):
    with torch.no_grad():
        return [round(completion_loss(model, ex, "cuda").item(), 5) for ex in examples]


def main():
    tok = AutoTokenizer.from_pretrained(MERGED)
    diag = load_jsonl(f"{OUT}/diag8.jsonl")
    examples = [build_example(tok, d) for d in diag]
    results = {}

    # 1a) training setup: rounded base + adapter
    from generate import load
    _, model = load(ADAPTER, "q4f16_1")
    results["training setup (rounded base + adapter)"] = losses(model, examples)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # 1b) merged fp16, then the unpacked browser weights copied into the same model
    model = AutoModelForCausalLM.from_pretrained(MERGED, dtype=torch.float16, device_map={"": 0})
    model.eval()
    results["merged fp16"] = losses(model, examples)

    # 2) does the converter's rounding keep the LoRA change? (one tensor)
    n0 = "model.layers.0.self_attn.q_proj.weight"
    base, _ = _read_hf(snapshot_download("Qwen/Qwen3-1.7B", allow_patterns=["*.safetensors", "*.json"]))
    mlc, base_mlc = mlc_source(MLC), mlc_source(BASE_MLC)
    d_true = model.get_parameter(n0).float().cpu() - base(n0).float()
    d_mlc = mlc(n0) - base_mlc(n0)
    cos = torch.nn.functional.cosine_similarity(d_true.flatten(), d_mlc.flatten(), dim=0).item()
    changed = (d_mlc != 0).float().mean().item()
    print(f"{n0}: LoRA change kept after rounding, cosine {cos:.3f}; weights that moved to another grid point {changed:.1%}", flush=True)

    with torch.no_grad():
        for n, p in model.named_parameters():
            p.data.copy_(mlc(n))
    results["browser weights unpacked"] = losses(model, examples)
    for k, v in results.items():
        print(f"{k:42} mean loss {np.mean(v):.5f}  per doc {v}", flush=True)

    # 3) generate with the unpacked browser weights
    with open(f"{OUT}/q4grid-mlc-unpacked-python_diag8.jsonl", "w", encoding="utf-8") as f:
        for d in diag:
            p = tok.apply_chat_template(messages_for(d["text"]), tokenize=False, add_generation_prompt=True, enable_thinking=False)
            enc = tok(p, return_tensors="pt", add_special_tokens=False).to(model.device)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=1024, do_sample=False, temperature=None, top_p=None, top_k=None)
            new = g[0, enc["input_ids"].shape[1]:]
            f.write(json.dumps({"doc_id": d["doc_id"], "raw": tok.decode(new, skip_special_tokens=True), "new_tokens": len(new)}, ensure_ascii=False) + "\n")
            f.flush()
            print("generated", d["doc_id"], len(new), "tokens", flush=True)
    json.dump({"docs": [d["doc_id"] for d in diag], "loss": results, "mean_loss": {k: float(np.mean(v)) for k, v in results.items()},
               "lora_change_cosine_" + n0: cos, "share_of_weights_on_new_grid_point_" + n0: changed, "compute": "cuda float16"},
              open(f"{OUT}/mlc_q4grid_check.json", "w"), indent=2)


if __name__ == "__main__":
    main()
