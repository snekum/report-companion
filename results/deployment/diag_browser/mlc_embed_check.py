"""Diagnostic: is the 4-bit embedding / output layer what breaks the converted model?

Runs the unpacked browser (q4f16_1) weights for every layer EXCEPT the
embedding table, which stays at the merged model's fp16 values. Qwen3-1.7B
ties the embedding to the output layer, so this also restores the output
layer. bitsandbytes nf4 (where the merged model works) also leaves the
embedding unquantised. Loss on the gold answer + greedy generation on the
8 docs the browser failed.

  .venv-train/Scripts/python.exe results/deployment/diag_browser/mlc_embed_check.py
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlc_weight_check import OUT, ROOT, mlc_source  # noqa: E402  (also chdirs to ROOT)

from common import load_jsonl, messages_for  # noqa: E402
from train import build_example, completion_loss  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def main():
    final_mlc = mlc_source("model/mlc/ReportCompanion-Qwen3-1.7B-q4f16_1-MLC")
    tok = AutoTokenizer.from_pretrained("model/merged/final")
    model = AutoModelForCausalLM.from_pretrained("model/merged/final", dtype=torch.float16, device_map={"": 0})
    model.eval()
    embed = "model.embed_tokens.weight"
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n != embed:
                p.data.copy_(final_mlc(n))
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr(), "output layer not tied"
    docs = load_jsonl(f"{OUT}/diag8.jsonl")
    with torch.no_grad():
        loss = [round(completion_loss(model, build_example(tok, d), "cuda").item(), 5) for d in docs[:4]]
    print("4-bit layers + fp16 embedding/output: loss per doc", loss, "mean", round(float(np.mean(loss)), 5), flush=True)
    with open(f"{OUT}/mlc-fp16embed-python_diag8.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            p = tok.apply_chat_template(messages_for(d["text"]), tokenize=False, add_generation_prompt=True, enable_thinking=False)
            enc = tok(p, return_tensors="pt", add_special_tokens=False).to(model.device)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=1024, do_sample=False, temperature=None, top_p=None, top_k=None)
            new = g[0, enc["input_ids"].shape[1]:]
            f.write(json.dumps({"doc_id": d["doc_id"], "raw": tok.decode(new, skip_special_tokens=True), "new_tokens": len(new)}, ensure_ascii=False) + "\n")
            f.flush()
            print("generated", d["doc_id"], len(new), "tokens", flush=True)
    json.dump({"loss_first4": loss}, open(f"{OUT}/mlc_embed_check.json", "w"), indent=2)


if __name__ == "__main__":
    main()
