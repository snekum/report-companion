"""Diagnostic: is the adapter tied to the rounding grid it was trained on?

  ref  : adapter on the nf4 base (the training setup)            -> loss only
  q4   : adapter on the q4f16_1-rounded base (unpacked from the
         converted ORIGINAL model), unmerged                      -> loss + greedy generation
If q4 fails like the browser while ref is near-perfect, the adapter learned
the nf4 grid's rounding errors along with the task, and retraining on the
q4f16_1 grid is the fix.

  .venv-train/Scripts/python.exe results/deployment/diag_browser/grid_mismatch_check.py ref|q4
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlc_weight_check import OUT, mlc_source  # noqa: E402  (also chdirs to the repo root)

from common import load_jsonl, messages_for  # noqa: E402
from train import build_example, completion_loss  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402

ADAPTER = "model/outputs/final/adapter"


def main():
    mode = sys.argv[1]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    if mode == "ref":
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", quantization_config=bnb, dtype=torch.float16, device_map={"": 0})
    else:
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.float16, device_map={"": 0})
        src = mlc_source("model/mlc/check-base-q4f16_1-MLC")
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.data.copy_(src(n))
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()
    docs = load_jsonl(f"{OUT}/diag8.jsonl")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        loss = [round(completion_loss(model, build_example(tok, d), "cuda").item(), 5) for d in docs[:4]]
    print(f"{mode}: adapter loss per doc {loss} mean {np.mean(loss):.5f}", flush=True)
    if mode == "q4":
        with open(f"{OUT}/q4base-adapter-python_diag8.jsonl", "w", encoding="utf-8") as f:
            for d in docs:
                p = tok.apply_chat_template(messages_for(d["text"]), tokenize=False, add_generation_prompt=True, enable_thinking=False)
                enc = tok(p, return_tensors="pt", add_special_tokens=False).to("cuda")
                with torch.no_grad():
                    g = model.generate(**enc, max_new_tokens=1024, do_sample=False, temperature=None, top_p=None, top_k=None)
                new = g[0, enc["input_ids"].shape[1]:]
                f.write(json.dumps({"doc_id": d["doc_id"], "raw": tok.decode(new, skip_special_tokens=True), "new_tokens": len(new)}, ensure_ascii=False) + "\n")
                f.flush()
                print("generated", d["doc_id"], len(new), "tokens", flush=True)
    json.dump({"mode": mode, "loss_first4": loss}, open(f"{OUT}/grid_mismatch_{mode}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
