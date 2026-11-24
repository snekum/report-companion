"""Run Qwen3-1.7B (optionally with a LoRA adapter) over a split.

Both the original and fine-tuned model are loaded the same way: 4-bit nf4
weights, fp16 compute, the shared system prompt, thinking disabled, greedy
decoding and the same max_new_tokens. Output rows are appended, so an
interrupted run resumes where it stopped.

  python model/generate.py --split test --tag base
  python model/generate.py --split test --tag ft --adapter model/outputs/pilot/adapter
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, ROOT, load_jsonl, messages_for  # noqa: E402

BASE_MODEL = "Qwen/Qwen3-1.7B"


def load(adapter: str | None, base_grid: str = "nf4"):
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.padding_side = "left"
    if base_grid == "nf4":
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, dtype=torch.float16, device_map={"": 0})
    else:  # fp16 weights rounded exactly like WebLLM's q4f16_1 browser format
        from q4grid import apply_q4f16_1_grid
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float16, device_map={"": 0})
        apply_q4f16_1_grid(model)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model


def prompt_text(tok, text: str) -> str:
    return tok.apply_chat_template(messages_for(text), tokenize=False, add_generation_prompt=True, enable_thinking=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--tag", required=True, help="name used in the output file, e.g. base or ft-pilot")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--base-grid", choices=["nf4", "q4f16_1"], default="nf4", help="see model/train.py")
    a = ap.parse_args()

    path = a.split if a.split.endswith(".jsonl") else os.path.join(DATA, f"{a.split}.jsonl")
    name = os.path.splitext(os.path.basename(path))[0]
    out = a.out or os.path.join(ROOT, "results", "predictions", f"{a.tag}_{name}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    docs = load_jsonl(path)[: a.limit]
    done = {json.loads(l)["doc_id"] for l in open(out, encoding="utf-8")} if os.path.exists(out) else set()
    todo = [d for d in docs if d["doc_id"] not in done]
    print(f"{len(todo)} to do ({len(done)} already in {out})")
    if not todo:
        return
    torch.manual_seed(0)
    tok, model = load(a.adapter, a.base_grid)
    prompts = {d["doc_id"]: prompt_text(tok, d["text"]) for d in todo}
    todo.sort(key=lambda d: len(prompts[d["doc_id"]]))
    torch.cuda.reset_peak_memory_stats()
    t_all = time.perf_counter()
    with open(out, "a", encoding="utf-8") as f:
        for b in range(0, len(todo), a.batch_size):
            batch = todo[b: b + a.batch_size]
            enc = tok([prompts[d["doc_id"]] for d in batch], return_tensors="pt", padding=True).to(model.device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000
            new = gen[:, enc["input_ids"].shape[1]:]
            for d, row in zip(batch, new):
                ids = [t for t in row.tolist() if t != tok.pad_token_id]
                text = tok.decode(ids, skip_special_tokens=True)
                f.write(json.dumps({
                    "doc_id": d["doc_id"], "raw": text,
                    "new_tokens": len(ids), "hit_token_limit": len(ids) >= a.max_new_tokens,
                    "latency_ms": round(dt / len(batch)), "batch_ms": round(dt), "batch_size": len(batch),
                    "prompt_tokens": int(enc["attention_mask"][0].sum()) if len(batch) == 1 else None,
                    "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20),
                    "model": BASE_MODEL, "adapter": a.adapter, "quant": "bnb-nf4-fp16" if a.base_grid == "nf4" else "q4f16_1-grid-fp16", "max_new_tokens": a.max_new_tokens,
                }, ensure_ascii=False) + "\n")
            f.flush()
            done_n = b + len(batch)
            el = time.perf_counter() - t_all
            print(f"  {done_n}/{len(todo)}  {dt/1000:.1f}s/batch  eta {el/done_n*(len(todo)-done_n)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
