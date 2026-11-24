"""Merge a trained LoRA adapter into full-precision Qwen3-1.7B weights (CPU).

The adapter was trained on the 4-bit (nf4) base. For export it is merged into
the original fp16 weights, which is the standard route but not bit-identical
to the training setup. That is why the browser model is evaluated again after
conversion instead of assuming its results.

  python model/merge_adapter.py --adapter model/outputs/final/adapter --out model/merged/final
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen3-1.7B"


def write_base_config(out: str):
    """Use the original Qwen config.json (dtype set to float16) instead of the one
    transformers 5 writes. transformers 5 moves rope_theta into rope_parameters,
    which mlc-llm 0.26.dev6's Qwen3 loader can't read. Merging LoRA doesn't change
    the architecture, so the original config describes the merged weights exactly."""
    from huggingface_hub import hf_hub_download
    cfg = json.load(open(hf_hub_download(BASE_MODEL, "config.json")))
    cfg["torch_dtype"] = "float16"
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


def write_index(out: str):
    """MLC's converter wants model.safetensors.index.json and tries to create it
    next to the weights if missing, which fails on the read-only mount in
    convert_mlc.sh. A single-file save has no index, so write one."""
    path = os.path.join(out, "model.safetensors")
    with open(path, "rb") as f:  # safetensors layout: 8-byte header length, JSON header, tensor bytes
        n = int.from_bytes(f.read(8), "little")
        keys = [k for k in json.loads(f.read(n)) if k != "__metadata__"]
    total = os.path.getsize(path) - 8 - n
    with open(os.path.join(out, "model.safetensors.index.json"), "w") as fh:
        json.dump({"metadata": {"total_size": total}, "weight_map": {k: "model.safetensors" for k in keys}}, fh, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float16, device_map={"": "cpu"})
    model = PeftModel.from_pretrained(model, a.adapter)
    model = model.merge_and_unload()
    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(BASE_MODEL).save_pretrained(a.out)
    write_base_config(a.out)
    write_index(a.out)
    with open(os.path.join(a.out, "merge_info.json"), "w") as f:
        json.dump({"base_model": BASE_MODEL, "adapter": a.adapter, "dtype": "float16", "note": "LoRA trained on nf4 base, merged into fp16 base"}, f, indent=2)
    print("merged ->", a.out)


if __name__ == "__main__":
    main()
