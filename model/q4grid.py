"""Round Qwen3 weights exactly the way WebLLM's q4f16_1 format does.

Why: an adapter trained on the bitsandbytes nf4 base did not survive the
browser's q4f16_1 rounding (see results/deployment/diag_browser/). Training on
top of a base rounded exactly like the browser's removes that mismatch.

q4f16_1 (mlc-llm group quantisation): weights in fp16, groups of 32 along the
input dimension, scale = max|w| / 7 (fp16), code = clamp(round(w / scale + 7),
0, 14) with the division done in fp16, value = (code - 7) * scale. Every
linear layer and the embedding table (tied to the output layer in
Qwen3-1.7B) are rounded; norms stay as they are. `--verify` checks the
codes and scales against a converted model byte for byte.

  python model/q4grid.py --verify model/mlc/check-base-q4f16_1-MLC
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch

GROUP = 32
QUANTISED = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens")


def q4f16_1_codes(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Codes (0..14, int64) and fp16 scales for a [out, in] weight."""
    g = w.to(torch.float16).reshape(w.shape[0], -1, GROUP)
    scale = g.abs().amax(-1, keepdim=True) / torch.tensor(7.0, dtype=torch.float16, device=w.device)
    x = g / scale + torch.tensor(7.0, dtype=torch.float16, device=w.device)
    codes = torch.clamp(torch.round(x.float()), 0, 14).to(torch.int64)
    return codes.reshape(w.shape), scale.reshape(w.shape[0], -1)


def q4f16_1_round(w: torch.Tensor) -> torch.Tensor:
    """The weight as the browser sees it after q4f16_1 rounding, in fp16."""
    out = torch.empty(w.shape, dtype=torch.float16, device=w.device)
    for r in range(0, w.shape[0], 8192):  # row chunks keep the embedding table cheap
        codes, scale = q4f16_1_codes(w[r:r + 8192])
        out[r:r + 8192] = ((codes.to(torch.float16) - 7) * scale.repeat_interleave(GROUP, dim=1))
    return out


@torch.no_grad()
def apply_q4f16_1_grid(model) -> int:
    """Round every quantised weight of a Qwen3 model in place. Returns how many tensors were rounded."""
    n = 0
    for name, p in model.named_parameters():
        if name.endswith(".weight") and name.split(".")[-2] in QUANTISED:
            p.data.copy_(q4f16_1_round(p.data))
            n += 1
    return n


def _read_hf(d):
    """Tensor reader that avoids safetensors' mmap (it crashes on Windows under memory pressure)."""
    where = {}
    for f in glob.glob(os.path.join(d, "*.safetensors")):
        with open(f, "rb") as fh:
            n = int.from_bytes(fh.read(8), "little")
            hdr = json.loads(fh.read(n))
        where.update({k: (f, 8 + n, v) for k, v in hdr.items() if k != "__metadata__"})

    def get(name):
        f, base, v = where[name]
        a, b = v["data_offsets"]
        dt = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32}[v["dtype"]]
        arr = np.fromfile(f, dtype=dt, count=(b - a) // np.dtype(dt).itemsize, offset=base + a).reshape(v["shape"])
        if v["dtype"] == "BF16":
            return torch.from_numpy((arr.astype(np.uint32) << 16).view(np.float32)).to(torch.bfloat16)
        return torch.from_numpy(arr.copy())
    return get, where


def verify(mlc_dir: str, hf_id: str = "Qwen/Qwen3-1.7B") -> bool:
    from huggingface_hub import snapshot_download
    cache = json.load(open(os.path.join(mlc_dir, "tensor-cache.json")))
    index = {r["name"]: (os.path.join(mlc_dir, s["dataPath"]), r) for s in cache["records"] for r in s["records"]}

    def raw(name):
        path, r = index[name]
        dt = {"uint32": np.uint32, "float16": np.float16}[r["dtype"]]
        return np.fromfile(path, dtype=dt, count=int(np.prod(r["shape"])), offset=r["byteOffset"]).reshape(r["shape"])

    get, where = _read_hf(snapshot_download(hf_id, allow_patterns=["*.safetensors"]))
    fused = {"c_attn": ["q_proj", "k_proj", "v_proj"], "gate_up_proj": ["gate_proj", "up_proj"]}
    ok, checked = True, 0
    for name in index:
        if not name.endswith(".q_weight"):
            continue
        prefix = name[: -len(".q_weight")]
        head, _, leaf = prefix.rpartition(".")
        parts = [f"{head}.{p}.weight" for p in fused[leaf]] if leaf in fused else [prefix + ".weight"]
        w = torch.cat([get(p) for p in parts], 0)
        codes, scale = q4f16_1_codes(w)
        q = torch.from_numpy(raw(name).astype(np.int64))
        mlc_codes = ((q.unsqueeze(-1) >> (torch.arange(8) * 4)) & 0xF).reshape(q.shape[0], -1)
        mlc_scale = torch.from_numpy(raw(prefix + ".q_scale").copy())
        checked += 1
        # exact comparison: a float32 mean over the 311M embedding values isn't exactly 1.0 even when all match
        if not (torch.equal(codes, mlc_codes) and torch.equal(scale, mlc_scale)):
            ok = False
            print(f"MISMATCH {prefix}: {(codes != mlc_codes).sum().item()} codes, {(scale != mlc_scale).sum().item()} scales differ", flush=True)
    print(f"checked {checked} quantised tensors: {'all codes and scales identical' if ok else 'MISMATCHES above'}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", required=True, help="a converted q4f16_1 MLC folder of the unmodified base model")
    raise SystemExit(0 if verify(ap.parse_args().verify) else 1)
