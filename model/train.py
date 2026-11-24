"""QLoRA fine-tune of Qwen3-1.7B for document -> structured facts + evidence.

Stack: transformers + peft + bitsandbytes, with a short explicit training loop.
Why not TRL's SFTTrainer: on a 4 GB GPU the full-sequence logits
(seq_len x 151,936 vocab, upcast to fp32 for the loss) do not fit. Because
only the answer is trained (completion-only loss), this loop asks the model
for logits at the answer positions only (`logits_to_keep`), which cuts that
memory by ~3x. Everything else is standard: 4-bit nf4 base, fp16 autocast,
LoRA on all linear projections, AdamW, cosine schedule, gradient
accumulation, fixed seeds, config + per-step log saved next to the adapter.

  python model/train.py --run pilot --n-train 64 --epochs 1          # smoke test
  python model/train.py --run r1 --n-train 400 --epochs 2
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, ROOT, load_jsonl, messages_for, target_json  # noqa: E402

BASE_MODEL = "Qwen/Qwen3-1.7B"


def load_base(base_grid: str):
    """Frozen base model on GPU 0: nf4 (bitsandbytes) or fp16 rounded to WebLLM's q4f16_1 grid."""
    if base_grid == "nf4":
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        return AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, dtype=torch.float16, device_map={"": 0})
    from q4grid import apply_q4f16_1_grid
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float16, device_map={"": 0})
    n = apply_q4f16_1_grid(model)
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"base rounded to the q4f16_1 grid: {n} tensors")
    return model


def build_example(tok, doc):
    prompt = tok.apply_chat_template(messages_for(doc["text"]), tokenize=False, add_generation_prompt=True, enable_thinking=False)
    completion = target_json(doc["target"]) + "<|im_end|>"
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    c_ids = tok(completion, add_special_tokens=False)["input_ids"]
    return {"doc_id": doc["doc_id"], "input_ids": p_ids + c_ids, "n_completion": len(c_ids)}


def completion_loss(model, ex, device):
    ids = torch.tensor([ex["input_ids"]], device=device)
    n = ex["n_completion"]
    # logits for the last n+1 positions: position t predicts token t+1, so the
    # n+1 kept logits cover the n completion tokens (the last logit is unused).
    out = model(input_ids=ids, logits_to_keep=n + 1, use_cache=False)
    logits = out.logits[0, :-1, :].float()
    targets = ids[0, -n:]
    return F.cross_entropy(logits, targets)


@torch.no_grad()
def eval_loss(model, examples, device):
    model.eval()
    tot, n = 0.0, 0
    for ex in examples:
        with torch.autocast("cuda", dtype=torch.float16):
            loss = completion_loss(model, ex, device)
        tot += loss.item() * ex["n_completion"]
        n += ex["n_completion"]
    model.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--warmup-steps", type=int, default=5)
    ap.add_argument("--n-val-loss", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-steps", type=int, default=None, help="stop after this many optimizer steps (smoke tests)")
    ap.add_argument("--base-grid", choices=["nf4", "q4f16_1"], default="nf4",
                    help="frozen base under the adapter: bitsandbytes nf4 (default, as in the pilot and 'final' runs) or "
                         "fp16 weights rounded exactly like WebLLM's q4f16_1 browser format (model/q4grid.py)")
    a = ap.parse_args()

    out_dir = os.path.join(ROOT, "model", "outputs", a.run)
    os.makedirs(out_dir, exist_ok=True)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = "cuda"

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    train_docs = load_jsonl(os.path.join(DATA, "train.jsonl"))
    # deterministic subset: shuffle with the seed, then take the first n
    random.Random(a.seed).shuffle(train_docs)
    train_docs = train_docs[: a.n_train]
    val_docs = load_jsonl(os.path.join(DATA, "val.jsonl"))[: a.n_val_loss]
    train = [build_example(tok, d) for d in train_docs]
    too_long = [e["doc_id"] for e in train if len(e["input_ids"]) > a.max_len]
    train = [e for e in train if len(e["input_ids"]) <= a.max_len]
    val = [build_example(tok, d) for d in val_docs]
    lens = sorted(len(e["input_ids"]) for e in train)
    print(f"train examples {len(train)} (dropped {len(too_long)} over {a.max_len} tokens); "
          f"length median {lens[len(lens)//2]} max {lens[-1]}; completion tokens mean {sum(e['n_completion'] for e in train)/len(train):.0f}")

    model = load_base(a.base_grid)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lcfg = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg)
    for p in model.parameters():  # LoRA weights in fp32 so the fp16 grad scaler can unscale them
        if p.requires_grad:
            p.data = p.data.float()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params {trainable:,} of {total:,}")

    steps_per_epoch = math.ceil(len(train) / a.grad_accum)
    total_steps = math.ceil(steps_per_epoch * a.epochs)
    if a.max_steps:
        total_steps = min(total_steps, a.max_steps)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, a.warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda")

    quant = ("bnb nf4, double quant, fp16 compute" if a.base_grid == "nf4" else
             "fp16 weights rounded to the q4f16_1 grid (model/q4grid.py), frozen; fp16 compute")
    config = {**vars(a), "base_model": BASE_MODEL, "quant": quant, "target_modules": lcfg.target_modules and sorted(lcfg.target_modules),
              "train_examples": len(train), "dropped_too_long": too_long, "total_steps": total_steps, "effective_batch": a.grad_accum,
              "loss": "cross-entropy on completion tokens only (JSON answer + <|im_end|>)", "trainable_params": trainable,
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    with open(os.path.join(out_dir, "train_config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)

    log = open(os.path.join(out_dir, "train_log.jsonl"), "w")
    v0 = eval_loss(model, val, device)
    print(f"val loss before training: {v0:.4f}")
    log.write(json.dumps({"step": 0, "val_loss": v0}) + "\n")
    torch.cuda.reset_peak_memory_stats()
    model.train()
    step, micro, t0 = 0, 0, time.perf_counter()
    tokens_seen, running = 0, []
    order_rng = random.Random(a.seed + 1)
    done = False
    epoch = 0
    while not done:
        order = list(range(len(train)))
        order_rng.shuffle(order)
        for idx in order:
            ex = train[idx]
            with torch.autocast("cuda", dtype=torch.float16):
                loss = completion_loss(model, ex, device)
            scaler.scale(loss / a.grad_accum).backward()
            running.append(loss.item())
            tokens_seen += len(ex["input_ids"])
            micro += 1
            if micro % a.grad_accum == 0:
                scaler.unscale_(opt)
                gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0).item()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                el = time.perf_counter() - t0
                rec = {"step": step, "epoch": round(micro / len(train), 3), "loss": round(sum(running) / len(running), 4), "grad_norm": round(gn, 3),
                       "lr": sched.get_last_lr()[0], "elapsed_s": round(el), "examples_per_s": round(micro / el, 3), "tokens_per_s": round(tokens_seen / el),
                       "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20)}
                running = []
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(f"step {step}/{total_steps} loss {rec['loss']} gn {rec['grad_norm']} {rec['examples_per_s']} ex/s peak {rec['peak_mem_mb']}MB "
                      f"eta {(total_steps - step) * el / step / 60:.1f} min", flush=True)
                if step % steps_per_epoch == 0 or step == total_steps:
                    vl = eval_loss(model, val, device)
                    print(f"  val loss {vl:.4f}")
                    log.write(json.dumps({"step": step, "val_loss": vl}) + "\n")
                    log.flush()
                if step >= total_steps:
                    done = True
                    break
        epoch += 1
    adapter_dir = os.path.join(out_dir, "adapter")
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    summary = {"steps": step, "elapsed_min": round((time.perf_counter() - t0) / 60, 1), "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20),
               "examples_per_s": round(micro / (time.perf_counter() - t0), 3), "val_loss_before": v0}
    with open(os.path.join(out_dir, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(summary)


if __name__ == "__main__":
    main()
