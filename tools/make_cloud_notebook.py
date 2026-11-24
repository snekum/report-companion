"""Writes model/cloud_train_eval.ipynb.

  python -I tools/make_cloud_notebook.py model/cloud_train_eval.ipynb

Run settings live in the "Settings" cell below (RUN, RUN_ARGS, GRID).
"""
import json
import sys

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md(r"""
# Report Companion: LoRA run on the browser-rounded base + evaluation on a free Kaggle GPU

Runs the same `model/train.py`, `model/generate.py` and `model/evaluate.py` used locally, on a bigger GPU.

**Accelerator:** pick **GPU T4 x2** (free). With two GPUs the original model's predictions run on the second T4 while training runs on the first. With one GPU (Colab, or Kaggle "T4 x1") everything still works, just one job after another.

**Cost:** $0 on the Kaggle free tier (it uses part of the weekly 30 GPU-hour quota). Nothing here creates paid resources.

**Maximum runtime:** Kaggle stops a run after 12 h. Expect well under that; the training cell prints an ETA after every step.

**How to run on Kaggle:** turn on *Settings > Internet* (the notebook downloads `Qwen/Qwen3-1.7B` from Hugging Face), then *Save Version > Save & Run All (Commit)* so it keeps running with the browser closed. When it finishes, download `rc/outputs.zip` from the version's *Output* tab and unzip it at the repository root.

**Data:** the bundle holds only synthetic (Synthea-derived) and hand-written fictional documents. Nothing is uploaded or published anywhere.

**Test set rule:** this notebook *generates* test and challenge predictions but only *scores* val40. Score the test set locally, and only after the val40 check passes.
""")

code(r"""
!nvidia-smi
import torch, sys
N_GPU = torch.cuda.device_count()
print(sys.version, torch.__version__, "GPUs:", N_GPU, [torch.cuda.get_device_name(i) for i in range(N_GPU)])
assert N_GPU >= 1, "No GPU: set Accelerator to GPU T4 x2 in the notebook settings"
""")

code(r"""
# Find the bundle. Kaggle unzips uploaded .zip files into folders, so accept either form.
import glob, os, shutil, zipfile
WORK = "/kaggle/working/rc" if os.path.exists("/kaggle") else "/content/rc"
os.makedirs(WORK, exist_ok=True)
zips = glob.glob("/kaggle/input/**/report-companion-cloud.zip", recursive=True)
dirs = [os.path.dirname(os.path.dirname(p)) for p in glob.glob("/kaggle/input/**/model/train.py", recursive=True)]
if zips:
    zipfile.ZipFile(zips[0]).extractall(WORK)
elif dirs:
    shutil.copytree(dirs[0], WORK, dirs_exist_ok=True)
else:
    from google.colab import files  # Colab: pick dist/report-companion-cloud.zip
    up = files.upload()
    zipfile.ZipFile(os.path.abspath(next(iter(up)))).extractall(WORK)
os.chdir(WORK)
for f in ["model/train.py", "model/generate.py", "model/evaluate.py", "data/processed/train.jsonl", "data/processed/val40.jsonl",
          "data/processed/test.jsonl", "data/processed/challenge.jsonl", "data/processed/demo_journeys.jsonl"]:
    assert os.path.exists(f), f"missing {f}: rebuild the bundle with tools/make_cloud_bundle.sh"
print(os.getcwd(), sorted(os.listdir(".")))
""")

code(r"""
# Pinned libraries (torch comes preinstalled; its version is printed above).
!pip -q install -r model/requirements.txt
import transformers, peft, bitsandbytes
print(transformers.__version__, peft.__version__, bitsandbytes.__version__)
# Download the base model once, so two jobs don't download it at the same time.
from huggingface_hub import snapshot_download
print(snapshot_download("Qwen/Qwen3-1.7B"))
""")

md(r"""
## Settings
Run `q4grid`: same data and LoRA settings as the `final` run (951 documents, 1 epoch, rank 16, alpha 32, lr 2e-4, effective batch 8, cosine schedule, seed 1234, loss on the answer only). The one change: the frozen base is rounded exactly like WebLLM's q4f16_1 browser format (`model/q4grid.py`) instead of bitsandbytes nf4, because the `final` adapter did not survive the browser's rounding. All predictions below use that same rounded base, for the original model too, so the Python numbers are what the browser should see.
""")

code(r"""
import subprocess
RUN = "q4grid"
GRID = "--base-grid q4f16_1"
RUN_ARGS = f"--n-train 951 --epochs 1 {GRID}"
BS = 16
LOGS = f"model/outputs/{RUN}/cloud_logs"
os.makedirs(LOGS, exist_ok=True)

# Original model on the same rounded base (what the browser's original model computes).
BASE_JOBS = " && ".join([
    f"python model/generate.py --split val40 --tag base-q4grid {GRID} --batch-size {BS}",
    f"python model/generate.py --split challenge --tag base-q4grid {GRID} --batch-size {BS}",
    f"python model/generate.py --split test --tag base-q4grid {GRID} --batch-size {BS}",
])
base_proc = None
if N_GPU >= 2:
    base_proc = subprocess.Popen(["bash", "-c", BASE_JOBS], env={**os.environ, "CUDA_VISIBLE_DEVICES": "1"},
                                 stdout=open(f"{LOGS}/base_generate.log", "w"), stderr=subprocess.STDOUT)
    print("original model running on GPU 1 in the background, log:", f"{LOGS}/base_generate.log")
else:
    print("one GPU: the original model runs after the fine-tuned model")
""")

md(r"""
## 1. Training (GPU 0)
""")

code(r"""
!CUDA_VISIBLE_DEVICES=0 python model/train.py --run {RUN} {RUN_ARGS} 2>&1 | tee {LOGS}/train.log
assert os.path.exists(f"model/outputs/{RUN}/adapter/adapter_config.json"), "training failed: see the log above"
print(open(f"model/outputs/{RUN}/train_summary.json").read())
""")

md(r"""
## 2. Fine-tuned model predictions (GPU 0)
Rounded base + adapter (unmerged), same prompt, thinking off, greedy decoding, max 1024 new tokens.
""")

code(r"""
FT = f"--tag ft-{RUN} --adapter model/outputs/{RUN}/adapter {GRID}"
for split in ["val40", "data/processed/demo_journeys.jsonl", "challenge", "test"]:
    !CUDA_VISIBLE_DEVICES=0 python model/generate.py --split {split} {FT} --batch-size {BS} 2>&1 | tail -3
""")

md(r"""
## 3. Original model predictions
Waits for the background job (two GPUs) or runs it now (one GPU).
""")

code(r"""
if base_proc is not None:
    print("waiting for the original model on GPU 1 ...")
    rc = base_proc.wait()
else:
    rc = subprocess.run(["bash", "-c", BASE_JOBS], env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
                        stdout=open(f"{LOGS}/base_generate.log", "w"), stderr=subprocess.STDOUT).returncode
print("exit code", rc)
!grep -E "to do|Error|Traceback" {LOGS}/base_generate.log
!wc -l results/predictions/*.jsonl
""")

md(r"""
## 4. Val40 check (the go/no-go)
Both models here run on the browser-rounded base. Pass if, compared with the original on the same 40 documents, the fine-tuned model: (a) clearly improves format, document date and follow-up; (b) keeps lab F1 at or above ~0.95; (c) doesn't add unsupported facts. Test and challenge are **not** scored here.
""")

code(r"""
!python model/evaluate.py --pred results/predictions/base-q4grid_val40.jsonl --split val40
!python model/evaluate.py --pred results/predictions/ft-{RUN}_val40.jsonl --split val40
""")

code(r"""
# Package results. Unzip at the repository root.
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
with open(f"model/outputs/{RUN}/cloud_gpu.txt", "w") as f:
    f.write(gpu + "\n")
with zipfile.ZipFile("outputs.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for root in ["model/outputs", "results/predictions"]:
        for dirpath, _, fnames in os.walk(root):
            for fn in fnames:
                if "checkpoint-" not in dirpath:
                    z.write(os.path.join(dirpath, fn))
    print("\n".join(sorted(z.namelist())))
print(os.path.getsize("outputs.zip") / 1e6, "MB")
if not os.path.exists("/kaggle"):
    from google.colab import files
    files.download("outputs.zip")
else:
    print("Kaggle: download rc/outputs.zip from the Output tab of this version")
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"c{i:02d}"
with open(sys.argv[1], "w", encoding="utf-8", newline="\n") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")
print("wrote", sys.argv[1], len(cells), "cells")
