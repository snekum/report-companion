#!/usr/bin/env bash
# Convert Hugging Face weights to WebLLM's q4f16_1 format inside a Linux
# container (the Windows MLC wheels are missing SPIRV-Tools.dll).
#
#   bash model/convert_mlc.sh <hf_model_dir_or_id> <output_name>
#   e.g. bash model/convert_mlc.sh model/merged/final ReportCompanion-Qwen3-1.7B-q4f16_1-MLC
#        bash model/convert_mlc.sh Qwen/Qwen3-1.7B check-base-q4f16_1-MLC   (toolchain check)
#
# Only the weights are converted. mlc-chat-config.json and tokenizer files are
# copied from the official prebuilt mlc-ai/Qwen3-1.7B-q4f16_1-MLC so they match
# the prebuilt WebGPU library (Qwen3-1.7B-q4f16_1_cs1k-webgpu.wasm) exactly.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$1"; NAME="$2"
OUT="$ROOT/model/mlc/$NAME"
mkdir -p "$OUT"
docker image inspect rc-mlc:2026-11-23 >/dev/null 2>&1 || docker build -t rc-mlc:2026-11-23 -f "$ROOT/model/Dockerfile.mlc" "$ROOT/model"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
if [ -d "$ROOT/$SRC" ]; then MOUNT_SRC="-v $(cygpath -m "$ROOT/$SRC" 2>/dev/null || echo "$ROOT/$SRC"):/src:ro"; SRC_IN=/src; else MOUNT_SRC=""; SRC_IN="$SRC"; fi
export MSYS_NO_PATHCONV=1
docker run --rm \
  -v "$(cygpath -m "$OUT" 2>/dev/null || echo "$OUT"):/out" \
  -v "$(cygpath -m "$HF_CACHE" 2>/dev/null || echo "$HF_CACHE"):/hf" \
  $MOUNT_SRC -e HF_HOME=/hf \
  rc-mlc:2026-11-23 bash -c "
    set -e
    pip freeze | grep -iE 'mlc|tvm|torch|numpy' > /out/convert_env.txt
    grep -qx 'apache-tvm-ffi==0.1.13' /out/convert_env.txt || { echo 'image has the wrong apache-tvm-ffi (0.1.16 segfaults): docker build -t rc-mlc:2026-11-23 -f model/Dockerfile.mlc model'; exit 1; }
    SRC_DIR=\"$SRC_IN\"
    if [ ! -d \"\$SRC_DIR\" ]; then SRC_DIR=\$(python -c 'from huggingface_hub import snapshot_download as s; print(s(\"$SRC_IN\"))'); fi
    python -m mlc_llm convert_weight \"\$SRC_DIR\" --quantization q4f16_1 --device cpu -o /out
  "
# config + tokenizer from the prebuilt model
for f in mlc-chat-config.json tokenizer.json tokenizer_config.json vocab.json merges.txt; do
  # all five exist in the prebuilt repo; a failure here is a network problem and must not pass silently
  curl -sfL --retry 5 -o "$OUT/$f" "https://huggingface.co/mlc-ai/Qwen3-1.7B-q4f16_1-MLC/resolve/main/$f" || { echo "download failed: $f"; exit 1; }
done
ls "$OUT"/params_shard_0.bin >/dev/null  # fail loudly if conversion produced nothing
( cd "$OUT" && sha256sum params_shard_*.bin tensor-cache.json ndarray-cache.json 2>/dev/null > SHA256SUMS || true )
du -sh "$OUT"; ls "$OUT" | head
