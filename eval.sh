#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_PYTHON="${TRAIN_PYTHON:-python}"
VLLM_PYTHON="${VLLM_PYTHON:-python}"
BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-2-7b-hf}"
ADAPTER="${1:?usage: ./eval.sh ADAPTER [OUTPUT_JSON] [MERGED_MODEL_DIR]}"
OUTPUT="${2:-${ROOT_DIR}/gsm8k_results.json}"
MERGED_MODEL="${3:-${ROOT_DIR}/merged_model}"

export HF_HOME="${HF_HOME:-/root/autodl-fs/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM=false

rm -rf "${MERGED_MODEL}"
"${TRAIN_PYTHON}" "${ROOT_DIR}/merge_adapter.py" \
  --base-model "${BASE_MODEL}" \
  --adapter "${ADAPTER}" \
  --output "${MERGED_MODEL}"

"${VLLM_PYTHON}" "${ROOT_DIR}/eval_gsm8k.py" \
  --model "${MERGED_MODEL}" \
  --output "${OUTPUT}"
