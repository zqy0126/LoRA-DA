#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${TRAIN_PYTHON:-python}"

export HF_HOME="${HF_HOME:-/root/autodl-fs/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM=false

cd "${ROOT_DIR}"
"${PYTHON}" run_exp.py "$@"

echo "Adapter: ${ROOT_DIR}/safe_results/xxx_meta_math/lora-da-metamath100k-gsm8k-grad256-fisher256-beta0p03-gamma1024-scale1-final/9/final_checkpoint"
