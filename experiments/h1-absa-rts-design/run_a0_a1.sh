#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   RAW_ROOT=/path/to/semeval14 INPUT_FORMAT=csv bash experiments/h1-absa-rts-design/run_a0_a1.sh

RAW_ROOT="${RAW_ROOT:-data/raw/SemEval14}"
INPUT_FORMAT="${INPUT_FORMAT:-csv}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
WORKDIR="${WORKDIR:-experiments/h1-absa-rts-design/results}"

if [[ -z "${RAW_ROOT}" ]]; then
  echo "ERROR: set RAW_ROOT to SemEval-2014 XML directory"
  exit 1
fi

mkdir -p "${WORKDIR}/a0" "${WORKDIR}/a1_adapter" "${WORKDIR}/a1_eval"

python src/prepare_semeval2014.py \
  --raw-root "${RAW_ROOT}" \
  --input-format "${INPUT_FORMAT}" \
  --out-dir data/semeval14 \
  --drop-conflict

# A0: prompt-only baseline
python src/eval_baseline.py \
  --model-name "${MODEL_NAME}" \
  --test-file data/semeval14/all_test.jsonl \
  --output-dir "${WORKDIR}/a0"

# A1: QLoRA baseline
python src/train_qlora.py \
  --model-name "${MODEL_NAME}" \
  --train-file data/semeval14/all_train.jsonl \
  --eval-file data/semeval14/all_test.jsonl \
  --output-dir "${WORKDIR}/a1_adapter"

python src/eval_baseline.py \
  --model-name "${MODEL_NAME}" \
  --adapter-path "${WORKDIR}/a1_adapter" \
  --test-file data/semeval14/all_test.jsonl \
  --output-dir "${WORKDIR}/a1_eval"
