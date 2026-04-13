#!/usr/bin/env bash
set -euo pipefail

# Example:
#   CONDITION=c2_contrast ADD_PER_CONDITION=600 bash experiments/h3-targeted-augmentation-plan/run_h3_condition.sh

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
CONDITION="${CONDITION:-c2_contrast}"
ADD_PER_CONDITION="${ADD_PER_CONDITION:-600}"
BASE_TRAIN="${BASE_TRAIN:-data/semeval14/all_train.jsonl}"
TEST_FILE="${TEST_FILE:-data/semeval14/all_test.jsonl}"
METAMORPHIC_FILE="${METAMORPHIC_FILE:-data/absa_rts/metamorphic_suite_v0.1.jsonl}"
WORKDIR="${WORKDIR:-experiments/h3-targeted-augmentation-plan/results/${CONDITION}}"

mkdir -p "${WORKDIR}"

python src/build_h3_datasets.py \
  --input-train "${BASE_TRAIN}" \
  --out-dir data/h3 \
  --add-per-condition "${ADD_PER_CONDITION}"

TRAIN_FILE="data/h3/${CONDITION}_train.jsonl"
if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "Missing training file: ${TRAIN_FILE}"
  exit 1
fi

python src/train_qlora.py \
  --model-name "${MODEL_NAME}" \
  --train-file "${TRAIN_FILE}" \
  --eval-file "${TEST_FILE}" \
  --output-dir "${WORKDIR}/adapter"

python src/eval_baseline.py \
  --model-name "${MODEL_NAME}" \
  --adapter-path "${WORKDIR}/adapter" \
  --test-file "${TEST_FILE}" \
  --output-dir "${WORKDIR}/semeval_eval"

python src/eval_metamorphic.py \
  --model-name "${MODEL_NAME}" \
  --adapter-path "${WORKDIR}/adapter" \
  --input-jsonl "${METAMORPHIC_FILE}" \
  --output-dir "${WORKDIR}/metamorphic_eval"
