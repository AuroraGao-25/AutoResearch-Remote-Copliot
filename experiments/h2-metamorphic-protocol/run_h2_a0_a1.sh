#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
WORKDIR="${WORKDIR:-experiments/h2-metamorphic-protocol/results}"
A1_ADAPTER="${A1_ADAPTER:-experiments/h1-absa-rts-design/results/a1_adapter}"

mkdir -p "${WORKDIR}/a0" "${WORKDIR}/a1"

python src/build_absa_rts.py \
  --input-jsonl data/semeval14/all_test.jsonl \
  --output-jsonl data/absa_rts/absa_rts_v0.1.jsonl \
  --summary-json data/absa_rts/summary_v0.1.json \
  --per-category-per-domain 100

python src/make_metamorphic_suite.py \
  --input-rts data/absa_rts/absa_rts_v0.1.jsonl \
  --output-jsonl data/absa_rts/metamorphic_suite_v0.1.jsonl

python src/eval_metamorphic.py \
  --model-name "${MODEL_NAME}" \
  --input-jsonl data/absa_rts/metamorphic_suite_v0.1.jsonl \
  --output-dir "${WORKDIR}/a0"

if [[ -d "${A1_ADAPTER}" ]]; then
  python src/eval_metamorphic.py \
    --model-name "${MODEL_NAME}" \
    --adapter-path "${A1_ADAPTER}" \
    --input-jsonl data/absa_rts/metamorphic_suite_v0.1.jsonl \
    --output-dir "${WORKDIR}/a1"
else
  echo "A1 adapter not found at ${A1_ADAPTER}. Skipping A1 metamorphic eval."
fi

