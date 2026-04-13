#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
ADD_PER_CONDITION="${ADD_PER_CONDITION:-600}"
ROOT="experiments/h3-targeted-augmentation-plan/results"

mkdir -p "${ROOT}"

run_condition() {
  local cond="$1"
  local out="${ROOT}/${cond}"
  local semeval_metrics="${out}/semeval_eval/metrics.json"
  local metamorphic_metrics="${out}/metamorphic_eval/metamorphic_metrics.json"
  mkdir -p "${out}"

  if [[ -f "${semeval_metrics}" && -f "${metamorphic_metrics}" ]]; then
    echo "[skip] ${cond} already complete"
    return
  fi

  echo "[run] ${cond}"
  CONDITION="${cond}" MODEL_NAME="${MODEL_NAME}" ADD_PER_CONDITION="${ADD_PER_CONDITION}" \
    bash experiments/h3-targeted-augmentation-plan/run_h3_condition.sh \
    2>&1 | tee "${out}/run.log"
}

# Continue any incomplete run first, then execute the remaining conditions.
for cond in c1_negation c3_multi_aspect c4_all; do
  run_condition "${cond}"
done

# Build a compact ablation summary from available outputs.
python - <<'PY'
import csv
import json
from pathlib import Path

root = Path("experiments/h3-targeted-augmentation-plan/results")
a1_semeval = Path("experiments/h1-absa-rts-design/results/a1_eval/metrics.json")
a1_meta = Path("experiments/h2-metamorphic-protocol/results/a1/metamorphic_metrics.json")

def read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())

rows = []
baseline = {
    "condition": "a1_baseline",
    "accuracy": None,
    "macro_f1": None,
    "pass_rate": None,
}
if a1_semeval.exists():
    d = read_json(a1_semeval)
    baseline["accuracy"] = d.get("accuracy")
    baseline["macro_f1"] = d.get("macro_f1")
if a1_meta.exists():
    d = read_json(a1_meta)
    baseline["pass_rate"] = d.get("pass_rate")
rows.append(baseline)

for cond in ["c1_negation", "c2_contrast", "c3_multi_aspect", "c4_all"]:
    sem = read_json(root / cond / "semeval_eval" / "metrics.json")
    met = read_json(root / cond / "metamorphic_eval" / "metamorphic_metrics.json")
    if not sem and not met:
        continue
    rows.append(
        {
            "condition": cond,
            "accuracy": sem.get("accuracy") if sem else None,
            "macro_f1": sem.get("macro_f1") if sem else None,
            "pass_rate": met.get("pass_rate") if met else None,
        }
    )

a1 = rows[0]
for r in rows[1:]:
    r["delta_macro_f1_vs_a1"] = (
        (r["macro_f1"] - a1["macro_f1"]) if r.get("macro_f1") is not None and a1.get("macro_f1") is not None else None
    )
    r["delta_pass_rate_vs_a1"] = (
        (r["pass_rate"] - a1["pass_rate"]) if r.get("pass_rate") is not None and a1.get("pass_rate") is not None else None
    )

out_json = root / "ablation_summary.json"
out_csv = root / "ablation_summary.csv"
out_json.write_text(json.dumps(rows, indent=2))

fieldnames = [
    "condition",
    "accuracy",
    "macro_f1",
    "pass_rate",
    "delta_macro_f1_vs_a1",
    "delta_pass_rate_vs_a1",
]
with out_csv.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k) for k in fieldnames})

print(f"Wrote {out_json}")
print(f"Wrote {out_csv}")
PY

echo "[done] H3 remaining runs and summary complete"
