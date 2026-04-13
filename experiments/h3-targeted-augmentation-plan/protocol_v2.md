# Protocol — H3-v2: Lower-Strength Targeted Augmentation

## Why v2

Interim H3-v1 results are negative so far (C1, C2 both below A1 on macro-F1 and metamorphic pass rate).  
Likely cause: augmentation strength/noise at `ADD_PER_CONDITION=600` is too aggressive for this train recipe.

## v2 Objective

Recover robustness gains while preserving SemEval guardrail by reducing augmentation pressure and isolating effects.

## Fixed Settings

- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Method: QLoRA (same pipeline/scripts)
- Data discipline: ABSA-RTS remains evaluation-only (no training leakage)
- Evaluation: same SemEval + metamorphic suite as H3-v1

## v2 Experiment Set (small, controlled)

1. `c4_all` with `ADD_PER_CONDITION=200`
2. `c4_all` with `ADD_PER_CONDITION=400`
3. `c3_multi_aspect` with `ADD_PER_CONDITION=200`
4. `c2_contrast` with `ADD_PER_CONDITION=200`

Rationale:
- Test whether lower augmentation strength removes regression.
- Prioritize `c4_all` for robustness balancing and `c3_multi_aspect` for known weak category.
- Keep total runs small before any broader sweep.

## Success / Rejection Criteria

- **Keep candidate** only if:
  - metamorphic pass rate improves vs A1 baseline, and
  - macro-F1 regression is within small tolerance (<= 0.005 absolute drop).
- **Reject candidate** if:
  - metamorphic pass rate does not improve, or
  - macro-F1 drops > 0.005.

## Execution Order

Run only after H3-v1 C3/C4 completes.

1. `c4_all@200`
2. `c4_all@400`
3. `c3_multi_aspect@200`
4. `c2_contrast@200`

## Run Commands (single-condition)

```bash
ADD_PER_CONDITION=200 CONDITION=c4_all WORKDIR=experiments/h3-targeted-augmentation-plan/results_v2/c4_all_add200 bash experiments/h3-targeted-augmentation-plan/run_h3_condition.sh
ADD_PER_CONDITION=400 CONDITION=c4_all WORKDIR=experiments/h3-targeted-augmentation-plan/results_v2/c4_all_add400 bash experiments/h3-targeted-augmentation-plan/run_h3_condition.sh
ADD_PER_CONDITION=200 CONDITION=c3_multi_aspect WORKDIR=experiments/h3-targeted-augmentation-plan/results_v2/c3_multi_aspect_add200 bash experiments/h3-targeted-augmentation-plan/run_h3_condition.sh
ADD_PER_CONDITION=200 CONDITION=c2_contrast WORKDIR=experiments/h3-targeted-augmentation-plan/results_v2/c2_contrast_add200 bash experiments/h3-targeted-augmentation-plan/run_h3_condition.sh
```

Use separate output roots for v2 runs to avoid overwriting v1 artifacts.
