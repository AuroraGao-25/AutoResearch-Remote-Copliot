# Analysis Notes — H3

Completed run: **C2 (+Contrast only)**.

## C2 metrics (A2-c2 vs A1 baseline)

- SemEval accuracy: 0.8737 vs 0.8777 (**delta -0.0040**)
- SemEval macro-F1: 0.8133 vs 0.8278 (**delta -0.0146**)
- Metamorphic pass rate: 0.8198 vs 0.8586 (**delta -0.0388**)

## C2 interpretation

- Contrast-only augmentation did **not** improve the contrast weakness observed in A1.
- `flip_contrast_template` decreased further (A1 0.7195 -> C2 0.6148).
- C2 currently weakens both aggregate metric and robustness metric, so it is not a candidate final setting.

## Interim light analysis (while C3 in progress)

Available completed conditions vs A1:

- C1: macro-F1 0.8151 (**-0.0127**), pass rate 0.8121 (**-0.0465**)
- C2: macro-F1 0.8133 (**-0.0146**), pass rate 0.8198 (**-0.0388**)

Interpretation:

- Under current recipe, single-category targeted augmentation at strength 600 has a consistent regression pattern.
- This supports a practical failure hypothesis: **augmentation pressure/noise is too high** for this finetuning setup.
- Decision: finish H3-v1 (C3/C4) first, then run a lower-strength H3-v2 grid.

## H3-v2 prep

Prepared next protocol: `experiments/h3-targeted-augmentation-plan/protocol_v2.md`

Key design:

- Lower-strength additions (`ADD_PER_CONDITION=200/400`) before any larger sweep
- Small controlled run set to isolate effect size
- Keep only variants that improve metamorphic pass rate without meaningful macro-F1 regression

Planned outputs:
- Ablation table C1-C4 vs A1 baseline
- Heatmap of category gains by augmentation condition
- Trade-off view: robustness gain vs standard metric delta
