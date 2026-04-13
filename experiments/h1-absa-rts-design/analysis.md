# Analysis Notes — H1

Completed baseline comparison on SemEval-2014 test (n=1758):

- A0 (prompt-only): accuracy 0.8447, macro-F1 0.7639
- A1 (QLoRA): accuracy 0.8777, macro-F1 0.8278
- Delta (A1 - A0): +0.0330 accuracy, +0.0639 macro-F1

Planned outputs:
- Category-wise confusion matrices
- Per-category confidence intervals
- Error exemplars for each failure mode
