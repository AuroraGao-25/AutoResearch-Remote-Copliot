# Analysis Notes — H2

Completed A0/A1 metamorphic evaluation (n=3352 transformations each).

## Overall pass rate

- A0: 0.8598
- A1: 0.8586
- Delta (A1 - A0): -0.0012

## Pass rate by transform

- flip_contrast_template: A0 0.8924, A1 0.7195, delta -0.1730
- flip_negation_template: A0 1.0000, A1 1.0000, delta +0.0000
- invariance_irrelevant_clause: A0 0.7915, A1 0.8603, delta +0.0688
- invariance_reorder_context: A0 0.8077, A1 0.8553, delta +0.0476

## Pass rate by source category

- contrast: A0 0.8750, A1 0.8535, delta -0.0215
- implicit_sentiment: A0 0.9700, A1 0.9400, delta -0.0300
- intensifier: A0 0.9464, A1 0.9405, delta -0.0060
- multi_aspect_conflict: A0 0.8841, A1 0.8152, delta -0.0688
- negation: A0 0.8261, A1 0.8060, delta -0.0201
- neutral_irrelevant: A0 0.5126, A1 0.6843, delta +0.1717

Interpretation: despite strong A1 gains on macro-F1 (H1), metamorphic robustness shifts are mixed and phenomenon-specific, supporting H2's claim that aggregate metrics and behavioral robustness can diverge.

Planned outputs:
- Transformation-level pass/fail tables
- Correlation plot (macro-F1 vs metamorphic pass rate)
- Case studies where aggregate metrics hide behavioral failure
