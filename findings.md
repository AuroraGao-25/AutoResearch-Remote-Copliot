# Research Findings

## Research Question

How can a testing-oriented evaluation pipeline (diagnostic suites + metamorphic tests) and targeted instruction tuning improve the robustness of instruction-tuned LLMs for aspect sentiment classification?

## Current Understanding

The proposal and literature jointly support a clear claim: **standard aggregate metrics are insufficient for ABSA reliability**. SemEval-style evaluation is necessary but not diagnostic; it does not directly isolate behavior under known linguistic stressors (negation, contrast, multi-aspect conflict, implicit sentiment, neutral/irrelevant aspect handling). CheckList provides a validated NLP testing philosophy that should transfer well to ABSA when transformed into domain-specific categories and deterministic metamorphic rules.

Instruction tuning has already shown strong ABSA performance (InstructABSA), and efficient adaptation methods (LoRA/QLoRA) make iterative targeted augmentation feasible on modest hardware. This creates a practical path: first reveal category-specific brittleness, then apply size-matched targeted augmentation and measure whether gains are localized and reproducible.

## Key Results

Completed baseline experiments (SemEval-2014 test, n=1758) show clear gains from QLoRA tuning:

- **A0 (prompt-only Qwen2.5-7B-Instruct):** accuracy 0.8447, macro-F1 0.7639.
- **A1 (QLoRA on SemEval-2014 train):** accuracy 0.8777, macro-F1 0.8278.
- **Delta (A1 - A0):** +0.0330 accuracy, +0.0639 macro-F1.

Operationally, the A0→A1 pipeline is now stable in this environment after resolving a Transformers API mismatch (`evaluation_strategy` vs `eval_strategy`) in training arguments.

Completed H2 metamorphic evaluation (n=3352 transformations/model) shows:

- **Overall pass rate:** A0 0.8598 vs A1 0.8586 (near-equal, slight A1 drop of 0.0012).
- **Strong divergence by transform family:** A1 improves invariance transforms but degrades `flip_contrast_template`.
- **Largest category gain:** `neutral_irrelevant` (A0 0.5126 → A1 0.6843).
- **Largest category drop:** `multi_aspect_conflict` (A0 0.8841 → A1 0.8152).

Completed first H3 ablation run (**C2 contrast-only augmentation**) shows a negative result:

- SemEval: accuracy 0.8737, macro-F1 0.8133 (both below A1).
- Metamorphic pass rate: 0.8198 (below A1 0.8586).
- `flip_contrast_template` pass rate dropped further (A1 0.7195 → C2 0.6148).

## Patterns and Insights

1. Parameter-efficient adaptation (A1 QLoRA) gives a meaningful quality lift over prompt-only A0 on standard SemEval metrics.
2. H2 is supported: macro-F1 improves substantially in A1, but metamorphic behavior is mixed and transformation-dependent.
3. Robustness improvements are not monotonic; A1 helps invariance consistency but introduces a large contrast-flip weakness.
4. Initial H3 evidence suggests naive contrast-only augmentation is not sufficient and can overfit in the wrong direction.

## Lessons and Constraints

- Reliability work needs strict train/test boundary discipline; ABSA-RTS must remain leakage-free.
- Metamorphic tests must be deterministic by construction; ambiguous transformations reduce interpretability.
- Size matching in ablations is mandatory to attribute gains to targeting rather than data volume.
- Strict JSON output is necessary for deterministic scoring and reproducible analysis.

## Open Questions

- What specific prompt/output patterns cause A1 failures on `flip_contrast_template` and `multi_aspect_conflict`?
- Can C1 (negation), C3 (multi-aspect), or C4 (all categories) recover robustness without C2-style regressions?
- Does category-targeted augmentation generalize across domains (Restaurants ↔ Laptops), or is transfer narrow?
- Which failure classes persist even after targeted augmentation (candidate future-work boundary)?

## Optimization Trajectory

Execution has started. Guardrail metric trajectory so far:

- A0 macro-F1: 0.7639
- A1 macro-F1: 0.8278

Primary trajectory metric is now measured:

- A0 metamorphic pass rate: 0.8598
- A1 metamorphic pass rate: 0.8586
- H3 C2 metamorphic pass rate: 0.8198
