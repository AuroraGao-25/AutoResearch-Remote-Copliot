# Research Findings

## Research Question

How can a testing-oriented evaluation pipeline (diagnostic suites + metamorphic tests) and targeted instruction tuning improve the robustness of instruction-tuned LLMs for aspect sentiment classification?

## Current Understanding

The proposal and literature jointly support a clear claim: **standard aggregate metrics are insufficient for ABSA reliability**. SemEval-style evaluation is necessary but not diagnostic; it does not directly isolate behavior under known linguistic stressors (negation, contrast, multi-aspect conflict, implicit sentiment, neutral/irrelevant aspect handling). CheckList provides a validated NLP testing philosophy that should transfer well to ABSA when transformed into domain-specific categories and deterministic metamorphic rules.

Instruction tuning has already shown strong ABSA performance (InstructABSA), and efficient adaptation methods (LoRA/QLoRA) make iterative targeted augmentation feasible on modest hardware. This creates a practical path: first reveal category-specific brittleness, then apply size-matched targeted augmentation and measure whether gains are localized and reproducible.

## Key Results

No model experiments have been executed yet in this environment. The main completed output is a **research-operational blueprint**:

- Locked high-priority hypotheses (H1-H3) tied to measurable outcomes.
- Defined ABSA-RTS category set and metamorphic testing behavior expectations.
- Defined ablation structure (single-category and all-category targeted augmentation with size matching).
- Prepared paper draft and progress report artifacts for immediate execution phase.

## Patterns and Insights

1. The strongest methodological opportunity is not a new architecture but **better test design** and **targeted data intervention**.
2. Metamorphic pass rate is likely to surface hidden brittleness even when macro-F1 remains stable.
3. A robust contribution can be made even if standard metrics move modestly, provided per-category reliability gains are clear and reproducible.

## Lessons and Constraints

- Reliability work needs strict train/test boundary discipline; ABSA-RTS must remain leakage-free.
- Metamorphic tests must be deterministic by construction; ambiguous transformations reduce interpretability.
- Size matching in ablations is mandatory to attribute gains to targeting rather than data volume.
- Strict JSON output is necessary for deterministic scoring and reproducible analysis.

## Open Questions

- Which ABSA-RTS categories have the largest baseline failure rate for A0 vs A1?
- How strongly does metamorphic pass rate correlate with macro-F1 at overall and per-category levels?
- Does category-targeted augmentation generalize across domains (Restaurants ↔ Laptops), or is transfer narrow?
- Which failure classes persist even after targeted augmentation (candidate future-work boundary)?

## Optimization Trajectory

Execution phase not started yet. The planned trajectory metric is metamorphic pass rate (overall and per category) with macro-F1 guardrails.

