# Protocol — H3: Targeted Augmentation + QLoRA

## Hypothesis
Size-matched category-targeted instruction augmentation improves robustness on targeted phenomena without harming standard SemEval performance.

## Prediction
Single-category augmentations improve corresponding ABSA-RTS category metrics; combined augmentation improves overall metamorphic pass rate with minimal macro-F1 regression.

## Conditions (size-matched additions)
- C1: +Negation only
- C2: +Contrast only
- C3: +Multi-aspect only
- C4: +All categories

## Training
- Base: 7B instruct model (fallback: ~3B instruct if resource-limited)
- Method: QLoRA
- Data discipline: ABSA-RTS kept strictly out of training

## Evaluation
- Standard SemEval metrics (guardrail)
- ABSA-RTS per-category metrics
- Metamorphic pass rate by transformation family

## Confirmatory Criteria
- H3 supported if targeted categories improve under matching augmentations and overall standard metrics are maintained within acceptable tolerance.

