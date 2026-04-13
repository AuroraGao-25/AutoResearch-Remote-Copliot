# Protocol — H2: Metamorphic Signal Beyond Macro-F1

## Hypothesis
Metamorphic pass rate captures robustness differences not visible in conventional aggregate metrics.

## Prediction
Pairs of model variants with similar macro-F1 will still differ materially in metamorphic pass rate for selected transformation families.

## Transformation Families

### Invariance (label should remain unchanged)
- Add irrelevant clause not referencing target aspect.
- Reorder non-target context.
- Neutral paraphrase preserving target-aspect sentiment.

### Directional/flip (label should change deterministically)
- Controlled negation insertion for target-aspect predicate.
- Contrast insertion where post-contrast target clause determines label.

## Evaluation
- Metamorphic pass rate overall and by transformation type.
- Correlation analysis between macro-F1 and pass rate.
- Delta diagnostics across A0/A1/A2.

## Confirmatory Criteria
- H2 supported if at least one condition shows near-equal macro-F1 but clear pass-rate separation.

