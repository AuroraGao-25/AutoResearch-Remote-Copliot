# Protocol — H1: ABSA-RTS Diagnostic Failure Mapping

## Hypothesis
Instruction-tuned ABSA baselines have largest robustness drops in negation, contrast/discourse, and multi-aspect conflict categories.

## Prediction
Compared with overall benchmark metrics, per-category ABSA-RTS metrics will show significantly lower performance on at least two of these three categories.

## Setup
- Task: ASC `(text, aspect) -> sentiment`
- Domains: Restaurants, Laptops
- Models: A0 prompt-only, A1 standard QLoRA SFT
- Output contract: strict JSON `{"aspect": "...", "sentiment": "positive|negative|neutral"}`

## Diagnostic Categories
1. Negation
2. Contrast/discourse markers
3. Intensifiers/diminishers
4. Multi-aspect conflict
5. Implicit sentiment
6. Neutral/irrelevant aspect

## Evaluation
- Standard: macro-F1 and/or accuracy on SemEval test split
- Diagnostic: ABSA-RTS overall + per-category metrics with confidence intervals

## Confirmatory Criteria
- H1 supported if category deficits are consistent across domains/models and exceed overall-average drop.
- H1 inconclusive if category counts are insufficient or confidence intervals overlap broadly.

