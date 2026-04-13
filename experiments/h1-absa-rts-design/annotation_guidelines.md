# ABSA-RTS v0.1 Annotation Guidelines

Target task: ASC `(text, aspect) -> sentiment` with labels `positive`, `negative`, `neutral`.

## Category Definitions

1. **negation**: explicit negators affecting sentiment scope (e.g., "not good", "never works").
2. **contrast**: discourse opposition markers (e.g., "but", "however", "although") where clause interaction matters.
3. **intensifier**: scalar modifiers changing sentiment strength (e.g., "very", "extremely", "slightly").
4. **multi_aspect_conflict**: same sentence contains at least two aspects with opposite polarities.
5. **implicit_sentiment**: sentiment is inferred without explicit opinion lexicon cues.
6. **neutral_irrelevant**: target-aspect sentiment is neutral or irrelevant.

## Labeling Rules

1. Primary category is assigned by precedence: multi_aspect_conflict > negation > contrast > intensifier > implicit_sentiment > neutral_irrelevant.
2. Keep all matching categories in `rts_categories`; store one `rts_primary_category`.
3. Exclude examples with invalid/none aspect terms and conflict labels from model training/eval baseline files.
4. Preserve strict split discipline: ABSA-RTS is built from held-out/test material and is never used in finetuning.

## Quality Control

1. Double-annotate sampled subset.
2. Report Cohen's kappa.
3. Resolve disagreements through adjudication notes and update rule definitions.

