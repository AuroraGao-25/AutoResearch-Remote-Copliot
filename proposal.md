# Master’s Research Proposal

## Working Title
**Beyond F1: Diagnostic Test Suites and Targeted Instruction Tuning for Robust Aspect-Based Sentiment Analysis (ABSA)**

## 1. Background and Motivation
Aspect-Based Sentiment Analysis (ABSA) identifies sentiment *towards a specific aspect* (e.g., **battery**, **service**) within a review. Compared to document-level sentiment, ABSA is harder because a single text may contain multiple aspects with different polarities.

Recent instruction-tuned large language models (LLMs) are promising for ABSA due to strong language understanding and flexible prompting. However, ABSA performance measured on a standard test split (e.g., SemEval) may hide systematic failure modes such as:
- **Negation** (e.g., “not good”) 
- **Contrast / discourse** markers (e.g., “good, **but** expensive”)
- **Multi-aspect conflicts** (e.g., “screen is great, battery is terrible”)

From a software testing perspective, these are “bug patterns” that should be tested explicitly, tracked, and improved via targeted interventions.

## 2. Problem Statement
Standard ABSA evaluation often focuses on aggregate metrics (accuracy / macro-F1) on a held-out test set. This can under-represent brittle model behavior under small but meaningful wording changes.

**The problem:** Instruction-tuned ABSA models may show acceptable overall scores while failing predictably on specific linguistic phenomena. A thesis-level contribution is to (i) *diagnose* these reliability boundaries and (ii) *improve* them with a targeted, reproducible fine-tuning approach.

## 3. Aim and Objectives
### Aim
To develop a testing-oriented methodology for evaluating and improving robustness of instruction-tuned LLMs for ABSA.

### Objectives
1. **Construct ABSA-RTS**: a diagnostic robustness test suite for ABSA (ASC setting) with labeled categories (negation, contrast, multi-aspect, etc.).
2. **Define metamorphic tests**: controlled text transformations with expected invariance or label flips; measure **metamorphic pass rate**.
3. **Apply targeted instruction augmentation + QLoRA**: fine-tune a 7B instruct model with category-specific examples and measure robustness gains.
4. Provide **ablation evidence** mapping each targeted augmentation to improvements in specific failure categories.

## 4. Research Questions
- **RQ1:** Which linguistic/discourse phenomena cause the largest robustness drops for instruction-tuned ABSA models? -- RO1
- **RQ2:** Can targeted instruction augmentation measurably improve robustness on these phenomena without harming standard test performance? -- RO3 RO4
- **RQ3:** Do metamorphic pass rates provide information beyond conventional metrics (accuracy/macro-F1), and how do they relate? -- RO2

## 5. Scope and Assumptions
- **Primary task:** Aspect Sentiment Classification (ASC): given *(text, aspect)* → predict sentiment polarity.
- **Datasets:** SemEval ABSA domains (Restaurants and Laptops).
- **Compute (primary + fallbacks):**
  - Primary: **Colab T4 16GB** with **QLoRA/LoRA**.
  - Fallbacks (to reduce disconnection risk): **Colab pro**.
  - If resources remain constrained: switch the main model to a **~3B instruct** checkpoint (e.g., Llama-3.2-3B-Instruct-class) using the same QLoRA pipeline.
- **Language:** English.
- **Out of scope (initially):** full end-to-end triplet extraction; cross-lingual transfer (possible future extensions).

## 6. Proposed Methodology
### 6.1 Task Formulation and Output Format
To reduce ambiguity and support automatic evaluation, the model will be prompted/trained to output **strict JSON**, e.g.:
```json
{"aspect": "battery", "sentiment": "negative"}
```

### 6.2 Baseline Models
- **A0 Prompt-only:** instruction prompting without fine-tuning.
- **A1 Standard QLoRA SFT:** QLoRA fine-tuning on SemEval training data.

### 6.3 ABSA-RTS: Robustness Test Suite (construction details)
ABSA-RTS will be a curated diagnostic evaluation set organized into labeled categories:
- Negation
- Contrast / discourse (e.g., “but/however”)
- Intensifiers/diminishers (e.g., “very”, “slightly”)
- Multi-aspect conflict
- Implicit sentiment (sentiment implied without explicit polarity words)
- Neutral/irrelevant aspect cases

**Sample size (target):** at least **50-100 examples per category per domain** (Restaurants, Laptops) where feasible; if a category is naturally rare, we will report confidence intervals and treat results as exploratory for that category.

**Labeling and quality control:**
- The **sentiment labels** come from SemEval where applicable.
- The **category label** (e.g., negation vs contrast) will be assigned using written guidelines.
- A subset will be **double-annotated** and inter-annotator agreement will be reported using **Cohen’s κ** (target κ ≥ 0.75). Disagreements will be adjudicated and guidelines refined.

**Data separation:** ABSA-RTS examples will be kept separate from training (no leakage) to support a testing/regression framing.

### 6.4 Metamorphic Testing Protocol (objective expected behavior)
Define transformations **T(x)** with pre-specified expected behavior:
- **Invariance tests** (label should not change): add irrelevant clause; reorder non-target information; neutral paraphrase not affecting target-aspect sentiment.
- **Directional/flip tests** (label should change in a defined way): insert negation in a controlled template; add a contrast clause where the *post-contrast* clause expresses sentiment about the target aspect.

To reduce subjectivity, transformations will be designed so that the expected behavior is **deterministic by construction** (template-based where needed). Any ambiguous cases will be filtered out during dataset creation.

The key metric is **metamorphic pass rate**: the percentage of transformed examples where the model behavior matches the expected invariance/flip rule.

### 6.5 Targeted Instruction Augmentation
Create additional instruction-style training examples specifically covering each ABSA-RTS category. Fine-tune with QLoRA using:
- **Single-category augmentation** (negation-only, contrast-only, etc.)
- **All-category augmentation** (combined)

All ablations will be **size-matched** to ensure improvements are not merely from “more data”.

## 7. Experimental Design and Ablations
### 7.1 Main Results
Evaluate A0/A1/A2 on:
- Standard SemEval test sets (accuracy / macro-F1 as appropriate)
- ABSA-RTS overall + per-category metrics
- Metamorphic pass rate (overall + by transformation type)

### 7.2 Targeted Augmentation Ablations (Key Table)
Train with size-matched additions:
- **C1:** +Negation only
- **C2:** +Contrast only
- **C3:** +Multi-aspect only
- **C4:** +All categories

**Practical compute plan:** run the full grid first on one domain for debugging, then replicate on the second domain once the pipeline is stable.

### 7.3 Optional (time-permitting): Instruction Design Ablation
Compare inference prompts **with vs without explicit “rules”** (e.g., contrast-handling guideline) to separate learned robustness from prompt scaffolding. This is explicitly treated as an extension after the mainline (A0/A1/A2 + C1–C4) is complete.

## 8. Evaluation Metrics
- **ASC standard metrics:** accuracy and/or macro-F1 (depending on dataset label distribution and standard practice).
- **ABSA-RTS metrics:** per-category accuracy/macro-F1 (+ confidence intervals where appropriate).
- **Metamorphic pass rate:** pass/fail correctness under defined transformations.
- **Error analysis:** taxonomy of common failures with representative before/after examples.

## 9. Expected Contributions / Deliverables
1. **ABSA-RTS** robustness test suite with category labels and annotation guidelines.
2. **Metamorphic testing framework** for ABSA with scoring scripts.
3. **Targeted instruction augmentation dataset** (or generation procedure) and training configuration.
4. Reproducible results with ablations and diagnostic analysis.

## 10. Feasibility (Resources)
- **Compute:** Colab T4 or pro is sufficient for 7B QLoRA experiments and multiple ablations.
- **Tools:** HuggingFace Transformers + PEFT + bitsandbytes for QLoRA; standard NLP evaluation scripts.

## 11. Risks and Mitigation
- **Risk:** Minimal improvement on standard metrics.
  - **Mitigation:** thesis emphasis is reliability; report ABSA-RTS per-category and metamorphic improvements with clear diagnosis→fix mapping.
- **Risk:** Subjectivity in robustness labels or metamorphic expectations.
  - **Mitigation:** written labeling guidelines; κ agreement reporting; template-based transformations with deterministic expected behavior; filter ambiguous cases.
- **Risk:** Overfitting to synthetic test cases.
  - **Mitigation:** keep ABSA-RTS anchored to SemEval-style text; keep strict separation from training; report both standard test and ABSA-RTS.
- **Risk:** Evaluation ambiguity due to free-form outputs.
  - **Mitigation:** strict JSON outputs and parsable evaluation.

## 12. References (Starter List)
- Pontiki et al. (2014). *SemEval-2014 Task 4: Aspect Based Sentiment Analysis.*
- Pontiki et al. (2015). *SemEval-2015 Task 12: Aspect Based Sentiment Analysis.*
- Pontiki et al. (2016). *SemEval-2016 Task 5: Aspect Based Sentiment Analysis.*
- Ribeiro et al. (2020). *Beyond Accuracy: Behavioral Testing of NLP Models with CheckList.*
- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.*
- Wei et al. (2022). *Finetuned Language Models Are Zero-Shot Learners (FLAN).*
- Dettmers et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.*
- Scaria, K., Gupta, H., Goyal, S., Sawant, S. A., Mishra, S., & Baral, C. (2023). *InstructABSA: Instruction Learning for Aspect Based Sentiment Analysis.* arXiv:2302.08624.
- Simmering, P. F., & Huoviala, P. (2023). *Large language models for aspect-based sentiment analysis.* arXiv:2310.18025.
