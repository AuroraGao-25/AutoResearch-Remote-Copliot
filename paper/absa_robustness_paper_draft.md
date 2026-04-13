# Beyond F1: Diagnostic Test Suites and Targeted Instruction Tuning for Robust Aspect-Based Sentiment Analysis

## Abstract

Aspect-Based Sentiment Analysis (ABSA) systems can achieve competitive aggregate performance while failing on specific linguistic phenomena that are critical for reliability in practice. We propose a testing-oriented ABSA robustness framework combining: (1) **ABSA-RTS**, a category-labeled diagnostic test suite; (2) **deterministic metamorphic tests** with pass-rate scoring; and (3) **targeted instruction augmentation with QLoRA** under size-matched ablations. The framework is designed to reveal and repair brittle behavior in instruction-tuned LLMs for aspect sentiment classification. We provide a reproducible protocol over SemEval domains (Restaurants, Laptops), explicit leakage controls, strict JSON output formatting for deterministic evaluation, and ablations mapping augmentation categories to robustness gains. This manuscript presents the finalized experimental design and literature-grounded contribution claims; empirical sections are structured and ready to be populated with execution results.

## 1. Introduction

ABSA predicts sentiment toward a given aspect within a sentence. Unlike document-level sentiment, ABSA requires localized reasoning over syntax, discourse, and target-aspect alignment. A single sentence can contain conflicting sentiments across aspects, making robustness to linguistic variation essential.

Recent instruction-tuned LLMs improve ABSA quality, but standard held-out metrics (accuracy, macro-F1) can hide systematic failures. Inspired by behavioral testing in NLP, we treat robustness categories as software-style failure patterns and test them explicitly. Our central premise is that robust ABSA requires both **diagnosis** and **targeted intervention**, not only higher aggregate benchmark scores.

## 2. Related Work

### 2.1 ABSA benchmarks
SemEval ABSA tasks established the canonical benchmark landscape for aspect-level sentiment classification and remain standard evaluation references across domains.

### 2.2 Behavioral testing for NLP
CheckList showed that held-out accuracy can overestimate model reliability and introduced capability-oriented testing strategies. This motivates ABSA-specific diagnostic and metamorphic testing.

### 2.3 Instruction tuning and efficient adaptation
Instruction finetuning improves broad task generalization. LoRA and QLoRA enable practical parameter-efficient adaptation under limited hardware budgets, making targeted robustness-oriented finetuning feasible.

### 2.4 LLMs for ABSA
InstructABSA demonstrated strong ABSA gains from instruction learning. Recent LLM-based ABSA analyses further highlight trade-offs among prompt-only, few-shot, and fine-tuned setups, supporting our A0/A1/A2 evaluation framing.

## 3. Method

### 3.1 Task and output contract
Primary task is aspect sentiment classification: given `(text, aspect)`, predict sentiment polarity. Model outputs are constrained to strict JSON:

```json
{"aspect": "battery", "sentiment": "negative"}
```

This enforces deterministic parsing and reproducible scoring.

### 3.2 Model conditions
- **A0**: Prompt-only instruction baseline
- **A1**: Standard QLoRA SFT baseline
- **A2**: Targeted instruction augmentation + QLoRA

### 3.3 ABSA-RTS diagnostic suite
ABSA-RTS is organized into six categories: negation, contrast/discourse, intensifiers/diminishers, multi-aspect conflict, implicit sentiment, and neutral/irrelevant-aspect cases. Target sample density is 50-100 items/category/domain where feasible, with confidence intervals for sparse classes.

### 3.4 Metamorphic testing
We define deterministic transformations with pre-specified expected behavior:
- **Invariance transforms** should preserve label.
- **Directional/flip transforms** should change label in a known direction.

The principal robustness metric is metamorphic pass rate (overall and by transformation family).

### 3.5 Targeted augmentation ablations
We evaluate size-matched augmentation regimes:
- C1 (+Negation),
- C2 (+Contrast),
- C3 (+Multi-aspect),
- C4 (+All categories).

Size matching controls for pure data-volume effects.

## 4. Experimental Design

### 4.1 Datasets and scope
SemEval ABSA domains: Restaurants and Laptops (English). ABSA-RTS remains strictly separate from all training data.

### 4.2 Metrics
- Standard: accuracy / macro-F1
- Robustness: ABSA-RTS overall + per-category metrics
- Behavioral: metamorphic pass rate
- Diagnostics: error taxonomy with representative examples

### 4.3 Hypotheses
- **H1**: Largest robustness deficits occur in negation, contrast, and multi-aspect conflict.
- **H2**: Metamorphic pass rate provides robustness signal beyond macro-F1.
- **H3**: Targeted augmentation improves corresponding robustness categories without harming standard test performance.

## 5. Reproducibility and Validity Controls

1. Strict train/test separation to prevent leakage.
2. Written category-labeling guidelines and adjudication workflow.
3. Inter-annotator agreement checks on sampled subsets (Cohen's kappa target >= 0.75).
4. Deterministic transformation templates for metamorphic tests.
5. Output schema constraints (strict JSON).

## 6. Expected Contributions

1. ABSA-RTS: a reusable diagnostic robustness suite for ASC.
2. ABSA metamorphic testing framework and scoring scripts.
3. Targeted instruction augmentation protocol with size-matched ablations.
4. Evidence linking specific failure modes to specific interventions.

## 7. Current Status

This draft captures the finalized research design and literature-backed positioning. Empirical execution and result tables/figures are pending the next inner-loop phase (dataset construction, model runs, and ablation evaluation).

## References

- Pontiki et al. (2014). *SemEval-2014 Task 4: Aspect Based Sentiment Analysis*. DOI: 10.3115/v1/S14-2004.
- Pontiki et al. (2015). *SemEval-2015 Task 12: Aspect Based Sentiment Analysis*. DOI: 10.18653/v1/S15-2082.
- Pontiki et al. (2016). *SemEval-2016 Task 5: Aspect Based Sentiment Analysis*. DOI: 10.18653/v1/S16-1002.
- Ribeiro et al. (2020). *Beyond Accuracy: Behavioral Testing of NLP Models with CheckList*. ACL.
- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685.
- Chung et al. (2022). *Scaling Instruction-Finetuned Language Models*. arXiv:2210.11416.
- Dettmers et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. arXiv:2305.14314.
- Scaria et al. (2023). *InstructABSA: Instruction Learning for Aspect Based Sentiment Analysis*. arXiv:2302.08624.
- Simmering & Huoviala (2023). *Large language models for aspect-based sentiment analysis*. arXiv:2310.18025.

