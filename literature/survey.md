# Literature Survey — ABSA Robustness and Instruction Tuning

## Scope

This survey focuses on work needed to execute the proposal:
1. ABSA benchmark/task framing
2. Behavioral robustness testing
3. Instruction tuning and efficient adaptation
4. LLM-based ABSA empirical baselines

## Core Papers and Relevance

### 1) ABSA benchmark foundation
- **Pontiki et al. (2014), SemEval-2014 Task 4 (DOI: 10.3115/v1/S14-2004)**  
  Defines core ABSA tasks including aspect category/term sentiment settings and provides canonical benchmark data.
- **Pontiki et al. (2015), SemEval-2015 Task 12 (DOI: 10.18653/v1/S15-2082)**  
  Extends ABSA evaluation settings and datasets.
- **Pontiki et al. (2016), SemEval-2016 Task 5 (DOI: 10.18653/v1/S16-1002)**  
  Multi-domain and multilingual expansion that reinforces benchmark continuity.

### 2) Behavioral testing
- **Ribeiro et al. (2020), CheckList (ACL 2020)**  
  Shows that held-out accuracy can overestimate reliability and introduces capability × test-type behavioral testing. Strong conceptual anchor for ABSA-RTS + metamorphic testing.

### 3) Efficient adaptation + instruction tuning
- **Hu et al. (2021), LoRA (arXiv:2106.09685)**  
  Introduces low-rank adaptation to reduce trainable parameters and memory while retaining quality.
- **Dettmers et al. (2023), QLoRA (arXiv:2305.14314)**  
  4-bit quantized backprop through frozen base model + LoRA adapters; practical for limited GPU settings.
- **Chung et al. (2022), FLAN/Instruction finetuning (arXiv:2210.11416)**  
  Demonstrates broad generalization gains from scaling instruction finetuning.

### 4) ABSA with instruction/LLMs
- **Scaria et al. (2023), InstructABSA (arXiv:2302.08624)**  
  Instruction-learning setup tailored to ABSA subtasks; reports strong gains over prior work.
- **Simmering & Huoviala (2023), LLMs for ABSA (arXiv:2310.18025)**  
  Compares GPT settings for ABSA and discusses prompt-vs-finetune trade-offs and cost-performance implications.

## Gap Synthesis

The reviewed literature supports high ABSA performance potential and robust adaptation methods, but still leaves a practical gap:

1. **Phenomenon-level reliability is under-measured** relative to aggregate metrics.
2. **ABSA-specific metamorphic testing is not standardized** in common evaluation pipelines.
3. **Targeted augmentation effects are often not disentangled** from data-size effects with strict size-matched ablations.

## Research Positioning

This project contributes by combining:
- A curated ABSA diagnostic robustness suite (ABSA-RTS),
- Deterministic metamorphic tests and pass-rate metrics,
- Targeted instruction augmentation under size-matched QLoRA ablations,
- Joint reporting of standard metrics + robustness diagnostics.

## Candidate Novelty Claim (to validate experimentally)

Targeted robustness-oriented instruction augmentation can deliver measurable category-specific reliability gains (metamorphic and per-category metrics) with minimal trade-off on standard SemEval metrics, and metamorphic pass rate reveals failure modes hidden by macro-F1.

