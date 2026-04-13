# Research Log

Chronological record of research decisions and actions. Append-only.

| # | Date | Type | Summary |
|---|------|------|---------|
| 1 | 2026-04-13 | bootstrap | Initialized autoresearch workspace from project proposal and created persistent state files (research-state, findings, log, literature, experiments, paper, to_human). |
| 2 | 2026-04-13 | bootstrap | Surveyed core references (SemEval ABSA tasks, CheckList, LoRA/QLoRA, InstructABSA, LLMs-for-ABSA). Identified gaps: phenomenon-level robustness diagnostics and metamorphic testing standardization for ABSA. |
| 3 | 2026-04-13 | outer-loop | Direction decision: DEEPEN current question rather than pivot. Immediate focus is H1/H2 protocol locking: ABSA-RTS categories + deterministic metamorphic transformations + size-matched augmentation ablations. |
| 4 | 2026-04-13 | report | Generated initial progress report (`to_human/progress-001.html`) and manuscript draft (`paper/absa_robustness_paper_draft.md`). |
| 5 | 2026-04-13 | inner-loop | Implemented executable baseline pipeline for SemEval-2014 with Qwen: XML parser + JSONL prep, QLoRA training script, A0/A1 evaluation script, and one-command runner (`run_a0_a1.sh`). |
| 6 | 2026-04-13 | inner-loop | Updated pipeline to SemEval-2014 CSV input, prepared 5,915 train / 1,758 test ASC samples, and launched full A0→A1 baseline run for `Qwen/Qwen2.5-7B-Instruct`. |
| 7 | 2026-04-13 | inner-loop | Implemented next-step robustness tooling: ABSA-RTS v0.1 builder, metamorphic suite generator, metamorphic pass-rate evaluator, and H2 runner script for A0/A1. |
