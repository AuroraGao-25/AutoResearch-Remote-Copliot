#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from absa.prompting import build_prompt, parse_json_sentiment


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate metamorphic pass rate for ABSA.")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--adapter-path", type=Path, default=None)
    p.add_argument("--input-jsonl", type=Path, default=Path("data/absa_rts/metamorphic_suite_v0.1.jsonl"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _predict_one(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, text: str, aspect: str, max_new_tokens: int) -> str:
    prompt = build_prompt(text, aspect)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    pred = parse_json_sentiment(decoded)
    return pred if pred else "neutral"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, str(args.adapter_path))
    model.eval()

    rows = _read_jsonl(args.input_jsonl)
    if args.limit > 0:
        rows = rows[: args.limit]

    details: list[dict] = []
    pass_flags: list[int] = []
    by_transform: dict[str, list[int]] = defaultdict(list)
    by_category: dict[str, list[int]] = defaultdict(list)

    for row in tqdm(rows, desc="Metamorphic eval"):
        pred = _predict_one(
            model=model,
            tokenizer=tokenizer,
            text=row["text"],
            aspect=row["aspect"],
            max_new_tokens=args.max_new_tokens,
        )
        passed = int(pred == row["expected_sentiment"])
        pass_flags.append(passed)
        by_transform[row["transform_type"]].append(passed)
        by_category[row.get("source_primary_category", "unknown")].append(passed)
        details.append(
            {
                **row,
                "predicted_sentiment": pred,
                "passed": bool(passed),
            }
        )

    def _rate(vals: list[int]) -> float:
        return (sum(vals) / len(vals)) if vals else 0.0

    summary = {
        "n": len(rows),
        "pass_rate": _rate(pass_flags),
        "model_name": args.model_name,
        "adapter_path": str(args.adapter_path) if args.adapter_path else None,
        "transform_distribution": dict(Counter(r["transform_type"] for r in rows)),
        "pass_rate_by_transform": {k: _rate(v) for k, v in sorted(by_transform.items())},
        "pass_rate_by_source_category": {k: _rate(v) for k, v in sorted(by_category.items())},
    }

    with (args.output_dir / "metamorphic_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with (args.output_dir / "metamorphic_predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in details:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

