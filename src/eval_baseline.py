#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from absa.prompting import build_prompt, normalize_sentiment, parse_json_sentiment


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate ABSA baseline (A0/A1).")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--adapter-path", type=Path, default=None, help="QLoRA adapter dir for A1.")
    p.add_argument("--test-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=0, help="Optional cap for quick smoke runs.")
    return p.parse_args()


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _predict_one(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer, text: str, aspect: str, max_new_tokens: int
) -> str | None:
    prompt = build_prompt(text, aspect)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return parse_json_sentiment(decoded)


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

    rows = _read_jsonl(args.test_file)
    if args.limit > 0:
        rows = rows[: args.limit]

    y_true: list[str] = []
    y_pred: list[str] = []
    pred_rows: list[dict] = []

    for row in tqdm(rows, desc="Evaluating"):
        gold = normalize_sentiment(row["sentiment"])
        pred = _predict_one(
            model=model,
            tokenizer=tokenizer,
            text=row["text"],
            aspect=row["aspect"],
            max_new_tokens=args.max_new_tokens,
        )
        if pred is None:
            pred = "neutral"

        y_true.append(gold)
        y_pred.append(pred)
        pred_rows.append(
            {
                "id": row.get("id"),
                "domain": row.get("domain"),
                "text": row["text"],
                "aspect": row["aspect"],
                "gold": gold,
                "pred": pred,
            }
        )

    metrics = {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "model_name": args.model_name,
        "adapter_path": str(args.adapter_path) if args.adapter_path else None,
        "test_file": str(args.test_file),
    }

    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in pred_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

