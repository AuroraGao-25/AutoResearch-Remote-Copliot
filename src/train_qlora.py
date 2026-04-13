#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from absa.prompting import build_training_example


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ABSA QLoRA baseline (A1).")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--train-file", type=Path, required=True)
    p.add_argument("--eval-file", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--num-train-epochs", type=float, default=3.0)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--per-device-train-batch-size", type=int, default=2)
    p.add_argument("--per-device-eval-batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    return p.parse_args()


def _format_record(example: dict) -> dict:
    text = build_training_example(
        text=example["text"],
        aspect=example["aspect"],
        sentiment=example["sentiment"],
    )
    return {"text": text}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data_files = {"train": str(args.train_file)}
    if args.eval_file:
        data_files["validation"] = str(args.eval_file)
    dataset = load_dataset("json", data_files=data_files)
    dataset = dataset.map(_format_record)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(batch: dict) -> dict:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset["train"].column_names)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[x.strip() for x in args.target_modules.split(",") if x.strip()],
    )
    model = get_peft_model(model, lora_cfg)

    training_kwargs = dict(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if "validation" in tokenized else None,
        save_strategy="steps",
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to=[],
        seed=args.seed,
    )
    eval_key = "eval_strategy"
    if "eval_strategy" not in inspect.signature(TrainingArguments.__init__).parameters:
        eval_key = "evaluation_strategy"
    training_kwargs[eval_key] = "steps" if "validation" in tokenized else "no"
    train_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    config_out = args.output_dir / "run_config.json"
    with config_out.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)
    print(f"Saved adapter and config to {args.output_dir}")


if __name__ == "__main__":
    main()
