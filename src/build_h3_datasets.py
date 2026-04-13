#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


NEGATION_RE = re.compile(r"\b(not|never|no|n't|hardly|barely)\b", re.IGNORECASE)
CONTRAST_RE = re.compile(r"\b(but|however|although|though|yet|whereas)\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build H3 size-matched targeted augmentation datasets.")
    p.add_argument("--input-train", type=Path, default=Path("data/semeval14/all_train.jsonl"))
    p.add_argument("--out-dir", type=Path, default=Path("data/h3"))
    p.add_argument("--add-per-condition", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _has_multi_aspect_signal(text: str) -> bool:
    lowered = text.lower()
    clause_markers = lowered.count(",") + lowered.count(";")
    conj_markers = sum(lowered.count(tok) for tok in [" and ", " but ", " while ", " although "])
    return clause_markers + conj_markers >= 2


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sample_additions(pool: list[dict], k: int, rng: random.Random) -> list[dict]:
    if not pool:
        return []
    if len(pool) >= k:
        return rng.sample(pool, k)
    # Size-matched target with replacement if pool is smaller.
    return [rng.choice(pool) for _ in range(k)]


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    rows = _read_jsonl(args.input_train)

    neg_pool = [r for r in rows if NEGATION_RE.search(r["text"] or "")]
    contrast_pool = [r for r in rows if CONTRAST_RE.search(r["text"] or "")]
    multi_pool = [r for r in rows if _has_multi_aspect_signal(r["text"] or "")]

    c1_add = _sample_additions(neg_pool, args.add_per_condition, rng)
    c2_add = _sample_additions(contrast_pool, args.add_per_condition, rng)
    c3_add = _sample_additions(multi_pool, args.add_per_condition, rng)

    # C4 stays size-matched by splitting additions across three categories.
    each = args.add_per_condition // 3
    rem = args.add_per_condition - (each * 3)
    c4_add = (
        _sample_additions(neg_pool, each + (1 if rem > 0 else 0), rng)
        + _sample_additions(contrast_pool, each + (1 if rem > 1 else 0), rng)
        + _sample_additions(multi_pool, each, rng)
    )

    outputs = {
        "c1_negation": rows + c1_add,
        "c2_contrast": rows + c2_add,
        "c3_multi_aspect": rows + c3_add,
        "c4_all": rows + c4_add,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, out_rows in outputs.items():
        _write_jsonl(args.out_dir / f"{name}_train.jsonl", out_rows)

    summary = {
        "base_size": len(rows),
        "add_per_condition": args.add_per_condition,
        "pool_sizes": {
            "negation": len(neg_pool),
            "contrast": len(contrast_pool),
            "multi_aspect": len(multi_pool),
        },
        "output_sizes": {name: len(out_rows) for name, out_rows in outputs.items()},
        "input_train": str(args.input_train),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
