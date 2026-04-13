#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _append_clause(text: str) -> str:
    clause = " By the way, I checked the weather this morning."
    if text.endswith("."):
        return text + clause
    return text + "." + clause


def _neutral_reorder(text: str) -> str:
    if "," not in text:
        return text
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) < 2:
        return text
    return ", ".join(parts[1:] + parts[:1])


def _negation_flip(aspect: str, sentiment: str) -> tuple[str, str] | None:
    if sentiment == "positive":
        return f"The {aspect} is not good.", "negative"
    if sentiment == "negative":
        return f"The {aspect} is good.", "positive"
    return None


def _contrast_flip(aspect: str, sentiment: str) -> tuple[str, str] | None:
    if sentiment == "positive":
        return f"The {aspect} seemed okay at first, but it is actually terrible.", "negative"
    if sentiment == "negative":
        return f"The {aspect} looked bad at first, but it is actually excellent.", "positive"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic metamorphic ABSA suite.")
    parser.add_argument("--input-rts", type=Path, default=Path("data/absa_rts/absa_rts_v0.1.jsonl"))
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("data/absa_rts/metamorphic_suite_v0.1.jsonl"),
    )
    args = parser.parse_args()

    rows = _read_jsonl(args.input_rts)
    transformed: list[dict] = []

    for row in rows:
        base = {
            "source_id": row["id"],
            "domain": row["domain"],
            "aspect": row["aspect"],
            "source_sentiment": row["sentiment"],
            "source_primary_category": row.get("rts_primary_category"),
        }

        transformed.append(
            {
                **base,
                "transform_type": "invariance_irrelevant_clause",
                "expected_sentiment": row["sentiment"],
                "text": _append_clause(row["text"]),
            }
        )
        transformed.append(
            {
                **base,
                "transform_type": "invariance_reorder_context",
                "expected_sentiment": row["sentiment"],
                "text": _neutral_reorder(row["text"]),
            }
        )

        neg = _negation_flip(row["aspect"], row["sentiment"])
        if neg is not None:
            t, label = neg
            transformed.append(
                {
                    **base,
                    "transform_type": "flip_negation_template",
                    "expected_sentiment": label,
                    "text": t,
                }
            )

        con = _contrast_flip(row["aspect"], row["sentiment"])
        if con is not None:
            t, label = con
            transformed.append(
                {
                    **base,
                    "transform_type": "flip_contrast_template",
                    "expected_sentiment": label,
                    "text": t,
                }
            )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for row in transformed:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "input_rows": len(rows),
                "transformed_rows": len(transformed),
                "output_jsonl": str(args.output_jsonl),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

