#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


NEGATION_RE = re.compile(r"\b(?:not|n't|never|no|hardly|barely|without)\b", flags=re.IGNORECASE)
CONTRAST_RE = re.compile(r"\b(?:but|however|though|although|yet|while)\b", flags=re.IGNORECASE)
INTENSIFIER_RE = re.compile(
    r"\b(?:very|extremely|really|too|so|slightly|somewhat|quite|highly|barely)\b",
    flags=re.IGNORECASE,
)
EXPLICIT_OPINION_RE = re.compile(
    r"\b(?:good|great|excellent|awesome|amazing|bad|terrible|awful|poor|love|hate|like|dislike)\b",
    flags=re.IGNORECASE,
)


CATEGORY_PRIORITY = [
    "multi_aspect_conflict",
    "negation",
    "contrast",
    "intensifier",
    "implicit_sentiment",
    "neutral_irrelevant",
]


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _sentence_key(row_id: str) -> str:
    # id format from prep script: {domain}-{split}-{sentence_id}-{idx}
    return row_id.rsplit("-", 1)[0]


def _infer_categories(row: dict, sentence_rows: list[dict]) -> list[str]:
    text = row["text"]
    sentiment = row["sentiment"]
    cats: list[str] = []

    group_pols = {r["sentiment"] for r in sentence_rows}
    if {"positive", "negative"}.issubset(group_pols):
        cats.append("multi_aspect_conflict")
    if NEGATION_RE.search(text):
        cats.append("negation")
    if CONTRAST_RE.search(text):
        cats.append("contrast")
    if INTENSIFIER_RE.search(text):
        cats.append("intensifier")
    if sentiment == "neutral":
        cats.append("neutral_irrelevant")
    if sentiment in {"positive", "negative"} and not EXPLICIT_OPINION_RE.search(text):
        cats.append("implicit_sentiment")

    if not cats:
        cats.append("implicit_sentiment")
    return cats


def _pick_primary(cats: list[str]) -> str:
    for c in CATEGORY_PRIORITY:
        if c in cats:
            return c
    return cats[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ABSA-RTS v0.1 from prepared SemEval JSONL.")
    parser.add_argument("--input-jsonl", type=Path, default=Path("data/semeval14/all_test.jsonl"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("data/absa_rts/absa_rts_v0.1.jsonl"))
    parser.add_argument("--summary-json", type=Path, default=Path("data/absa_rts/summary_v0.1.json"))
    parser.add_argument("--per-category-per-domain", type=int, default=100)
    parser.add_argument(
        "--categories",
        default="negation,contrast,intensifier,multi_aspect_conflict,implicit_sentiment,neutral_irrelevant",
    )
    args = parser.parse_args()

    allowed_categories = {x.strip() for x in args.categories.split(",") if x.strip()}
    rows = _read_jsonl(args.input_jsonl)

    by_sentence: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_sentence[_sentence_key(row["id"])].append(row)

    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        sentence_rows = by_sentence[_sentence_key(row["id"])]
        all_cats = _infer_categories(row, sentence_rows)
        all_cats = [c for c in all_cats if c in allowed_categories]
        if not all_cats:
            continue
        primary = _pick_primary(all_cats)
        enriched = dict(row)
        enriched["rts_categories"] = all_cats
        enriched["rts_primary_category"] = primary
        buckets[(row["domain"], primary)].append(enriched)

    selected: list[dict] = []
    for key, items in sorted(buckets.items()):
        selected.extend(items[: args.per_category_per_domain])

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input_rows": len(rows),
        "selected_rows": len(selected),
        "per_category_per_domain": args.per_category_per_domain,
        "counts": dict(
            Counter(f"{r['domain']}::{r['rts_primary_category']}" for r in selected)
        ),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

