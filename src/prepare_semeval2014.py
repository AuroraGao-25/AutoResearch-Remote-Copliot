#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter
from pathlib import Path

from absa.io_semeval import parse_semeval_xml, write_jsonl


def _pick_file(root: Path, domain: str, split: str, input_format: str) -> Path:
    split_token = "Train" if split == "train" else "Test"
    ext = "xml" if input_format == "xml" else "csv"
    patterns = [
        f"*{domain}*{split_token}*.{ext}",
        f"*{domain.capitalize()}*{split_token}*.{ext}",
        f"*{domain.lower()}*{split_token.lower()}*.{ext}",
    ]
    for pat in patterns:
        hits = sorted(root.rglob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"Could not find {ext.upper()} for domain={domain}, split={split} under {root}"
    )


def _stats(rows: list[dict]) -> dict:
    return {
        "count": len(rows),
        "sentiment_dist": dict(Counter(r["sentiment"] for r in rows)),
    }


def _parse_aspect_terms(value: str) -> list[dict]:
    if value is None:
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    if isinstance(parsed, list):
        return [x for x in parsed if isinstance(x, dict)]
    return []


def parse_semeval_csv(
    csv_path: Path, domain: str, split: str, drop_conflict: bool = True
) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentence_id = str(row.get("sentenceId", "unknown")).strip()
            text = (row.get("raw_text") or "").strip()
            if not text:
                continue
            terms = _parse_aspect_terms(row.get("aspectTerms", ""))
            for idx, term_item in enumerate(terms):
                term = (term_item.get("term") or "").strip()
                polarity = str(term_item.get("polarity", "")).strip().lower()
                if not term or term == "noaspectterm":
                    continue
                if polarity in {"", "none"}:
                    continue
                if drop_conflict and polarity == "conflict":
                    continue
                if polarity not in {"positive", "negative", "neutral", "conflict"}:
                    continue

                rows.append(
                    {
                        "id": f"{domain}-{split}-{sentence_id}-{idx}",
                        "domain": domain,
                        "split": split,
                        "text": text,
                        "aspect": term,
                        "sentiment": polarity,
                        "from": None,
                        "to": None,
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SemEval-2014 ABSA ASC JSONL files.")
    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="Directory containing SemEval files (XML or CSV).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/semeval14"),
        help="Output directory for JSONL files.",
    )
    parser.add_argument(
        "--drop-conflict",
        action="store_true",
        help="Drop 'conflict' labels.",
    )
    parser.add_argument(
        "--input-format",
        choices=["xml", "csv"],
        default="xml",
        help="Input file format under raw-root.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = {"train": [], "test": []}
    summary: dict[str, dict] = {}

    for domain in ("Restaurants", "Laptops"):
        domain_key = domain.lower()
        for split in ("train", "test"):
            source_path = _pick_file(args.raw_root, domain, split, args.input_format)
            if args.input_format == "xml":
                parsed_objs = parse_semeval_xml(
                    xml_path=source_path,
                    domain=domain_key,
                    split=split,
                    drop_conflict=args.drop_conflict,
                )
                parsed = [x.to_json() for x in parsed_objs]
            else:
                parsed = parse_semeval_csv(
                    csv_path=source_path,
                    domain=domain_key,
                    split=split,
                    drop_conflict=args.drop_conflict,
                )
            out_path = args.out_dir / f"{domain_key}_{split}.jsonl"
            if args.input_format == "xml":
                # Parsed as objects in XML path.
                write_jsonl(out_path, parsed_objs)
            else:
                with out_path.open("w", encoding="utf-8") as f:
                    for row in parsed:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")

            all_rows[split].extend(parsed)
            summary[f"{domain_key}_{split}"] = _stats(parsed)
            summary[f"{domain_key}_{split}"]["source_file"] = str(source_path)

    for split in ("train", "test"):
        out_path = args.out_dir / f"all_{split}.jsonl"
        count = 0
        with out_path.open("w", encoding="utf-8") as f:
            for row in all_rows[split]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
        summary[f"all_{split}"] = _stats(all_rows[split])
        summary[f"all_{split}"]["written_rows"] = count

    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
