from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


VALID_SENTIMENTS = {"positive", "negative", "neutral", "conflict"}


@dataclass(frozen=True)
class AspectSample:
    sample_id: str
    domain: str
    split: str
    text: str
    aspect: str
    sentiment: str
    char_from: int | None
    char_to: int | None

    def to_json(self) -> dict:
        return {
            "id": self.sample_id,
            "domain": self.domain,
            "split": self.split,
            "text": self.text,
            "aspect": self.aspect,
            "sentiment": self.sentiment,
            "from": self.char_from,
            "to": self.char_to,
        }


def _safe_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def parse_semeval_xml(
    xml_path: Path, domain: str, split: str, drop_conflict: bool = True
) -> list[AspectSample]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    rows: list[AspectSample] = []

    for sent in root.iter("sentence"):
        sent_id = sent.attrib.get("id", "unknown")
        text_node = sent.find("text")
        if text_node is None or text_node.text is None:
            continue
        text = text_node.text.strip()

        aspect_nodes = []
        terms = sent.find("aspectTerms")
        if terms is not None:
            aspect_nodes.extend(terms.findall("aspectTerm"))
        opinions = sent.find("Opinions")
        if opinions is not None:
            aspect_nodes.extend(opinions.findall("Opinion"))

        for idx, node in enumerate(aspect_nodes):
            aspect = node.attrib.get("term") or node.attrib.get("target")
            sentiment = node.attrib.get("polarity")
            if not aspect or not sentiment:
                continue
            sentiment = sentiment.lower().strip()
            if sentiment not in VALID_SENTIMENTS:
                continue
            if drop_conflict and sentiment == "conflict":
                continue
            if aspect.strip().upper() == "NULL":
                continue

            rows.append(
                AspectSample(
                    sample_id=f"{domain}-{split}-{sent_id}-{idx}",
                    domain=domain,
                    split=split,
                    text=text,
                    aspect=aspect.strip(),
                    sentiment=sentiment,
                    char_from=_safe_int(node.attrib.get("from")),
                    char_to=_safe_int(node.attrib.get("to")),
                )
            )

    return rows


def write_jsonl(path: Path, rows: Iterable[AspectSample]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_json(), ensure_ascii=False) + "\n")
            count += 1
    return count

