from __future__ import annotations

import json
import re


LABEL_SET = {"positive", "negative", "neutral"}

INSTRUCTION = (
    "You are an ABSA classifier. Given a review text and a target aspect, "
    "predict sentiment polarity toward that aspect only. "
    'Output strict JSON only in this format: {"aspect":"<aspect>","sentiment":"positive|negative|neutral"}.'
)


def build_prompt(text: str, aspect: str) -> str:
    return (
        f"{INSTRUCTION}\n"
        f"Text: {text}\n"
        f"Aspect: {aspect}\n"
        "Answer:"
    )


def build_training_example(text: str, aspect: str, sentiment: str) -> str:
    answer = json.dumps({"aspect": aspect, "sentiment": sentiment}, ensure_ascii=False)
    return f"{build_prompt(text, aspect)} {answer}"


def normalize_sentiment(label: str | None) -> str | None:
    if not label:
        return None
    v = label.strip().lower()
    if v in LABEL_SET:
        return v
    synonyms = {
        "pos": "positive",
        "neg": "negative",
        "neu": "neutral",
    }
    return synonyms.get(v)


def parse_json_sentiment(output_text: str) -> str | None:
    # Find the first JSON object in generated text.
    m = re.search(r"\{.*?\}", output_text, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return normalize_sentiment(obj.get("sentiment"))

