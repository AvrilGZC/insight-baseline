from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


TOKEN_RE = re.compile(r"[A-Za-z]+(?:[-_/][A-Za-z]+)*|\d+(?:\.\d+)?%?")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?")
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"\'])")

LABEL_ALIASES = {
    "support": "supports",
    "supports": "supports",
    "supported": "supports",
    "entailment": "supports",
    "true": "supports",
    "refute": "refutes",
    "refutes": "refutes",
    "refuted": "refutes",
    "contradiction": "refutes",
    "false": "refutes",
    "nei": "nei",
    "not enough info": "nei",
    "not_enough_info": "nei",
    "unknown": "nei",
    "unverifiable": "nei",
}

COMPARATIVE_TOKENS = {
    "higher",
    "lower",
    "better",
    "worse",
    "larger",
    "smaller",
    "more",
    "less",
    "improves",
    "improved",
    "increase",
    "increases",
    "decrease",
    "decreases",
    "greater",
    "fewer",
    "than",
}
SUPERLATIVE_TOKENS = {
    "best",
    "worst",
    "highest",
    "lowest",
    "largest",
    "smallest",
    "top",
    "maximum",
    "minimum",
    "most",
    "least",
}
NEGATION_TOKENS = {"not", "no", "never", "none", "without", "cannot"}

NUMERIC_FEATURE_NAMES = [
    "claim_number_count",
    "claim_table_number_match_ratio",
    "claim_caption_number_match_ratio",
    "claim_header_overlap_ratio",
    "claim_cell_overlap_ratio",
    "claim_caption_overlap_ratio",
    "has_comparative_language",
    "has_superlative_language",
    "has_negation",
    "claim_token_length",
    "table_token_length",
]


@dataclass
class Example:
    example_id: str
    claim: str
    caption: str
    table_text: str
    claim_tokens: List[str]
    text_tokens: List[str]
    table_tokens: List[str]
    numeric_features: List[float]
    label: Optional[str]
    raw_record: Dict[str, object]


class Vocabulary:
    PAD = "<pad>"
    UNK = "<unk>"
    SEP = "<sep>"

    def __init__(self, token_to_id: Dict[str, int]) -> None:
        self.token_to_id = token_to_id
        self.id_to_token = {idx: token for token, idx in token_to_id.items()}
        self.pad_id = token_to_id[self.PAD]
        self.unk_id = token_to_id[self.UNK]
        self.sep_id = token_to_id[self.SEP]

    @classmethod
    def build(
        cls,
        token_sequences: Iterable[Sequence[str]],
        min_freq: int = 1,
        max_size: int = 50000,
    ) -> "Vocabulary":
        counter: Counter[str] = Counter()
        for tokens in token_sequences:
            counter.update(tokens)
        token_to_id = {
            cls.PAD: 0,
            cls.UNK: 1,
            cls.SEP: 2,
        }
        sorted_tokens = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        for token, freq in sorted_tokens:
            if freq < min_freq:
                continue
            if token in token_to_id:
                continue
            token_to_id[token] = len(token_to_id)
            if len(token_to_id) >= max_size:
                break
        return cls(token_to_id)

    def encode(self, tokens: Sequence[str], max_length: int) -> List[int]:
        token_ids = [self.token_to_id.get(token, self.unk_id) for token in tokens[:max_length]]
        if len(token_ids) < max_length:
            token_ids.extend([self.pad_id] * (max_length - len(token_ids)))
        return token_ids

    def to_dict(self) -> Dict[str, object]:
        return {"token_to_id": self.token_to_id}

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "Vocabulary":
        token_to_id = payload.get("token_to_id")
        if not isinstance(token_to_id, dict):
            raise ValueError("Invalid vocabulary payload.")
        return cls({str(token): int(idx) for token, idx in token_to_id.items()})


def load_examples(path: str | Path) -> List[Example]:
    records = load_records(path)
    examples: List[Example] = []
    for index, record in enumerate(records):
        example = normalize_record(record, index=index)
        if example is not None:
            examples.append(example)
    return examples


def expand_records_to_sentence_claims(path: str | Path) -> List[Dict[str, object]]:
    records = load_records(path)
    sentence_claims: List[Dict[str, object]] = []
    for index, record in enumerate(records):
        claims = extract_sentence_claims(record, parent_index=index)
        sentence_claims.extend(claims)
    return sentence_claims


def load_records(path: str | Path) -> List[Dict[str, object]]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        records: List[Dict[str, object]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(payload)
        return records

    payload = json.loads(text)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        if "data" in payload and isinstance(payload["data"], list):
            return [item for item in payload["data"] if isinstance(item, dict)]
        records: List[Dict[str, object]] = []
        for key, value in payload.items():
            if not isinstance(value, dict):
                continue
            enriched = dict(value)
            if not any(isinstance(enriched.get(field), str) and clean_text(str(enriched.get(field))) for field in ("id", "uid", "example_id")):
                enriched["id"] = str(key)
            records.append(enriched)
        return records
    raise ValueError(f"Unsupported data format in {path}")


def normalize_record(record: Dict[str, object], index: int = 0) -> Optional[Example]:
    claim = clean_text(
        first_string(
            record,
            "claim",
            "statement",
            "sentence",
            "hypothesis",
        )
    )
    if not claim:
        return None

    caption = clean_text(
        first_string(
            record,
            "table_caption",
            "caption",
            "tableCaption",
        )
    )
    columns = extract_columns(record)
    rows = extract_rows(record)
    table_text = linearize_table(caption=caption, columns=columns, rows=rows)

    claim_tokens = tokenize(claim)
    caption_tokens = tokenize(caption)
    table_tokens = tokenize(table_text)
    text_tokens = claim_tokens + [Vocabulary.SEP] + caption_tokens
    table_branch_tokens = claim_tokens + [Vocabulary.SEP] + table_tokens
    features = build_numeric_features(
        claim_tokens=claim_tokens,
        caption_tokens=caption_tokens,
        columns=columns,
        rows=rows,
        claim=claim,
        caption=caption,
        table_text=table_text,
    )
    label = normalize_label(first_string(record, "label", "verdict", "gold_label"))
    example_id = clean_text(first_string(record, "id", "uid", "example_id"))
    if not example_id:
        example_id = f"example_{index:05d}"
    return Example(
        example_id=example_id,
        claim=claim,
        caption=caption,
        table_text=table_text,
        claim_tokens=claim_tokens,
        text_tokens=text_tokens,
        table_tokens=table_branch_tokens,
        numeric_features=features,
        label=label,
        raw_record=record,
    )


def split_examples(
    examples: Sequence[Example],
    dev_ratio: float,
    seed: int,
) -> Tuple[List[Example], List[Example]]:
    if not 0.0 < dev_ratio < 1.0:
        raise ValueError("dev_ratio must be between 0 and 1.")
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    dev_size = max(1, int(round(len(shuffled) * dev_ratio)))
    dev_examples = shuffled[:dev_size]
    train_examples = shuffled[dev_size:]
    if not train_examples:
        raise ValueError("Need at least one training example after the split.")
    return train_examples, dev_examples


def extract_sentence_claims(record: Dict[str, object], parent_index: int = 0) -> List[Dict[str, object]]:
    parent_id = clean_text(first_string(record, "id", "uid", "example_id"))
    if not parent_id:
        parent_id = f"record_{parent_index:05d}"
    caption = clean_text(first_string(record, "table_caption", "caption", "tableCaption"))
    columns = extract_columns(record)
    rows = extract_rows(record)
    sentence_sources = extract_text_sentences(record)
    claims: List[Dict[str, object]] = []
    for sentence_index, sentence in enumerate(sentence_sources):
        claim_id = f"{parent_id}_sent_{sentence_index:03d}"
        claims.append(
            {
                "id": claim_id,
                "parent_id": parent_id,
                "sentence_index": sentence_index,
                "claim": sentence,
                "table_caption": caption,
                "table_column_names": columns,
                "table_content_values": rows,
                "source_text": sentence,
                "paper": first_string(record, "paper", "paper_title", "title"),
                "paper_id": first_string(record, "paper_id"),
                "table_id": first_string(record, "table_id"),
            }
        )
    return claims


def first_string(record: Dict[str, object], *keys: str) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str):
            return value
    table = record.get("table")
    if isinstance(table, dict):
        for key in keys:
            value = table.get(key)
            if isinstance(value, str):
                return value
    return ""


def extract_text_sentences(record: Dict[str, object]) -> List[str]:
    raw_sentences = record.get("sentences")
    if isinstance(raw_sentences, list):
        sentences = [clean_text(str(item)) for item in raw_sentences if clean_text(str(item))]
        if sentences:
            return sentences
    text = clean_text(
        first_string(
            record,
            "text",
            "description",
            "table_description",
            "caption_text",
        )
    )
    if not text:
        return []
    parts = SENTENCE_BOUNDARY_RE.split(text)
    sentences = [clean_text(part) for part in parts if clean_text(part)]
    return [sentence for sentence in sentences if len(sentence.split()) >= 3]


def extract_columns(record: Dict[str, object]) -> List[str]:
    for key in ("table_column_names", "column_headers", "columns"):
        value = record.get(key)
        extracted = flatten_list_field(value)
        if extracted:
            return extracted
    table = record.get("table")
    if isinstance(table, dict):
        for key in ("table_column_names", "column_headers", "columns"):
            extracted = flatten_list_field(table.get(key))
            if extracted:
                return extracted
    return []


def extract_rows(record: Dict[str, object]) -> List[List[str]]:
    for key in ("table_content_values", "contents", "rows", "table_rows"):
        rows = normalize_rows(record.get(key))
        if rows:
            return rows
    table = record.get("table")
    if isinstance(table, dict):
        for key in ("table_content_values", "contents", "rows", "table_rows"):
            rows = normalize_rows(table.get(key))
            if rows:
                return rows
    return []


def flatten_list_field(value: object) -> List[str]:
    if not isinstance(value, list):
        return []
    flattened: List[str] = []
    for item in value:
        if isinstance(item, str):
            flattened.append(clean_text(item))
        elif isinstance(item, list):
            parts = [clean_text(str(part)) for part in item if str(part).strip()]
            if parts:
                flattened.append(" / ".join(parts))
    return [item for item in flattened if item]


def normalize_rows(value: object) -> List[List[str]]:
    if not isinstance(value, list):
        return []
    rows: List[List[str]] = []
    for row in value:
        if isinstance(row, list):
            normalized = [clean_text(str(cell)) for cell in row]
            if any(normalized):
                rows.append(normalized)
        elif isinstance(row, dict):
            ordered = [clean_text(str(cell)) for _, cell in sorted(row.items())]
            if any(ordered):
                rows.append(ordered)
    return rows


def normalize_label(value: str) -> Optional[str]:
    canonical = clean_text(value).lower().replace("-", " ").replace("_", " ")
    if not canonical:
        return None
    return LABEL_ALIASES.get(canonical)


def linearize_table(caption: str, columns: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    parts: List[str] = []
    if caption:
        parts.append(f"caption: {caption}")
    if columns:
        parts.append("columns: " + " | ".join(columns))
    for idx, row in enumerate(rows, start=1):
        row_text = " | ".join(cell for cell in row if cell)
        if row_text:
            parts.append(f"row {idx}: {row_text}")
    return " </s> ".join(parts)


def tokenize(text: str) -> List[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def clean_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def build_numeric_features(
    claim_tokens: Sequence[str],
    caption_tokens: Sequence[str],
    columns: Sequence[str],
    rows: Sequence[Sequence[str]],
    claim: str,
    caption: str,
    table_text: str,
) -> List[float]:
    claim_numbers = NUMBER_RE.findall(claim)
    caption_numbers = set(NUMBER_RE.findall(caption))
    table_numbers = set(NUMBER_RE.findall(table_text))

    claim_token_set = set(claim_tokens)
    caption_token_set = set(caption_tokens)
    header_tokens = set(tokenize(" ".join(columns)))
    cell_tokens = set(tokenize(" ".join(" ".join(row) for row in rows)))

    claim_len = max(1, len(claim_tokens))
    table_len = max(1, len(tokenize(table_text)))
    claim_number_count = float(len(claim_numbers))
    table_number_match_ratio = (
        sum(1 for number in claim_numbers if number in table_numbers) / max(1, len(claim_numbers))
    )
    caption_number_match_ratio = (
        sum(1 for number in claim_numbers if number in caption_numbers) / max(1, len(claim_numbers))
    )
    claim_header_overlap_ratio = len(claim_token_set & header_tokens) / claim_len
    claim_cell_overlap_ratio = len(claim_token_set & cell_tokens) / claim_len
    claim_caption_overlap_ratio = len(claim_token_set & caption_token_set) / claim_len
    has_comparative_language = float(any(token in COMPARATIVE_TOKENS for token in claim_token_set))
    has_superlative_language = float(any(token in SUPERLATIVE_TOKENS for token in claim_token_set))
    has_negation = float(any(token in NEGATION_TOKENS for token in claim_token_set))
    claim_token_length = min(1.0, len(claim_tokens) / 40.0)
    table_token_length = min(1.0, math.log1p(table_len) / math.log(200.0))
    return [
        claim_number_count,
        table_number_match_ratio,
        caption_number_match_ratio,
        claim_header_overlap_ratio,
        claim_cell_overlap_ratio,
        claim_caption_overlap_ratio,
        has_comparative_language,
        has_superlative_language,
        has_negation,
        claim_token_length,
        table_token_length,
    ]
