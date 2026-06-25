from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


TEST_FILE_CANDIDATES = {
    "test": ("test-CL.json", "test-Other.json"),
    "test-cl": ("test-CL.json",),
    "cl": ("test-CL.json",),
    "test-other": ("test-Other.json",),
    "other": ("test-Other.json",),
}

TYPE_PRIORITY = {
    "comparison": 5.0,
    "extremum": 4.0,
    "aggregate": 3.5,
    "trend": 3.0,
}

CANDIDATE_LIST_KEYS = (
    "insights",
    "reranked_top",
    "text_aware_top",
    "bidirectional_top",
    "table_only_top",
    "qtp_top_insights",
    "rule_top_insights",
    "pure_top_insights",
)


@dataclass(frozen=True)
class EvidenceCell:
    row_index: int
    column_index: int
    row_header: str
    column_header: str
    value: str
    numeric_value: float | None = None


@dataclass
class Insight:
    type: str
    subtype: str
    claim: str
    evidence: list[EvidenceCell]
    operation: str | None = None
    subject: str | None = None
    condition: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    score: dict[str, float] = field(default_factory=dict)
    source: str = "rule"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evidence"] = [asdict(cell) for cell in self.evidence]
        return payload


@dataclass
class SciGenRecord:
    record_id: str
    domain: str
    paper_id: str
    paper: str
    caption: str
    description: str
    column_headers: list[str]
    rows: list[list[str]]


def load_scigen_records(data_dir: str | Path, split: str = "test", limit: int | None = None) -> list[SciGenRecord]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"SciGen data directory does not exist: {root}")

    files: list[Path] = []
    if root.is_file():
        files = [root]
    else:
        for name in TEST_FILE_CANDIDATES.get(split.lower(), (f"{split}.json",)):
            files.extend(root.rglob(name))

    if not files:
        known = sorted(str(path.relative_to(root)) for path in root.rglob("*.json"))[:30]
        raise FileNotFoundError(f"No SciGen JSON files found for split={split!r}. Known JSON files: {known}")

    records: list[SciGenRecord] = []
    for file_path in sorted(set(files)):
        for idx, item in enumerate(_load_json_records(file_path)):
            records.append(_normalize_record(item, file_path.stem, idx))
            if limit and len(records) >= limit:
                return records
    return records


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if all(isinstance(value, dict) for value in data.values()):
            records = []
            for key, value in data.items():
                record = dict(value)
                record.setdefault("__record_key", str(key))
                records.append(record)
            return records
        for key in ("data", "records", "examples"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"Unsupported JSON shape in {path}")


def _normalize_record(item: dict[str, Any], source_name: str, index: int) -> SciGenRecord:
    columns = _flatten_headers(item.get("table_column_names") or item.get("columns") or item.get("column_headers"))
    rows = _normalize_rows(item.get("table_content_values") or item.get("rows") or item.get("table_content"))
    if rows:
        width = max([len(columns), *(len(row) for row in rows)])
        columns = _pad(columns, width)
        rows = [_pad(row, width) for row in rows]

    paper_id = str(item.get("paper_id") or "")
    table_id = str(item.get("__record_key") or item.get("table_id") or index)
    return SciGenRecord(
        record_id=f"{source_name}:{paper_id}:{table_id}:{index}",
        domain=_domain_from_source(source_name),
        paper_id=paper_id,
        paper=str(item.get("paper") or ""),
        caption=str(item.get("table_caption") or item.get("caption") or ""),
        description=str(item.get("text") or item.get("description") or ""),
        column_headers=columns,
        rows=rows,
    )


def _domain_from_source(source_name: str) -> str:
    lowered = source_name.lower()
    if "other" in lowered:
        return "Other"
    if "cl" in lowered:
        return "C&L"
    return source_name


def _flatten_headers(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        headers = []
        for value in raw:
            if isinstance(value, list):
                headers.append(" / ".join(str(part) for part in value if str(part).strip()))
            else:
                headers.append(str(value))
        return headers
    return [str(raw)]


def _normalize_rows(raw: Any) -> list[list[str]]:
    if not isinstance(raw, list):
        return []
    rows = []
    for row in raw:
        if isinstance(row, list):
            rows.append([_clean_cell(cell) for cell in row])
        elif isinstance(row, dict):
            rows.append([_clean_cell(value) for value in row.values()])
    return rows


def _clean_cell(value: Any) -> str:
    text = str(value or "")
    return re.sub(r"\s+", " ", text.replace("[BOLD]", "")).strip()


def _pad(values: list[str], width: int) -> list[str]:
    return [str(values[idx]) if idx < len(values) else "" for idx in range(width)]


def extract_candidate_insights(record: SciGenRecord, max_candidates: int | None = None) -> list[Insight]:
    numeric = _numeric_cells(record)
    insights: list[Insight] = []
    insights.extend(_column_extrema(record, numeric))
    insights.extend(_row_mean_extrema(record, numeric))
    insights.extend(_row_outperformance(record, numeric))
    insights.extend(_row_trends(record, numeric))
    ranked = _rank_table_only(_dedupe(insights))
    return ranked[:max_candidates] if max_candidates else ranked


def load_external_candidate_map(candidate_dir: str | Path) -> tuple[dict[str, list[Insight]], dict[str, Any]]:
    root = Path(candidate_dir)
    if not root.exists():
        raise FileNotFoundError(f"Candidate directory does not exist: {root}")

    candidate_map: dict[str, list[Insight]] = {}
    files = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.suffix.lower() in {".jsonl", ".json"}
        and not path.name.endswith(".summary.json")
    ]
    stats: dict[str, Any] = {
        "candidate_dir": str(root),
        "files": len(files),
        "rows": 0,
        "records_with_candidates": 0,
        "candidates": 0,
        "candidate_keys": {},
    }

    for path in files:
        for row in _load_candidate_rows(path):
            stats["rows"] += 1
            record_id = str(row.get("record_id") or "")
            if not record_id:
                continue
            for key in CANDIDATE_LIST_KEYS:
                raw_items = row.get(key)
                if not isinstance(raw_items, list):
                    continue
                stats["candidate_keys"][key] = int(stats["candidate_keys"].get(key, 0)) + len(raw_items)
                for idx, item in enumerate(raw_items):
                    insight = _external_item_to_insight(item, source_hint=f"{path.stem}:{key}", index=idx)
                    if insight is None:
                        continue
                    candidate_map.setdefault(record_id, []).append(insight)
                    stats["candidates"] += 1

    for record_id, insights in list(candidate_map.items()):
        candidate_map[record_id] = _dedupe(insights)
    stats["records_with_candidates"] = len(candidate_map)
    stats["deduped_candidates"] = sum(len(insights) for insights in candidate_map.values())
    return candidate_map, stats


def _load_candidate_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in ("data", "records", "examples", "rows"):
            if isinstance(data.get(key), list):
                return [row for row in data[key] if isinstance(row, dict)]
        if "record_id" in data:
            return [data]
        if all(isinstance(value, dict) for value in data.values()):
            rows = []
            for key, value in data.items():
                row = dict(value)
                row.setdefault("record_id", key)
                rows.append(row)
            return rows
    return []


def _external_item_to_insight(item: Any, source_hint: str, index: int) -> Insight | None:
    if not isinstance(item, dict):
        return None
    claim = str(item.get("claim") or item.get("insight") or item.get("text") or "").strip()
    if not claim:
        return None
    score = item.get("score") if isinstance(item.get("score"), dict) else {}
    if "final_score" in item:
        score = {**score, "external_final_score": _safe_float(item.get("final_score"))}
    if "confidence" in item:
        score = {**score, "confidence": _safe_float(item.get("confidence"))}
    details = item.get("details") if isinstance(item.get("details"), dict) else {}
    details = {
        **details,
        "external_original_index": index,
        "external_source_hint": source_hint,
    }
    return Insight(
        type=str(item.get("type") or "external"),
        subtype=str(item.get("subtype") or "candidate"),
        claim=claim,
        evidence=_external_evidence(item.get("evidence")),
        operation=item.get("operation") if isinstance(item.get("operation"), str) else None,
        subject=item.get("subject") if isinstance(item.get("subject"), str) else None,
        condition=item.get("condition") if isinstance(item.get("condition"), str) else None,
        details=details,
        score=score,
        source=str(item.get("source") or source_hint),
    )


def _external_evidence(raw: Any) -> list[EvidenceCell]:
    if not isinstance(raw, list):
        return []
    cells = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        cells.append(
            EvidenceCell(
                row_index=int(_safe_float(item.get("row_index"), default=-1)),
                column_index=int(_safe_float(item.get("column_index"), default=-1)),
                row_header=str(item.get("row_header") or ""),
                column_header=str(item.get("column_header") or ""),
                value=str(item.get("value") or ""),
                numeric_value=_optional_float(item.get("numeric_value")),
            )
        )
    return cells


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    parsed = _optional_float(value)
    return default if parsed is None else parsed


def _numeric_cells(record: SciGenRecord) -> dict[tuple[int, int], float]:
    values: dict[tuple[int, int], float] = {}
    for row_idx, row in enumerate(record.rows):
        for col_idx, cell in enumerate(row):
            parsed = _parse_number(cell)
            if parsed is not None:
                values[(row_idx, col_idx)] = parsed
    return values


def _column_extrema(record: SciGenRecord, numeric: dict[tuple[int, int], float]) -> list[Insight]:
    insights: list[Insight] = []
    for col_idx, column in enumerate(record.column_headers):
        column_cells = [(row_idx, value) for (row_idx, c_idx), value in numeric.items() if c_idx == col_idx]
        if len(column_cells) < 2:
            continue
        max_row, max_value = max(column_cells, key=lambda item: item[1])
        min_row, min_value = min(column_cells, key=lambda item: item[1])
        if math.isclose(max_value, min_value):
            continue
        spread = max_value - min_value
        insights.append(
            Insight(
                type="extremum",
                subtype="metric_best",
                claim=f"{_row_label(record, max_row)} has the highest {column} value ({max_value:g}).",
                evidence=[_cell(record, max_row, col_idx, max_value)],
                operation="argmax",
                subject=_row_label(record, max_row),
                condition=column,
                details={"value": max_value, "spread_to_lowest": round(spread, 4)},
                score={"support": 1.0, "importance": abs(spread)},
            )
        )
        insights.append(
            Insight(
                type="extremum",
                subtype="metric_lowest",
                claim=f"{_row_label(record, min_row)} has the lowest {column} value ({min_value:g}).",
                evidence=[_cell(record, min_row, col_idx, min_value)],
                operation="argmin",
                subject=_row_label(record, min_row),
                condition=column,
                details={"value": min_value, "spread_to_highest": round(spread, 4)},
                score={"support": 1.0, "importance": abs(spread) * 0.6},
            )
        )
    return insights


def _row_mean_extrema(record: SciGenRecord, numeric: dict[tuple[int, int], float]) -> list[Insight]:
    row_values = _row_numeric_values(numeric)
    candidates = [(row_idx, values) for row_idx, values in row_values.items() if len(values) >= 2]
    if len(candidates) < 2:
        return []
    means = [(row_idx, statistics.fmean(value for _, value in values)) for row_idx, values in candidates]
    max_row, max_mean = max(means, key=lambda item: item[1])
    min_row, min_mean = min(means, key=lambda item: item[1])
    if math.isclose(max_mean, min_mean):
        return []
    return [
        Insight(
            type="aggregate",
            subtype="highest_average",
            claim=f"{_row_label(record, max_row)} has the highest average numeric value across reported metrics ({max_mean:.4g}).",
            evidence=[_cell(record, max_row, col_idx, value) for col_idx, value in row_values[max_row]],
            operation="mean_argmax",
            subject=_row_label(record, max_row),
            condition="all numeric columns",
            score={"support": 1.0, "importance": abs(max_mean - min_mean)},
        )
    ]


def _row_outperformance(record: SciGenRecord, numeric: dict[tuple[int, int], float]) -> list[Insight]:
    row_values = _row_numeric_values(numeric)
    insights: list[Insight] = []
    for left in sorted(row_values):
        for right in sorted(row_values):
            if left == right:
                continue
            right_by_col = dict(row_values[right])
            shared = [(col_idx, lv, right_by_col[col_idx]) for col_idx, lv in row_values[left] if col_idx in right_by_col]
            if len(shared) < 2:
                continue
            wins = [(col_idx, lv, rv) for col_idx, lv, rv in shared if lv > rv]
            if len(wins) / len(shared) < 0.7:
                continue
            avg_diff = statistics.fmean(lv - rv for _, lv, rv in shared)
            if avg_diff <= 0:
                continue
            evidence = []
            for col_idx, left_value, right_value in wins[:5]:
                evidence.append(_cell(record, left, col_idx, left_value))
                evidence.append(_cell(record, right, col_idx, right_value))
            insights.append(
                Insight(
                    type="comparison",
                    subtype="outperformance",
                    claim=(
                        f"{_row_label(record, left)} outperforms {_row_label(record, right)} "
                        f"on {len(wins)}/{len(shared)} shared numeric columns, with an average difference of {avg_diff:.4g}."
                    ),
                    evidence=evidence,
                    operation="majority_difference",
                    subject=f"{_row_label(record, left)} vs {_row_label(record, right)}",
                    condition="shared numeric columns",
                    details={"wins": len(wins), "total": len(shared), "average_difference": round(avg_diff, 4)},
                    score={"support": round(len(wins) / len(shared), 4), "importance": abs(avg_diff)},
                )
            )
    return insights


def _row_trends(record: SciGenRecord, numeric: dict[tuple[int, int], float]) -> list[Insight]:
    if not _headers_look_ordered(record.column_headers):
        return []
    insights = []
    for row_idx, values in _row_numeric_values(numeric).items():
        if len(values) < 3:
            continue
        ordered = sorted(values)
        first_col, first = ordered[0]
        last_col, last = ordered[-1]
        delta = last - first
        if math.isclose(delta, 0.0):
            continue
        direction = "increases" if delta > 0 else "decreases"
        insights.append(
            Insight(
                type="trend",
                subtype="ordered_trend",
                claim=(
                    f"{_row_label(record, row_idx)} {direction} from {record.column_headers[first_col]} "
                    f"to {record.column_headers[last_col]} ({first:g} to {last:g})."
                ),
                evidence=[_cell(record, row_idx, first_col, first), _cell(record, row_idx, last_col, last)],
                operation="difference",
                subject=_row_label(record, row_idx),
                condition="ordered columns",
                details={"change": round(delta, 4), "direction": "up" if delta > 0 else "down"},
                score={"support": 1.0, "importance": abs(delta)},
            )
        )
    return insights


def _row_numeric_values(numeric: dict[tuple[int, int], float]) -> dict[int, list[tuple[int, float]]]:
    values: dict[int, list[tuple[int, float]]] = {}
    for (row_idx, col_idx), value in numeric.items():
        values.setdefault(row_idx, []).append((col_idx, value))
    return values


def _headers_look_ordered(headers: list[str]) -> bool:
    parsed = [_parse_number(header) for header in headers]
    present = [value for value in parsed if value is not None]
    return len(present) >= max(3, len(headers) // 2)


def _cell(record: SciGenRecord, row_idx: int, col_idx: int, numeric_value: float) -> EvidenceCell:
    row = record.rows[row_idx] if row_idx < len(record.rows) else []
    return EvidenceCell(
        row_index=row_idx,
        column_index=col_idx,
        row_header=_row_label(record, row_idx),
        column_header=record.column_headers[col_idx] if col_idx < len(record.column_headers) else f"col_{col_idx}",
        value=row[col_idx] if col_idx < len(row) else "",
        numeric_value=numeric_value,
    )


def _row_label(record: SciGenRecord, row_idx: int) -> str:
    row = record.rows[row_idx] if row_idx < len(record.rows) else []
    label_cells = [cell for cell in row[:3] if _looks_like_label(cell)]
    if label_cells:
        return " / ".join(label_cells)
    return f"row {row_idx}"


def _parse_number(value: str) -> float | None:
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    if re.search(r"[A-Za-z]", text):
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _looks_like_label(value: str) -> bool:
    text = str(value).strip()
    if not text or text in {"-", "--", "–", "—", "[EMPTY]"}:
        return False
    return bool(re.search(r"[A-Za-z]", text))


def _rank_table_only(insights: Sequence[Insight]) -> list[Insight]:
    return sorted(insights, key=_table_rank_key, reverse=True)


def _table_rank_key(insight: Insight) -> tuple[float, float, float]:
    score = insight.score or {}
    importance = min(float(score.get("importance", 0.0)), 1000.0)
    support = float(score.get("support", 1.0))
    return (TYPE_PRIORITY.get(insight.type, 1.0), support, importance)


def _dedupe(insights: Sequence[Insight]) -> list[Insight]:
    seen = set()
    result = []
    for insight in insights:
        key = (insight.type, insight.subtype, insight.claim.lower())
        if key in seen:
            continue
        seen.add(key)
        result.append(insight)
    return result


def rerank_text_aware(
    insights: Sequence[Insight],
    description: str,
    table_weight: float,
    text_weight: float,
    evidence_threshold: float,
) -> list[dict[str, Any]]:
    if not insights:
        return []

    sentences, similarities = _claim_sentence_similarities(insights, description)

    total = len(insights)
    scored = []
    for idx, insight in enumerate(insights):
        row = similarities[idx]
        best_idx = int(row.argmax()) if len(row) else 0
        text_support = float(row[best_idx]) if len(row) else 0.0
        evidence_count = int((row >= evidence_threshold).sum()) if len(row) else 0
        table_rank_score = (total - idx) / total
        final_score = table_weight * table_rank_score + text_weight * text_support
        payload = insight.to_dict()
        payload["rerank"] = {
            "original_rank": idx + 1,
            "table_rank_score": round(table_rank_score, 6),
            "text_support": round(text_support, 6),
            "best_sentence": sentences[best_idx] if sentences else "",
            "evidence_count": evidence_count,
            "final_score": round(final_score, 6),
        }
        scored.append(payload)

    return sorted(scored, key=lambda item: item["rerank"]["final_score"], reverse=True)


def rerank_bidirectional_salience(
    insights: Sequence[Insight],
    description: str,
    table_weight: float,
    candidate_to_text_weight: float,
    text_to_candidate_weight: float,
    evidence_threshold: float,
) -> list[dict[str, Any]]:
    if not insights:
        return []

    sentences, similarities = _claim_sentence_similarities(insights, description)
    total = len(insights)
    sentence_count = max(len(sentences), 1)

    sentence_best_candidate = similarities.argmax(axis=0) if len(sentences) else []
    sentence_best_scores = similarities.max(axis=0) if len(sentences) else []
    sentence_credit = [0.0 for _ in insights]
    sentence_coverage_count = [0 for _ in insights]
    sentence_weight_total = float(sentence_best_scores.sum()) or 1.0

    for sentence_idx, candidate_idx in enumerate(sentence_best_candidate):
        score = float(sentence_best_scores[sentence_idx])
        sentence_credit[int(candidate_idx)] += score
        if score >= evidence_threshold:
            sentence_coverage_count[int(candidate_idx)] += 1

    scored = []
    for idx, insight in enumerate(insights):
        row = similarities[idx]
        best_idx = int(row.argmax()) if len(row) else 0
        candidate_to_text = float(row[best_idx]) if len(row) else 0.0
        evidence_count = int((row >= evidence_threshold).sum()) if len(row) else 0
        table_rank_score = (total - idx) / total
        text_to_candidate = sentence_credit[idx] / sentence_weight_total
        sentence_coverage = sentence_coverage_count[idx] / sentence_count
        final_score = (
            table_weight * table_rank_score
            + candidate_to_text_weight * candidate_to_text
            + text_to_candidate_weight * text_to_candidate
        )
        payload = insight.to_dict()
        payload["rerank"] = {
            "mode": "bidirectional",
            "original_rank": idx + 1,
            "table_rank_score": round(table_rank_score, 6),
            "candidate_to_text_support": round(candidate_to_text, 6),
            "text_support": round(candidate_to_text, 6),
            "text_to_candidate_salience": round(text_to_candidate, 6),
            "text_to_candidate_coverage": round(sentence_coverage, 6),
            "best_sentence": sentences[best_idx] if sentences else "",
            "evidence_count": evidence_count,
            "covered_sentence_count": sentence_coverage_count[idx],
            "final_score": round(final_score, 6),
        }
        scored.append(payload)

    return sorted(scored, key=lambda item: item["rerank"]["final_score"], reverse=True)


def _claim_sentence_similarities(insights: Sequence[Insight], description: str):
    sentences = split_description(description)
    if not sentences:
        sentences = [description]

    claims = [insight.claim for insight in insights]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, token_pattern=r"(?u)\b[\w.+-]+\b")
    matrix = vectorizer.fit_transform([*claims, *sentences])
    claim_matrix = matrix[: len(claims)]
    sentence_matrix = matrix[len(claims) :]
    return sentences, cosine_similarity(claim_matrix, sentence_matrix)


def split_description(description: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", description.replace("[CONTINUE]", ". ")).strip()
    if not normalized:
        return []
    pieces = re.split(r"(?<=[.!?])\s+", normalized)
    sentences = [piece.strip(" .") for piece in pieces if piece.strip(" .")]
    return sentences or [normalized]


def run_baseline(
    data_dir: str,
    split: str,
    output: str,
    limit: int | None,
    top_k: int,
    max_candidates: int | None,
    candidate_source: str,
    candidate_dir: str | None,
    rerank_mode: str,
    table_weight: float,
    text_weight: float,
    candidate_to_text_weight: float,
    text_to_candidate_weight: float,
    evidence_threshold: float,
) -> dict[str, Any]:
    records = load_scigen_records(data_dir, split=split, limit=limit)
    external_candidate_map: dict[str, list[Insight]] = {}
    external_candidate_stats: dict[str, Any] = {}
    if candidate_dir:
        external_candidate_map, external_candidate_stats = load_external_candidate_map(candidate_dir)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    totals = {
        "records": 0,
        "candidate_count": 0,
        "internal_candidate_count": 0,
        "external_candidate_count": 0,
        "records_without_candidates": 0,
        "table_only_avg_text_support": 0.0,
        "reranked_avg_text_support": 0.0,
        "reranked_avg_text_to_candidate_salience": 0.0,
        "reranked_avg_text_to_candidate_coverage": 0.0,
        "top1_changed": 0,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            internal_candidates = extract_candidate_insights(record, max_candidates=None)
            external_candidates = external_candidate_map.get(record.record_id, [])
            if candidate_source == "external":
                raw_candidates = list(external_candidates)
            elif candidate_source == "combined":
                raw_candidates = [*internal_candidates, *external_candidates]
            else:
                raw_candidates = list(internal_candidates)

            candidates = _rank_table_only(_dedupe(raw_candidates))
            if max_candidates:
                candidates = candidates[:max_candidates]

            if rerank_mode == "bidirectional":
                reranked = rerank_bidirectional_salience(
                    candidates,
                    record.description,
                    table_weight=table_weight,
                    candidate_to_text_weight=candidate_to_text_weight,
                    text_to_candidate_weight=text_to_candidate_weight,
                    evidence_threshold=evidence_threshold,
                )
            else:
                reranked = rerank_text_aware(
                    candidates,
                    record.description,
                    table_weight=table_weight,
                    text_weight=text_weight,
                    evidence_threshold=evidence_threshold,
                )
            table_only_payload = rerank_text_aware(
                candidates,
                record.description,
                table_weight=1.0,
                text_weight=0.0,
                evidence_threshold=evidence_threshold,
            )

            table_only_top = table_only_payload[:top_k]
            reranked_top = reranked[:top_k]
            table_avg = _avg_text_support(table_only_top)
            reranked_avg = _avg_text_support(reranked_top)

            totals["records"] += 1
            totals["candidate_count"] += len(candidates)
            totals["internal_candidate_count"] += len(internal_candidates)
            totals["external_candidate_count"] += len(external_candidates)
            totals["records_without_candidates"] += int(not candidates)
            totals["table_only_avg_text_support"] += table_avg
            totals["reranked_avg_text_support"] += reranked_avg
            totals["reranked_avg_text_to_candidate_salience"] += _avg_metric(
                reranked_top, "text_to_candidate_salience"
            )
            totals["reranked_avg_text_to_candidate_coverage"] += _avg_metric(
                reranked_top, "text_to_candidate_coverage"
            )
            if table_only_top and reranked_top and table_only_top[0]["claim"] != reranked_top[0]["claim"]:
                totals["top1_changed"] += 1

            handle.write(
                json.dumps(
                    {
                        "record_id": record.record_id,
                        "domain": record.domain,
                        "paper_id": record.paper_id,
                        "paper": record.paper,
                        "caption": record.caption,
                        "description": record.description,
                        "candidate_count": len(candidates),
                        "internal_candidate_count": len(internal_candidates),
                        "external_candidate_count": len(external_candidates),
                        "candidate_source": candidate_source,
                        "top_k": top_k,
                        "rerank_mode": rerank_mode,
                        "table_only_top": table_only_top,
                        "reranked_top": reranked_top,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    record_count = totals["records"]
    summary = {
        "records": record_count,
        "average_candidates": round(totals["candidate_count"] / record_count, 4) if record_count else 0.0,
        "average_internal_candidates": round(totals["internal_candidate_count"] / record_count, 4)
        if record_count
        else 0.0,
        "average_external_candidates": round(totals["external_candidate_count"] / record_count, 4)
        if record_count
        else 0.0,
        "records_without_candidates": totals["records_without_candidates"],
        "table_only_avg_text_support_at_k": round(totals["table_only_avg_text_support"] / record_count, 6)
        if record_count
        else 0.0,
        "reranked_avg_text_support_at_k": round(totals["reranked_avg_text_support"] / record_count, 6)
        if record_count
        else 0.0,
        "reranked_avg_text_to_candidate_salience_at_k": round(
            totals["reranked_avg_text_to_candidate_salience"] / record_count, 6
        )
        if record_count
        else 0.0,
        "reranked_avg_text_to_candidate_coverage_at_k": round(
            totals["reranked_avg_text_to_candidate_coverage"] / record_count, 6
        )
        if record_count
        else 0.0,
        "top1_changed_rate": round(totals["top1_changed"] / record_count, 6) if record_count else 0.0,
        "candidate_source": candidate_source,
        "candidate_dir": candidate_dir,
        "external_candidate_stats": external_candidate_stats,
        "rerank_mode": rerank_mode,
        "table_weight": table_weight,
        "text_weight": text_weight,
        "candidate_to_text_weight": candidate_to_text_weight,
        "text_to_candidate_weight": text_to_candidate_weight,
        "output": str(output_path),
    }
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _avg_text_support(items: Sequence[dict[str, Any]]) -> float:
    if not items:
        return 0.0
    return statistics.fmean(float(item["rerank"]["text_support"]) for item in items)


def _avg_metric(items: Sequence[dict[str, Any]], metric: str) -> float:
    values = [float(item["rerank"].get(metric, 0.0)) for item in items]
    return statistics.fmean(values) if values else 0.0


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal text-aware reranking baseline on SciGen.")
    parser.add_argument("--data-dir", default="SciGen-main/dataset")
    parser.add_argument("--split", default="test", choices=sorted(TEST_FILE_CANDIDATES))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=40)
    parser.add_argument(
        "--candidate-source",
        choices=("internal", "external", "combined"),
        default="internal",
        help="internal uses rule candidates, external uses --candidate-dir, combined merges both.",
    )
    parser.add_argument(
        "--candidate-dir",
        help="Directory containing JSONL/JSON files with record_id and insight lists from other baselines.",
    )
    parser.add_argument("--rerank-mode", choices=("text-aware", "bidirectional"), default="text-aware")
    parser.add_argument("--table-weight", type=positive_float, default=0.35)
    parser.add_argument("--text-weight", type=positive_float, default=0.65)
    parser.add_argument("--candidate-to-text-weight", type=positive_float, default=0.45)
    parser.add_argument("--text-to-candidate-weight", type=positive_float, default=0.20)
    parser.add_argument("--evidence-threshold", type=positive_float, default=0.12)
    parser.add_argument("--output", default="text_aware_rerank_baseline/outputs/scigen_text_aware_test.jsonl")
    args = parser.parse_args()
    if args.candidate_source in {"external", "combined"} and not args.candidate_dir:
        parser.error("--candidate-dir is required when --candidate-source is external or combined")

    summary = run_baseline(
        data_dir=args.data_dir,
        split=args.split,
        output=args.output,
        limit=args.limit,
        top_k=args.top_k,
        max_candidates=args.max_candidates,
        candidate_source=args.candidate_source,
        candidate_dir=args.candidate_dir,
        rerank_mode=args.rerank_mode,
        table_weight=args.table_weight,
        text_weight=args.text_weight,
        candidate_to_text_weight=args.candidate_to_text_weight,
        text_to_candidate_weight=args.text_to_candidate_weight,
        evidence_threshold=args.evidence_threshold,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
