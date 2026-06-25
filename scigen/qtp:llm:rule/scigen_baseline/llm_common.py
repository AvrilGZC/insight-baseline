from __future__ import annotations

from typing import Any, Dict, List

from .schema import EvidenceCell, Insight, SciGenRecord


ALLOWED_TYPES = {"comparison", "extremum", "aggregate", "trend", "relationship"}


def normalize_llm_insights(record: SciGenRecord, raw_items: List[Dict[str, Any]], source: str) -> List[Insight]:
    insights: List[Insight] = []
    for item in raw_items:
        claim = str(item.get("claim") or "").strip()
        if not claim:
            continue
        insight_type = str(item.get("type") or "relationship").strip().lower()
        if insight_type not in ALLOWED_TYPES:
            insight_type = "relationship"
        subtype = str(item.get("subtype") or item.get("insight_subtype") or "llm_generated").strip().lower()
        evidence = [_normalize_evidence_cell(record, cell) for cell in _as_list(item.get("evidence"))]
        details = item.get("details") if isinstance(item.get("details"), dict) else {}
        score = item.get("score") if isinstance(item.get("score"), dict) else None
        insights.append(
            Insight(
                type=insight_type,
                subtype=subtype,
                claim=claim,
                evidence=[cell for cell in evidence if cell is not None],
                operation=str(item.get("operation")) if item.get("operation") else None,
                subject=str(item.get("subject")) if item.get("subject") else None,
                condition=str(item.get("condition")) if item.get("condition") else None,
                details=details,
                score=score,
                source=source,
            )
        )
    return insights


def _normalize_evidence_cell(record: SciGenRecord, item: Any) -> EvidenceCell | None:
    if not isinstance(item, dict):
        return None
    row_index = _to_int(item.get("row_index"))
    column_index = _to_int(item.get("column_index"))
    row_header = str(item.get("row_header") or item.get("row") or "")
    column_header = str(item.get("column_header") or item.get("column") or "")
    value = str(item.get("value") or "")
    numeric_value = _to_float(item.get("numeric_value") or item.get("number") or item.get("value"))
    if row_index is None:
        row_index = _find_row(record, row_header)
    if column_index is None:
        column_index = _find_column(record, column_header)
    if row_index is None or column_index is None:
        return None
    if row_index < len(record.rows) and column_index < len(record.rows[row_index]):
        value = value or record.rows[row_index][column_index]
    if column_index < len(record.column_headers):
        column_header = column_header or record.column_headers[column_index]
    return EvidenceCell(
        row_index=row_index,
        column_index=column_index,
        row_header=row_header or f"row {row_index}",
        column_header=column_header or f"col_{column_index}",
        value=value,
        numeric_value=numeric_value,
    )


def _find_row(record: SciGenRecord, row_header: str) -> int | None:
    needle = row_header.lower().strip()
    if not needle:
        return None
    for idx, row in enumerate(record.rows):
        if any(needle in cell.lower() or cell.lower() in needle for cell in row[:3] if cell):
            return idx
    return None


def _find_column(record: SciGenRecord, column_header: str) -> int | None:
    needle = column_header.lower().strip()
    if not needle:
        return None
    for idx, header in enumerate(record.column_headers):
        lowered = header.lower()
        if needle in lowered or lowered in needle:
            return idx
    return None


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _to_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None

