from __future__ import annotations

import math
import re
import statistics
from typing import List

from .ranking import rank_insights
from .schema import EvidenceCell, Insight, SciGenRecord


def extract_rule_insights(record: SciGenRecord, top_k: int | None = None) -> List[Insight]:
    numeric = _numeric_cells(record)
    insights: List[Insight] = []
    insights.extend(_column_extrema(record, numeric))
    insights.extend(_row_mean_extrema(record, numeric))
    insights.extend(_row_outperformance(record, numeric))
    insights.extend(_row_trends(record, numeric))
    return rank_insights(insights, top_k=top_k)


def _numeric_cells(record: SciGenRecord) -> dict[tuple[int, int], float]:
    values: dict[tuple[int, int], float] = {}
    for row_idx, row in enumerate(record.rows):
        for col_idx, value in enumerate(row):
            parsed = _parse_number(value)
            if parsed is not None:
                values[(row_idx, col_idx)] = parsed
    return values


def _column_extrema(record: SciGenRecord, numeric: dict[tuple[int, int], float]) -> List[Insight]:
    insights: List[Insight] = []
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


def _row_mean_extrema(record: SciGenRecord, numeric: dict[tuple[int, int], float]) -> List[Insight]:
    row_values = _row_numeric_values(record, numeric)
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


def _row_outperformance(record: SciGenRecord, numeric: dict[tuple[int, int], float]) -> List[Insight]:
    row_values = _row_numeric_values(record, numeric)
    insights: List[Insight] = []
    row_indices = sorted(row_values)
    for left in row_indices:
        for right in row_indices:
            if left == right:
                continue
            shared = []
            right_by_col = dict(row_values[right])
            for col_idx, left_value in row_values[left]:
                if col_idx in right_by_col:
                    shared.append((col_idx, left_value, right_by_col[col_idx]))
            if len(shared) < 2:
                continue
            wins = [(col_idx, lv, rv) for col_idx, lv, rv in shared if lv > rv]
            if len(wins) / len(shared) < 0.7:
                continue
            avg_diff = statistics.fmean(lv - rv for _, lv, rv in shared)
            if avg_diff <= 0:
                continue
            evidence = []
            for col_idx, lv, rv in wins[:6]:
                evidence.append(_cell(record, left, col_idx, lv))
                evidence.append(_cell(record, right, col_idx, rv))
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


def _row_trends(record: SciGenRecord, numeric: dict[tuple[int, int], float]) -> List[Insight]:
    if not _headers_look_ordered(record.column_headers):
        return []
    insights: List[Insight] = []
    for row_idx, values in _row_numeric_values(record, numeric).items():
        if len(values) < 3:
            continue
        values = sorted(values)
        first_col, first = values[0]
        last_col, last = values[-1]
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


def _row_numeric_values(record: SciGenRecord, numeric: dict[tuple[int, int], float]) -> dict[int, List[tuple[int, float]]]:
    values: dict[int, List[tuple[int, float]]] = {}
    for (row_idx, col_idx), value in numeric.items():
        values.setdefault(row_idx, []).append((col_idx, value))
    return values


def _headers_look_ordered(headers: List[str]) -> bool:
    if len(headers) < 3:
        return False
    parsed = [_parse_number(header) for header in headers]
    present = [value for value in parsed if value is not None]
    return len(present) >= max(3, len(headers) // 2)


def _cell(record: SciGenRecord, row_idx: int, col_idx: int, numeric_value: float) -> EvidenceCell:
    row = record.rows[row_idx] if row_idx < len(record.rows) else []
    value = row[col_idx] if col_idx < len(row) else ""
    return EvidenceCell(
        row_index=row_idx,
        column_index=col_idx,
        row_header=_row_label(record, row_idx),
        column_header=record.column_headers[col_idx] if col_idx < len(record.column_headers) else f"col_{col_idx}",
        value=value,
        numeric_value=numeric_value,
    )


def _row_label(record: SciGenRecord, row_idx: int) -> str:
    row = record.rows[row_idx] if row_idx < len(record.rows) else []
    non_numeric = [cell for cell in row[:3] if _parse_number(cell) is None and cell.strip()]
    if non_numeric:
        return " / ".join(non_numeric)
    return f"row {row_idx}"


def _parse_number(value: str) -> float | None:
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None

