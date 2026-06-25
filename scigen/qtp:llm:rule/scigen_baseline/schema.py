from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class EvidenceCell:
    row_index: int
    column_index: int
    row_header: str
    column_header: str
    value: str
    numeric_value: float | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Insight:
    type: str
    subtype: str
    claim: str
    evidence: List[EvidenceCell]
    operation: str | None = None
    subject: str | None = None
    condition: str | None = None
    details: Dict[str, Any] | None = None
    score: Dict[str, float] | None = None
    source: str = "rule"

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["evidence"] = [cell.to_dict() for cell in self.evidence]
        return {key: value for key, value in payload.items() if value is not None}


@dataclass
class SciGenRecord:
    record_id: str
    domain: str
    paper_id: str
    caption: str
    description: str
    column_headers: List[str]
    rows: List[List[str]]
    raw: Dict[str, Any]

    def to_table_text(self, max_rows: int | None = None) -> str:
        rows = self.rows if max_rows is None else self.rows[:max_rows]
        widths = [len(self.column_headers)]
        widths.extend(len(row) for row in rows)
        width = max(widths) if widths else 0
        headers = _pad(self.column_headers, width)
        lines = [" | ".join(headers)]
        for idx, row in enumerate(rows):
            lines.append(f"{idx}: " + " | ".join(_pad(row, width)))
        if max_rows is not None and len(self.rows) > max_rows:
            lines.append(f"... ({len(self.rows) - max_rows} more rows)")
        return "\n".join(lines)


def _pad(values: List[str], width: int) -> List[str]:
    return [str(values[idx]) if idx < len(values) else "" for idx in range(width)]

