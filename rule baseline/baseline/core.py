from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from typing import Dict, List


@dataclass
class EvidenceCell:
    row: str
    column: str
    value: float


@dataclass
class Insight:
    type: str
    subject: str
    condition: str
    claim: str
    evidence: List[EvidenceCell]
    axis: str | None = None
    scope: str | None = None
    operation: str | None = None
    details: Dict[str, object] | None = None
    verification: str = "supported"
    score: Dict[str, float] | None = None

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["evidence"] = [asdict(cell) for cell in self.evidence]
        return {key: value for key, value in payload.items() if value is not None}


@dataclass
class TableData:
    row_label_name: str
    row_labels: List[str]
    columns: List[str]
    values: List[List[float]]

    def row_means(self) -> List[float]:
        return [statistics.fmean(row) for row in self.values]

    def column_values(self, column_index: int) -> List[float]:
        return [row[column_index] for row in self.values]
