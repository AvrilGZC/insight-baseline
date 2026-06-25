from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


RESPONSE_SECTIONS = (
    "table_facts",
    "supported_claims",
    "missing_insights",
    "contradicted_claims",
)


def audit_results(
    results_file: str | Path,
    records_file: str | Path,
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    records = _load_records(records_file)
    records_by_id = {str(record.get("table_id_paper") or ""): record for record in records}
    payload = json.loads(Path(results_file).read_text(encoding="utf-8"))
    results = payload.get("results") or payload.get("examples") or []

    issue_counts: Counter[str] = Counter()
    per_record: List[Dict[str, Any]] = []
    examples: Dict[str, List[Any]] = defaultdict(list)

    for item in results:
        table_id_paper = str(item.get("table_id_paper") or "")
        record = records_by_id.get(table_id_paper)
        record_issues: Counter[str] = Counter()
        if not record:
            _add_issue(issue_counts, record_issues, examples, "missing_source_record", table_id_paper)
            per_record.append(_record_report(item, record_issues))
            continue

        table = record.get("table") or {}
        table_index = _table_index(table)
        _audit_table_shape(table_id_paper, table_index, issue_counts, record_issues, examples)
        if "ranked_table_facts" in item or "evaluated_insights" in item:
            _audit_v2_item(table_id_paper, item, table_index, issue_counts, record_issues, examples)
            per_record.append(_record_report(item, record_issues))
            continue

        response = item.get("llm_response") or {}
        if not response:
            _add_issue(issue_counts, record_issues, examples, "missing_llm_response", table_id_paper)
            per_record.append(_record_report(item, record_issues))
            continue
        for section in RESPONSE_SECTIONS:
            claims = response.get(section)
            if not isinstance(claims, list):
                _add_issue(issue_counts, record_issues, examples, f"{section}_not_list", table_id_paper)
                continue
            for claim_index, claim in enumerate(claims):
                if not isinstance(claim, dict):
                    _add_issue(issue_counts, record_issues, examples, f"{section}_claim_not_object", table_id_paper)
                    continue
                evidence = claim.get("evidence") or claim.get("table_evidence") or []
                if not isinstance(evidence, list) or not evidence:
                    _add_issue(
                        issue_counts,
                        record_issues,
                        examples,
                        f"{section}_missing_evidence",
                        {"table_id_paper": table_id_paper, "claim_index": claim_index},
                    )
                    continue
                for ev in evidence:
                    _audit_evidence(
                        table_id_paper=table_id_paper,
                        section=section,
                        claim_index=claim_index,
                        evidence=ev,
                        table_index=table_index,
                        issue_counts=issue_counts,
                        record_issues=record_issues,
                        examples=examples,
                    )

        per_record.append(_record_report(item, record_issues))

    report = {
        "results_file": str(results_file),
        "records_file": str(records_file),
        "evaluated_results": len(results),
        "source_records": len(records),
        "records_with_issues": sum(1 for item in per_record if item["issue_count"] > 0),
        "issue_counts": dict(issue_counts.most_common()),
        "examples": {key: value[:10] for key, value in examples.items()},
        "per_record": per_record,
    }
    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit pure LLM baseline results against source tables.")
    parser.add_argument("results_file", help="Pure LLM result JSON with a results list.")
    parser.add_argument("--records-file", required=True, help="single_header_tables JSON/JSONL source records.")
    parser.add_argument("--output", default="", help="Optional audit report JSON path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = audit_results(
        results_file=args.results_file,
        records_file=args.records_file,
        output_path=args.output or None,
    )
    printable = {key: value for key, value in report.items() if key not in {"examples", "per_record"}}
    printable["top_examples"] = {
        key: value[:3]
        for key, value in report["examples"].items()
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2))


def _load_records(records_file: str | Path) -> List[Dict[str, Any]]:
    path = Path(records_file)
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return payload["records"]
    if isinstance(payload, list):
        return payload
    raise ValueError("Records file must be JSONL, a JSON list, or a JSON object with records.")


def _table_index(table: Dict[str, Any]) -> Dict[str, Any]:
    row_labels = [_label(header) for header in table.get("row_headers") or []]
    column_labels = [_label(header) for header in table.get("column_headers") or []]
    cells = set()
    indexed_cells = set()
    for row_index, row in enumerate(table.get("contents") or []):
        row_label = row_labels[row_index] if row_index < len(row_labels) else ""
        for column_index, value in enumerate(row):
            column_label = column_labels[column_index] if column_index < len(column_labels) else ""
            cells.add((_norm(row_label), _norm(column_label), _value_norm(value)))
            indexed_cells.add((row_index, column_index, _value_norm(value)))
    return {
        "row_counts": Counter(_norm(label) for label in row_labels),
        "column_counts": Counter(_norm(label) for label in column_labels),
        "cells": cells,
        "indexed_cells": indexed_cells,
    }


def _audit_v2_item(
    table_id_paper: str,
    item: Dict[str, Any],
    table_index: Dict[str, Any],
    issue_counts: Counter[str],
    record_issues: Counter[str],
    examples: Dict[str, List[Any]],
) -> None:
    ranked_facts = item.get("ranked_table_facts")
    if not isinstance(ranked_facts, list):
        _add_issue(issue_counts, record_issues, examples, "ranked_table_facts_not_list", table_id_paper)
        ranked_facts = []
    for fact_index, fact in enumerate(ranked_facts):
        _audit_fact_evidence(
            table_id_paper,
            section="ranked_table_facts",
            fact_index=fact_index,
            fact=fact,
            table_index=table_index,
            issue_counts=issue_counts,
            record_issues=record_issues,
            examples=examples,
        )

    evaluated = item.get("evaluated_insights")
    if not isinstance(evaluated, list):
        _add_issue(issue_counts, record_issues, examples, "evaluated_insights_not_list", table_id_paper)
        return
    for fact_index, evaluated_item in enumerate(evaluated):
        if not isinstance(evaluated_item, dict):
            _add_issue(issue_counts, record_issues, examples, "evaluated_insight_not_object", table_id_paper)
            continue
        status = str(evaluated_item.get("description_status") or "")
        if status not in {"covered", "partially_covered", "missing", "contradicted"}:
            _add_issue(
                issue_counts,
                record_issues,
                examples,
                "evaluated_insight_bad_status",
                {"table_id_paper": table_id_paper, "fact_index": fact_index, "status": status},
            )
        table_fact = evaluated_item.get("table_fact")
        _audit_fact_evidence(
            table_id_paper,
            section="evaluated_insights",
            fact_index=fact_index,
            fact=table_fact,
            table_index=table_index,
            issue_counts=issue_counts,
            record_issues=record_issues,
            examples=examples,
        )


def _audit_fact_evidence(
    table_id_paper: str,
    section: str,
    fact_index: int,
    fact: Any,
    table_index: Dict[str, Any],
    issue_counts: Counter[str],
    record_issues: Counter[str],
    examples: Dict[str, List[Any]],
) -> None:
    if not isinstance(fact, dict):
        _add_issue(
            issue_counts,
            record_issues,
            examples,
            f"{section}_fact_not_object",
            {"table_id_paper": table_id_paper, "fact_index": fact_index},
        )
        return
    evidence = fact.get("evidence") or []
    if not isinstance(evidence, list) or not evidence:
        _add_issue(
            issue_counts,
            record_issues,
            examples,
            f"{section}_missing_evidence",
            {"table_id_paper": table_id_paper, "fact_index": fact_index},
        )
        return
    for ev in evidence:
        _audit_evidence(
            table_id_paper=table_id_paper,
            section=section,
            claim_index=fact_index,
            evidence=ev,
            table_index=table_index,
            issue_counts=issue_counts,
            record_issues=record_issues,
            examples=examples,
        )


def _audit_table_shape(
    table_id_paper: str,
    table_index: Dict[str, Any],
    issue_counts: Counter[str],
    record_issues: Counter[str],
    examples: Dict[str, List[Any]],
) -> None:
    if any(count > 1 for count in table_index["row_counts"].values()):
        _add_issue(issue_counts, record_issues, examples, "table_duplicate_row_labels", table_id_paper)
    if any(count > 1 for count in table_index["column_counts"].values()):
        _add_issue(issue_counts, record_issues, examples, "table_duplicate_column_labels", table_id_paper)


def _audit_evidence(
    table_id_paper: str,
    section: str,
    claim_index: int,
    evidence: Any,
    table_index: Dict[str, Any],
    issue_counts: Counter[str],
    record_issues: Counter[str],
    examples: Dict[str, List[Any]],
) -> None:
    if not isinstance(evidence, dict):
        _add_issue(
            issue_counts,
            record_issues,
            examples,
            f"{section}_evidence_not_object",
            {"table_id_paper": table_id_paper, "claim_index": claim_index, "evidence": evidence},
        )
        return
    row = _norm(evidence.get("row") or "")
    column = _norm(evidence.get("column") or "")
    value = _value_norm(evidence.get("value") or "")
    row_index = evidence.get("row_index")
    column_index = evidence.get("column_index")
    has_indices = isinstance(row_index, int) and isinstance(column_index, int)
    if has_indices and (row_index, column_index, value) not in table_index["indexed_cells"]:
        _add_issue(
            issue_counts,
            record_issues,
            examples,
            f"{section}_evidence_not_exact_indexed_cell",
            {
                "table_id_paper": table_id_paper,
                "claim_index": claim_index,
                "evidence": evidence,
            },
        )
    if not has_indices and ("row_index" in evidence or "column_index" in evidence):
        _add_issue(
            issue_counts,
            record_issues,
            examples,
            f"{section}_evidence_bad_indices",
            {
                "table_id_paper": table_id_paper,
                "claim_index": claim_index,
                "evidence": evidence,
            },
        )
    if (row, column, value) not in table_index["cells"]:
        _add_issue(
            issue_counts,
            record_issues,
            examples,
            f"{section}_evidence_not_exact_cell",
            {
                "table_id_paper": table_id_paper,
                "claim_index": claim_index,
                "evidence": evidence,
            },
        )
    if row and not has_indices and table_index["row_counts"][row] > 1:
        _add_issue(
            issue_counts,
            record_issues,
            examples,
            f"{section}_ambiguous_duplicate_row_evidence",
            {"table_id_paper": table_id_paper, "claim_index": claim_index, "evidence": evidence},
        )
    if column and not has_indices and table_index["column_counts"][column] > 1:
        _add_issue(
            issue_counts,
            record_issues,
            examples,
            f"{section}_ambiguous_duplicate_column_evidence",
            {"table_id_paper": table_id_paper, "claim_index": claim_index, "evidence": evidence},
        )


def _record_report(item: Dict[str, Any], record_issues: Counter[str]) -> Dict[str, Any]:
    return {
        "record_index": item.get("record_index"),
        "table_id_paper": item.get("table_id_paper"),
        "issue_count": sum(record_issues.values()),
        "issues": dict(record_issues.most_common()),
    }


def _add_issue(
    issue_counts: Counter[str],
    record_issues: Counter[str],
    examples: Dict[str, List[Any]],
    issue: str,
    example: Any,
) -> None:
    issue_counts[issue] += 1
    record_issues[issue] += 1
    if len(examples[issue]) < 10:
        examples[issue].append(example)


def _label(header: Any) -> str:
    if isinstance(header, list):
        return " | ".join(str(part) for part in header)
    return str(header)


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", _label(value)).strip().lower()


def _value_norm(value: Any) -> str:
    return re.sub(r"[^0-9a-z.+\-%]", "", str(value).strip().lower())


if __name__ == "__main__":
    main()
