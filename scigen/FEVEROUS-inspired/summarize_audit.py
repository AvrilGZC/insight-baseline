from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize scitab_verifier_baseline audit output and export review CSV."
    )
    parser.add_argument("--audit-file", required=True)
    parser.add_argument("--summary-file", required=True)
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    audit_path = Path(args.audit_file)
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    summary = summarize_audit_payload(payload, top_k=args.top_k)

    summary_path = Path(args.summary_file)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    review_csv_path = Path(args.review_csv)
    review_csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_review_csv(review_csv_path, summary["review_rows"])

    print(
        json.dumps(
            {
                "audit_file": str(audit_path),
                "summary_file": str(summary_path),
                "review_csv": str(review_csv_path),
                "review_rows": len(summary["review_rows"]),
            },
            indent=2,
        )
    )


def summarize_audit_payload(payload: Dict[str, object], top_k: int) -> Dict[str, object]:
    results = payload.get("results") or []
    records = list(results) if isinstance(results, list) else []
    supported_rows: List[Dict[str, object]] = []
    refuted_rows: List[Dict[str, object]] = []
    nei_rows: List[Dict[str, object]] = []

    for record in records:
        parent_context = {
            "parent_id": record.get("parent_id") or "",
            "paper": record.get("paper") or "",
            "paper_id": record.get("paper_id") or "",
            "table_id": record.get("table_id") or "",
            "caption": record.get("caption") or "",
            "table_preview": record.get("table_preview") or {},
        }
        for item in record.get("supported_sentences") or []:
            supported_rows.append(flatten_item(parent_context, item))
        for item in record.get("refuted_sentences") or []:
            refuted_rows.append(flatten_item(parent_context, item))
        for item in record.get("nei_sentences") or []:
            nei_rows.append(flatten_item(parent_context, item))

    supported_rows.sort(key=lambda row: (-float(row["support_score"]), int(row["sentence_index"])))
    refuted_rows.sort(
        key=lambda row: (
            -float(row["contradiction_score"]),
            -float(row["refutes_score"]),
            int(row["sentence_index"]),
        )
    )
    nei_rows.sort(key=lambda row: (-float(row["uncertainty_score"]), int(row["sentence_index"])))

    summary = {
        "summary": payload.get("summary") or {},
        "record_count": len(records),
        "average_supported_per_record": round(len(supported_rows) / max(1, len(records)), 3),
        "average_refuted_per_record": round(len(refuted_rows) / max(1, len(records)), 3),
        "average_nei_per_record": round(len(nei_rows) / max(1, len(records)), 3),
        "top_supported": supported_rows[:top_k],
        "top_refuted": refuted_rows[:top_k],
        "top_nei": nei_rows[:top_k],
        "review_rows": build_review_rows(refuted_rows, nei_rows, top_k=top_k),
    }
    return summary


def flatten_item(parent_context: Dict[str, object], item: Dict[str, object]) -> Dict[str, object]:
    scores = item.get("scores") or {}
    return {
        **parent_context,
        "example_id": item.get("example_id") or "",
        "sentence_index": int(item.get("sentence_index") or 0),
        "claim": item.get("claim") or "",
        "predicted_label": item.get("predicted_label") or "",
        "support_score": float(item.get("support_score") or 0.0),
        "contradiction_score": float(item.get("contradiction_score") or 0.0),
        "uncertainty_score": float(item.get("uncertainty_score") or 0.0),
        "supports_score": float(scores.get("supports") or 0.0),
        "refutes_score": float(scores.get("refutes") or 0.0),
        "nei_score": float(scores.get("nei") or 0.0),
    }


def build_review_rows(
    refuted_rows: Sequence[Dict[str, object]],
    nei_rows: Sequence[Dict[str, object]],
    top_k: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for rank, row in enumerate(refuted_rows[:top_k], start=1):
        rows.append(
            {
                "bucket": "refuted",
                "rank": rank,
                **row,
                "manual_label": "",
                "review_notes": "",
            }
        )
    for rank, row in enumerate(nei_rows[:top_k], start=1):
        rows.append(
            {
                "bucket": "nei",
                "rank": rank,
                **row,
                "manual_label": "",
                "review_notes": "",
            }
        )
    return rows


def write_review_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "bucket",
        "rank",
        "parent_id",
        "paper",
        "paper_id",
        "table_id",
        "caption",
        "example_id",
        "sentence_index",
        "claim",
        "predicted_label",
        "support_score",
        "contradiction_score",
        "uncertainty_score",
        "supports_score",
        "refutes_score",
        "nei_score",
        "manual_label",
        "review_notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
