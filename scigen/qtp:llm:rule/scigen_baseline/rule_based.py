from __future__ import annotations

import argparse
import json
from pathlib import Path

from .rules import extract_rule_insights
from .scigen_loader import load_scigen_records
from .text_match import score_insights_against_description


def run_rule_baseline(data_dir: str, split: str, output: str, limit: int | None, top_k: int) -> dict[str, object]:
    records = load_scigen_records(data_dir, split=split, limit=limit)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    totals = {"records": 0, "insights": 0, "covered": 0}
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            insights = extract_rule_insights(record, top_k=top_k)
            audit = score_insights_against_description(insights, record.description)
            totals["records"] += 1
            totals["insights"] += len(insights)
            totals["covered"] += int(audit["covered_or_partial"])
            handle.write(
                json.dumps(
                    {
                        "record_id": record.record_id,
                        "domain": record.domain,
                        "paper_id": record.paper_id,
                        "caption": record.caption,
                        "description": record.description,
                        "top_k": top_k,
                        "insights": [insight.to_dict() for insight in insights],
                        "heuristic_description_audit": audit,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary = {
        **totals,
        "average_insights": round(totals["insights"] / totals["records"], 4) if totals["records"] else 0.0,
        "heuristic_coverage": round(totals["covered"] / totals["insights"], 4) if totals["insights"] else 0.0,
        "output": str(output_path),
    }
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the rule-based SciGen insight baseline.")
    parser.add_argument("data_dir", help="SciGen dataset root containing test-CL.json and test-Other.json.")
    parser.add_argument("--split", default="test", help="test, test-cl, or test-other.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--output", default="outputs/scigen_rule_test.jsonl")
    args = parser.parse_args()
    summary = run_rule_baseline(args.data_dir, args.split, args.output, args.limit, args.top_k)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

