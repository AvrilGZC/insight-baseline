from __future__ import annotations

import argparse
import json
from pathlib import Path


def merge_outputs(inputs: list[str], output: str) -> dict[str, object]:
    merged: dict[str, dict[str, object]] = {}
    order: list[str] = []
    for input_path in inputs:
        for line in Path(input_path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            record_id = str(row.get("record_id") or "")
            if not record_id:
                continue
            if record_id not in merged:
                order.append(record_id)
                merged[record_id] = row
                continue
            old = merged[record_id]
            old_failed = bool(old.get("error"))
            new_failed = bool(row.get("error"))
            if old_failed and not new_failed:
                merged[record_id] = row
            elif old_failed == new_failed:
                merged[record_id] = row

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record_id in order:
            handle.write(json.dumps(merged[record_id], ensure_ascii=False) + "\n")

    rows = list(merged.values())
    summary = {
        "records": len(rows),
        "failures": sum(1 for row in rows if row.get("error")),
        "successes": sum(1 for row in rows if not row.get("error")),
        "output": str(output_path),
    }
    output_path.with_suffix(output_path.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge SciGen JSONL outputs by record_id.")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(merge_outputs(args.inputs, args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

