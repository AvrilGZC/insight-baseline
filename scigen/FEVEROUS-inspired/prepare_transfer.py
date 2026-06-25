from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare SciGen or numericNLG records for scitab_verifier_baseline audit."
    )
    parser.add_argument("--source", required=True, choices=["scigen", "numericnlg"])
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit.")
    parser.add_argument(
        "--single-level-only",
        action="store_true",
        help="Keep only single-level tables. For SciGen this uses a conservative header heuristic.",
    )
    parser.add_argument("--input-file", help="Used for SciGen source.")
    parser.add_argument("--table-file", help="Used for numericNLG source.")
    parser.add_argument("--desc-file", help="Used for numericNLG source.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.source == "scigen":
        if not args.input_file:
            raise ValueError("--input-file is required for --source scigen")
        records = prepare_scigen_records(
            args.input_file,
            limit=args.limit,
            single_level_only=args.single_level_only,
        )
    else:
        if not args.table_file or not args.desc_file:
            raise ValueError("--table-file and --desc-file are required for --source numericnlg")
        records = prepare_numericnlg_records(
            args.table_file,
            args.desc_file,
            limit=args.limit,
            single_level_only=args.single_level_only,
        )

    output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "source": args.source,
                "records": len(records),
                "output_file": str(output_path),
            },
            indent=2,
        )
    )


def prepare_scigen_records(
    path: str | Path,
    limit: int = 0,
    single_level_only: bool = False,
) -> List[Dict[str, object]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        items = list(payload.items())
    elif isinstance(payload, list):
        items = [(str(index), record) for index, record in enumerate(payload)]
    else:
        raise ValueError("Unsupported SciGen JSON structure.")

    records: List[Dict[str, object]] = []
    for key, record in items:
        if not isinstance(record, dict):
            continue
        if single_level_only and not scigen_is_single_level(record):
            continue
        prepared = {
            "id": str(record.get("id") or key),
            "paper": record.get("paper") or "",
            "paper_id": record.get("paper_id") or "",
            "table_caption": record.get("table_caption") or "",
            "table_column_names": record.get("table_column_names") or [],
            "table_content_values": record.get("table_content_values") or [],
            "text": record.get("text") or "",
        }
        records.append(prepared)
        if limit and len(records) >= limit:
            break
    return records


def prepare_numericnlg_records(
    table_file: str | Path,
    desc_file: str | Path,
    limit: int = 0,
    single_level_only: bool = False,
) -> List[Dict[str, object]]:
    tables = json.loads(Path(table_file).read_text(encoding="utf-8"))
    descs = json.loads(Path(desc_file).read_text(encoding="utf-8"))
    desc_by_id = {str(item.get("table_id_paper") or ""): item for item in descs if isinstance(item, dict)}
    records: List[Dict[str, object]] = []
    for table_record in tables:
        if not isinstance(table_record, dict):
            continue
        if single_level_only and not numericnlg_is_single_level(table_record):
            continue
        table_id_paper = str(table_record.get("table_id_paper") or "")
        desc_record = desc_by_id.get(table_id_paper)
        if not desc_record:
            continue
        records.append(_merge_numericnlg_record(table_record, desc_record))
        if limit and len(records) >= limit:
            break
    return records


def _merge_numericnlg_record(
    table_record: Dict[str, object],
    desc_record: Dict[str, object],
) -> Dict[str, object]:
    row_headers = table_record.get("row_headers") or []
    row_labels = [_flatten_path(path, fallback=f"row_{index + 1}") for index, path in enumerate(row_headers)]
    column_headers = table_record.get("column_headers") or []
    columns = [_flatten_path(path, fallback=f"column_{index + 1}") for index, path in enumerate(column_headers)]
    contents = table_record.get("contents") or []
    merged_rows: List[List[str]] = []
    for index, row in enumerate(contents):
        if isinstance(row, list):
            label = row_labels[index] if index < len(row_labels) else f"row_{index + 1}"
            merged_rows.append([label] + [str(cell) for cell in row])
    if columns:
        merged_columns = ["row_label"] + columns
    else:
        width = len(merged_rows[0]) - 1 if merged_rows else 0
        merged_columns = ["row_label"] + [f"column_{index + 1}" for index in range(width)]

    return {
        "id": table_record.get("table_id_paper") or "",
        "paper_id": table_record.get("paper_id") or desc_record.get("paper_id") or "",
        "table_id": table_record.get("table_id") or desc_record.get("table_id") or "",
        "table_caption": table_record.get("caption") or "",
        "table_column_names": merged_columns,
        "table_content_values": merged_rows,
        "description": desc_record.get("description") or "",
        "sentences": desc_record.get("sentences") or [],
        "class_sentence": desc_record.get("class_sentence") or [],
        "header_mention": desc_record.get("header_mention") or [],
    }


def numericnlg_is_single_level(table_record: Dict[str, object]) -> bool:
    return table_record.get("row_header_level") == 1 and table_record.get("column_header_level") == 1


def scigen_is_single_level(record: Dict[str, object]) -> bool:
    columns = record.get("table_column_names") or []
    if not isinstance(columns, list) or not columns:
        return False
    rows = record.get("table_content_values") or []
    if len(set(str(item) for item in columns)) < len(columns):
        return False
    if not isinstance(rows, list) or not rows:
        return True
    first_row = rows[0]
    if not isinstance(first_row, list) or len(first_row) != len(columns):
        return True
    non_numeric_cells = 0
    for cell in first_row:
        cell_text = str(cell).strip()
        if not cell_text:
            non_numeric_cells += 1
            continue
        normalized = cell_text.replace(",", "").replace("%", "")
        try:
            float(normalized)
        except ValueError:
            non_numeric_cells += 1
    # If the first content row mostly behaves like a second header row,
    # we treat the table as multi-level.
    return non_numeric_cells < max(1, len(first_row) - 1)


def _flatten_path(path: object, fallback: str) -> str:
    if isinstance(path, list):
        parts = [str(item).strip() for item in path if str(item).strip()]
        return " / ".join(parts) if parts else fallback
    if isinstance(path, str) and path.strip():
        return path.strip()
    return fallback


if __name__ == "__main__":
    main()
