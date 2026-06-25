from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, List

from .schema import SciGenRecord


TEST_FILE_CANDIDATES = {
    "test": ("test-CL.json", "test-Other.json", "test_CL.json", "test_Other.json"),
    "test-cl": ("test-CL.json", "test_CL.json"),
    "test-other": ("test-Other.json", "test_Other.json"),
    "cl": ("test-CL.json", "test_CL.json"),
    "other": ("test-Other.json", "test_Other.json"),
}


def load_scigen_records(data_dir: str | Path, split: str = "test", limit: int | None = None) -> List[SciGenRecord]:
    root = Path(data_dir)
    if root.is_file():
        raw_records = _load_json_records(root)
        records = [_normalize_record(item, root.stem, idx) for idx, item in enumerate(raw_records)]
        return records[:limit] if limit else records

    if not root.exists():
        raise FileNotFoundError(f"SciGen data directory does not exist: {root}")

    split_key = split.lower()
    files: List[Path] = []
    for name in TEST_FILE_CANDIDATES.get(split_key, (f"{split}.json", f"{split}.jsonl")):
        files.extend(root.rglob(name))

    if not files:
        known = sorted(path.name for path in root.rglob("*.json"))[:30]
        raise FileNotFoundError(
            f"Could not find SciGen split files for split={split!r} under {root}. "
            f"Expected files like test-CL.json and test-Other.json. Found JSON files: {known}"
        )

    records: List[SciGenRecord] = []
    for file_path in sorted(set(files)):
        raw_records = _load_json_records(file_path)
        for idx, item in enumerate(raw_records):
            records.append(_normalize_record(item, file_path.stem, idx))
            if limit and len(records) >= limit:
                return records
    return records


def _load_json_records(path: Path) -> List[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "records", "examples"):
            if isinstance(data.get(key), list):
                return data[key]
        if all(isinstance(value, dict) for value in data.values()):
            records = []
            for key, value in data.items():
                record = dict(value)
                record.setdefault("__record_key", str(key))
                records.append(record)
            return records
    raise ValueError(f"Unsupported SciGen JSON shape in {path}")


def _normalize_record(item: dict[str, Any], source_name: str, index: int) -> SciGenRecord:
    paper_id = str(item.get("paper_id") or item.get("paper") or item.get("id") or "")
    table_id = str(item.get("table_id") or item.get("__record_key") or item.get("table") or index)
    record_id = str(item.get("record_id") or item.get("example_id") or f"{source_name}:{paper_id}:{table_id}:{index}")
    domain = _domain_from_source(source_name)
    caption = _first_string(item, ("table_caption", "caption", "table_title", "title"))
    description = _first_string(item, ("text", "description", "target", "table_description"))

    column_headers = _flatten_headers(item.get("table_column_names") or item.get("column_headers") or item.get("columns"))
    rows = _normalize_rows(item.get("table_content_values") or item.get("rows") or item.get("table") or item.get("table_content"))
    if not column_headers and rows:
        column_headers = [f"col_{idx}" for idx in range(max(len(row) for row in rows))]
    if column_headers and rows:
        width = max(len(column_headers), max(len(row) for row in rows))
        column_headers = _pad(column_headers, width)
        rows = [_pad(row, width) for row in rows]

    return SciGenRecord(
        record_id=record_id,
        domain=domain,
        paper_id=paper_id,
        caption=caption,
        description=description,
        column_headers=column_headers,
        rows=rows,
        raw=item,
    )


def _domain_from_source(source_name: str) -> str:
    lowered = source_name.lower()
    if "other" in lowered:
        return "Other"
    if "cl" in lowered or "c&l" in lowered:
        return "C&L"
    return source_name


def _first_string(item: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return " ".join(str(part) for part in value if str(part).strip())
    return ""


def _flatten_headers(raw: Any) -> List[str]:
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


def _normalize_rows(raw: Any) -> List[List[str]]:
    if raw is None:
        return []
    if isinstance(raw, list):
        rows: List[List[str]] = []
        for row in raw:
            if isinstance(row, list):
                rows.append([_cell_to_str(cell) for cell in row])
            elif isinstance(row, dict):
                rows.append([_cell_to_str(value) for value in row.values()])
            else:
                rows.append([_cell_to_str(row)])
        return rows
    return [[_cell_to_str(raw)]]


def _cell_to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " / ".join(str(part) for part in value if str(part).strip())
    return str(value)


def _pad(values: List[str], width: int) -> List[str]:
    return [str(values[idx]) if idx < len(values) else "" for idx in range(width)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect SciGen test records.")
    parser.add_argument("data_dir")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    records = load_scigen_records(args.data_dir, split=args.split, limit=args.limit)
    print(json.dumps({"records": len(records), "domains": sorted({r.domain for r in records})}, indent=2))
    if records:
        first = records[0]
        print(first.record_id)
        print(first.caption)
        print(first.to_table_text(max_rows=5))


if __name__ == "__main__":
    main()
