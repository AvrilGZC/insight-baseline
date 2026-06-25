from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from .openai_client import DEFAULT_API_BASE, DEFAULT_MODEL, call_chat_completion, parse_json_object


SYSTEM_PROMPT = """You audit whether table-supported insights are covered by a scientific table description.
Return compact valid JSON only. Use the description to classify coverage. Do not invent new table facts."""


def build_prompt(row: Dict[str, Any]) -> str:
    compact_insights = [
        {
            "index": idx,
            "type": item.get("type"),
            "subtype": item.get("subtype"),
            "claim": item.get("claim"),
            "evidence": item.get("evidence", []),
        }
        for idx, item in enumerate(row.get("insights", []))
    ]
    return f"""Classify each candidate table insight against the gold description.

Labels:
- covered: the description clearly states the same insight.
- partially_covered: the description states part of the insight or a weaker version.
- missing: the insight is table-supported but absent from the description.
- contradicted: the description says the opposite or conflicts with the insight.

Return compact valid JSON only. Do not use Markdown code fences. Classify every candidate insight exactly once.

Return JSON:
{{
  "classifications": [
    {{"index": 0, "label": "covered|partially_covered|missing|contradicted", "reason": "short reason"}}
  ],
  "summary": {{"covered": 0, "partially_covered": 0, "missing": 0, "contradicted": 0}}
}}

Table caption:
{row.get("caption", "")}

Gold description:
{row.get("description", "")}

Candidate insights:
{json.dumps(compact_insights, ensure_ascii=False, indent=2)}
"""


def run_rule_llm_verifier(
    input_jsonl: str,
    output: str,
    limit: int | None,
    model: str,
    api_base: str,
    dry_run: bool,
    max_tokens: int = 2500,
    json_repair_retries: int = 1,
    retry_failed_from: str | None = None,
) -> dict[str, object]:
    input_path = Path(input_jsonl)
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if retry_failed_from:
        failed_ids = _load_failed_record_ids(retry_failed_from)
        rows = [row for row in rows if str(row.get("record_id") or "") in failed_ids]
    if limit:
        rows = rows[:limit]
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    totals = {"records": 0, "failures": 0, "covered": 0, "partially_covered": 0, "missing": 0, "contradicted": 0}

    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            raw_text = ""
            prompt = build_prompt(row)
            payload = {
                "record_id": row.get("record_id"),
                "domain": row.get("domain"),
                "paper_id": row.get("paper_id"),
                "model": model,
                "prompt": prompt if dry_run else None,
            }
            try:
                if dry_run:
                    raw_text = json.dumps({"classifications": [], "summary": {}})
                else:
                    raw_text = call_chat_completion(
                        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                        model=model,
                        api_base=api_base,
                        max_tokens=max_tokens,
                    )
                parsed = _parse_or_repair_json(
                    raw_text=raw_text,
                    prompt=prompt,
                    model=model,
                    api_base=api_base,
                    max_tokens=max_tokens,
                    repair_retries=json_repair_retries if not dry_run else 0,
                )
                classifications = parsed.get("classifications", [])
                summary = _count_labels(classifications)
                payload.update(
                    {
                        "raw_model_text": raw_text,
                        "classifications": classifications,
                        "summary": summary,
                    }
                )
                for key in ("covered", "partially_covered", "missing", "contradicted"):
                    totals[key] += summary[key]
            except Exception as exc:  # noqa: BLE001
                payload.update({"error": str(exc), "raw_model_text": raw_text, "classifications": [], "summary": {}})
                totals["failures"] += 1
            totals["records"] += 1
            handle.write(json.dumps({k: v for k, v in payload.items() if v is not None}, ensure_ascii=False) + "\n")

    total_labels = totals["covered"] + totals["partially_covered"] + totals["missing"] + totals["contradicted"]
    summary_payload = {
        **totals,
        "coverage_rate": round((totals["covered"] + totals["partially_covered"]) / total_labels, 4) if total_labels else 0.0,
        "output": str(output_path),
        "dry_run": dry_run,
        "api_base": api_base,
        "model": model,
        "max_tokens": max_tokens,
        "json_repair_retries": json_repair_retries,
        "retry_failed_from": retry_failed_from or "",
    }
    output_path.with_suffix(output_path.suffix + ".summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary_payload


def _count_labels(classifications: Any) -> Dict[str, int]:
    counts = {"covered": 0, "partially_covered": 0, "missing": 0, "contradicted": 0}
    if not isinstance(classifications, list):
        return counts
    for item in classifications:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip().lower()
        if label in counts:
            counts[label] += 1
    return counts


def _parse_or_repair_json(
    raw_text: str,
    prompt: str,
    model: str,
    api_base: str,
    max_tokens: int,
    repair_retries: int,
) -> Dict[str, Any]:
    last_error: Exception | None = None
    candidate = raw_text
    for attempt in range(repair_retries + 1):
        try:
            return parse_json_object(candidate)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= repair_retries:
                break
            candidate = call_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You fix model outputs into valid compact JSON. "
                            "Return only valid JSON with top-level keys classifications and summary."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "The previous answer was not valid JSON. Convert it into valid JSON matching "
                            "the requested schema. If one classification cannot be repaired, drop it.\n\n"
                            f"Original task prompt:\n{prompt[:6000]}\n\n"
                            f"Invalid answer:\n{candidate[:10000]}"
                        ),
                    },
                ],
                model=model,
                api_base=api_base,
                max_tokens=max_tokens,
            )
    raise last_error or ValueError("Model response did not contain valid JSON.")


def _load_failed_record_ids(path: str) -> set[str]:
    failed = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("error") and row.get("record_id"):
            failed.add(str(row["record_id"]))
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Use an LLM to audit rule-based SciGen insights against descriptions.")
    parser.add_argument("input_jsonl", help="Output from scigen_baseline.rule_based.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--max-tokens", type=int, default=2500)
    parser.add_argument("--json-repair-retries", type=int, default=1)
    parser.add_argument("--retry-failed-from")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default="outputs/scigen_rule_llm_verified_test.jsonl")
    args = parser.parse_args()
    summary = run_rule_llm_verifier(
        args.input_jsonl,
        args.output,
        args.limit,
        args.model,
        args.api_base,
        args.dry_run,
        args.max_tokens,
        args.json_repair_retries,
        args.retry_failed_from,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
