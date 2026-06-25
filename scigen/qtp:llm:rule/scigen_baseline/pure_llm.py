from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .llm_common import normalize_llm_insights
from .openai_client import DEFAULT_API_BASE, DEFAULT_MODEL, call_chat_completion, parse_json_object
from .ranking import rank_insights
from .scigen_loader import load_scigen_records
from .text_match import score_insights_against_description


SYSTEM_PROMPT = """You extract important, table-supported insights from scientific tables.
Return only valid JSON. Every insight must be grounded in table cells."""


def build_prompt(record, max_insights: int, include_description: bool = False) -> str:
    if include_description:
        return _build_table_description_prompt(record, max_insights)
    return _build_table_only_prompt(record, max_insights)


def _build_table_only_prompt(record, max_insights: int) -> str:
    return f"""Extract up to {max_insights} high-level insights from this scientific table.

Focus on analytical findings, not one fact per cell. Prefer comparisons, best/worst results,
improvements over baselines, majority wins across datasets/metrics, trends across ordered columns,
and aggregate patterns.

Return JSON with this shape:
{{
  "insights": [
    {{
      "type": "comparison|extremum|aggregate|trend|relationship",
      "subtype": "outperformance|metric_best|baseline_improvement|relative_gain|majority_win|ordered_trend|ablation_effect|other",
      "claim": "one concise table-supported claim",
      "operation": "difference|argmax|argmin|mean|count|ratio|none",
      "evidence": [
        {{"row_index": 0, "column_index": 1, "row_header": "...", "column_header": "...", "value": "..."}}
      ],
      "score": {{"importance": 1.0, "support": 1.0}}
    }}
  ]
}}

Table caption:
{record.caption}

Table:
{record.to_table_text(max_rows=80)}
"""


def _build_table_description_prompt(record, max_insights: int) -> str:
    return f"""Given a scientific table and its gold description, extract up to {max_insights}
important high-level table-supported insights and audit whether the description covers each insight.

Focus on analytical findings, not one fact per cell. Prefer comparisons, best/worst results,
improvements over baselines, majority wins across datasets/metrics, trends across ordered columns,
and aggregate patterns.

Every insight must be directly supported by the table. The description is used only to assign
coverage labels, not to invent unsupported claims.
Return compact valid JSON only. Do not use Markdown code fences. Escape all quotation marks inside strings.

Use these coverage labels:
- covered: the description clearly states the same table insight.
- partially_covered: the description states part of the insight or a weaker version.
- missing: the insight is table-supported but absent from the description.
- contradicted: the description says the opposite or conflicts with the table-supported insight.

Return JSON with this shape:
{{
  "insights": [
    {{
      "type": "comparison|extremum|aggregate|trend|relationship",
      "subtype": "outperformance|metric_best|baseline_improvement|relative_gain|majority_win|ordered_trend|ablation_effect|other",
      "claim": "one concise table-supported claim",
      "operation": "difference|argmax|argmin|mean|count|ratio|none",
      "description_status": "covered|partially_covered|missing|contradicted",
      "description_rationale": "brief reason for the coverage label",
      "evidence": [
        {{"row_index": 0, "column_index": 1, "row_header": "...", "column_header": "...", "value": "..."}}
      ],
      "score": {{"importance": 1.0, "support": 1.0}}
    }}
  ],
  "summary": {{
    "covered": 0,
    "partially_covered": 0,
    "missing": 0,
    "contradicted": 0
  }}
}}

Table caption:
{record.caption}

Table:
{record.to_table_text(max_rows=60)}

Gold description:
{record.description}
"""


def run_pure_llm(
    data_dir: str,
    split: str,
    output: str,
    limit: int | None,
    model: str,
    api_base: str,
    max_insights: int,
    top_k: int,
    dry_run: bool,
    include_description: bool = False,
    max_tokens: int = 4000,
    json_repair_retries: int = 1,
    retry_failed_from: str | None = None,
) -> dict[str, object]:
    records = load_scigen_records(data_dir, split=split, limit=limit)
    if retry_failed_from:
        retry_ids = _load_failed_record_ids(retry_failed_from)
        records = [record for record in records if record.record_id in retry_ids]
        if limit:
            records = records[:limit]
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    totals = {
        "records": 0,
        "insights": 0,
        "failures": 0,
        "covered": 0,
        "partially_covered": 0,
        "missing": 0,
        "contradicted": 0,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            raw_text = ""
            prompt = build_prompt(record, max_insights=max_insights, include_description=include_description)
            payload = {
                "record_id": record.record_id,
                "domain": record.domain,
                "paper_id": record.paper_id,
                "caption": record.caption,
                "description": record.description,
                "model": model,
                "prompt": prompt if dry_run else None,
            }
            try:
                if dry_run:
                    raw_text = json.dumps({"insights": []})
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
                raw_insight_items = _prepare_raw_insight_items(parsed.get("insights", []), include_description)
                insights = rank_insights(
                    normalize_llm_insights(record, raw_insight_items, source="pure_llm_with_description" if include_description else "pure_llm"),
                    top_k=top_k,
                )
                audit = _llm_description_audit(insights) if include_description else score_insights_against_description(insights, record.description)
                payload.update(
                    {
                        "raw_model_text": raw_text,
                        "insights": [insight.to_dict() for insight in insights],
                        "description_audit": audit,
                        "include_description": include_description,
                    }
                )
                totals["insights"] += len(insights)
                if include_description:
                    for key in ("covered", "partially_covered", "missing", "contradicted"):
                        totals[key] += int(audit.get(key, 0))
            except Exception as exc:  # noqa: BLE001 - batch runner should keep going.
                payload.update({"error": str(exc), "raw_model_text": locals().get("raw_text", ""), "insights": []})
                totals["failures"] += 1
            totals["records"] += 1
            handle.write(json.dumps({k: v for k, v in payload.items() if v is not None}, ensure_ascii=False) + "\n")

    summary = {
        **totals,
        "average_insights": round(totals["insights"] / totals["records"], 4) if totals["records"] else 0.0,
        "output": str(output_path),
        "dry_run": dry_run,
        "api_base": api_base,
        "model": model,
        "include_description": include_description,
        "max_tokens": max_tokens,
        "json_repair_retries": json_repair_retries,
        "retry_failed_from": retry_failed_from or "",
    }
    output_path.with_suffix(output_path.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pure LLM SciGen insight extraction.")
    parser.add_argument("data_dir")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--max-insights", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--json-repair-retries", type=int, default=1)
    parser.add_argument(
        "--retry-failed-from",
        help="Only rerun records that have an error in a previous JSONL output.",
    )
    parser.add_argument(
        "--include-description",
        action="store_true",
        help="Give both table and gold description to the LLM and ask it to label coverage for each insight.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default="outputs/scigen_pure_llm_test.jsonl")
    args = parser.parse_args()
    summary = run_pure_llm(
        args.data_dir,
        args.split,
        args.output,
        args.limit,
        args.model,
        args.api_base,
        args.max_insights,
        args.top_k,
        args.dry_run,
        args.include_description,
        args.max_tokens,
        args.json_repair_retries,
        args.retry_failed_from,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _prepare_raw_insight_items(items, include_description: bool):
    if not isinstance(items, list):
        return []
    if not include_description:
        return items
    prepared = []
    for item in items:
        if not isinstance(item, dict):
            continue
        copied = dict(item)
        details = copied.get("details") if isinstance(copied.get("details"), dict) else {}
        status = _normalize_status(copied.get("description_status"))
        if status:
            details["description_status"] = status
        if copied.get("description_rationale"):
            details["description_rationale"] = str(copied.get("description_rationale"))
        copied["details"] = details
        prepared.append(copied)
    return prepared


def _llm_description_audit(insights):
    counts = {"covered": 0, "partially_covered": 0, "missing": 0, "contradicted": 0}
    matches = []
    for insight in insights:
        details = insight.details or {}
        status = _normalize_status(details.get("description_status")) or "missing"
        counts[status] += 1
        matches.append(
            {
                "label": status,
                "rationale": details.get("description_rationale", ""),
                "insight": insight.to_dict(),
            }
        )
    total = sum(counts.values())
    return {
        **counts,
        "insight_count": total,
        "coverage_rate": round((counts["covered"] + counts["partially_covered"]) / total, 4) if total else 0.0,
        "matches": matches,
    }


def _normalize_status(value):
    text = str(value or "").strip().lower().replace(" ", "_")
    if text in {"covered", "partially_covered", "missing", "contradicted"}:
        return text
    if text in {"partial", "partially-covered", "partiallycovered"}:
        return "partially_covered"
    return ""


def _parse_or_repair_json(
    raw_text: str,
    prompt: str,
    model: str,
    api_base: str,
    max_tokens: int,
    repair_retries: int,
) -> dict[str, Any]:
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
                            "Return only valid JSON with top-level key insights."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "The previous answer was not valid JSON. Convert it into valid JSON that matches "
                            "the requested schema. If an item cannot be repaired, drop it.\n\n"
                            f"Original task prompt:\n{prompt[:6000]}\n\n"
                            f"Invalid answer:\n{candidate[:12000]}"
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


if __name__ == "__main__":
    main()
