from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .pure_llm_baseline import (
    DEFAULT_API_BASE,
    DEFAULT_MODEL,
    OpenAICompatibleClient,
    _complete_json_with_retries,
    _load_baseline_records,
    _load_env_file,
    _table_payload,
)


FACT_SYSTEM_PROMPT = (
    "You are a careful table analysis assistant. "
    "Use only the provided numericNLG table. "
    "Return only valid JSON."
)

EVAL_SYSTEM_PROMPT = (
    "You are a careful table-text fact coverage assistant. "
    "Use only the provided ranked table insights and description text. "
    "Return only valid JSON."
)


def run_pure_llm_baseline_v2(
    data_dir: str | Path,
    split: str,
    limit: int,
    examples: int = 3,
    header_level: str = "single",
    client: OpenAICompatibleClient | None = None,
    dry_run: bool = False,
    sleep_seconds: float = 0.0,
    retries: int = 2,
    start_index: int = 0,
    output_path: str | Path | None = None,
    records_file: str | Path | None = None,
    failure_retries: int = 2,
    save_all_results: bool = False,
    max_table_facts: int = 16,
    min_top_k: int = 3,
    max_top_k: int = 8,
    top_k_fraction: float = 0.5,
) -> Dict[str, Any]:
    candidate_records, total_source_records = _load_baseline_records(
        data_dir=data_dir,
        split=split,
        header_level=header_level,
        records_file=records_file,
    )
    selected_records = candidate_records[start_index : start_index + limit]
    evaluated: List[Dict[str, Any]] = []
    pending_records = [
        (start_index + offset, record)
        for offset, record in enumerate(selected_records)
    ]
    failed: List[Dict[str, Any]] = []

    for retry_round in range(max(0, failure_retries) + 1):
        failed = []
        next_pending = []
        for record_index, record in pending_records:
            result, error = _evaluate_record_v2(
                record=record,
                record_index=record_index,
                client=client,
                dry_run=dry_run,
                retries=retries,
                retry_round=retry_round,
                max_table_facts=max_table_facts,
                min_top_k=min_top_k,
                max_top_k=max_top_k,
                top_k_fraction=top_k_fraction,
            )
            if error:
                failed.append(error)
                next_pending.append((record_index, record))
                continue
            assert result is not None
            evaluated.append(result)
            if sleep_seconds > 0 and not dry_run:
                time.sleep(sleep_seconds)
        pending_records = next_pending
        if not pending_records:
            break

    evaluated.sort(key=lambda item: int(item.get("record_index", 0)))
    summary = _summarize_v2_run(
        split=split,
        header_level=header_level,
        total_source_records=total_source_records,
        candidate_count=len(candidate_records),
        start_index=start_index,
        requested=len(selected_records),
        evaluated=evaluated,
        failed=failed,
        examples=examples,
        dry_run=dry_run,
        records_file=str(records_file) if records_file else "",
        failure_retries=failure_retries,
        save_all_results=save_all_results,
        max_table_facts=max_table_facts,
        min_top_k=min_top_k,
        max_top_k=max_top_k,
        top_k_fraction=top_k_fraction,
    )
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _evaluate_record_v2(
    record: Dict[str, Any],
    record_index: int,
    client: OpenAICompatibleClient | None,
    dry_run: bool,
    retries: int,
    retry_round: int,
    max_table_facts: int,
    min_top_k: int,
    max_top_k: int,
    top_k_fraction: float,
) -> tuple[Dict[str, Any] | None, Dict[str, Any] | None]:
    table_record = record["table_record"]
    desc_record = record.get("desc_record") or {}
    table_id_paper = str(record.get("table_id_paper") or table_record.get("table_id_paper") or "")
    try:
        fact_prompt = build_fact_generation_prompt(table_record, desc_record, max_table_facts)
        if dry_run:
            result = {
                "record_index": record_index,
                "source_split": record.get("split") or "",
                "table_id_paper": table_id_paper,
                "paper_id": str(desc_record.get("paper_id") or ""),
                "table_id": str(desc_record.get("table_id") or ""),
                "dry_run": True,
                "fact_generation_prompt": fact_prompt,
                "retry_round": retry_round,
            }
            return result, None
        if client is None:
            raise ValueError("LLM client is required unless dry_run=True.")

        fact_response = _complete_json_with_retries(client, fact_prompt, retries=retries)
        ranked_table_facts = _normalize_ranked_table_facts(fact_response, max_table_facts)
        top_k = dynamic_top_k(
            fact_count=len(ranked_table_facts),
            min_top_k=min_top_k,
            max_top_k=max_top_k,
            top_k_fraction=top_k_fraction,
        )
        selected_facts = ranked_table_facts[:top_k]
        coverage_prompt = build_coverage_prompt(table_record, desc_record, selected_facts, top_k)
        coverage_response = _complete_json_with_retries(client, coverage_prompt, retries=retries)
        evaluation = _normalize_coverage_response(coverage_response, selected_facts)
        result = {
            "record_index": record_index,
            "source_split": record.get("split") or "",
            "table_id_paper": table_id_paper,
            "paper_id": str(desc_record.get("paper_id") or ""),
            "table_id": str(desc_record.get("table_id") or ""),
            "dry_run": False,
            "top_k_formula": {
                "min_top_k": min_top_k,
                "max_top_k": max_top_k,
                "top_k_fraction": top_k_fraction,
                "fact_count": len(ranked_table_facts),
                "selected_top_k": top_k,
            },
            "ranked_table_facts": ranked_table_facts,
            "selected_top_k_facts": selected_facts,
            "evaluated_insights": evaluation["evaluated_insights"],
            "supported_claims": evaluation["supported_claims"],
            "partially_covered_insights": evaluation["partially_covered_insights"],
            "missing_insights": evaluation["missing_insights"],
            "contradicted_claims": evaluation["contradicted_claims"],
            "retry_round": retry_round,
        }
    except Exception as exc:  # noqa: BLE001 - batch summary should keep going.
        return None, {
            "record_index": record_index,
            "source_split": record.get("split") or "",
            "table_id_paper": table_id_paper,
            "error": str(exc),
            "retry_round": retry_round,
        }
    return result, None


def build_fact_generation_prompt(
    table_record: Dict[str, Any],
    desc_record: Dict[str, Any] | None,
    max_table_facts: int = 16,
) -> str:
    desc_record = desc_record or {}
    payload = {
        "task": (
            "Generate a deduplicated, ranked set of important high-level table insights. "
            "Do not compare against the description in this step."
        ),
        "max_table_facts": max_table_facts,
        "output_schema": {
            "ranked_table_facts": [
                {
                    "rank": 1,
                    "type": "extremum | comparison | tie | ranking | pattern | gap | other",
                    "importance_score": 0.0,
                    "claim": "string",
                    "evidence": [
                        {
                            "row_index": 0,
                            "column_index": 0,
                            "row": "string",
                            "column": "string",
                            "value": "string",
                        }
                    ],
                    "reason_for_importance": "string",
                }
            ]
        },
        "instructions": [
            "Return at most max_table_facts facts.",
            "Do not create one fact per cell.",
            "Do not list raw cell values unless they support a higher-level insight.",
            "Deduplicate overlapping facts; keep the broader or more important version.",
            "Prefer extrema, comparisons, ties, rankings, notable gaps, or consistent patterns.",
            "Copy every evidence value exactly as it appears in the table, including letters, symbols, percent signs, commas, and suffixes.",
            "Every evidence object must include row_index and column_index from the provided table.",
            "Rank facts by importance for checking whether the description covers the table.",
        ],
        "table_metadata": {
            "table_id_paper": str(table_record.get("table_id_paper") or desc_record.get("table_id_paper") or ""),
            "paper_id": str(desc_record.get("paper_id") or ""),
            "table_id": str(desc_record.get("table_id") or ""),
            "caption": str(table_record.get("caption") or ""),
            "metrics_type": table_record.get("metrics_type") or [],
        },
        "table": _table_payload(table_record),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_coverage_prompt(
    table_record: Dict[str, Any],
    desc_record: Dict[str, Any] | None,
    selected_facts: Sequence[Dict[str, Any]],
    top_k: int,
) -> str:
    desc_record = desc_record or {}
    payload = {
        "task": (
            "Evaluate whether the description covers each selected top-K table insight. "
            "Do not create new table insights in this step."
        ),
        "selected_top_k": top_k,
        "output_schema": {
            "evaluated_insights": [
                {
                    "rank": 1,
                    "claim": "string",
                    "description_status": "covered | partially_covered | missing | contradicted",
                    "matched_text": "string",
                    "reason": "string",
                }
            ]
        },
        "instructions": [
            "Evaluate every selected insight exactly once.",
            "Use covered only when the description states the same table insight.",
            "Use partially_covered when the description states a weaker, less specific, or incomplete version.",
            "Use missing when the description does not mention the selected insight.",
            "Use contradicted when the description conflicts with the selected insight.",
            "Do not mark an insight missing merely because the description omits exact numbers if it clearly covers the same comparison or pattern.",
        ],
        "table_metadata": {
            "table_id_paper": str(table_record.get("table_id_paper") or desc_record.get("table_id_paper") or ""),
            "paper_id": str(desc_record.get("paper_id") or ""),
            "table_id": str(desc_record.get("table_id") or ""),
            "caption": str(table_record.get("caption") or ""),
        },
        "selected_table_insights": list(selected_facts),
        "description": str(desc_record.get("description") or ""),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def dynamic_top_k(
    fact_count: int,
    min_top_k: int = 3,
    max_top_k: int = 8,
    top_k_fraction: float = 0.5,
) -> int:
    if fact_count <= 0:
        return 0
    bounded_min = max(0, min_top_k)
    bounded_max = max(bounded_min, max_top_k)
    computed = math.ceil(fact_count * top_k_fraction)
    return min(fact_count, bounded_max, max(bounded_min, computed))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run pure LLM baseline v2 with ranked facts and dynamic top-K coverage."
    )
    parser.add_argument("data_dir", help="Path to numericNLG data directory.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test", "all"])
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--header-level", default="single", choices=["single", "all"])
    parser.add_argument("--records-file", default="")
    parser.add_argument("--model", default=os.environ.get("NUMERIC_NLG_LLM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE", DEFAULT_API_BASE))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--env-file", default="")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--failure-retries", type=int, default=2)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-all-results", action="store_true")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only a compact summary to the terminal. The output file still contains full results.",
    )
    parser.add_argument("--output", default="")
    parser.add_argument("--max-table-facts", type=int, default=16)
    parser.add_argument("--min-top-k", type=int, default=3)
    parser.add_argument("--max-top-k", type=int, default=8)
    parser.add_argument("--top-k-fraction", type=float, default=0.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.env_file:
        _load_env_file(args.env_file)
    api_key = os.environ.get(args.api_key_env, "")
    client = None
    if not args.dry_run:
        if not api_key:
            raise SystemExit(
                f"{args.api_key_env} is not set. Use --dry-run to inspect prompts without calling an LLM."
            )
        client = OpenAICompatibleClient(
            api_key=api_key,
            model=args.model,
            api_base=args.api_base,
            timeout=args.timeout,
        )

    summary = run_pure_llm_baseline_v2(
        data_dir=args.data_dir,
        split=args.split,
        limit=args.limit,
        examples=args.examples,
        header_level=args.header_level,
        client=client,
        dry_run=args.dry_run,
        sleep_seconds=args.sleep_seconds,
        retries=args.retries,
        start_index=args.start_index,
        output_path=args.output or None,
        records_file=args.records_file or None,
        failure_retries=args.failure_retries,
        save_all_results=args.save_all_results,
        max_table_facts=args.max_table_facts,
        min_top_k=args.min_top_k,
        max_top_k=args.max_top_k,
        top_k_fraction=args.top_k_fraction,
    )
    printable = _compact_summary(summary) if args.quiet else summary
    print(json.dumps(printable, ensure_ascii=False, indent=2))


def _normalize_ranked_table_facts(response: Dict[str, Any], max_table_facts: int) -> List[Dict[str, Any]]:
    raw_facts = response.get("ranked_table_facts") or response.get("table_facts") or []
    if not isinstance(raw_facts, list):
        return []
    normalized = []
    seen_claims = set()
    for index, fact in enumerate(raw_facts[:max_table_facts], start=1):
        if not isinstance(fact, dict):
            continue
        claim = str(fact.get("claim") or "").strip()
        if not claim:
            continue
        claim_key = claim.lower()
        if claim_key in seen_claims:
            continue
        seen_claims.add(claim_key)
        evidence = fact.get("evidence") if isinstance(fact.get("evidence"), list) else []
        normalized.append(
            {
                "rank": _safe_int(fact.get("rank"), index),
                "type": str(fact.get("type") or "other"),
                "importance_score": _safe_float(fact.get("importance_score"), None),
                "claim": claim,
                "evidence": evidence,
                "reason_for_importance": str(fact.get("reason_for_importance") or ""),
            }
        )
    normalized.sort(key=lambda item: item["rank"])
    for index, fact in enumerate(normalized, start=1):
        fact["rank"] = index
    return normalized


def _normalize_coverage_response(
    response: Dict[str, Any],
    selected_facts: Sequence[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    raw_items = response.get("evaluated_insights") or []
    if not isinstance(raw_items, list):
        raw_items = []
    by_rank = {}
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        rank = _safe_int(item.get("rank"), 0)
        if rank:
            by_rank[rank] = item

    evaluated = []
    for fact in selected_facts:
        rank = int(fact.get("rank") or len(evaluated) + 1)
        item = by_rank.get(rank, {})
        status = _normalize_status(item.get("description_status"))
        evaluated.append(
            {
                "rank": rank,
                "claim": str(item.get("claim") or fact.get("claim") or ""),
                "description_status": status,
                "matched_text": str(item.get("matched_text") or ""),
                "reason": str(item.get("reason") or ""),
                "table_fact": fact,
            }
        )

    return {
        "evaluated_insights": evaluated,
        "supported_claims": [item for item in evaluated if item["description_status"] == "covered"],
        "partially_covered_insights": [
            item for item in evaluated if item["description_status"] == "partially_covered"
        ],
        "missing_insights": [item for item in evaluated if item["description_status"] == "missing"],
        "contradicted_claims": [item for item in evaluated if item["description_status"] == "contradicted"],
    }


def _summarize_v2_run(
    split: str,
    header_level: str,
    total_source_records: int,
    candidate_count: int,
    start_index: int,
    requested: int,
    evaluated: Sequence[Dict[str, Any]],
    failed: Sequence[Dict[str, Any]],
    examples: int,
    dry_run: bool,
    records_file: str,
    failure_retries: int,
    save_all_results: bool,
    max_table_facts: int,
    min_top_k: int,
    max_top_k: int,
    top_k_fraction: float,
) -> Dict[str, Any]:
    fact_counts = [len(item.get("ranked_table_facts") or []) for item in evaluated if not dry_run]
    top_k_counts = [int((item.get("top_k_formula") or {}).get("selected_top_k") or 0) for item in evaluated if not dry_run]
    supported_counts = [len(item.get("supported_claims") or []) for item in evaluated if not dry_run]
    partial_counts = [len(item.get("partially_covered_insights") or []) for item in evaluated if not dry_run]
    missing_counts = [len(item.get("missing_insights") or []) for item in evaluated if not dry_run]
    contradicted_counts = [len(item.get("contradicted_claims") or []) for item in evaluated if not dry_run]
    summary = {
        "dataset": "numericNLG",
        "baseline": "pure_llm_table_text_correction_v2",
        "uses_rule_baseline": False,
        "dry_run": dry_run,
        "split": split,
        "header_level": header_level,
        "records_file": records_file,
        "total_source_records": total_source_records,
        "candidate_tables": candidate_count,
        "start_index": start_index,
        "requested": requested,
        "evaluated": len(evaluated),
        "failed": len(failed),
        "failure_retries": failure_retries,
        "max_table_facts": max_table_facts,
        "top_k_formula": {
            "min_top_k": min_top_k,
            "max_top_k": max_top_k,
            "top_k_fraction": top_k_fraction,
        },
        "average_ranked_table_facts": _mean(fact_counts),
        "average_selected_top_k": _mean(top_k_counts),
        "average_supported_claims": _mean(supported_counts),
        "average_partially_covered_insights": _mean(partial_counts),
        "average_missing_insights": _mean(missing_counts),
        "average_contradicted_claims": _mean(contradicted_counts),
        "examples": list(evaluated[:examples]),
        "failures": list(failed),
    }
    if save_all_results:
        summary["results"] = list(evaluated)
    return summary


def _normalize_status(status: Any) -> str:
    value = str(status or "").strip().lower()
    if value in {"covered", "partially_covered", "missing", "contradicted"}:
        return value
    if value in {"partial", "partially covered"}:
        return "partially_covered"
    return "missing"


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float | None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: Sequence[int]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 2)


def _compact_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "dataset",
        "baseline",
        "dry_run",
        "split",
        "records_file",
        "requested",
        "evaluated",
        "failed",
        "failure_retries",
        "max_table_facts",
        "top_k_formula",
        "average_ranked_table_facts",
        "average_selected_top_k",
        "average_supported_claims",
        "average_partially_covered_insights",
        "average_missing_insights",
        "average_contradicted_claims",
        "failures",
    ]
    return {key: summary.get(key) for key in keys if key in summary}


if __name__ == "__main__":
    main()
