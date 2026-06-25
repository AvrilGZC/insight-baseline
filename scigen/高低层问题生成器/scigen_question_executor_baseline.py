from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nlg.baseline.pure_llm_baseline import (  # noqa: E402
    DEFAULT_API_BASE,
    DEFAULT_MODEL,
    OpenAICompatibleClient,
    _complete_json_with_retries,
    _load_env_file,
)


SUPPORTED_OPERATIONS = {
    "column_extrema",
    "row_extrema",
    "row_comparison",
    "column_correlation",
    "threshold_count",
    "trend",
}


def run_scigen_question_executor_baseline(
    dataset_file: str | Path,
    limit: int,
    start_index: int = 0,
    examples: int = 3,
    client: OpenAICompatibleClient | None = None,
    dry_run: bool = False,
    planner_mode: str = "rule",
    max_high_questions: int = 4,
    max_low_questions: int = 12,
    use_llm_summarizer: bool = True,
    use_llm_verifier: bool = True,
    use_llm_matching: bool = True,
    retries: int = 2,
    failure_retries: int = 2,
    sleep_seconds: float = 0.0,
    output_path: str | Path | None = None,
    save_all_results: bool = False,
) -> Dict[str, Any]:
    records = load_scigen_records(dataset_file)
    selected = records[start_index:] if limit <= 0 else records[start_index : start_index + limit]
    evaluated: List[Dict[str, Any]] = []
    pending = [(start_index + offset, record) for offset, record in enumerate(selected)]
    failed: List[Dict[str, Any]] = []

    for retry_round in range(max(0, failure_retries) + 1):
        failed = []
        next_pending = []
        for record_index, record in pending:
            result, error = _evaluate_record(
                record=record,
                record_index=record_index,
                client=client,
                dry_run=dry_run,
                planner_mode=planner_mode,
                max_high_questions=max_high_questions,
                max_low_questions=max_low_questions,
                use_llm_summarizer=use_llm_summarizer,
                use_llm_verifier=use_llm_verifier,
                use_llm_matching=use_llm_matching,
                retries=retries,
                retry_round=retry_round,
            )
            if error:
                failed.append(error)
                next_pending.append((record_index, record))
                continue
            assert result is not None
            evaluated.append(result)
            if sleep_seconds > 0 and not dry_run:
                time.sleep(sleep_seconds)
        pending = next_pending
        if not pending:
            break

    evaluated.sort(key=lambda item: int(item.get("record_index", 0)))
    summary = _summarize_run(
        dataset_file=str(dataset_file),
        total_records=len(records),
        start_index=start_index,
        requested=len(selected),
        evaluated=evaluated,
        failed=failed,
        examples=examples,
        dry_run=dry_run,
        planner_mode=planner_mode,
        max_high_questions=max_high_questions,
        max_low_questions=max_low_questions,
        use_llm_summarizer=use_llm_summarizer,
        use_llm_verifier=use_llm_verifier,
        use_llm_matching=use_llm_matching,
        failure_retries=failure_retries,
        save_all_results=save_all_results,
    )
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def load_scigen_records(dataset_file: str | Path) -> List[Dict[str, Any]]:
    path = Path(dataset_file)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            json_names = [name for name in archive.namelist() if name.endswith(".json")]
            if not json_names:
                raise ValueError(f"No JSON file found in {path}")
            data = json.loads(archive.read(json_names[0]).decode("utf-8"))
    else:
        data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, dict):
        items = sorted(data.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else str(item[0]))
        return [{**record, "record_id": str(record_id)} for record_id, record in items]
    if isinstance(data, list):
        return [{**record, "record_id": str(index)} for index, record in enumerate(data)]
    raise ValueError("SciGen file must be a JSON object, JSON list, or ZIP containing one JSON file.")


def _evaluate_record(
    record: Dict[str, Any],
    record_index: int,
    client: OpenAICompatibleClient | None,
    dry_run: bool,
    planner_mode: str,
    max_high_questions: int,
    max_low_questions: int,
    use_llm_summarizer: bool,
    use_llm_verifier: bool,
    use_llm_matching: bool,
    retries: int,
    retry_round: int,
) -> tuple[Dict[str, Any] | None, Dict[str, Any] | None]:
    record_id = str(record.get("record_id") or record_index)
    try:
        table = normalize_scigen_table(record)
        high_prompt = build_high_level_question_prompt(record, table, max_high_questions)
        if dry_run:
            low_prompt = build_low_level_plan_prompt(record, table, [], max_low_questions)
            return {
                "record_index": record_index,
                "record_id": record_id,
                "paper_id": str(record.get("paper_id") or ""),
                "dry_run": True,
                "planner_mode": planner_mode,
                "normalized_table": table_for_json(table),
                "high_level_question_prompt": high_prompt,
                "low_level_plan_prompt": low_prompt,
                "retry_round": retry_round,
            }, None

        if planner_mode not in {"rule", "llm", "hybrid"}:
            raise ValueError(f"Unsupported planner mode: {planner_mode}")
        if planner_mode in {"llm", "hybrid"} and client is None:
            raise ValueError("LLM client is required for llm or hybrid planner mode.")

        high_questions = _generate_high_questions(
            record=record,
            table=table,
            client=client,
            retries=retries,
            planner_mode=planner_mode,
            max_high_questions=max_high_questions,
            prompt=high_prompt,
        )
        low_prompt = build_low_level_plan_prompt(record, table, high_questions, max_low_questions)
        low_level_questions = _generate_low_level_questions(
            record=record,
            table=table,
            high_questions=high_questions,
            client=client,
            retries=retries,
            planner_mode=planner_mode,
            max_low_questions=max_low_questions,
            prompt=low_prompt,
        )
        executed_answers = execute_low_level_questions(table, low_level_questions)
        raw_candidate_insights = _summarize_answers(
            record=record,
            table=table,
            high_questions=high_questions,
            low_level_questions=low_level_questions,
            executed_answers=executed_answers,
            client=client,
            retries=retries,
            use_llm_summarizer=use_llm_summarizer,
        )
        candidate_insights, verifier_decisions = _verify_candidate_insights(
            record=record,
            table=table,
            candidate_insights=raw_candidate_insights,
            executed_answers=executed_answers,
            client=client,
            retries=retries,
            use_llm_verifier=use_llm_verifier,
        )
        matching = _match_candidates(
            record=record,
            table=table,
            candidate_insights=candidate_insights,
            client=client,
            retries=retries,
            use_llm_matching=use_llm_matching,
        )
        return {
            "record_index": record_index,
            "record_id": record_id,
            "paper_id": str(record.get("paper_id") or ""),
            "paper": str(record.get("paper") or ""),
            "dry_run": False,
            "planner_mode": planner_mode,
            "normalized_table": table_for_json(table),
            "high_level_questions": high_questions,
            "low_level_questions": low_level_questions,
            "executed_answers": executed_answers,
            "raw_candidate_insights": raw_candidate_insights,
            "verifier_decisions": verifier_decisions,
            "candidate_insights": candidate_insights,
            "evaluated_insights": matching["evaluated_insights"],
            "supported_claims": matching["supported_claims"],
            "partially_covered_insights": matching["partially_covered_insights"],
            "missing_insights": matching["missing_insights"],
            "contradicted_claims": matching["contradicted_claims"],
            "retry_round": retry_round,
        }, None
    except Exception as exc:  # noqa: BLE001 - batch runs should keep going.
        return None, {
            "record_index": record_index,
            "record_id": record_id,
            "paper_id": str(record.get("paper_id") or ""),
            "error": str(exc),
            "retry_round": retry_round,
        }


def normalize_scigen_table(record: Dict[str, Any]) -> Dict[str, Any]:
    raw_columns = [_clean_cell(value) for value in record.get("table_column_names") or []]
    rows = [[_clean_cell(value) for value in row] for row in record.get("table_content_values") or []]
    if not raw_columns or not rows:
        raise ValueError("SciGen record has no table columns or rows.")

    columns = list(raw_columns)
    if _looks_like_subheader_row(rows[0], rows[1:] if len(rows) > 1 else [], columns):
        subheader = rows[0]
        columns = [_combine_header(top, lower) for top, lower in zip(columns, subheader)]
        rows = rows[1:]

    numeric_counts = []
    for col_idx in range(len(columns)):
        numeric_counts.append(sum(_parse_number(row[col_idx]) is not None for row in rows if col_idx < len(row)))
    numeric_indices = [idx for idx, count in enumerate(numeric_counts) if count > 0]
    if not numeric_indices:
        raise ValueError("SciGen table has no executable numeric columns.")

    first_numeric_idx = min(numeric_indices)
    label_indices = list(range(first_numeric_idx)) or [0]
    value_indices = [idx for idx in numeric_indices if idx not in label_indices]
    if not value_indices:
        raise ValueError("SciGen table has numeric labels but no numeric metric columns.")

    row_labels = []
    values = []
    value_columns = _dedupe_labels([columns[idx] for idx in value_indices])
    for row_idx, row in enumerate(rows):
        label_parts = [row[idx] for idx in label_indices if idx < len(row) and row[idx]]
        row_label = " | ".join(label_parts) or f"row_{row_idx}"
        parsed = [_parse_number(row[idx]) if idx < len(row) else None for idx in value_indices]
        if all(value is None for value in parsed):
            continue
        row_labels.append(row_label)
        values.append([float(value) if value is not None else math.nan for value in parsed])

    if not values:
        raise ValueError("SciGen table has no numeric rows after normalization.")
    row_labels = _dedupe_labels(row_labels)
    df = pd.DataFrame(values, index=row_labels, columns=value_columns).dropna(axis=1, how="all")
    if df.empty:
        raise ValueError("SciGen table has no non-empty numeric metric columns.")

    return {
        "df": df,
        "row_label_columns": [columns[idx] for idx in label_indices],
        "source_columns": columns,
        "caption": str(record.get("table_caption") or ""),
    }


def table_for_json(table: Dict[str, Any]) -> Dict[str, Any]:
    df = table["df"]
    return {
        "caption": table.get("caption", ""),
        "row_label_columns": table.get("row_label_columns", []),
        "columns": list(df.columns),
        "rows": [
            {
                "row": str(row_label),
                "values": {
                    str(column): None if pd.isna(value) else float(value)
                    for column, value in df.loc[row_label].items()
                },
            }
            for row_label in df.index
        ],
    }


def build_high_level_question_prompt(
    record: Dict[str, Any],
    table: Dict[str, Any],
    max_high_questions: int = 4,
) -> str:
    payload = {
        "task": (
            "Generate high-level analytical questions for a SciGen scientific table. "
            "The questions should guide executable table reasoning and candidate insight generation."
        ),
        "max_high_questions": max_high_questions,
        "output_schema": {
            "high_level_questions": [
                {
                    "id": "h1",
                    "question": "string",
                    "rationale": "string",
                    "suggested_operations": [
                        "column_extrema | row_extrema | row_comparison | column_correlation | threshold_count | trend"
                    ],
                }
            ]
        },
        "instructions": [
            "Use only the paper title, table caption, headers, and table cells.",
            "Prefer questions about maxima, minima, performance gaps, trends, correlations, or threshold counts.",
            "Do not ask questions that require information outside the provided table and caption.",
            "Return only valid JSON.",
        ],
        "paper": str(record.get("paper") or ""),
        "paper_id": str(record.get("paper_id") or ""),
        "table_caption": str(record.get("table_caption") or ""),
        "normalized_table": table_for_json(table),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_low_level_plan_prompt(
    record: Dict[str, Any],
    table: Dict[str, Any],
    high_level_questions: Sequence[Dict[str, Any]],
    max_low_questions: int = 12,
) -> str:
    payload = {
        "task": "Decompose high-level SciGen table questions into executable low-level operations.",
        "max_low_questions": max_low_questions,
        "supported_operations": sorted(SUPPORTED_OPERATIONS),
        "operation_schema": {
            "column_extrema": {"column": "exact metric column", "kind": "max | min"},
            "row_extrema": {"row": "exact row label", "kind": "max | min"},
            "row_comparison": {
                "row_a": "exact row label",
                "row_b": "exact row label",
                "column": "exact metric column or empty for row average",
            },
            "column_correlation": {"column_x": "exact metric column", "column_y": "exact metric column"},
            "threshold_count": {"column": "exact metric column", "operator": "> | >= | < | <=", "threshold": 0.0},
            "trend": {"row": "exact row label"},
        },
        "output_schema": {
            "low_level_questions": [
                {
                    "id": "q1",
                    "parent_id": "h1",
                    "question": "string",
                    "operation": "column_extrema",
                    "params": {},
                    "importance_score": 0.0,
                }
            ]
        },
        "instructions": [
            "Use exact row labels and metric columns from normalized_table.",
            "Emit only operations answerable by the executor.",
            "Do not emit pandas code or SQL.",
            "Return only valid JSON.",
        ],
        "high_level_questions": list(high_level_questions),
        "normalized_table": table_for_json(table),
        "table_caption": str(record.get("table_caption") or ""),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_summarization_prompt(
    record: Dict[str, Any],
    table: Dict[str, Any],
    high_questions: Sequence[Dict[str, Any]],
    low_level_questions: Sequence[Dict[str, Any]],
    executed_answers: Sequence[Dict[str, Any]],
) -> str:
    payload = {
        "task": "Summarize executable SciGen table answers into candidate insights.",
        "output_schema": {
            "candidate_insights": [
                {
                    "rank": 1,
                    "claim": "string",
                    "supporting_answer_ids": ["q1"],
                    "evidence": [],
                    "confidence": "high | medium | low",
                }
            ]
        },
        "instructions": [
            "Use only successful executed answers.",
            "Do not invent values, comparisons, or outside-paper context.",
            "Prefer concise claims that could plausibly appear in a scientific table description.",
            "Return only valid JSON.",
        ],
        "paper": str(record.get("paper") or ""),
        "table_caption": str(record.get("table_caption") or ""),
        "normalized_table": table_for_json(table),
        "high_level_questions": list(high_questions),
        "low_level_questions": list(low_level_questions),
        "executed_answers": list(executed_answers),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_candidate_verification_prompt(
    record: Dict[str, Any],
    table: Dict[str, Any],
    candidate_insights: Sequence[Dict[str, Any]],
    executed_answers: Sequence[Dict[str, Any]],
) -> str:
    payload = {
        "task": "Verify candidate SciGen insights against executor answers before gold-text matching.",
        "output_schema": {
            "verified_insights": [
                {
                    "rank": 1,
                    "decision": "supported | rewrite | drop",
                    "claim": "string",
                    "supporting_answer_ids": ["q1"],
                    "reason": "string",
                }
            ]
        },
        "instructions": [
            "Evaluate every candidate insight exactly once.",
            "A kept insight must be fully supported by its listed executor answers.",
            "Use supported only when the candidate claim does not add facts, entities, trends, causes, rankings, thresholds, or interpretations beyond those answers.",
            "Use rewrite when a narrower claim can be written using only the listed executor answers; put the rewritten claim in claim.",
            "Use drop when the claim is unsupported, over-generalized, causal, interpretive, or uses answer ids that do not support it.",
            "Every supported or rewritten insight must have at least one valid supporting_answer_ids entry.",
            "Do not use the gold description, paper knowledge, caption implications, or unstated table facts.",
            "Return only valid JSON.",
        ],
        "paper": str(record.get("paper") or ""),
        "table_caption": str(record.get("table_caption") or ""),
        "normalized_table": table_for_json(table),
        "candidate_insights": list(candidate_insights),
        "executed_answers": [
            answer for answer in executed_answers if answer.get("status") == "answered"
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_matching_prompt(
    record: Dict[str, Any],
    table: Dict[str, Any],
    candidate_insights: Sequence[Dict[str, Any]],
) -> str:
    payload = {
        "task": "Match candidate SciGen table insights against the gold table description.",
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
            "Evaluate every candidate insight exactly once.",
            "Use covered only when the gold text states the same table insight.",
            "Use partially_covered when it states a weaker or incomplete version.",
            "Use contradicted only when the gold text conflicts with the candidate.",
            "Do not create new candidate insights in this step.",
            "Return only valid JSON.",
        ],
        "table_caption": str(record.get("table_caption") or ""),
        "normalized_table": table_for_json(table),
        "candidate_insights": list(candidate_insights),
        "gold_description": str(record.get("text") or ""),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def execute_low_level_questions(
    table: Dict[str, Any],
    low_level_questions: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    answers = []
    for index, question in enumerate(low_level_questions, start=1):
        qid = str(question.get("id") or f"q{index}")
        operation = str(question.get("operation") or "")
        params = question.get("params") if isinstance(question.get("params"), dict) else {}
        base = {
            "id": qid,
            "parent_id": str(question.get("parent_id") or ""),
            "question": str(question.get("question") or ""),
            "operation": operation,
            "params": params,
        }
        try:
            if operation not in SUPPORTED_OPERATIONS:
                raise ValueError(f"Unsupported operation: {operation}")
            answers.append({**base, "status": "answered", **_execute_operation(table, operation, params)})
        except Exception as exc:  # noqa: BLE001 - preserve failures for inspection.
            answers.append({**base, "status": "failed", "error": str(exc)})
    return answers


def _execute_operation(table: Dict[str, Any], operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
    df = table["df"]
    rows = list(df.index)
    columns = list(df.columns)
    if operation == "column_extrema":
        column = _resolve_label(params.get("column"), columns)
        kind = _normalize_kind(params.get("kind"))
        series = df[column].dropna()
        row = str(series.idxmax() if kind == "max" else series.idxmin())
        value = float(series.loc[row])
        return _answer(
            f"{row} has the {kind} value for {column} at {_format_number(value)}.",
            value,
            [{"row": row, "column": column, "value": _format_number(value)}],
        )
    if operation == "row_extrema":
        row = _resolve_label(params.get("row"), rows)
        kind = _normalize_kind(params.get("kind"))
        series = df.loc[row].dropna()
        column = str(series.idxmax() if kind == "max" else series.idxmin())
        value = float(series.loc[column])
        return _answer(
            f"{row} has its {kind} value on {column} at {_format_number(value)}.",
            value,
            [{"row": row, "column": column, "value": _format_number(value)}],
        )
    if operation == "row_comparison":
        row_a = _resolve_label(params.get("row_a"), rows)
        row_b = _resolve_label(params.get("row_b"), rows)
        column_param = str(params.get("column") or "").strip()
        if column_param:
            column = _resolve_label(column_param, columns)
            value_a = float(df.loc[row_a, column])
            value_b = float(df.loc[row_b, column])
            metric = column
        else:
            value_a = float(df.loc[row_a].mean())
            value_b = float(df.loc[row_b].mean())
            metric = "average numeric value"
        diff = value_b - value_a
        direction = "higher than" if diff > 0 else "lower than" if diff < 0 else "equal to"
        return _answer(
            f"{row_b} is {_format_number(abs(diff))} {direction} {row_a} on {metric}.",
            diff,
            [
                {"row": row_a, "column": metric, "value": _format_number(value_a)},
                {"row": row_b, "column": metric, "value": _format_number(value_b)},
            ],
        )
    if operation == "column_correlation":
        column_x = _resolve_label(params.get("column_x"), columns)
        column_y = _resolve_label(params.get("column_y"), columns)
        pair = df[[column_x, column_y]].dropna()
        if len(pair) < 2 or pair[column_x].nunique() < 2 or pair[column_y].nunique() < 2:
            raise ValueError(f"Not enough variation to correlate {column_x} and {column_y}.")
        corr = float(pair[column_x].corr(pair[column_y]))
        direction = "positive" if corr > 0 else "negative" if corr < 0 else "zero"
        return _answer(
            f"{column_x} and {column_y} have a {direction} correlation of {_format_number(corr)}.",
            corr,
            [
                {"column": column_x, "values": [_format_number(value) for value in pair[column_x].tolist()]},
                {"column": column_y, "values": [_format_number(value) for value in pair[column_y].tolist()]},
            ],
        )
    if operation == "threshold_count":
        column = _resolve_label(params.get("column"), columns)
        operator = str(params.get("operator") or ">").strip()
        threshold = float(params.get("threshold"))
        mask = _threshold_mask(df[column], operator, threshold)
        matching_rows = [str(row) for row in df.index[mask.fillna(False)].tolist()]
        return _answer(
            f"{len(matching_rows)} rows have {column} {operator} {_format_number(threshold)}.",
            len(matching_rows),
            [{"rows": matching_rows, "column": column, "threshold": _format_number(threshold)}],
        )
    if operation == "trend":
        row = _resolve_label(params.get("row"), rows)
        series = df.loc[row].dropna()
        values = [float(value) for value in series.tolist()]
        diffs = [b - a for a, b in zip(values, values[1:])]
        if all(diff >= 0 for diff in diffs) and any(diff > 0 for diff in diffs):
            trend = "increases"
        elif all(diff <= 0 for diff in diffs) and any(diff < 0 for diff in diffs):
            trend = "decreases"
        else:
            trend = "varies"
        return _answer(
            f"{row} {trend} across the ordered numeric columns.",
            values[-1] - values[0] if values else 0.0,
            [
                {"row": row, "column": column, "value": _format_number(value)}
                for column, value in zip(series.index, values)
            ],
        )
    raise ValueError(f"Unsupported operation: {operation}")


def _generate_high_questions(
    record: Dict[str, Any],
    table: Dict[str, Any],
    client: OpenAICompatibleClient | None,
    retries: int,
    planner_mode: str,
    max_high_questions: int,
    prompt: str,
) -> List[Dict[str, Any]]:
    questions = _rule_high_questions(record, table, max_high_questions) if planner_mode in {"rule", "hybrid"} else []
    if planner_mode in {"llm", "hybrid"}:
        assert client is not None
        response = _complete_json_with_retries(client, prompt, retries=retries)
        questions = _merge_by_question(questions, _normalize_high_questions(response, max_high_questions), max_high_questions)
    return questions[:max_high_questions]


def _generate_low_level_questions(
    record: Dict[str, Any],
    table: Dict[str, Any],
    high_questions: Sequence[Dict[str, Any]],
    client: OpenAICompatibleClient | None,
    retries: int,
    planner_mode: str,
    max_low_questions: int,
    prompt: str,
) -> List[Dict[str, Any]]:
    questions = _rule_low_level_questions(table, max_low_questions) if planner_mode in {"rule", "hybrid"} else []
    if planner_mode in {"llm", "hybrid"}:
        assert client is not None
        response = _complete_json_with_retries(client, prompt, retries=retries)
        questions = _merge_by_question(questions, _normalize_low_level_questions(response, max_low_questions), max_low_questions)
    return questions[:max_low_questions]


def _summarize_answers(
    record: Dict[str, Any],
    table: Dict[str, Any],
    high_questions: Sequence[Dict[str, Any]],
    low_level_questions: Sequence[Dict[str, Any]],
    executed_answers: Sequence[Dict[str, Any]],
    client: OpenAICompatibleClient | None,
    retries: int,
    use_llm_summarizer: bool,
) -> List[Dict[str, Any]]:
    if use_llm_summarizer and client is not None:
        prompt = build_summarization_prompt(record, table, high_questions, low_level_questions, executed_answers)
        response = _complete_json_with_retries(client, prompt, retries=retries)
        return _normalize_candidate_insights(response, executed_answers)

    candidates = []
    for answer in executed_answers:
        if answer.get("status") != "answered":
            continue
        candidates.append(
            {
                "rank": len(candidates) + 1,
                "claim": str(answer.get("answer") or ""),
                "supporting_answer_ids": [str(answer.get("id") or "")],
                "evidence": answer.get("evidence") if isinstance(answer.get("evidence"), list) else [],
                "confidence": "high",
            }
        )
    return candidates


def _verify_candidate_insights(
    record: Dict[str, Any],
    table: Dict[str, Any],
    candidate_insights: Sequence[Dict[str, Any]],
    executed_answers: Sequence[Dict[str, Any]],
    client: OpenAICompatibleClient | None,
    retries: int,
    use_llm_verifier: bool,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if use_llm_verifier and client is not None:
        prompt = build_candidate_verification_prompt(record, table, candidate_insights, executed_answers)
        response = _complete_json_with_retries(client, prompt, retries=retries)
        return _normalize_verifier_response(response, candidate_insights, executed_answers)
    return _rule_verify_candidate_insights(candidate_insights, executed_answers)


def _match_candidates(
    record: Dict[str, Any],
    table: Dict[str, Any],
    candidate_insights: Sequence[Dict[str, Any]],
    client: OpenAICompatibleClient | None,
    retries: int,
    use_llm_matching: bool,
) -> Dict[str, List[Dict[str, Any]]]:
    if use_llm_matching and client is not None:
        prompt = build_matching_prompt(record, table, candidate_insights)
        response = _complete_json_with_retries(client, prompt, retries=retries)
        evaluated = _normalize_matching_response(response, candidate_insights)
    else:
        evaluated = _lexical_match_candidates(candidate_insights, str(record.get("text") or ""))
    return {
        "evaluated_insights": evaluated,
        "supported_claims": [item for item in evaluated if item["description_status"] == "covered"],
        "partially_covered_insights": [item for item in evaluated if item["description_status"] == "partially_covered"],
        "missing_insights": [item for item in evaluated if item["description_status"] == "missing"],
        "contradicted_claims": [item for item in evaluated if item["description_status"] == "contradicted"],
    }


def _rule_high_questions(record: Dict[str, Any], table: Dict[str, Any], max_high_questions: int) -> List[Dict[str, Any]]:
    caption = str(record.get("table_caption") or "the table")
    questions = [
        {
            "id": "h1",
            "question": f"What are the strongest results or extrema in {caption}?",
            "rationale": "SciGen descriptions often mention best-performing systems or highest scores.",
            "suggested_operations": ["column_extrema"],
        },
        {
            "id": "h2",
            "question": "Which systems or settings differ most on the key metrics?",
            "rationale": "Arithmetic gaps are central to reasoning-aware table descriptions.",
            "suggested_operations": ["row_comparison"],
        },
    ]
    if len(table["df"].columns) >= 2:
        questions.append(
            {
                "id": "h3",
                "question": "Do metrics move together or show tradeoffs across rows?",
                "rationale": "Metric relationships can support higher-level scientific claims.",
                "suggested_operations": ["column_correlation", "trend"],
            }
        )
    return questions[:max_high_questions]


def _rule_low_level_questions(table: Dict[str, Any], max_low_questions: int) -> List[Dict[str, Any]]:
    df = table["df"]
    questions: List[Dict[str, Any]] = []
    for column in df.columns:
        questions.append(
            {
                "id": f"q{len(questions) + 1}",
                "parent_id": "h1",
                "question": f"Which row has the maximum value for {column}?",
                "operation": "column_extrema",
                "params": {"column": str(column), "kind": "max"},
                "importance_score": 0.9,
            }
        )
        if len(questions) >= max_low_questions:
            return questions
    if len(df.index) >= 2:
        first, last = str(df.index[0]), str(df.index[-1])
        for column in list(df.columns)[:3]:
            questions.append(
                {
                    "id": f"q{len(questions) + 1}",
                    "parent_id": "h2",
                    "question": f"How does {last} compare with {first} on {column}?",
                    "operation": "row_comparison",
                    "params": {"row_a": first, "row_b": last, "column": str(column)},
                    "importance_score": 0.7,
                }
            )
            if len(questions) >= max_low_questions:
                return questions
    if len(df.columns) >= 2:
        questions.append(
            {
                "id": f"q{len(questions) + 1}",
                "parent_id": "h3",
                "question": f"What is the correlation between {df.columns[0]} and {df.columns[1]}?",
                "operation": "column_correlation",
                "params": {"column_x": str(df.columns[0]), "column_y": str(df.columns[1])},
                "importance_score": 0.6,
            }
        )
    return questions[:max_low_questions]


def _normalize_high_questions(response: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
    raw = response.get("high_level_questions") or response.get("questions") or []
    if not isinstance(raw, list):
        return []
    normalized = []
    for index, item in enumerate(raw[:limit], start=1):
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "").strip()
        if not question:
            continue
        normalized.append(
            {
                "id": str(item.get("id") or f"h{index}"),
                "question": question,
                "rationale": str(item.get("rationale") or ""),
                "suggested_operations": [
                    str(op) for op in item.get("suggested_operations", []) if str(op) in SUPPORTED_OPERATIONS
                ],
            }
        )
    return normalized


def _normalize_low_level_questions(response: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
    raw = response.get("low_level_questions") or response.get("questions") or []
    if not isinstance(raw, list):
        return []
    normalized = []
    for index, item in enumerate(raw[:limit], start=1):
        if not isinstance(item, dict):
            continue
        operation = str(item.get("operation") or "")
        if operation not in SUPPORTED_OPERATIONS:
            continue
        normalized.append(
            {
                "id": str(item.get("id") or f"q{index}"),
                "parent_id": str(item.get("parent_id") or ""),
                "question": str(item.get("question") or ""),
                "operation": operation,
                "params": item.get("params") if isinstance(item.get("params"), dict) else {},
                "importance_score": _safe_float(item.get("importance_score"), 0.0),
            }
        )
    return normalized


def _normalize_candidate_insights(response: Dict[str, Any], executed_answers: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    raw = response.get("candidate_insights") or response.get("insights") or []
    if not isinstance(raw, list):
        return []
    answer_ids = {str(answer.get("id") or "") for answer in executed_answers}
    normalized = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim") or "").strip()
        if not claim:
            continue
        supporting_ids = [str(answer_id) for answer_id in item.get("supporting_answer_ids", []) if str(answer_id) in answer_ids]
        normalized.append(
            {
                "rank": _safe_int(item.get("rank"), index),
                "claim": claim,
                "supporting_answer_ids": supporting_ids,
                "evidence": item.get("evidence") if isinstance(item.get("evidence"), list) else [],
                "confidence": str(item.get("confidence") or "medium"),
            }
        )
    normalized.sort(key=lambda item: item["rank"])
    for index, item in enumerate(normalized, start=1):
        item["rank"] = index
    return normalized


def _normalize_verifier_response(
    response: Dict[str, Any],
    candidate_insights: Sequence[Dict[str, Any]],
    executed_answers: Sequence[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    raw = response.get("verified_insights") or response.get("verifier_decisions") or []
    if not isinstance(raw, list):
        raw = []
    valid_answers = _answer_by_id(executed_answers)
    candidates_by_rank = {
        _safe_int(candidate.get("rank"), index): candidate
        for index, candidate in enumerate(candidate_insights, start=1)
    }
    decisions = []
    kept = []
    seen_ranks = set()
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            continue
        rank = _safe_int(item.get("rank"), index)
        candidate = candidates_by_rank.get(rank, {})
        seen_ranks.add(rank)
        decision = str(item.get("decision") or "drop").strip().lower()
        if decision not in {"supported", "rewrite", "drop"}:
            decision = "drop"
        supporting_ids = _valid_supporting_ids(item.get("supporting_answer_ids"), candidate, valid_answers)
        claim = str(item.get("claim") or candidate.get("claim") or "").strip()
        reason = str(item.get("reason") or "")
        if decision in {"supported", "rewrite"} and (not supporting_ids or not claim):
            decision = "drop"
            reason = reason or "Verifier did not provide a supported claim with valid executor answer ids."
        decision_item = {
            "rank": rank,
            "decision": decision,
            "claim": claim,
            "supporting_answer_ids": supporting_ids,
            "reason": reason,
            "original_candidate": candidate,
        }
        decisions.append(decision_item)
        if decision in {"supported", "rewrite"}:
            kept.append(
                _verified_candidate_from_decision(
                    candidate=candidate,
                    decision=decision_item,
                    valid_answers=valid_answers,
                )
            )

    for rank, candidate in candidates_by_rank.items():
        if rank in seen_ranks:
            continue
        fallback_kept, fallback_decisions = _rule_verify_candidate_insights([candidate], executed_answers)
        decisions.extend(fallback_decisions)
        kept.extend(fallback_kept)

    kept.sort(key=lambda item: item["rank"])
    for index, item in enumerate(kept, start=1):
        item["rank"] = index
    return kept, decisions


def _rule_verify_candidate_insights(
    candidate_insights: Sequence[Dict[str, Any]],
    executed_answers: Sequence[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    valid_answers = _answer_by_id(executed_answers)
    kept = []
    decisions = []
    for index, candidate in enumerate(candidate_insights, start=1):
        rank = _safe_int(candidate.get("rank"), index)
        supporting_ids = _valid_supporting_ids(candidate.get("supporting_answer_ids"), candidate, valid_answers)
        claim = str(candidate.get("claim") or "").strip()
        if not supporting_ids:
            decision = {
                "rank": rank,
                "decision": "drop",
                "claim": claim,
                "supporting_answer_ids": [],
                "reason": "Candidate has no valid executor answer binding.",
                "original_candidate": candidate,
            }
            decisions.append(decision)
            continue

        answer_text = " ".join(
            json.dumps(valid_answers[answer_id], ensure_ascii=False)
            for answer_id in supporting_ids
        )
        claim_numbers = set(re.findall(r"-?\d+(?:\.\d+)?", claim))
        answer_numbers = set(re.findall(r"-?\d+(?:\.\d+)?", answer_text))
        claim_tokens = set(_tokens(claim))
        answer_tokens = set(_tokens(answer_text))
        token_overlap = len(claim_tokens & answer_tokens) / max(1, len(claim_tokens))
        unsupported_numbers = sorted(claim_numbers - answer_numbers)
        if unsupported_numbers:
            reason = f"Candidate includes numbers not present in supporting executor answers: {unsupported_numbers}."
            decision_name = "drop"
        elif token_overlap < 0.2:
            reason = f"Candidate has low lexical overlap with supporting executor answers: {token_overlap:.2f}."
            decision_name = "drop"
        else:
            reason = "Candidate has valid executor bindings and no unsupported numeric values."
            decision_name = "supported"

        decision = {
            "rank": rank,
            "decision": decision_name,
            "claim": claim,
            "supporting_answer_ids": supporting_ids,
            "reason": reason,
            "original_candidate": candidate,
        }
        decisions.append(decision)
        if decision_name == "supported":
            kept.append(
                _verified_candidate_from_decision(
                    candidate=candidate,
                    decision=decision,
                    valid_answers=valid_answers,
                )
            )
    kept.sort(key=lambda item: item["rank"])
    for index, item in enumerate(kept, start=1):
        item["rank"] = index
    return kept, decisions


def _verified_candidate_from_decision(
    candidate: Dict[str, Any],
    decision: Dict[str, Any],
    valid_answers: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    supporting_ids = list(decision.get("supporting_answer_ids") or [])
    evidence = []
    for answer_id in supporting_ids:
        answer_evidence = valid_answers.get(answer_id, {}).get("evidence")
        if isinstance(answer_evidence, list):
            evidence.extend(answer_evidence)
    return {
        "rank": _safe_int(candidate.get("rank"), _safe_int(decision.get("rank"), 0)),
        "claim": str(decision.get("claim") or candidate.get("claim") or ""),
        "supporting_answer_ids": supporting_ids,
        "evidence": evidence or (candidate.get("evidence") if isinstance(candidate.get("evidence"), list) else []),
        "confidence": str(candidate.get("confidence") or "medium"),
        "verifier_status": str(decision.get("decision") or ""),
        "verifier_reason": str(decision.get("reason") or ""),
    }


def _answer_by_id(executed_answers: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        str(answer.get("id") or ""): answer
        for answer in executed_answers
        if answer.get("status") == "answered" and str(answer.get("id") or "")
    }


def _valid_supporting_ids(
    raw_ids: Any,
    candidate: Dict[str, Any],
    valid_answers: Dict[str, Dict[str, Any]],
) -> List[str]:
    if isinstance(raw_ids, list):
        ids = [str(answer_id) for answer_id in raw_ids]
    else:
        ids = []
    if not ids and isinstance(candidate.get("supporting_answer_ids"), list):
        ids = [str(answer_id) for answer_id in candidate.get("supporting_answer_ids", [])]
    seen = set()
    valid = []
    for answer_id in ids:
        if answer_id in valid_answers and answer_id not in seen:
            seen.add(answer_id)
            valid.append(answer_id)
    return valid


def _normalize_matching_response(response: Dict[str, Any], candidate_insights: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    raw = response.get("evaluated_insights") or []
    if not isinstance(raw, list):
        raw = []
    by_rank = {}
    for item in raw:
        if isinstance(item, dict):
            rank = _safe_int(item.get("rank"), 0)
            if rank:
                by_rank[rank] = item
    evaluated = []
    for index, candidate in enumerate(candidate_insights, start=1):
        rank = _safe_int(candidate.get("rank"), index)
        item = by_rank.get(rank, {})
        evaluated.append(
            {
                "rank": rank,
                "claim": str(item.get("claim") or candidate.get("claim") or ""),
                "description_status": _normalize_status(item.get("description_status")),
                "matched_text": str(item.get("matched_text") or ""),
                "reason": str(item.get("reason") or ""),
                "candidate_insight": candidate,
            }
        )
    return evaluated


def _lexical_match_candidates(candidate_insights: Sequence[Dict[str, Any]], description: str) -> List[Dict[str, Any]]:
    description_tokens = set(_tokens(description.replace("[CONTINUE]", " ")))
    evaluated = []
    for index, candidate in enumerate(candidate_insights, start=1):
        claim = str(candidate.get("claim") or "")
        claim_tokens = set(_tokens(claim))
        overlap = len(claim_tokens & description_tokens) / max(1, len(claim_tokens))
        status = "covered" if overlap >= 0.6 else "partially_covered" if overlap >= 0.3 else "missing"
        evaluated.append(
            {
                "rank": _safe_int(candidate.get("rank"), index),
                "claim": claim,
                "description_status": status,
                "matched_text": "",
                "reason": f"Lexical overlap ratio: {overlap:.2f}",
                "candidate_insight": candidate,
            }
        )
    return evaluated


def _looks_like_subheader_row(row: Sequence[str], remaining_rows: Sequence[Sequence[str]], columns: Sequence[str]) -> bool:
    if len(row) != len(columns) or not remaining_rows:
        return False
    if _clean_key(row[0]) == _clean_key(columns[0]):
        return True
    nonnumeric = sum(_parse_number(value) is None for value in row)
    numeric_after = sum(
        _parse_number(value) is not None
        for next_row in remaining_rows[:3]
        for value in next_row[1:]
    )
    return nonnumeric >= max(2, len(row) // 2) and numeric_after >= 2


def _combine_header(top: str, lower: str) -> str:
    top = _clean_cell(top)
    lower = _clean_cell(lower)
    if not top:
        return lower
    if not lower or _clean_key(top) == _clean_key(lower):
        return top
    if lower.lower() in top.lower():
        return top
    return f"{top} / {lower}"


def _clean_cell(value: Any) -> str:
    text = str(value or "").replace("[BOLD]", "").replace("[EMPTY]", "").strip()
    return re.sub(r"\s+", " ", text)


def _clean_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", _clean_cell(value).lower())


def _parse_number(value: Any) -> float | None:
    text = _clean_cell(value)
    if not text or text in {"-", "—", "–"}:
        return None
    text = text.replace(",", "")
    if text.startswith("(") and ")" in text:
        text = text[1 : text.index(")")]
    match = re.search(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", text, flags=re.IGNORECASE)
    if not match:
        return None
    return float(match.group(0))


def _resolve_label(value: Any, choices: Sequence[str]) -> str:
    text = str(value or "").strip()
    if text in choices:
        return text
    lowered = text.lower()
    for choice in choices:
        if choice.lower() == lowered:
            return choice
    for choice in choices:
        if choice.lower().split("#", 1)[0] == lowered:
            return choice
    raise ValueError(f"Unknown label {text!r}; choices are {list(choices)!r}")


def _dedupe_labels(labels: Sequence[str]) -> List[str]:
    counts: Dict[str, int] = {}
    deduped = []
    for label in labels:
        base = str(label).strip() or "item"
        counts[base] = counts.get(base, 0) + 1
        deduped.append(base if counts[base] == 1 else f"{base}#{counts[base]}")
    return deduped


def _normalize_kind(value: Any) -> str:
    kind = str(value or "max").strip().lower()
    if kind not in {"max", "min"}:
        raise ValueError(f"Unsupported extrema kind: {kind}")
    return kind


def _threshold_mask(series: pd.Series, operator: str, threshold: float) -> pd.Series:
    if operator == ">":
        return series > threshold
    if operator == ">=":
        return series >= threshold
    if operator == "<":
        return series < threshold
    if operator == "<=":
        return series <= threshold
    raise ValueError(f"Unsupported threshold operator: {operator}")


def _answer(answer: str, value: float, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"answer": answer, "value": None if pd.isna(value) else float(value), "evidence": evidence}


def _merge_by_question(first: Sequence[Dict[str, Any]], second: Sequence[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    merged = []
    seen = set()
    for item in list(first) + list(second):
        key = str(item.get("question") or item.get("operation") or "").lower()
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(dict(item))
        if len(merged) >= limit:
            break
    return merged


def _normalize_status(status: Any) -> str:
    value = str(status or "").strip().lower()
    if value in {"covered", "partially_covered", "missing", "contradicted"}:
        return value
    if value in {"partial", "partially covered"}:
        return "partially_covered"
    return "missing"


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9.]+", text.lower())


def _format_number(value: float) -> str:
    if pd.isna(value):
        return "nan"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: Sequence[int]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 2)


def _summarize_run(
    dataset_file: str,
    total_records: int,
    start_index: int,
    requested: int,
    evaluated: Sequence[Dict[str, Any]],
    failed: Sequence[Dict[str, Any]],
    examples: int,
    dry_run: bool,
    planner_mode: str,
    max_high_questions: int,
    max_low_questions: int,
    use_llm_summarizer: bool,
    use_llm_verifier: bool,
    use_llm_matching: bool,
    failure_retries: int,
    save_all_results: bool,
) -> Dict[str, Any]:
    low_counts = [len(item.get("low_level_questions") or []) for item in evaluated if not dry_run]
    answer_counts = [len(item.get("executed_answers") or []) for item in evaluated if not dry_run]
    raw_candidate_counts = [len(item.get("raw_candidate_insights") or []) for item in evaluated if not dry_run]
    candidate_counts = [len(item.get("candidate_insights") or []) for item in evaluated if not dry_run]
    verifier_drop_counts = [
        sum(1 for decision in item.get("verifier_decisions") or [] if decision.get("decision") == "drop")
        for item in evaluated
        if not dry_run
    ]
    missing_counts = [len(item.get("missing_insights") or []) for item in evaluated if not dry_run]
    summary = {
        "dataset": "SciGen",
        "baseline": "scigen_question_executor_table_text_matching",
        "paper_basis": "2025.naacl-long.24 high-level questions -> low-level executable questions -> tool answers -> summary -> matching",
        "dataset_file": dataset_file,
        "uses_executor": True,
        "dry_run": dry_run,
        "total_records": total_records,
        "start_index": start_index,
        "requested": requested,
        "evaluated": len(evaluated),
        "failed": len(failed),
        "failure_retries": failure_retries,
        "planner_mode": planner_mode,
        "max_high_questions": max_high_questions,
        "max_low_questions": max_low_questions,
        "use_llm_summarizer": use_llm_summarizer,
        "use_llm_verifier": use_llm_verifier,
        "use_llm_matching": use_llm_matching,
        "average_low_level_questions": _mean(low_counts),
        "average_executed_answers": _mean(answer_counts),
        "average_raw_candidate_insights": _mean(raw_candidate_counts),
        "average_candidate_insights": _mean(candidate_counts),
        "average_verifier_drops": _mean(verifier_drop_counts),
        "average_missing_insights": _mean(missing_counts),
        "examples": list(evaluated[:examples]),
        "failures": list(failed),
    }
    if save_all_results:
        summary["results"] = list(evaluated)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SciGen question-executor insight baseline.")
    parser.add_argument(
        "--dataset-file",
        default="SciGen-main/dataset/development/few-shot/dev.json",
        help="Path to a SciGen JSON file or train/large ZIP.",
    )
    parser.add_argument("--limit", type=int, default=3, help="Number of records to run. Use 0 to run all remaining records.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--model", default=os.environ.get("SCIGEN_LLM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE", DEFAULT_API_BASE))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--env-file", default="")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--failure-retries", type=int, default=2)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-all-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output", default="")
    parser.add_argument("--planner-mode", default="rule", choices=["rule", "llm", "hybrid"])
    parser.add_argument("--max-high-questions", type=int, default=4)
    parser.add_argument("--max-low-questions", type=int, default=12)
    parser.add_argument("--no-llm-summarizer", action="store_true")
    parser.add_argument("--no-llm-verifier", action="store_true")
    parser.add_argument("--no-llm-matching", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.env_file:
        _load_env_file(args.env_file)
    needs_client = (
        not args.dry_run
        and (
            args.planner_mode in {"llm", "hybrid"}
            or not args.no_llm_summarizer
            or not args.no_llm_verifier
            or not args.no_llm_matching
        )
    )
    client = None
    if needs_client:
        api_key = os.environ.get(args.api_key_env, "")
        if not api_key:
            raise SystemExit(
                f"{args.api_key_env} is not set. Use --dry-run, --planner-mode rule, "
                "--no-llm-summarizer, --no-llm-verifier, or --no-llm-matching to avoid LLM calls."
            )
        client = OpenAICompatibleClient(
            api_key=api_key,
            model=args.model,
            api_base=args.api_base,
            timeout=args.timeout,
        )
    summary = run_scigen_question_executor_baseline(
        dataset_file=args.dataset_file,
        limit=args.limit,
        start_index=args.start_index,
        examples=args.examples,
        client=client,
        dry_run=args.dry_run,
        planner_mode=args.planner_mode,
        max_high_questions=args.max_high_questions,
        max_low_questions=args.max_low_questions,
        use_llm_summarizer=not args.no_llm_summarizer,
        use_llm_verifier=not args.no_llm_verifier,
        use_llm_matching=not args.no_llm_matching,
        retries=args.retries,
        failure_retries=args.failure_retries,
        sleep_seconds=args.sleep_seconds,
        output_path=args.output or None,
        save_all_results=args.save_all_results,
    )
    printable = _compact_summary(summary) if args.quiet else summary
    print(json.dumps(printable, ensure_ascii=False, indent=2))


def _compact_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "dataset",
        "baseline",
        "dataset_file",
        "dry_run",
        "requested",
        "evaluated",
        "failed",
        "planner_mode",
        "use_llm_summarizer",
        "use_llm_verifier",
        "use_llm_matching",
        "average_low_level_questions",
        "average_executed_answers",
        "average_raw_candidate_insights",
        "average_candidate_insights",
        "average_verifier_drops",
        "average_missing_insights",
        "failures",
    ]
    return {key: summary.get(key) for key in keys if key in summary}


if __name__ == "__main__":
    main()
