from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

from .core import Insight, TableData
from .ranking import rank_insights
from .v2_insight_rules import extract_insights_v2


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "back",
    "be",
    "by",
    "chart",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "over",
    "shown",
    "the",
    "there",
    "this",
    "to",
    "with",
    "within",
    "year",
    "years",
}

SIGNAL_PATTERNS = {
    "trend_up": [
        r"\bincreas(?:e|ed|es|ing)\b",
        r"\bimprov(?:e|ed|es|ing|ement|ements)?\b",
        r"\bgain(?:s|ed|ing)?\b",
        r"\br(?:i|o)se\b",
        r"\brising\b",
        r"\bgrow(?:s|th|ing|n)?\b",
        r"\bupward\b",
    ],
    "trend_down": [
        r"\bdecreas(?:e|ed|es|ing)\b",
        r"\bdrop(?:s|ped|ping)?\b",
        r"\bdeclin(?:e|ed|es|ing)\b",
        r"\bfall(?:s|ing)?\b",
        r"\bfell\b",
        r"\bdownward\b",
    ],
    "stable": [
        r"\bstable\b",
        r"\bstagnant\b",
        r"\bflat\b",
        r"\bsteady\b",
        r"\bremain(?:s|ed|ing)?\b",
        r"\bsimilar\b",
    ],
    "extremum_high": [
        r"\bhighest\b",
        r"\blargest\b",
        r"\bmaximum\b",
        r"\bmost\b",
        r"\btop\b",
        r"\bpeak(?:s|ed)?\b",
    ],
    "extremum_low": [
        r"\blowest\b",
        r"\bsmallest\b",
        r"\bminimum\b",
        r"\bleast\b",
        r"\bbottom\b",
        r"\btrough(?:s)?\b",
    ],
    "comparison": [
        r"\bhigher\b",
        r"\blower\b",
        r"\bmore\b",
        r"\bless\b",
        r"\babove\b",
        r"\bbelow\b",
        r"\bthan\b",
        r"\bcompared\b",
        r"\boutperform(?:s|ed|ing)?\b",
        r"\bbeat(?:s|ing)?\b",
        r"\bbetter\b",
        r"\bbaseline\b",
    ],
    "proportion": [
        r"\bshare\b",
        r"\bportion\b",
        r"\bproportion\b",
        r"\bpercent(?:age)?\b",
        r"\baccounts?\s+for\b",
        r"\bmajority\b",
    ],
    "relationship": [
        r"\bcorrelat(?:e|es|ed|ion)\b",
        r"\blinear\b",
        r"\brelationship\b",
        r"\brelated\b",
        r"\bassociat(?:e|es|ed|ion)\b",
    ],
}

OPPOSITE_SIGNALS = {
    "trend_up": "trend_down",
    "trend_down": "trend_up",
    "extremum_high": "extremum_low",
    "extremum_low": "extremum_high",
}

LOWER_IS_BETTER_PATTERNS = (
    "error",
    "perplexity",
    "loss",
    "wer",
    "ter",
    "med",
    "rank",
    "time",
    "cost",
    "distance",
)

HEADER_ROLE_TOKENS = {
    "-",
    "",
    "class",
    "dataset",
    "data",
    "language",
    "metric",
    "method",
    "model",
    "setting",
    "system",
    "task",
    "test",
}


@dataclass
class ParsedNumericNLGRecord:
    table_id_paper: str
    table_id: str
    paper_id: str
    caption: str
    description: str
    table: TableData
    metric_types: List[str]
    lower_is_better_columns: List[str]
    target_entities: List[str]


def load_numeric_split(data_dir: str | Path, split: str) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, object]]]:
    data_dir = Path(data_dir)
    tables = json.loads((data_dir / f"table_{split}.json").read_text(encoding="utf-8"))
    descriptions = json.loads((data_dir / f"table_desc_{split}.json").read_text(encoding="utf-8"))
    desc_by_id = {str(item["table_id_paper"]): item for item in descriptions}
    return tables, desc_by_id


def parse_numeric_record(
    table_record: Dict[str, object],
    desc_record: Dict[str, object] | None = None,
) -> ParsedNumericNLGRecord:
    row_headers = _header_paths(table_record.get("row_headers"))
    column_headers = _header_paths(table_record.get("column_headers"))
    raw_contents = table_record.get("contents")
    if not isinstance(raw_contents, list) or not raw_contents:
        raise ValueError("numericNLG record is missing contents.")

    metric_types = [str(item) for item in table_record.get("metrics_type") or []]
    if len(metric_types) != len(column_headers):
        metric_types = [_metric_from_header(path) for path in column_headers]

    columns = _make_unique_labels(
        [
            _flatten_header(path, drop_trailing_numeric=True, fallback=f"column_{idx + 1}")
            for idx, path in enumerate(column_headers)
        ]
    )
    rows = [
        _flatten_header(path, drop_trailing_numeric=False, fallback=f"row_{idx + 1}")
        for idx, path in enumerate(row_headers)
    ]

    values, kept_columns, kept_rows = _numeric_matrix(raw_contents)
    if not kept_columns or not kept_rows:
        raise ValueError("No fully numeric table slice could be parsed.")

    parsed_columns = [columns[idx] for idx in kept_columns]
    parsed_rows = [rows[idx] if idx < len(rows) else f"row_{idx + 1}" for idx in kept_rows]
    parsed_metric_types = [metric_types[idx] if idx < len(metric_types) else parsed_columns[pos] for pos, idx in enumerate(kept_columns)]
    lower_is_better_columns = [
        column
        for column, metric in zip(parsed_columns, parsed_metric_types)
        if _is_lower_is_better(metric)
    ]

    desc_record = desc_record or {}
    table_id_paper = str(table_record.get("table_id_paper") or desc_record.get("table_id_paper") or "")
    return ParsedNumericNLGRecord(
        table_id_paper=table_id_paper,
        table_id=str(desc_record.get("table_id") or ""),
        paper_id=str(desc_record.get("paper_id") or ""),
        caption=str(table_record.get("caption") or ""),
        description=str(desc_record.get("description") or ""),
        table=TableData(
            row_label_name=_row_label_name(row_headers),
            row_labels=parsed_rows,
            columns=parsed_columns,
            values=values,
        ),
        metric_types=parsed_metric_types,
        lower_is_better_columns=lower_is_better_columns,
        target_entities=_target_entities(table_record.get("target_entity")),
    )


def evaluate_numeric_record(
    table_record: Dict[str, object],
    desc_record: Dict[str, object] | None = None,
    max_matches: int = 5,
    max_corrections: int = 5,
    top_k: int | None = None,
) -> Dict[str, object]:
    parsed = parse_numeric_record(table_record, desc_record)
    all_insights = _extract_numeric_insights(
        parsed.table,
        metric_types=parsed.metric_types,
        lower_is_better_columns=parsed.lower_is_better_columns,
        target_entities=parsed.target_entities,
    )
    insights = rank_insights(all_insights, top_k=top_k)
    text = parsed.description or parsed.caption
    caption_signals = extract_signals(text)

    matches = []
    covered_signals: Set[str] = set()
    for insight in insights:
        scored = score_insight_against_caption(insight, text, caption_signals)
        if scored["score"] <= 0:
            continue
        matches.append(scored)
        covered_signals.update(scored["covered_signals"])

    matches.sort(key=lambda item: item["score"], reverse=True)
    coverage_score = (
        len(covered_signals & caption_signals) / len(caption_signals)
        if caption_signals
        else 0.0
    )

    return {
        "table_id_paper": parsed.table_id_paper,
        "paper_id": parsed.paper_id,
        "table_id": parsed.table_id,
        "rule_version": "latest",
        "caption": parsed.caption,
        "description": parsed.description,
        "table": {
            "row_label_name": parsed.table.row_label_name,
            "row_labels": parsed.table.row_labels,
            "columns": parsed.table.columns,
            "values": parsed.table.values,
        },
        "metric_types": parsed.metric_types,
        "lower_is_better_columns": parsed.lower_is_better_columns,
        "target_entities": parsed.target_entities,
        "caption_signals": sorted(caption_signals),
        "generated_insight_count": len(all_insights),
        "ranked_insight_count": len(insights),
        "top_k": top_k or 0,
        "matched_insight_count": len(matches),
        "coverage_score": round(coverage_score, 4),
        "covered_signals": sorted(covered_signals & caption_signals),
        "unmatched_signals": sorted(caption_signals - covered_signals),
        "matches": matches[:max_matches],
        "text_correction": build_text_correction_report(
            insights=insights,
            matches=matches,
            text=text,
            caption_signals=caption_signals,
            max_items=max_corrections,
        ),
    }


def run_numeric_nlg(
    data_dir: str | Path,
    split: str,
    limit: int,
    examples: int,
    max_matches: int,
    max_corrections: int = 5,
    top_k: int | None = None,
    header_level: str = "all",
) -> Dict[str, object]:
    tables, desc_by_id = load_numeric_split(data_dir, split)
    total_split_tables = len(tables)
    candidate_tables = _filter_by_header_level(tables, header_level)
    requested = min(limit, len(candidate_tables))
    evaluated = []
    failed = []
    signal_counts = Counter()
    unmatched_signal_counts = Counter()
    coverage_buckets = Counter()
    generated_counts = []
    ranked_counts = []
    lower_is_better_tables = 0
    missing_counts = []
    potential_error_counts = []

    for table_record in candidate_tables[:requested]:
        table_id_paper = str(table_record.get("table_id_paper") or "")
        try:
            result = evaluate_numeric_record(
                table_record,
                desc_by_id.get(table_id_paper),
                max_matches=max_matches,
                max_corrections=max_corrections,
                top_k=top_k,
            )
        except Exception as exc:  # noqa: BLE001 - summary should keep going.
            failed.append({"table_id_paper": table_id_paper, "error": str(exc)})
            continue

        evaluated.append(result)
        signal_counts.update(result["caption_signals"])
        unmatched_signal_counts.update(result["unmatched_signals"])
        coverage_buckets[_coverage_bucket(result["coverage_score"])] += 1
        generated_counts.append(result["generated_insight_count"])
        ranked_counts.append(result["ranked_insight_count"])
        missing_counts.append(result["text_correction"]["missing_insight_count"])
        potential_error_counts.append(result["text_correction"]["potential_text_error_count"])
        if result["lower_is_better_columns"]:
            lower_is_better_tables += 1

    average_coverage = (
        sum(item["coverage_score"] for item in evaluated) / len(evaluated)
        if evaluated
        else math.nan
    )

    return {
        "dataset": "numericNLG",
        "split": split,
        "header_level": header_level,
        "rule_version": "latest",
        "top_k": top_k or 0,
        "total_split_tables": total_split_tables,
        "candidate_tables": len(candidate_tables),
        "requested": requested,
        "evaluated": len(evaluated),
        "failed": len(failed),
        "parse_rate": len(evaluated) / requested if requested else math.nan,
        "average_coverage_score": round(average_coverage, 4) if evaluated else math.nan,
        "average_generated_insights": round(sum(generated_counts) / len(generated_counts), 2)
        if generated_counts
        else math.nan,
        "average_ranked_insights": round(sum(ranked_counts) / len(ranked_counts), 2)
        if ranked_counts
        else math.nan,
        "average_missing_insights": round(sum(missing_counts) / len(missing_counts), 2)
        if missing_counts
        else math.nan,
        "average_potential_text_errors": round(sum(potential_error_counts) / len(potential_error_counts), 2)
        if potential_error_counts
        else math.nan,
        "tables_with_lower_is_better_metrics": lower_is_better_tables,
        "caption_signal_counts": dict(signal_counts),
        "unmatched_signal_counts": dict(unmatched_signal_counts),
        "coverage_buckets": dict(coverage_buckets),
        "examples": evaluated[:examples],
        "failures": failed[:10],
        "notes": [
            "Rule v2 uses general insight types such as comparison, extremum, proportion, trend, and relationship, with experiment-table subtypes stored in details.insight_subtype.",
            "Experiment-table subtypes include method/system wins, baseline improvements, ablations, ties, group contrasts, dominant shares, ordered trends, and linear correlations.",
            "Metric direction is applied when lower-is-better metrics such as error, loss, perplexity, WER, TER, MED, rank, time, cost, or distance are detected.",
            "text_correction is a heuristic baseline: potential_text_errors are candidates for manual/LLM verification, not final labels.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run table-insight and text-coverage baseline on numericNLG."
    )
    parser.add_argument("data_dir", help="Path to numeric-nlg-main/data.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--max-matches", type=int, default=5)
    parser.add_argument("--max-corrections", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--header-level",
        default="all",
        choices=["all", "single"],
        help="Use 'single' to evaluate only row_header_level=1 and column_header_level=1 tables.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_numeric_nlg(
        data_dir=args.data_dir,
        split=args.split,
        limit=args.limit,
        examples=args.examples,
        max_matches=args.max_matches,
        max_corrections=args.max_corrections,
        top_k=args.top_k or None,
        header_level=args.header_level,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _extract_numeric_insights(
    table: TableData,
    metric_types: Sequence[str] | None = None,
    lower_is_better_columns: Sequence[str] | None = None,
    target_entities: Sequence[str] | None = None,
) -> List[Insight]:
    return extract_insights_v2(
        table,
        metric_types=metric_types,
        lower_is_better_columns=lower_is_better_columns,
        target_entities=target_entities,
    )


def _filter_by_header_level(
    tables: Sequence[Dict[str, object]],
    header_level: str,
) -> List[Dict[str, object]]:
    if header_level == "all":
        return list(tables)
    if header_level == "single":
        return [
            table
            for table in tables
            if table.get("row_header_level") == 1 and table.get("column_header_level") == 1
        ]
    raise ValueError(f"Unsupported header level filter: {header_level}")


def score_insight_against_caption(
    insight: Insight,
    caption: str,
    caption_signals: Set[str] | None = None,
) -> Dict[str, object]:
    if caption_signals is None:
        caption_signals = extract_signals(caption)

    insight_signals = extract_insight_signals(insight)
    covered_signals = caption_signals & insight_signals
    token_overlap = _token_overlap(caption, _insight_text(insight))
    number_overlap = _number_overlap(caption, _insight_text(insight))

    signal_score = len(covered_signals) / len(caption_signals) if caption_signals else 0.0
    score = (0.7 * signal_score) + (0.2 * token_overlap) + (0.1 * number_overlap)

    if score < 0.12:
        return {
            "score": 0.0,
            "covered_signals": [],
            "insight": insight.to_dict(),
            "reason": "below_match_threshold",
        }

    return {
        "score": round(score, 4),
        "covered_signals": sorted(covered_signals),
        "token_overlap": round(token_overlap, 4),
        "number_overlap": round(number_overlap, 4),
        "precision_label": _precision_label(covered_signals, token_overlap, number_overlap),
        "insight": insight.to_dict(),
    }


def build_text_correction_report(
    insights: Sequence[Insight],
    matches: Sequence[Dict[str, object]],
    text: str,
    caption_signals: Set[str],
    max_items: int = 5,
) -> Dict[str, object]:
    supported_keys = {
        _insight_key_from_dict(match["insight"])
        for match in matches
        if match.get("score", 0.0) > 0
    }
    potential_errors = []
    missing = []

    for insight in insights:
        key = _insight_key(insight)
        contradiction = detect_potential_text_error(insight, text, caption_signals)
        if contradiction:
            potential_errors.append(contradiction)
            continue
        if key in supported_keys:
            continue
        missing.append(
            {
                "reason": "table_supported_insight_not_matched_in_text",
                "suggested_correction": insight.claim,
                "insight": insight.to_dict(),
            }
        )

    return {
        "supported_insight_count": len(supported_keys),
        "missing_insight_count": len(missing),
        "potential_text_error_count": len(potential_errors),
        "supported_insights": list(matches[:max_items]),
        "missing_insights": missing[:max_items],
        "potential_text_errors": potential_errors[:max_items],
    }


def detect_potential_text_error(
    insight: Insight,
    text: str,
    caption_signals: Set[str] | None = None,
) -> Dict[str, object] | None:
    if caption_signals is None:
        caption_signals = extract_signals(text)

    insight_signals = extract_insight_signals(insight)
    token_overlap = _token_overlap(text, _insight_text(insight))
    number_overlap = _number_overlap(text, _insight_text(insight))
    subject_mentioned = _phrase_in_text(insight.subject, text)

    for signal in insight_signals:
        opposite = OPPOSITE_SIGNALS.get(signal)
        if not opposite or opposite not in caption_signals:
            continue
        if not subject_mentioned and number_overlap <= 0:
            continue
        return {
            "reason": "opposite_direction_signal_in_text",
            "text_signal": opposite,
            "table_signal": signal,
            "table_supported_correction": insight.claim,
            "heuristics": {
                "subject_mentioned": subject_mentioned,
                "token_overlap": round(token_overlap, 4),
                "number_overlap": round(number_overlap, 4),
            },
            "insight": insight.to_dict(),
        }

    reversed_comparison = _detect_reversed_comparison(insight, text)
    if reversed_comparison:
        return reversed_comparison

    return None


def extract_signals(text: str) -> Set[str]:
    normalized = text.lower()
    signals = set()
    for signal, patterns in SIGNAL_PATTERNS.items():
        if any(re.search(pattern, normalized) for pattern in patterns):
            signals.add(signal)
    return signals


def extract_insight_signals(insight: Insight) -> Set[str]:
    text = _insight_text(insight).lower()
    subtype = str((insight.details or {}).get("insight_subtype") or "")
    signals = set()

    if insight.type == "trend":
        direction = (insight.details or {}).get("direction")
        if direction == "up":
            signals.add("trend_up")
        if direction == "down":
            signals.add("trend_down")
    if insight.type == "comparison":
        signals.add("comparison")
    if insight.type == "proportion":
        signals.add("proportion")
    if insight.type == "relationship":
        signals.add("relationship")
    if subtype in {"baseline_improvement", "ablation_gain"}:
        signals.add("trend_up")
    if subtype == "near_tie":
        signals.add("stable")
        signals.add("comparison")
    if insight.type == "extremum" and subtype in {"metric_best", "overall_best"}:
        signals.add("extremum_high")
    if insight.type == "extremum" and subtype == "metric_worst":
        signals.add("extremum_low")
    if insight.type == "extremum":
        if "highest" in text:
            signals.add("extremum_high")
        if "lowest" in text:
            signals.add("extremum_low")

    signals.update(extract_signals(text))
    return signals


def _coverage_bucket(score: float) -> str:
    if score >= 1.0:
        return "full"
    if score >= 0.67:
        return "high"
    if score >= 0.34:
        return "partial"
    if score > 0:
        return "low"
    return "none"


def _precision_label(covered_signals: Set[str], token_overlap: float, number_overlap: float) -> str:
    if covered_signals and (token_overlap >= 0.15 or number_overlap >= 0.2):
        return "strong"
    if covered_signals:
        return "medium"
    if token_overlap >= 0.2 or number_overlap >= 0.3:
        return "weak"
    return "weak"


def _insight_text(insight: Insight) -> str:
    parts = [insight.type, insight.subject, insight.condition, insight.claim]
    for cell in insight.evidence:
        parts.extend([cell.row, cell.column, str(cell.value)])
    if insight.details:
        parts.append(json.dumps(insight.details, ensure_ascii=False))
    return " ".join(parts)


def _token_overlap(left: str, right: str) -> float:
    left_tokens = _content_tokens(left)
    right_tokens = _content_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens))


def _number_overlap(left: str, right: str) -> float:
    left_numbers = _numbers(left)
    right_numbers = _numbers(right)
    if not left_numbers or not right_numbers:
        return 0.0
    return len(left_numbers & right_numbers) / min(len(left_numbers), len(right_numbers))


def _content_tokens(text: str) -> Set[str]:
    tokens = set()
    for token in re.findall(r"[a-zA-Z][a-zA-Z0-9']+", text.lower()):
        if len(token) < 3 or token in STOPWORDS:
            continue
        tokens.add(token)
    return tokens


def _numbers(text: str) -> Set[str]:
    values = set()
    for token in re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?%?", text):
        cleaned = token.replace(",", "").replace("%", "")
        if cleaned:
            values.add(cleaned)
    return values


def _detect_reversed_comparison(insight: Insight, text: str) -> Dict[str, object] | None:
    if insight.type != "comparison":
        return None
    details = insight.details or {}
    winner = str(details.get("winner") or "")
    loser = str(details.get("loser") or "")
    if not winner or not loser:
        return None

    normalized_text = _normalize_text(text)
    normalized_winner = _normalize_text(winner)
    normalized_loser = _normalize_text(loser)
    if not _normalized_phrase_in_text(normalized_winner, normalized_text):
        return None
    if not _normalized_phrase_in_text(normalized_loser, normalized_text):
        return None

    loser_beats_winner = _ordered_claim_present(
        normalized_text,
        normalized_loser,
        normalized_winner,
        r"\b(outperform(?:s|ed|ing)?|beat(?:s|ing)?|better|higher|more)\b",
    )
    winner_lower_than_loser = _ordered_claim_present(
        normalized_text,
        normalized_winner,
        normalized_loser,
        r"\b(lower|worse|less|below)\b",
    )
    if not loser_beats_winner and not winner_lower_than_loser:
        return None

    return {
        "reason": "reversed_pairwise_comparison_in_text",
        "table_supported_correction": insight.claim,
        "heuristics": {
            "winner": winner,
            "loser": loser,
            "loser_beats_winner_pattern": loser_beats_winner,
            "winner_lower_than_loser_pattern": winner_lower_than_loser,
        },
        "insight": insight.to_dict(),
    }


def _ordered_claim_present(text: str, first: str, second: str, relation_pattern: str) -> bool:
    for start in _normalized_phrase_positions(text, first):
        for end in _normalized_phrase_positions(text, second):
            if end <= start or end - start > 180:
                continue
            window = text[start:end]
            if re.search(relation_pattern, window):
                return True
    return False


def _phrase_in_text(phrase: str, text: str) -> bool:
    normalized_phrase = _normalize_text(phrase)
    return bool(normalized_phrase) and _normalized_phrase_in_text(normalized_phrase, _normalize_text(text))


def _normalize_text(text: str) -> str:
    return " ".join(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def _normalized_phrase_in_text(phrase: str, text: str) -> bool:
    return bool(_normalized_phrase_positions(text, phrase))


def _normalized_phrase_positions(text: str, phrase: str) -> List[int]:
    if not phrase:
        return []
    pattern = re.compile(rf"(?<![a-zA-Z0-9]){re.escape(phrase)}(?![a-zA-Z0-9])")
    return [match.start() for match in pattern.finditer(text)]


def _insight_key(insight: Insight) -> tuple[object, ...]:
    return (
        insight.type,
        insight.axis,
        insight.scope,
        insight.subject,
        insight.condition,
        insight.claim,
    )


def _insight_key_from_dict(insight: Dict[str, object]) -> tuple[object, ...]:
    return (
        insight.get("type"),
        insight.get("axis"),
        insight.get("scope"),
        insight.get("subject"),
        insight.get("condition"),
        insight.get("claim"),
    )


def _header_paths(value: object) -> List[List[str]]:
    if not isinstance(value, list):
        return []
    paths = []
    for item in value:
        if isinstance(item, list):
            paths.append([str(part) for part in item])
        else:
            paths.append([str(item)])
    return paths


def _target_entities(value: object) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]


def _flatten_header(path: Sequence[str], drop_trailing_numeric: bool, fallback: str) -> str:
    parts = [str(part).strip() for part in path if str(part).strip()]
    if drop_trailing_numeric and len(parts) > 1 and _parse_number(parts[-1]) is not None:
        parts = parts[:-1]

    compact = []
    for part in parts:
        if part.lower() in HEADER_ROLE_TOKENS and compact:
            continue
        compact.append(part)

    if len(compact) > 1 and compact[0].lower() in HEADER_ROLE_TOKENS:
        compact = compact[1:]
    return " | ".join(compact) if compact else fallback


def _row_label_name(row_headers: Sequence[Sequence[str]]) -> str:
    if not row_headers or not row_headers[0]:
        return "row"
    first = str(row_headers[0][0]).strip()
    return first if first and first != "-" else "row"


def _metric_from_header(path: Sequence[str]) -> str:
    for part in reversed(path):
        text = str(part).strip()
        if text and _parse_number(text) is None:
            return text
    return str(path[-1]) if path else ""


def _make_unique_labels(labels: Sequence[str]) -> List[str]:
    counts: Dict[str, int] = {}
    unique = []
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
        if counts[label] == 1:
            unique.append(label)
        else:
            unique.append(f"{label} ({counts[label]})")
    return unique


def _numeric_matrix(raw_contents: object) -> Tuple[List[List[float]], List[int], List[int]]:
    if not isinstance(raw_contents, list):
        return [], [], []

    raw_rows = raw_contents
    parsed_rows: List[List[float | None]] = []
    max_width = 0
    for row in raw_rows:
        if not isinstance(row, list):
            continue
        parsed = [_parse_number(str(cell)) for cell in row]
        parsed_rows.append(parsed)
        max_width = max(max_width, len(parsed))

    if not parsed_rows or max_width == 0:
        return [], [], []

    kept_rows = [
        idx
        for idx, row in enumerate(parsed_rows)
        if any(value is not None for value in row)
    ]
    kept_columns = []
    for column_idx in range(max_width):
        values = []
        valid = True
        for row_idx in kept_rows:
            row = parsed_rows[row_idx]
            value = row[column_idx] if column_idx < len(row) else None
            if value is None:
                valid = False
                break
            values.append(value)
        if valid and values:
            kept_columns.append(column_idx)

    values = [
        [parsed_rows[row_idx][column_idx] for column_idx in kept_columns]
        for row_idx in kept_rows
    ]
    return values, kept_columns, kept_rows


def _parse_number(text: str) -> float | None:
    cleaned = text.strip().replace(",", "").replace("%", "").replace("*", "")
    cleaned = cleaned.replace("−", "-").replace("–", "-")
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = "-" + cleaned[1:-1]
    if not re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", cleaned):
        return None
    return float(cleaned)


def _is_lower_is_better(metric: str) -> bool:
    normalized = metric.lower()
    return any(pattern in normalized for pattern in LOWER_IS_BETTER_PATTERNS)


if __name__ == "__main__":
    main()
