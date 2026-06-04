from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from typing import Iterable, List, Sequence

from .core import EvidenceCell, Insight, TableData
from .ranking import rank_insights


LOWER_IS_BETTER_PATTERNS = (
    "error",
    "err",
    "loss",
    "perplexity",
    "ppl",
    "wer",
    "ter",
    "med",
    "rank",
    "time",
    "cost",
    "distance",
)

BASELINE_TOKENS = (
    "baseline",
    "base",
    "w/o",
    "without",
    "no context",
)

METHOD_TOKEN_PATTERN = re.compile(
    r"\b(?:cnn|rnn|lstm|blstm|bert|crf|svm|hmm|model|method|system|baseline|proposed|ours?)\b",
    re.IGNORECASE,
)


def extract_insights_v2(
    table: TableData,
    metric_types: Sequence[str] | None = None,
    lower_is_better_columns: Sequence[str] | None = None,
    target_entities: Sequence[str] | None = None,
) -> List[Insight]:
    """Extract numericNLG-oriented insights for scientific result tables.

    The rules favor claims common in paper table descriptions: best/worst
    systems by metric, overall leaders, improvements over baselines,
    ablation/component gains, comparable systems, and coarse group contrasts.
    """

    metric_types = list(metric_types or table.columns)
    directions = _column_directions(table, metric_types, lower_is_better_columns or [])
    system_axis = _infer_system_axis(table, metric_types, target_entities or [])

    insights: List[Insight] = []
    if system_axis == "column":
        insights.extend(_extract_entity_rules_for_columns(table, metric_types, directions))
        insights.extend(_extract_entity_rules_for_rows(table, metric_types, directions, secondary=True))
    else:
        insights.extend(_extract_entity_rules_for_rows(table, metric_types, directions))
        insights.extend(_extract_entity_rules_for_columns(table, metric_types, directions, secondary=True))

    insights.extend(_extract_component_gains(table, directions))
    insights.extend(_extract_group_contrasts(table, directions))
    insights.extend(_extract_proportions(table))
    insights.extend(_extract_ordered_trends(table))
    insights.extend(_extract_linear_relationships(table))
    return rank_insights(_dedupe_and_sort(insights))


def _extract_entity_rules_for_rows(
    table: TableData,
    metric_types: Sequence[str],
    directions: Sequence[str],
    secondary: bool = False,
) -> List[Insight]:
    entities = table.row_labels
    contexts = table.columns
    matrix = table.values
    return _extract_entity_rules(
        entities=entities,
        contexts=contexts,
        matrix=matrix,
        directions=directions,
        axis="row",
        entity_kind="method",
        context_kind="metric",
        metric_types=metric_types,
        secondary=secondary,
    )


def _extract_entity_rules_for_columns(
    table: TableData,
    metric_types: Sequence[str],
    directions: Sequence[str],
    secondary: bool = False,
) -> List[Insight]:
    if len(table.columns) < 2:
        return []

    majority_direction = _majority_direction(directions)
    matrix = [
        [table.values[row_idx][col_idx] for row_idx in range(len(table.row_labels))]
        for col_idx in range(len(table.columns))
    ]
    metric_name = _common_metric_name(metric_types)
    contexts = [
        f"{row_label} ({metric_name})" if metric_name else row_label
        for row_label in table.row_labels
    ]
    return _extract_entity_rules(
        entities=table.columns,
        contexts=contexts,
        matrix=matrix,
        directions=[majority_direction] * len(contexts),
        axis="column",
        entity_kind="system",
        context_kind="row",
        metric_types=[metric_name or "value"] * len(contexts),
        secondary=secondary,
    )


def _extract_entity_rules(
    entities: Sequence[str],
    contexts: Sequence[str],
    matrix: Sequence[Sequence[float]],
    directions: Sequence[str],
    axis: str,
    entity_kind: str,
    context_kind: str,
    metric_types: Sequence[str],
    secondary: bool = False,
) -> List[Insight]:
    insights: List[Insight] = []
    if len(entities) < 2 or not contexts:
        return insights

    insights.extend(
        _extract_best_and_worst_by_context(
            entities,
            contexts,
            matrix,
            directions,
            axis,
            entity_kind,
            context_kind,
            metric_types,
            secondary,
        )
    )
    insights.extend(
        _extract_overall_leaders(
            entities,
            contexts,
            matrix,
            directions,
            axis,
            entity_kind,
            context_kind,
            secondary,
        )
    )
    insights.extend(
        _extract_pairwise_outperformance(
            entities,
            contexts,
            matrix,
            directions,
            axis,
            entity_kind,
            context_kind,
            secondary,
        )
    )
    insights.extend(
        _extract_near_ties(
            entities,
            contexts,
            matrix,
            axis,
            entity_kind,
            context_kind,
            secondary,
        )
    )
    return insights


def _extract_best_and_worst_by_context(
    entities: Sequence[str],
    contexts: Sequence[str],
    matrix: Sequence[Sequence[float]],
    directions: Sequence[str],
    axis: str,
    entity_kind: str,
    context_kind: str,
    metric_types: Sequence[str],
    secondary: bool,
) -> List[Insight]:
    insights: List[Insight] = []
    for context_idx, context in enumerate(contexts):
        values = [row[context_idx] for row in matrix]
        direction = directions[context_idx]
        scores = [_effective_value(value, direction) for value in values]
        best_idx = max(range(len(scores)), key=scores.__getitem__)
        worst_idx = min(range(len(scores)), key=scores.__getitem__)
        if math.isclose(scores[best_idx], scores[worst_idx]):
            continue

        margin = abs(values[best_idx] - values[worst_idx])
        metric_name = _metric_name(metric_types, context_idx, context)
        best_word = "lowest" if direction == "lower" else "highest"
        worst_word = "highest" if direction == "lower" else "lowest"
        operation_best = "argmin" if direction == "lower" else "argmax"
        operation_worst = "argmax" if direction == "lower" else "argmin"
        priority_penalty = 0.15 if secondary else 0.0

        insights.append(
            Insight(
                type="extremum",
                subject=entities[best_idx],
                condition=context,
                claim=(
                    f"{entities[best_idx]} achieves the best {metric_name} result "
                    f"on {context}, with the {best_word} value of {values[best_idx]:.4g}."
                ),
                evidence=[
                    _entity_cell(axis, entities[best_idx], context, values[best_idx]),
                    _entity_cell(axis, entities[worst_idx], context, values[worst_idx]),
                ],
                axis=axis,
                scope="single_metric",
                operation=operation_best,
                details={
                    "insight_subtype": "metric_best",
                    "metric": metric_name,
                    "best_entity": entities[best_idx],
                    "worst_entity": entities[worst_idx],
                    "metric_direction": f"{direction}_is_better",
                    "margin_to_worst": round(margin, 4),
                    "secondary_axis": secondary,
                },
                score={"support": 1.0 - priority_penalty, "importance": round(margin, 4)},
            )
        )
        insights.append(
            Insight(
                type="extremum",
                subject=entities[worst_idx],
                condition=context,
                claim=(
                    f"{entities[worst_idx]} has the weakest {metric_name} result "
                    f"on {context}, with the {worst_word} value of {values[worst_idx]:.4g}."
                ),
                evidence=[
                    _entity_cell(axis, entities[worst_idx], context, values[worst_idx]),
                    _entity_cell(axis, entities[best_idx], context, values[best_idx]),
                ],
                axis=axis,
                scope="single_metric",
                operation=operation_worst,
                details={
                    "insight_subtype": "metric_worst",
                    "metric": metric_name,
                    "weakest_entity": entities[worst_idx],
                    "best_entity": entities[best_idx],
                    "metric_direction": f"{direction}_is_better",
                    "margin_to_best": round(margin, 4),
                    "secondary_axis": secondary,
                },
                score={"support": 0.85 - priority_penalty, "importance": round(margin, 4)},
            )
        )
    return insights


def _extract_overall_leaders(
    entities: Sequence[str],
    contexts: Sequence[str],
    matrix: Sequence[Sequence[float]],
    directions: Sequence[str],
    axis: str,
    entity_kind: str,
    context_kind: str,
    secondary: bool,
) -> List[Insight]:
    best_counts: Counter[str] = Counter()
    best_contexts: dict[str, List[int]] = {entity: [] for entity in entities}
    for context_idx in range(len(contexts)):
        scores = [_effective_value(row[context_idx], directions[context_idx]) for row in matrix]
        best_score = max(scores)
        for entity_idx, score in enumerate(scores):
            if math.isclose(score, best_score):
                best_counts[entities[entity_idx]] += 1
                best_contexts[entities[entity_idx]].append(context_idx)

    if not best_counts:
        return []

    top_entity, top_count = best_counts.most_common(1)[0]
    if top_count == 0:
        return []

    top_idx = entities.index(top_entity)
    support = top_count / len(contexts)
    if support < 0.5 and top_count < 2:
        return []

    evidence = [
        _entity_cell(axis, top_entity, contexts[idx], matrix[top_idx][idx])
        for idx in best_contexts[top_entity]
    ]
    priority_penalty = 0.2 if secondary else 0.0
    descriptor = "overall strongest" if support < 1.0 else "best"
    return [
        Insight(
            type="extremum",
            subject=top_entity,
            condition=f"across {context_kind}s",
            claim=(
                f"{top_entity} is the {descriptor} {entity_kind}, ranking best on "
                f"{top_count}/{len(contexts)} {context_kind}s."
            ),
            evidence=evidence,
            axis=axis,
            scope="table",
            operation="count_argbest",
            details={
                "insight_subtype": "overall_best",
                "best_count": top_count,
                "total_count": len(contexts),
                "best_contexts": [contexts[idx] for idx in best_contexts[top_entity]],
                "secondary_axis": secondary,
            },
            score={"support": round(support - priority_penalty, 4), "importance": float(top_count)},
        )
    ]


def _extract_pairwise_outperformance(
    entities: Sequence[str],
    contexts: Sequence[str],
    matrix: Sequence[Sequence[float]],
    directions: Sequence[str],
    axis: str,
    entity_kind: str,
    context_kind: str,
    secondary: bool,
) -> List[Insight]:
    insights: List[Insight] = []
    priority_penalty = 0.15 if secondary else 0.0
    for winner_idx, winner in enumerate(entities):
        for loser_idx, loser in enumerate(entities):
            if winner_idx == loser_idx:
                continue
            signed_diffs = [
                _effective_value(matrix[winner_idx][context_idx], directions[context_idx])
                - _effective_value(matrix[loser_idx][context_idx], directions[context_idx])
                for context_idx in range(len(contexts))
            ]
            wins = [diff > 0 and not math.isclose(diff, 0.0) for diff in signed_diffs]
            win_count = sum(wins)
            support = win_count / len(contexts)
            avg_margin = statistics.fmean(signed_diffs)
            if support < 0.6 or avg_margin <= 0:
                continue

            insight_subtype = "baseline_improvement" if _is_baseline(loser) else "outperformance"
            relation = "improves over the baseline" if insight_subtype == "baseline_improvement" else "outperforms"
            evidence: List[EvidenceCell] = []
            for context_idx, is_win in enumerate(wins):
                if not is_win:
                    continue
                evidence.append(_entity_cell(axis, winner, contexts[context_idx], matrix[winner_idx][context_idx]))
                evidence.append(_entity_cell(axis, loser, contexts[context_idx], matrix[loser_idx][context_idx]))

            insights.append(
                Insight(
                    type="comparison",
                    subject=f"{winner} vs {loser}",
                    condition=f"across {context_kind}s",
                    claim=(
                        f"{winner} {relation} {loser} on "
                        f"{win_count}/{len(contexts)} {context_kind}s."
                    ),
                    evidence=evidence,
                    axis=axis,
                    scope="paired_entities",
                    operation="directional_difference",
                    details={
                        "insight_subtype": insight_subtype,
                        "winner": winner,
                        "loser": loser,
                        "win_count": win_count,
                        "total_count": len(contexts),
                        "average_effective_margin": round(avg_margin, 4),
                        "secondary_axis": secondary,
                    },
                    score={
                        "support": round(support - priority_penalty, 4),
                        "importance": round(avg_margin, 4),
                    },
                )
            )
    return insights


def _extract_near_ties(
    entities: Sequence[str],
    contexts: Sequence[str],
    matrix: Sequence[Sequence[float]],
    axis: str,
    entity_kind: str,
    context_kind: str,
    secondary: bool,
) -> List[Insight]:
    insights: List[Insight] = []
    if len(contexts) < 2:
        return insights

    for left_idx, left in enumerate(entities):
        if _is_baseline(left):
            continue
        for right_idx in range(left_idx + 1, len(entities)):
            right = entities[right_idx]
            if _is_baseline(right):
                continue
            abs_diffs = [
                abs(matrix[left_idx][context_idx] - matrix[right_idx][context_idx])
                for context_idx in range(len(contexts))
            ]
            avg_abs = statistics.fmean(abs_diffs)
            scale = statistics.fmean(
                abs(matrix[left_idx][context_idx]) + abs(matrix[right_idx][context_idx])
                for context_idx in range(len(contexts))
            ) / 2
            if math.isclose(scale, 0.0):
                continue
            relative_gap = avg_abs / scale
            if relative_gap > 0.025:
                continue

            priority_penalty = 0.1 if secondary else 0.0
            evidence = []
            for context_idx in range(len(contexts)):
                evidence.append(_entity_cell(axis, left, contexts[context_idx], matrix[left_idx][context_idx]))
                evidence.append(_entity_cell(axis, right, contexts[context_idx], matrix[right_idx][context_idx]))
            insights.append(
                Insight(
                    type="comparison",
                    subject=f"{left} vs {right}",
                    condition=f"across {context_kind}s",
                    claim=(
                        f"{left} and {right} are close across {context_kind}s, "
                        f"differing by {avg_abs:.4g} on average."
                    ),
                    evidence=evidence,
                    axis=axis,
                    scope="paired_entities",
                    operation="absolute_difference",
                    details={
                        "insight_subtype": "near_tie",
                        "left": left,
                        "right": right,
                        "average_absolute_difference": round(avg_abs, 4),
                        "relative_gap": round(relative_gap, 4),
                        "secondary_axis": secondary,
                    },
                    score={
                        "support": round(1 - relative_gap - priority_penalty, 4),
                        "importance": round(1 / (1 + avg_abs), 4),
                    },
                )
            )
    return insights


def _extract_component_gains(table: TableData, directions: Sequence[str]) -> List[Insight]:
    insights: List[Insight] = []
    if len(table.row_labels) < 2:
        return insights

    token_sets = [_label_tokens(label) for label in table.row_labels]
    for base_idx, base_label in enumerate(table.row_labels):
        base_tokens = token_sets[base_idx]
        if not base_tokens:
            continue
        for variant_idx, variant_label in enumerate(table.row_labels):
            if base_idx == variant_idx:
                continue
            variant_tokens = token_sets[variant_idx]
            if not base_tokens < variant_tokens:
                continue

            signed_diffs = [
                _effective_value(table.values[variant_idx][col_idx], directions[col_idx])
                - _effective_value(table.values[base_idx][col_idx], directions[col_idx])
                for col_idx in range(len(table.columns))
            ]
            wins = [diff > 0 and not math.isclose(diff, 0.0) for diff in signed_diffs]
            win_count = sum(wins)
            avg_margin = statistics.fmean(signed_diffs)
            if win_count == 0 or avg_margin <= 0:
                continue

            extras = sorted(variant_tokens - base_tokens)
            evidence: List[EvidenceCell] = []
            for col_idx, is_win in enumerate(wins):
                if not is_win:
                    continue
                evidence.append(EvidenceCell(row=variant_label, column=table.columns[col_idx], value=table.values[variant_idx][col_idx]))
                evidence.append(EvidenceCell(row=base_label, column=table.columns[col_idx], value=table.values[base_idx][col_idx]))
            insights.append(
                Insight(
                    type="comparison",
                    subject=f"{variant_label} vs {base_label}",
                    condition="component comparison",
                    claim=(
                        f"Adding {'+'.join(extras)} to {base_label} improves "
                        f"{win_count}/{len(table.columns)} metrics."
                    ),
                    evidence=evidence,
                    axis="row",
                    scope="paired_entities",
                    operation="directional_difference",
                    details={
                        "insight_subtype": "ablation_gain",
                        "variant": variant_label,
                        "base": base_label,
                        "added_components": extras,
                        "win_count": win_count,
                        "total_count": len(table.columns),
                        "average_effective_margin": round(avg_margin, 4),
                    },
                    score={
                        "support": round(win_count / len(table.columns), 4),
                        "importance": round(avg_margin, 4),
                    },
                )
            )
    return insights


def _extract_group_contrasts(table: TableData, directions: Sequence[str]) -> List[Insight]:
    groups: dict[str, List[int]] = {}
    for row_idx, label in enumerate(table.row_labels):
        group = _row_group(label)
        if not group:
            continue
        groups.setdefault(group, []).append(row_idx)

    if len(groups) < 2:
        return []

    insights: List[Insight] = []
    for col_idx, column in enumerate(table.columns):
        group_scores = []
        for group, row_indices in groups.items():
            values = [table.values[row_idx][col_idx] for row_idx in row_indices]
            raw_mean = statistics.fmean(values)
            effective_mean = _effective_value(raw_mean, directions[col_idx])
            group_scores.append((group, row_indices, raw_mean, effective_mean))
        best = max(group_scores, key=lambda item: item[3])
        weakest = min(group_scores, key=lambda item: item[3])
        if math.isclose(best[3], weakest[3]):
            continue

        evidence = [
            EvidenceCell(row=table.row_labels[row_idx], column=column, value=table.values[row_idx][col_idx])
            for row_idx in best[1] + weakest[1]
        ]
        raw_relation = "lower" if weakest[2] < best[2] else "higher"
        insights.append(
            Insight(
                type="comparison",
                subject=f"{best[0]} vs {weakest[0]}",
                condition=column,
                claim=(
                    f"The {best[0]} group is stronger than {weakest[0]} on {column}; "
                    f"{weakest[0]} has {raw_relation} average values."
                ),
                evidence=evidence,
                axis="row",
                scope="row_group",
                operation="mean_difference",
                details={
                    "insight_subtype": "group_contrast",
                    "stronger_group": best[0],
                    "weaker_group": weakest[0],
                    "stronger_mean": round(best[2], 4),
                    "weaker_mean": round(weakest[2], 4),
                    "metric_direction": f"{directions[col_idx]}_is_better",
                },
                score={
                    "support": 1.0,
                    "importance": round(abs(best[3] - weakest[3]), 4),
                },
            )
        )
    return insights


def _extract_proportions(table: TableData) -> List[Insight]:
    insights: List[Insight] = []
    if len(table.row_labels) >= 3:
        for col_idx, column in enumerate(table.columns):
            values = table.column_values(col_idx)
            insight = _dominant_share_insight(
                labels=table.row_labels,
                values=values,
                fixed_label=column,
                axis="row",
                scope="single_column",
            )
            if insight:
                insights.append(insight)

    if len(table.columns) >= 3:
        for row_label, row in zip(table.row_labels, table.values):
            insight = _dominant_share_insight(
                labels=table.columns,
                values=row,
                fixed_label=row_label,
                axis="column",
                scope="single_row",
            )
            if insight:
                insights.append(insight)

    return insights


def _dominant_share_insight(
    labels: Sequence[str],
    values: Sequence[float],
    fixed_label: str,
    axis: str,
    scope: str,
) -> Insight | None:
    if any(value < 0 for value in values):
        return None
    total = sum(values)
    if math.isclose(total, 0.0):
        return None

    max_idx = max(range(len(values)), key=values.__getitem__)
    share = values[max_idx] / total
    if share < 0.4:
        return None

    subject = labels[max_idx]
    share_pct = share * 100
    if axis == "row":
        evidence = [EvidenceCell(row=subject, column=fixed_label, value=values[max_idx])]
        condition = fixed_label
        claim = f"{subject} accounts for {share_pct:.1f}% of the total in {fixed_label}."
    else:
        evidence = [EvidenceCell(row=fixed_label, column=subject, value=values[max_idx])]
        condition = fixed_label
        claim = f"{subject} accounts for {share_pct:.1f}% of {fixed_label}'s total across columns."

    return Insight(
        type="proportion",
        subject=subject,
        condition=condition,
        claim=claim,
        evidence=evidence,
        axis=axis,
        scope=scope,
        operation="share_of_total",
        details={
            "insight_subtype": "dominant_share",
            "share": round(share, 4),
            "total": round(total, 4),
            "part_value": values[max_idx],
        },
        score={"support": round(share, 4), "importance": round(share, 4)},
    )


def _extract_ordered_trends(table: TableData) -> List[Insight]:
    insights: List[Insight] = []
    column_positions = _ordered_numeric_positions(table.columns)
    if column_positions:
        for row_label, row in zip(table.row_labels, table.values):
            insight = _trend_insight(
                subject=row_label,
                labels=table.columns,
                values=row,
                positions=column_positions,
                axis="column",
            )
            if insight:
                insights.append(insight)

    row_positions = _ordered_numeric_positions(table.row_labels)
    if row_positions:
        for col_idx, column in enumerate(table.columns):
            values = table.column_values(col_idx)
            insight = _trend_insight(
                subject=column,
                labels=table.row_labels,
                values=values,
                positions=row_positions,
                axis="row",
            )
            if insight:
                insights.append(insight)

    return insights


def _trend_insight(
    subject: str,
    labels: Sequence[str],
    values: Sequence[float],
    positions: Sequence[float],
    axis: str,
) -> Insight | None:
    if len(values) < 3:
        return None

    ordered = sorted(zip(positions, labels, values), key=lambda item: item[0])
    ordered_labels = [item[1] for item in ordered]
    ordered_values = [item[2] for item in ordered]
    step_diffs = [ordered_values[idx + 1] - ordered_values[idx] for idx in range(len(ordered_values) - 1)]
    non_negative = sum(diff >= 0 for diff in step_diffs)
    non_positive = sum(diff <= 0 for diff in step_diffs)
    upward_support = non_negative / len(step_diffs)
    downward_support = non_positive / len(step_diffs)
    total_change = ordered_values[-1] - ordered_values[0]
    span = max(ordered_values) - min(ordered_values)
    mean_abs = statistics.fmean(abs(value) for value in ordered_values)
    if math.isclose(span, 0.0) or math.isclose(mean_abs, 0.0):
        return None
    if abs(total_change) / mean_abs < 0.03:
        return None

    if upward_support >= 0.75 and total_change > 0:
        direction = "up"
        verb = "increases"
        support = upward_support
    elif downward_support >= 0.75 and total_change < 0:
        direction = "down"
        verb = "decreases"
        support = downward_support
    else:
        return None

    evidence = [
        EvidenceCell(row=subject, column=label, value=value)
        if axis == "column"
        else EvidenceCell(row=label, column=subject, value=value)
        for label, value in zip(ordered_labels, ordered_values)
    ]
    return Insight(
        type="trend",
        subject=subject,
        condition=f"ordered {axis} labels",
        claim=f"{subject} {verb} as the ordered {axis} labels progress.",
        evidence=evidence,
        axis=axis,
        scope="series",
        operation="monotonic_trend",
        details={
            "insight_subtype": "ordered_trend",
            "direction": direction,
            "ordered_labels": ordered_labels,
            "total_change": round(total_change, 4),
            "monotonicity": round(support, 4),
        },
        score={"support": round(support, 4), "importance": round(span, 4)},
    )


def _extract_linear_relationships(table: TableData) -> List[Insight]:
    insights: List[Insight] = []
    if len(table.row_labels) >= 3 and len(table.columns) >= 2:
        for left_idx in range(len(table.columns)):
            for right_idx in range(left_idx + 1, len(table.columns)):
                left_values = table.column_values(left_idx)
                right_values = table.column_values(right_idx)
                insight = _linear_relationship_insight(
                    left_label=table.columns[left_idx],
                    right_label=table.columns[right_idx],
                    point_labels=table.row_labels,
                    left_values=left_values,
                    right_values=right_values,
                    axis="column",
                )
                if insight:
                    insights.append(insight)

    if len(table.columns) >= 3 and len(table.row_labels) >= 2:
        for left_idx in range(len(table.row_labels)):
            for right_idx in range(left_idx + 1, len(table.row_labels)):
                insight = _linear_relationship_insight(
                    left_label=table.row_labels[left_idx],
                    right_label=table.row_labels[right_idx],
                    point_labels=table.columns,
                    left_values=table.values[left_idx],
                    right_values=table.values[right_idx],
                    axis="row",
                )
                if insight:
                    insights.append(insight)

    return insights


def _linear_relationship_insight(
    left_label: str,
    right_label: str,
    point_labels: Sequence[str],
    left_values: Sequence[float],
    right_values: Sequence[float],
    axis: str,
) -> Insight | None:
    correlation = _pearson_correlation(left_values, right_values)
    if correlation is None or abs(correlation) < 0.85:
        return None

    direction = "positive" if correlation > 0 else "negative"
    evidence: List[EvidenceCell] = []
    for point, left_value, right_value in zip(point_labels, left_values, right_values):
        if axis == "column":
            evidence.append(EvidenceCell(row=point, column=left_label, value=left_value))
            evidence.append(EvidenceCell(row=point, column=right_label, value=right_value))
        else:
            evidence.append(EvidenceCell(row=left_label, column=point, value=left_value))
            evidence.append(EvidenceCell(row=right_label, column=point, value=right_value))

    compared_kind = "columns" if axis == "column" else "rows"
    point_kind = "rows" if axis == "column" else "columns"
    return Insight(
        type="relationship",
        subject=f"{left_label} vs {right_label}",
        condition=f"across {point_kind}",
        claim=(
            f"{left_label} and {right_label} show a {direction} linear relationship "
            f"across {point_kind} (r={correlation:.2f})."
        ),
        evidence=evidence,
        axis=axis,
        scope=f"paired_{compared_kind}",
        operation="pearson_correlation",
        details={
            "insight_subtype": "linear_correlation",
            "left": left_label,
            "right": right_label,
            "correlation": round(correlation, 4),
            "direction": direction,
        },
        score={"support": round(abs(correlation), 4), "importance": round(abs(correlation), 4)},
    )


def _pearson_correlation(left_values: Sequence[float], right_values: Sequence[float]) -> float | None:
    if len(left_values) != len(right_values) or len(left_values) < 3:
        return None
    left_mean = statistics.fmean(left_values)
    right_mean = statistics.fmean(right_values)
    left_centered = [value - left_mean for value in left_values]
    right_centered = [value - right_mean for value in right_values]
    left_norm = math.sqrt(sum(value * value for value in left_centered))
    right_norm = math.sqrt(sum(value * value for value in right_centered))
    if math.isclose(left_norm, 0.0) or math.isclose(right_norm, 0.0):
        return None
    return sum(left * right for left, right in zip(left_centered, right_centered)) / (left_norm * right_norm)


def _ordered_numeric_positions(labels: Sequence[str]) -> List[float]:
    if len(labels) < 3:
        return []
    positions: List[float] = []
    for label in labels:
        numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", label)
        if not numbers:
            return []
        positions.append(float(numbers[-1]))
    if len(set(positions)) != len(positions):
        return []
    increasing = all(left < right for left, right in zip(positions, positions[1:]))
    decreasing = all(left > right for left, right in zip(positions, positions[1:]))
    return positions if increasing or decreasing else []


def _column_directions(
    table: TableData,
    metric_types: Sequence[str],
    lower_is_better_columns: Sequence[str],
) -> List[str]:
    explicit_lower = {_normalize_label(column) for column in lower_is_better_columns}
    directions = []
    for idx, column in enumerate(table.columns):
        metric = metric_types[idx] if idx < len(metric_types) else column
        if _normalize_label(column) in explicit_lower or _is_lower_is_better(metric) or _is_lower_is_better(column):
            directions.append("lower")
        else:
            directions.append("higher")
    return directions


def _infer_system_axis(
    table: TableData,
    metric_types: Sequence[str],
    target_entities: Sequence[str],
) -> str:
    row_target_hits = _target_hits(table.row_labels, target_entities)
    column_target_hits = _target_hits(table.columns, target_entities)
    if column_target_hits > row_target_hits:
        return "column"
    if row_target_hits > column_target_hits:
        return "row"

    row_baselines = sum(1 for label in table.row_labels if _is_baseline(label))
    column_baselines = sum(1 for label in table.columns if _is_baseline(label))
    if column_baselines > row_baselines:
        return "column"
    if row_baselines > column_baselines:
        return "row"

    metric_matches = 0
    for idx, column in enumerate(table.columns):
        metric = metric_types[idx] if idx < len(metric_types) else ""
        if metric and _normalize_label(metric) in _normalize_label(column):
            metric_matches += 1
    if table.columns and metric_matches / len(table.columns) >= 0.6:
        return "row"

    row_method_score = sum(_method_likeness(label) for label in table.row_labels)
    column_method_score = sum(_method_likeness(label) for label in table.columns)
    if column_method_score > row_method_score:
        return "column"
    return "row"


def _entity_cell(axis: str, entity: str, context: str, value: float) -> EvidenceCell:
    if axis == "row":
        return EvidenceCell(row=entity, column=context, value=value)
    row, metric = _split_context_metric(context)
    return EvidenceCell(row=row, column=entity if not metric else f"{entity} ({metric})", value=value)


def _split_context_metric(context: str) -> tuple[str, str]:
    match = re.fullmatch(r"(.+) \((.+)\)", context)
    if not match:
        return context, ""
    return match.group(1), match.group(2)


def _effective_value(value: float, direction: str) -> float:
    return -value if direction == "lower" else value


def _majority_direction(directions: Sequence[str]) -> str:
    if not directions:
        return "higher"
    return Counter(directions).most_common(1)[0][0]


def _common_metric_name(metric_types: Sequence[str]) -> str:
    normalized = [metric for metric in metric_types if metric]
    if not normalized:
        return ""
    counts = Counter(normalized)
    metric, count = counts.most_common(1)[0]
    return metric if count >= 2 else ""


def _metric_name(metric_types: Sequence[str], idx: int, fallback: str) -> str:
    if idx < len(metric_types) and metric_types[idx]:
        return str(metric_types[idx])
    return fallback


def _is_lower_is_better(text: str) -> bool:
    normalized = _normalize_label(text)
    return any(pattern in normalized for pattern in LOWER_IS_BETTER_PATTERNS)


def _is_baseline(label: str) -> bool:
    normalized = _normalize_label(label)
    return any(token in normalized for token in BASELINE_TOKENS)


def _method_likeness(label: str) -> int:
    score = 0
    if METHOD_TOKEN_PATTERN.search(label):
        score += 2
    if "+" in label or "/" in label:
        score += 1
    if _is_baseline(label):
        score += 2
    return score


def _target_hits(labels: Sequence[str], targets: Sequence[str]) -> int:
    hits = 0
    normalized_labels = [_normalize_label(label) for label in labels]
    for target in targets:
        normalized_target = _normalize_label(target)
        if not normalized_target:
            continue
        if any(normalized_target in label or label in normalized_target for label in normalized_labels):
            hits += 1
    return hits


def _label_tokens(label: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9]+", label.lower())
        if len(token) > 1 and token not in {"with", "without", "and", "the", "method", "model", "system"}
    }


def _row_group(label: str) -> str:
    parts = [part.strip() for part in label.split("|") if part.strip()]
    if len(parts) < 2:
        return ""
    for part in parts[:-1]:
        normalized = _normalize_label(part)
        if normalized not in {"set", "dataset", "task", "method", "model", "system"}:
            return part
    return parts[0]


def _normalize_label(text: str) -> str:
    return " ".join(re.findall(r"[a-zA-Z0-9@]+", str(text).lower()))


def _dedupe_and_sort(insights: Iterable[Insight]) -> List[Insight]:
    ranked = []
    seen = set()
    for insight in insights:
        key = (
            insight.type,
            insight.axis,
            insight.scope,
            insight.subject,
            insight.condition,
            insight.claim,
        )
        if key in seen:
            continue
        seen.add(key)
        ranked.append(insight)

    def score_value(item: Insight) -> float:
        if not item.score:
            return 0.0
        return item.score.get("support", 0.0) + item.score.get("importance", 0.0)

    ranked.sort(key=score_value, reverse=True)
    return ranked
