from __future__ import annotations

import math
from typing import List, Sequence

from .core import Insight


TYPE_WEIGHTS = {
    "comparison": 2.0,
    "extremum": 2.8,
    "relationship": 3.3,
    "proportion": 3.4,
    "trend": 3.5,
}

SUBTYPE_WEIGHTS = {
    "baseline_improvement": 5.3,
    "ablation_gain": 5.1,
    "outperformance": 4.9,
    "overall_best": 4.7,
    "metric_best": 4.5,
    "group_contrast": 4.4,
    "near_tie": 4.1,
    "metric_worst": 3.8,
}


def rank_insights(insights: Sequence[Insight], top_k: int | None = None) -> List[Insight]:
    ranked = sorted(insights, key=insight_rank_score, reverse=True)
    if top_k is None or top_k <= 0:
        return ranked
    return ranked[:top_k]


def insight_rank_score(insight: Insight) -> float:
    subtype = _insight_subtype(insight)
    type_score = SUBTYPE_WEIGHTS.get(subtype, TYPE_WEIGHTS.get(insight.type, 1.0))
    support = (insight.score or {}).get("support", 0.0)
    importance = (insight.score or {}).get("importance", 0.0)
    score = type_score + support + _scaled_importance(importance)

    if insight.type == "comparison":
        score += _comparison_bonus(insight)
    if subtype in {
        "baseline_improvement",
        "ablation_gain",
        "outperformance",
        "overall_best",
        "metric_best",
        "group_contrast",
        "near_tie",
    }:
        score += 0.5

    return score


def _insight_subtype(insight: Insight) -> str:
    return str((insight.details or {}).get("insight_subtype") or "")


def _scaled_importance(value: float) -> float:
    if value <= 0:
        return 0.0
    return min(math.log1p(value) / 8.0, 1.5)


def _comparison_bonus(insight: Insight) -> float:
    if insight.axis == "column" and insight.scope in {"single_row", "interval"}:
        return 0.5
    if insight.axis == "cell_pair":
        return 0.2
    if insight.axis == "row":
        return -0.4
    return 0.0
