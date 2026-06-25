from __future__ import annotations

from typing import List, Sequence

from .schema import Insight


TYPE_PRIORITY = {
    "comparison": 5.0,
    "extremum": 4.0,
    "aggregate": 3.5,
    "trend": 3.0,
}


def rank_insights(insights: Sequence[Insight], top_k: int | None = None) -> List[Insight]:
    deduped = _dedupe(insights)
    ranked = sorted(deduped, key=_rank_key, reverse=True)
    return ranked[:top_k] if top_k else ranked


def _rank_key(insight: Insight) -> tuple[float, float, int]:
    score = insight.score or {}
    importance = float(score.get("importance", 0.0))
    support = float(score.get("support", 1.0))
    return (TYPE_PRIORITY.get(insight.type, 1.0), support, min(importance, 1000.0))


def _dedupe(insights: Sequence[Insight]) -> List[Insight]:
    seen = set()
    result = []
    for insight in insights:
        key = (insight.type, insight.subtype, insight.claim.lower())
        if key in seen:
            continue
        seen.add(key)
        result.append(insight)
    return result

