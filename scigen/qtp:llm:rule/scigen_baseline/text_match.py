from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence

from .schema import Insight


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "is", "it", "of", "on", "or", "than", "the", "this", "to", "with",
}


SIGNAL_PATTERNS = {
    "increase": [r"\bimprov(?:e|ed|es|ing|ement)\b", r"\bincreas(?:e|ed|es|ing)\b", r"\bgain(?:s|ed)?\b"],
    "decrease": [r"\bdecreas(?:e|ed|es|ing)\b", r"\bdrop(?:s|ped|ping)?\b", r"\breduc(?:e|ed|es|ing|tion)\b"],
    "best": [r"\bbest\b", r"\btop\b", r"\bhighest\b", r"\blargest\b", r"\bmaximum\b"],
    "worst": [r"\bworst\b", r"\blowest\b", r"\bsmallest\b", r"\bminimum\b"],
    "comparison": [r"\boutperform(?:s|ed|ing)?\b", r"\bbetter\b", r"\bhigher\b", r"\blower\b", r"\bcompared\b"],
    "consistent": [r"\bconsistent(?:ly)?\b", r"\bmost\b", r"\bmajority\b", r"\bacross\b"],
}


def score_insights_against_description(insights: Sequence[Insight], description: str) -> Dict[str, object]:
    matches = [score_insight(insight, description) for insight in insights]
    covered = [item for item in matches if item["label"] in {"covered", "partially_covered"}]
    contradicted = [item for item in matches if item["label"] == "possibly_contradicted"]
    return {
        "insight_count": len(insights),
        "covered_or_partial": len(covered),
        "possibly_contradicted": len(contradicted),
        "coverage_rate": round(len(covered) / len(insights), 4) if insights else 0.0,
        "matches": matches,
    }


def score_insight(insight: Insight, description: str) -> Dict[str, object]:
    claim = insight.claim
    token_overlap = _token_overlap(claim, description)
    number_overlap = _number_overlap(claim, description)
    signal_overlap = _signal_overlap(claim, description)
    score = 0.55 * token_overlap + 0.25 * number_overlap + 0.20 * signal_overlap
    label = "missing"
    if score >= 0.45:
        label = "covered"
    elif score >= 0.22:
        label = "partially_covered"
    if _opposite_direction(claim, description) and token_overlap >= 0.12:
        label = "possibly_contradicted"
    return {
        "label": label,
        "score": round(score, 4),
        "token_overlap": round(token_overlap, 4),
        "number_overlap": round(number_overlap, 4),
        "signal_overlap": round(signal_overlap, 4),
        "insight": insight.to_dict(),
    }


def _tokens(text: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_+-]*", text.lower())
        if token not in STOPWORDS and len(token) > 1
    ]


def _numbers(text: str) -> List[str]:
    values = []
    for raw in re.findall(r"[-+]?\d+(?:\.\d+)?", text):
        try:
            values.append(f"{float(raw):.4g}")
        except ValueError:
            values.append(raw)
    return values


def _signals(text: str) -> set[str]:
    lowered = text.lower()
    found = set()
    for label, patterns in SIGNAL_PATTERNS.items():
        if any(re.search(pattern, lowered) for pattern in patterns):
            found.add(label)
    return found


def _token_overlap(left: str, right: str) -> float:
    left_counts = Counter(_tokens(left))
    right_counts = Counter(_tokens(right))
    if not left_counts:
        return 0.0
    overlap = sum(min(count, right_counts[token]) for token, count in left_counts.items())
    return overlap / sum(left_counts.values())


def _number_overlap(left: str, right: str) -> float:
    left_numbers = set(_numbers(left))
    right_numbers = set(_numbers(right))
    if not left_numbers:
        return 0.0
    return len(left_numbers & right_numbers) / len(left_numbers)


def _signal_overlap(left: str, right: str) -> float:
    left_signals = _signals(left)
    right_signals = _signals(right)
    if not left_signals:
        return 0.0
    return len(left_signals & right_signals) / len(left_signals)


def _opposite_direction(claim: str, description: str) -> bool:
    claim_signals = _signals(claim)
    desc_signals = _signals(description)
    return ("increase" in claim_signals and "decrease" in desc_signals) or (
        "decrease" in claim_signals and "increase" in desc_signals
    ) or ("best" in claim_signals and "worst" in desc_signals) or (
        "worst" in claim_signals and "best" in desc_signals
    )

