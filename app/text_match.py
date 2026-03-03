from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from app.ocr_service import OCRDetection

_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_WHITESPACE = re.compile(r"\s+")


@dataclass
class OCRMatch:
    detection: OCRDetection
    normalized_text: str
    fuzzy_score: float
    score: float


def normalize_text(value: str) -> str:
    text = value.lower().strip()
    text = _NON_ALNUM.sub(" ", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


def _fuzzy_score(query: str, candidate: str) -> float:
    try:
        from rapidfuzz import fuzz  # type: ignore[import-untyped]

        return max(
            fuzz.ratio(query, candidate) / 100.0,
            fuzz.partial_ratio(query, candidate) / 100.0,
            fuzz.token_set_ratio(query, candidate) / 100.0,
        )
    except Exception:
        return SequenceMatcher(a=query, b=candidate).ratio()


def _size_prior(det: OCRDetection, width: int, height: int) -> float:
    area = max(1, (det.x2 - det.x1) * (det.y2 - det.y1))
    image_area = max(1, width * height)
    area_ratio = area / image_area

    # Favor moderate clickable regions; too tiny/too huge boxes are less likely targets.
    if area_ratio <= 0:
        return 0.0
    if area_ratio < 0.0002:
        return 0.1
    if area_ratio < 0.005:
        return 0.7
    if area_ratio < 0.08:
        return 1.0
    return 0.4


def rank_ocr_matches(
    detections: list[OCRDetection],
    target_text: str,
    *,
    image_width: int,
    image_height: int,
) -> list[OCRMatch]:
    query = normalize_text(target_text)
    if not query:
        return []

    matches: list[OCRMatch] = []
    query_tokens = set(query.split())
    for det in detections:
        normalized = normalize_text(det.text)
        if not normalized:
            continue

        fuzzy = _fuzzy_score(query, normalized)
        token_overlap = 0.0
        if query_tokens:
            candidate_tokens = set(normalized.split())
            token_overlap = len(query_tokens & candidate_tokens) / len(query_tokens)

        # Hard floor to avoid ranking obviously unrelated text.
        if fuzzy < 0.45 and token_overlap < 0.5:
            continue

        size = _size_prior(det, image_width, image_height)
        score = (0.55 * fuzzy) + (0.35 * max(0.0, min(1.0, det.confidence))) + (0.10 * size)
        matches.append(
            OCRMatch(
                detection=det,
                normalized_text=normalized,
                fuzzy_score=fuzzy,
                score=round(score, 4),
            )
        )

    matches.sort(
        key=lambda m: (
            m.score,
            m.fuzzy_score,
            m.detection.confidence,
            -len(m.normalized_text),
        ),
        reverse=True,
    )
    return matches
