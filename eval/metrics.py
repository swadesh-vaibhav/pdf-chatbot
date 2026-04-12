#!/usr/bin/env python3
"""
metrics.py

Reusable scoring helpers for the PDF chatbot evaluation suite.

This module is intentionally pure:
- no API calls
- no file I/O
- no side effects

It is designed to be imported by run_eval.py.

Suggested use:
    from metrics import (
        keyword_recall,
        page_recall,
        groundedness_score,
        autocomplete_match,
        extract_pages_from_text,
    )
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence, List, Dict, Any


_PAGE_PATTERNS = [
    re.compile(r"\[page\s*([0-9]+(?:\s*&\s*[0-9]+)*)\]", re.IGNORECASE),
    re.compile(r"\(page\s*([0-9]+(?:\s*&\s*[0-9]+)*)\)", re.IGNORECASE),
    re.compile(r"\bpage\s*([0-9]+(?:\s*&\s*[0-9]+)*)\b", re.IGNORECASE),
]


def normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace for stable matching."""
    return " ".join(str(text).lower().split())


def tokenize(text: str) -> List[str]:
    """Simple word tokenizer used for overlap-based matching."""
    return re.findall(r"[a-z0-9@._-]+", normalize_text(text))


def _token_set(text: str) -> set[str]:
    return set(tokenize(text))


def jaccard_similarity(a: str, b: str) -> float:
    """Token-level Jaccard similarity."""
    ta = _token_set(a)
    tb = _token_set(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _substring_similarity(a: str, b: str) -> float:
    """
    Soft containment score that rewards good paraphrases.

    - 1.0 if one string is contained in the other
    - otherwise proportional to token overlap
    """
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 1.0

    ta = tokenize(a)
    tb = tokenize(b)
    if not ta or not tb:
        return 0.0

    inter = len(set(ta) & set(tb))
    denom = max(len(set(ta)), len(set(tb)))
    return inter / denom if denom else 0.0


def keyword_recall(text: str, keywords: Sequence[str]) -> float:
    """
    Fraction of expected keywords appearing in the text.

    Matching is substring-based over normalized text, which is more forgiving
    than exact token equality while still being easy to interpret.
    """
    kws = [k for k in keywords if str(k).strip()]
    if not kws:
        return 0.0

    norm = normalize_text(text)
    hits = 0
    for kw in kws:
        if normalize_text(kw) in norm:
            hits += 1
    return hits / len(kws)


def page_recall(found_pages: Sequence[int], expected_pages: Sequence[int]) -> float:
    """Fraction of required pages found in the cited/retrieved pages."""
    req = [int(p) for p in expected_pages if str(p).strip()]
    if not req:
        return 0.0

    found = {int(p) for p in found_pages if str(p).strip()}
    hits = sum(1 for p in req if p in found)
    return hits / len(req)


def extract_pages_from_text(text: str) -> List[int]:
    """
    Extract page numbers from answer text.

    Supports forms like:
      - [page 1]
      - (Page 3)
      - page 1 & 2
      - page 12
    """
    if not text:
        return []

    pages: set[int] = set()
    for pattern in _PAGE_PATTERNS:
        for match in pattern.finditer(text):
            raw = match.group(1)
            for part in re.split(r"\s*&\s*", raw):
                part = part.strip()
                if part.isdigit():
                    pages.add(int(part))
    return sorted(pages)


def groundedness_score(
    answer: str,
    expected_keywords: Sequence[str],
    found_pages: Sequence[int] | None = None,
    expected_pages: Sequence[int] | None = None,
) -> float:
    """
    Blend answer keyword recall and page recall.

    Intended as a simple, interpretable proxy:
      70% content overlap
      30% citation/page agreement

    If page info is unavailable, it falls back to keyword recall only.
    """
    kw = keyword_recall(answer, expected_keywords)

    found_pages = list(found_pages or [])
    expected_pages = list(expected_pages or [])
    pg = page_recall(found_pages, expected_pages) if expected_pages else 0.0

    if expected_pages:
        return 0.7 * kw + 0.3 * pg
    return kw


@dataclass(frozen=True)
class AutocompleteScore:
    any_match: bool
    top1_match: bool
    hit_count: int
    best_similarity: float


def autocomplete_match(
    suggestions: Sequence[str],
    expected_suggestions: Sequence[str],
    *,
    exact_threshold: float = 0.88,
    match_threshold: float = 0.55,
) -> AutocompleteScore:
    """
    Score autocomplete suggestions against expected suggestions.

    The logic is intentionally tolerant:
    - exact-ish matches are detected via high token similarity
    - semantically close suggestions get partial credit via overlap

    Parameters
    ----------
    suggestions:
        Model output suggestions, ordered best-first.
    expected_suggestions:
        Reference suggestions from the dataset.
    exact_threshold:
        Similarity threshold for a top-1 exact-ish match.
    match_threshold:
        Similarity threshold to count a hit for any suggestion.
    """
    suggs = [s for s in suggestions if str(s).strip()]
    expected = [e for e in expected_suggestions if str(e).strip()]

    if not suggs or not expected:
        return AutocompleteScore(
            any_match=False,
            top1_match=False,
            hit_count=0,
            best_similarity=0.0,
        )

    best_similarity = 0.0
    hit_count = 0

    for s in suggs:
        s_norm = normalize_text(s)
        local_best = 0.0
        for e in expected:
            e_norm = normalize_text(e)
            sim = max(
                jaccard_similarity(s_norm, e_norm),
                _substring_similarity(s_norm, e_norm),
            )
            local_best = max(local_best, sim)
            best_similarity = max(best_similarity, sim)
        if local_best >= match_threshold:
            hit_count += 1

    top1 = normalize_text(suggs[0])
    top1_best = 0.0
    for e in expected:
        e_norm = normalize_text(e)
        sim = max(
            jaccard_similarity(top1, e_norm),
            _substring_similarity(top1, e_norm),
        )
        top1_best = max(top1_best, sim)

    return AutocompleteScore(
        any_match=hit_count > 0,
        top1_match=top1_best >= exact_threshold,
        hit_count=hit_count,
        best_similarity=best_similarity,
    )


def score_qa_item(
    answer: str,
    expected_keywords: Sequence[str],
    expected_pages: Sequence[int] | None = None,
    found_pages: Sequence[int] | None = None,
) -> Dict[str, Any]:
    """
    Convenience helper that returns all QA scoring pieces in one dict.
    """
    expected_pages = list(expected_pages or [])
    found_pages = list(found_pages or extract_pages_from_text(answer))

    kw = keyword_recall(answer, expected_keywords)
    pg = page_recall(found_pages, expected_pages) if expected_pages else 0.0
    g = groundedness_score(answer, expected_keywords, found_pages, expected_pages)

    return {
        "keyword_recall": kw,
        "page_recall": pg,
        "groundedness_score": g,
        "found_pages": found_pages,
    }


def score_autocomplete_item(
    suggestions: Sequence[str],
    expected_suggestions: Sequence[str],
) -> Dict[str, Any]:
    """
    Convenience helper that returns autocomplete scoring pieces in one dict.
    """
    score = autocomplete_match(suggestions, expected_suggestions)
    return {
        "any_match": score.any_match,
        "top1_match": score.top1_match,
        "hit_count": score.hit_count,
        "best_similarity": score.best_similarity,
    }
