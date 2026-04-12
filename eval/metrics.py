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


# Regex patterns for extracting page references from LLM-generated answer text.
# Each pattern captures one or more page numbers (e.g. "1", "1 & 2") in group(1).
# Ordered from most specific to least specific to prefer bracketed citations.
_PAGE_PATTERNS = [
    re.compile(r"\[page\s*([0-9]+(?:\s*&\s*[0-9]+)*)\]", re.IGNORECASE),  # [page 3]
    re.compile(r"\(page\s*([0-9]+(?:\s*&\s*[0-9]+)*)\)", re.IGNORECASE),  # (page 3)
    re.compile(r"\bpage\s*([0-9]+(?:\s*&\s*[0-9]+)*)\b", re.IGNORECASE),  # page 3
]


def normalize_text(text: str) -> str:
    """
    Lowercase and collapse whitespace for stable matching.

    Args:
        text: Raw input string.

    Returns:
        Lowercased string with all runs of whitespace replaced by a single space.
    """
    return " ".join(str(text).lower().split())


def tokenize(text: str) -> List[str]:
    """
    Split text into lowercase alphanumeric tokens for overlap-based matching.

    Keeps email-like characters (@ . _ -) so that terms like "user@example.com"
    or "pre-existing" survive as single tokens.

    Args:
        text: Raw input string.

    Returns:
        List of lowercase token strings.
    """
    return re.findall(r"[a-z0-9@._-]+", normalize_text(text))


def _token_set(text: str) -> set[str]:
    """
    Return the unique set of tokens in *text* (used by Jaccard similarity).
    Args:
        text: Raw input string.

    Returns:
        Set of unique tokens.
    """
    return set(tokenize(text))


def jaccard_similarity(a: str, b: str) -> float:
    """
    Compute token-level Jaccard similarity between two strings.

    Jaccard index = |intersection| / |union| of the token sets.
    Returns 1.0 when both strings are empty (trivially identical).

    Args:
        a: First string.
        b: Second string.

    Returns:
        Similarity score in [0.0, 1.0].
    """
    ta = _token_set(a)
    tb = _token_set(b)
    if not ta and not tb:
        return 1.0  # Both empty → identical
    if not ta or not tb:
        return 0.0  # One empty, one not → no overlap
    return len(ta & tb) / len(ta | tb)


def _substring_similarity(a: str, b: str) -> float:
    """
    Soft containment score that rewards good paraphrases.

    Scoring logic:
      - 1.0 if one string is a literal substring of the other
      - Otherwise, the ratio of shared tokens to the larger token set

    This complements Jaccard by giving full credit to containment cases
    (e.g. expected="coverage" inside answer="full coverage plan details").

    Args:
        a: First string (already normalized is fine, but not required).
        b: Second string.

    Returns:
        Score in [0.0, 1.0].
    """
    if not a or not b:
        return 0.0
    # Full containment → perfect score
    if a in b or b in a:
        return 1.0

    ta = tokenize(a)
    tb = tokenize(b)
    if not ta or not tb:
        return 0.0

    # Proportional token overlap, normalized by the larger set
    inter = len(set(ta) & set(tb))
    denom = max(len(set(ta)), len(set(tb)))
    return inter / denom if denom else 0.0


def keyword_recall(text: str, keywords: Sequence[str]) -> float:
    """
    Fraction of expected keywords that appear in the answer text.

    Uses case-insensitive substring matching after whitespace normalization,
    which is more forgiving than exact token equality while still being
    easy to interpret.

    Args:
        text: The model-generated answer to search within.
        keywords: Ground-truth keywords that should appear in a correct answer.

    Returns:
        Recall score in [0.0, 1.0].  Returns 0.0 if *keywords* is empty.
    """
    # Filter out blank keywords
    kws = [k for k in keywords if str(k).strip()]
    if not kws:
        return 0.0

    norm = normalize_text(text)
    hits = 0
    for kw in kws:
        # Substring check after normalizing both sides
        if normalize_text(kw) in norm:
            hits += 1
    return hits / len(kws)


def page_recall(found_pages: Sequence[int], expected_pages: Sequence[int]) -> float:
    """
    Fraction of required pages found in the cited/retrieved pages.

    Args:
        found_pages: Page numbers extracted from the model answer or context.
        expected_pages: Ground-truth pages that a correct answer should cite.

    Returns:
        Recall score in [0.0, 1.0].  Returns 0.0 if *expected_pages* is empty.
    """
    req = [int(p) for p in expected_pages if str(p).strip()]
    if not req:
        return 0.0

    found = {int(p) for p in found_pages if str(p).strip()}
    hits = sum(1 for p in req if p in found)
    return hits / len(req)


def extract_pages_from_text(text: str) -> List[int]:
    """
    Extract page numbers cited in LLM-generated answer text.

    Supports common citation styles produced by language models:
      - Bracketed:    ``[page 1]``, ``[Page 1 & 2]``
      - Parenthesized: ``(page 3)``
      - Bare:          ``page 12``

    Args:
        text: The raw answer string to scan.

    Returns:
        Sorted list of unique page numbers found. Empty list if none detected.
    """
    if not text:
        return []

    pages: set[int] = set()
    for pattern in _PAGE_PATTERNS:
        for match in pattern.finditer(text):
            # group(1) may contain "1 & 2", so split on "&"
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
    Weighted blend of keyword recall and page recall.

    Provides a single "groundedness" number that captures both *content*
    correctness (did the answer mention the right terms?) and *citation*
    correctness (did it reference the right pages?).

    Weighting:
      - 70 % keyword recall  — rewards factual content overlap
      - 30 % page recall      — rewards correct source attribution

    Falls back to keyword recall alone when no expected pages are provided.

    Args:
        answer: The model-generated answer text.
        expected_keywords: Keywords a correct answer should contain.
        found_pages: Pages cited in or extracted from the answer.
        expected_pages: Ground-truth pages the answer should reference.

    Returns:
        Score in [0.0, 1.0].
    """
    kw = keyword_recall(answer, expected_keywords)

    found_pages = list(found_pages or [])
    expected_pages = list(expected_pages or [])
    pg = page_recall(found_pages, expected_pages) if expected_pages else 0.0

    if expected_pages:
        # Weighted combination: content (70%) + citation (30%)
        return 0.7 * kw + 0.3 * pg
    # No page ground-truth available — rely on keyword recall only
    return kw


@dataclass(frozen=True)
class AutocompleteScore:
    """
    Immutable result container for autocomplete evaluation.

    Attributes:
        any_match: True if at least one suggestion exceeds the match threshold.
        top1_match: True if the first (highest-ranked) suggestion is an exact-ish match.
        hit_count: Number of suggestions that exceed the match threshold.
        best_similarity: Highest similarity score observed across all suggestion–expected pairs.
    """

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
    # Strip blank suggestions / expected values
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

    # --- Pass 1: score every suggestion against every expected value ---
    for s in suggs:
        s_norm = normalize_text(s)
        local_best = 0.0  # Best score for *this* suggestion
        for e in expected:
            e_norm = normalize_text(e)
            # Take the better of Jaccard and substring similarity
            sim = max(
                jaccard_similarity(s_norm, e_norm),
                _substring_similarity(s_norm, e_norm),
            )
            local_best = max(local_best, sim)
            best_similarity = max(best_similarity, sim)
        # Count this suggestion as a "hit" if it's close enough to any expected
        if local_best >= match_threshold:
            hit_count += 1

    # --- Pass 2: separately evaluate the top-ranked suggestion ---
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
    Convenience helper that bundles all QA metrics into a single dict.

    If *found_pages* is not supplied, pages are extracted automatically from
    the answer text via :func:`extract_pages_from_text`.

    Args:
        answer: Model-generated answer.
        expected_keywords: Keywords a correct answer should contain.
        expected_pages: Ground-truth pages the answer should cite.
        found_pages: Pages already extracted/returned by the backend.

    Returns:
        Dict with keys ``keyword_recall``, ``page_recall``,
        ``groundedness_score``, and ``found_pages``.
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
    Convenience helper that bundles autocomplete metrics into a single dict.

    Args:
        suggestions: Model-produced autocomplete suggestions.
        expected_suggestions: Reference suggestions from the evaluation dataset.

    Returns:
        Dict with keys ``any_match``, ``top1_match``, ``hit_count``,
        and ``best_similarity``.
    """
    score = autocomplete_match(suggestions, expected_suggestions)
    return {
        "any_match": score.any_match,
        "top1_match": score.top1_match,
        "hit_count": score.hit_count,
        "best_similarity": score.best_similarity,
    }
