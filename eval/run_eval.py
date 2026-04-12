#!/usr/bin/env python3
"""
run_eval.py

End-to-end evaluation runner for the PDF chatbot.

Responsibilities:
- upload the reference PDF to the backend
- load QA and autocomplete datasets
- call /chat and /autocomplete
- score results using metrics.py
- write a markdown report

Backend endpoints:
  POST /upload
  POST /chat
  POST /autocomplete

Usage (macOS/Linux):
  python run_eval.py \
    --pdf data/sample_policy.pdf \
    --qa data/qa_dataset.jsonl \
    --autocomplete data/autocomplete_dataset.jsonl \
    --backend http://127.0.0.1:8000 \
    --out eval_report.md

Usage (Windows):
  python run_eval.py ^
    --pdf data\sample_policy.pdf ^
    --qa data\qa_dataset.jsonl ^
    --autocomplete data\autocomplete_dataset.jsonl ^
    --backend http://127.0.0.1:8000 ^
    --out eval_report.md
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Sequence

import requests

from metrics import (
    autocomplete_match,
    extract_pages_from_text,
    groundedness_score,
    keyword_recall,
    page_recall,
)


@dataclass
class QASample:
    question: str
    expected_keywords: List[str]
    must_reference_pages: List[int]
    expected_answer: str | None = None


@dataclass
class AutocompleteSample:
    prefix: str
    expected_suggestions: List[str]


@dataclass
class QAResult:
    question: str
    answer: str
    latency_ms: float
    keyword_recall: float
    page_recall: float
    groundedness_score: float
    found_pages: List[int]


@dataclass
class AutocompleteResult:
    prefix: str
    suggestions: List[str]
    latency_ms: float
    any_match: bool
    top1_match: bool
    hit_count: int
    best_similarity: float


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {e}") from e
    return rows


def load_qa_dataset(path: Path) -> List[QASample]:
    data = read_jsonl(path)
    return [
        QASample(
            question=row["question"],
            expected_keywords=list(row.get("expected_keywords", [])),
            must_reference_pages=list(row.get("must_reference_pages", [])),
            expected_answer=row.get("expected_answer"),
        )
        for row in data
    ]


def load_autocomplete_dataset(path: Path) -> List[AutocompleteSample]:
    data = read_jsonl(path)
    return [
        AutocompleteSample(
            prefix=row["prefix"],
            expected_suggestions=list(row.get("expected_suggestions", [])),
        )
        for row in data
    ]


def post_json(url: str, payload: Dict[str, Any], timeout: float = 120.0) -> Dict[str, Any]:
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def upload_pdf(backend: str, pdf_path: Path) -> Dict[str, Any]:
    with pdf_path.open("rb") as f:
        files = {"file": (pdf_path.name, f, "application/pdf")}
        resp = requests.post(f"{backend}/upload", files=files, timeout=300)
    resp.raise_for_status()
    return resp.json()


def mean(values: Sequence[float]) -> float:
    return float(stats.mean(values)) if values else 0.0


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    if len(vals) == 1:
        return float(vals[0])
    k = (len(vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    if f == c:
        return float(vals[f])
    return float(vals[f] + (vals[c] - vals[f]) * (k - f))


def summarize_qa(results: List[QAResult]) -> Dict[str, Any]:
    latencies = [r.latency_ms for r in results]
    keyword_recalls = [r.keyword_recall for r in results]
    page_recalls = [r.page_recall for r in results]
    groundedness = [r.groundedness_score for r in results]

    return {
        "count": len(results),
        "avg_latency_ms": mean(latencies),
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "avg_keyword_recall": mean(keyword_recalls),
        "avg_page_recall": mean(page_recalls),
        "avg_groundedness": mean(groundedness),
    }


def summarize_autocomplete(results: List[AutocompleteResult]) -> Dict[str, Any]:
    latencies = [r.latency_ms for r in results]
    any_match_rate = mean([1.0 if r.any_match else 0.0 for r in results])
    top1_match_rate = mean([1.0 if r.top1_match else 0.0 for r in results])
    avg_hit_count = mean([float(r.hit_count) for r in results])
    avg_best_similarity = mean([r.best_similarity for r in results])

    return {
        "count": len(results),
        "avg_latency_ms": mean(latencies),
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "any_match_rate": any_match_rate,
        "top1_match_rate": top1_match_rate,
        "avg_hit_count": avg_hit_count,
        "avg_best_similarity": avg_best_similarity,
    }


def run_qa_eval(backend: str, samples: List[QASample]) -> List[QAResult]:
    results: List[QAResult] = []

    for sample in samples:
        t0 = time.perf_counter()
        payload = post_json(f"{backend}/chat", {"query": sample.question}, timeout=180)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        answer = str(payload.get("answer", ""))
        context = payload.get("context", [])

        # Prefer backend-returned pages, but fall back to extracting citations from the answer text.
        found_pages: List[int] = []
        found_pages = extract_pages_from_text(answer)

        kw = keyword_recall(answer, sample.expected_keywords)
        pg = page_recall(found_pages, sample.must_reference_pages)
        grounded = groundedness_score(answer, sample.expected_keywords, found_pages, sample.must_reference_pages)

        results.append(
            QAResult(
                question=sample.question,
                answer=answer,
                latency_ms=latency_ms,
                keyword_recall=kw,
                page_recall=pg,
                groundedness_score=grounded,
                found_pages=found_pages,
            )
        )

    return results


def run_autocomplete_eval(backend: str, samples: List[AutocompleteSample]) -> List[AutocompleteResult]:
    results: List[AutocompleteResult] = []

    for sample in samples:
        t0 = time.perf_counter()
        payload = post_json(f"{backend}/autocomplete", {"prefix": sample.prefix}, timeout=180)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        suggestions = [str(s) for s in payload.get("suggestions", []) if str(s).strip()]
        score = autocomplete_match(suggestions, sample.expected_suggestions)

        results.append(
            AutocompleteResult(
                prefix=sample.prefix,
                suggestions=suggestions,
                latency_ms=latency_ms,
                any_match=score.any_match,
                top1_match=score.top1_match,
                hit_count=score.hit_count,
                best_similarity=score.best_similarity,
            )
        )

    return results


def write_report(
    out_path: Path,
    pdf_path: Path,
    qa_path: Path,
    autocomplete_path: Path,
    qa_results: List[QAResult],
    ac_results: List[AutocompleteResult],
    qa_summary: Dict[str, Any],
    ac_summary: Dict[str, Any],
) -> None:
    bad_qa = sorted(qa_results, key=lambda r: r.groundedness_score)[:5]
    bad_ac = sorted(ac_results, key=lambda r: (r.top1_match, r.any_match, -r.best_similarity, -r.latency_ms))[:5]

    lines: List[str] = []
    lines.append("# PDF Chatbot Evaluation Report")
    lines.append("")
    lines.append(f"- PDF: `{pdf_path}`")
    lines.append(f"- QA dataset: `{qa_path}`")
    lines.append(f"- Autocomplete dataset: `{autocomplete_path}`")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("### Question Answering")
    lines.append(f"- Samples: {qa_summary['count']}")
    lines.append(f"- Avg groundedness: {qa_summary['avg_groundedness']:.3f}")
    lines.append(f"- Avg keyword recall: {qa_summary['avg_keyword_recall']:.3f}")
    lines.append(f"- Avg page recall: {qa_summary['avg_page_recall']:.3f}")
    lines.append(f"- Median latency: {qa_summary['p50_latency_ms']:.1f} ms")
    lines.append(f"- P95 latency: {qa_summary['p95_latency_ms']:.1f} ms")
    lines.append("")

    lines.append("### Autocomplete")
    lines.append(f"- Samples: {ac_summary['count']}")
    lines.append(f"- Any-match rate: {ac_summary['any_match_rate'] * 100:.1f}%")
    lines.append(f"- Top-1 match rate: {ac_summary['top1_match_rate'] * 100:.1f}%")
    lines.append(f"- Avg best similarity: {ac_summary['avg_best_similarity']:.3f}")
    lines.append(f"- Median latency: {ac_summary['p50_latency_ms']:.1f} ms")
    lines.append(f"- P95 latency: {ac_summary['p95_latency_ms']:.1f} ms")
    lines.append("")

    lines.append("## Failure Analysis")
    lines.append("")
    lines.append("### Worst QA examples")
    for r in bad_qa:
        lines.append(f"- Q: {r.question}")
        lines.append(f"  - Groundedness: {r.groundedness_score:.3f}")
        lines.append(f"  - Keyword recall: {r.keyword_recall:.3f}")
        lines.append(f"  - Page recall: {r.page_recall:.3f}")
        lines.append(f"  - Found pages: {r.found_pages}")
        lines.append(f"  - Latency: {r.latency_ms:.1f} ms")
    lines.append("")

    lines.append("### Worst autocomplete examples")
    for r in bad_ac:
        lines.append(f"- Prefix: {r.prefix}")
        lines.append(f"  - Any-match: {r.any_match}")
        lines.append(f"  - Top-1 match: {r.top1_match}")
        lines.append(f"  - Hit count: {r.hit_count}")
        lines.append(f"  - Best similarity: {r.best_similarity:.3f}")
        lines.append(f"  - Latency: {r.latency_ms:.1f} ms")
        lines.append(f"  - Suggestions: {r.suggestions}")
    lines.append("")

    lines.append("## Raw JSON")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(
        {
            "qa_summary": qa_summary,
            "autocomplete_summary": ac_summary,
            "qa_results": [asdict(r) for r in qa_results],
            "autocomplete_results": [asdict(r) for r in ac_results],
        },
        indent=2,
        ensure_ascii=False,
    ))
    lines.append("```")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parser = argparse.ArgumentParser(description="Evaluate the PDF chatbot backend.")
    parser.add_argument("--pdf", type=Path, required=True, help="Path to the PDF used for evaluation.")
    parser.add_argument("--qa", type=Path, required=True, help="Path to qa_dataset.jsonl.")
    parser.add_argument("--autocomplete", type=Path, required=True, help="Path to autocomplete_dataset.jsonl.")
    parser.add_argument("--backend", type=str, default="http://127.0.0.1:8000", help="Backend base URL.")
    parser.add_argument("--out", type=Path, default=Path(f"reports/eval_report_{now}.md"), help="Output markdown report.")
    parser.add_argument("--skip-upload", action="store_true", help="Skip uploading the PDF first.")
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"PDF not found: {args.pdf}", file=sys.stderr)
        return 2
    if not args.qa.exists():
        print(f"QA dataset not found: {args.qa}", file=sys.stderr)
        return 2
    if not args.autocomplete.exists():
        print(f"Autocomplete dataset not found: {args.autocomplete}", file=sys.stderr)
        return 2

    backend = args.backend.rstrip("/")

    if not args.skip_upload:
        print(f"Uploading PDF: {args.pdf}")
        upload_result = upload_pdf(backend, args.pdf)
        print(f"Upload response: {upload_result}")

    print(f"Loading QA dataset: {args.qa}")
    qa_samples = load_qa_dataset(args.qa)
    print(f"Loading autocomplete dataset: {args.autocomplete}")
    ac_samples = load_autocomplete_dataset(args.autocomplete)

    print(f"Running QA eval on {len(qa_samples)} samples...")
    qa_results = run_qa_eval(backend, qa_samples)

    print(f"Running autocomplete eval on {len(ac_samples)} samples...")
    ac_results = run_autocomplete_eval(backend, ac_samples)

    qa_summary = summarize_qa(qa_results)
    ac_summary = summarize_autocomplete(ac_results)

    write_report(
        out_path=args.out,
        pdf_path=args.pdf,
        qa_path=args.qa,
        autocomplete_path=args.autocomplete,
        qa_results=qa_results,
        ac_results=ac_results,
        qa_summary=qa_summary,
        ac_summary=ac_summary,
    )

    print(f"Wrote report to: {args.out}")
    print("QA summary:", qa_summary)
    print("Autocomplete summary:", ac_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
