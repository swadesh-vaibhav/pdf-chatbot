#!/usr/bin/env python3
"""
run_eval.py

Evaluate a PDF chatbot backend on:
1) QA accuracy / groundedness / latency
2) Autocomplete usefulness / latency

Assumptions:
- Backend exposes:
    POST /upload
    POST /chat
    POST /autocomplete
- /upload accepts multipart form field named "file"
- /chat accepts JSON: {"query": "..."}
- /autocomplete accepts JSON: {"prefix": "..."}

Example:
    python run_eval.py \
      --pdf /mnt/data/sample_policy.pdf \
      --qa /mnt/data/qa_dataset.jsonl \
      --autocomplete /mnt/data/autocomplete_dataset.jsonl \
      --backend http://127.0.0.1:8000 \
      --out /mnt/data/eval_report.md
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


@dataclass
class QASample:
    question: str
    expected_keywords: List[str]
    must_reference_pages: List[int]
    expected_answer: Optional[str] = None


@dataclass
class AutocompleteSample:
    prefix: str
    expected_suggestions: List[str]


@dataclass
class QAResult:
    question: str
    answer: str
    context_pages: List[int]
    latency_ms: float
    keyword_hits: int
    keyword_total: int
    page_hits: int
    page_total: int
    groundedness_score: float


@dataclass
class AutocompleteResult:
    prefix: str
    suggestions: List[str]
    latency_ms: float
    any_match: bool
    top1_match: bool
    hit_count: int


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num} in {path}: {e}") from e
    return rows


def load_qa_dataset(path: Path) -> List[QASample]:
    samples: List[QASample] = []
    for row in read_jsonl(path):
        samples.append(
            QASample(
                question=row["question"],
                expected_keywords=list(row.get("expected_keywords", [])),
                must_reference_pages=list(row.get("must_reference_pages", [])),
                expected_answer=row.get("expected_answer"),
            )
        )
    return samples


def load_autocomplete_dataset(path: Path) -> List[AutocompleteSample]:
    samples: List[AutocompleteSample] = []
    for row in read_jsonl(path):
        samples.append(
            AutocompleteSample(
                prefix=row["prefix"],
                expected_suggestions=list(row.get("expected_suggestions", [])),
            )
        )
    return samples


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


def extract_pages_from_context(context: Any) -> List[int]:
    pages: List[int] = []
    if not isinstance(context, list):
        return pages
    for chunk in context:
        if isinstance(chunk, dict) and "page" in chunk:
            try:
                pages.append(int(chunk["page"]))
            except Exception:
                continue
    return pages


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def keyword_hits(text: str, keywords: Iterable[str]) -> Tuple[int, int]:
    norm = normalize_text(text)
    kws = [k for k in keywords if str(k).strip()]
    if not kws:
        return 0, 0
    hits = 0
    for kw in kws:
        if normalize_text(str(kw)) in norm:
            hits += 1
    return hits, len(kws)


def page_hits(found_pages: Iterable[int], required_pages: Iterable[int]) -> Tuple[int, int]:
    found = set(int(p) for p in found_pages)
    req = [int(p) for p in required_pages if str(p).strip()]
    if not req:
        return 0, 0
    hits = sum(1 for p in req if p in found)
    return hits, len(req)


def groundedness_score(answer: str, expected_keywords: List[str], context_pages: List[int], required_pages: List[int]) -> float:
    kw_hit, kw_total = keyword_hits(answer, expected_keywords)
    pg_hit, pg_total = page_hits(context_pages, required_pages)

    # Weighted blend: answer content matters most, but page alignment matters too.
    kw_score = kw_hit / kw_total if kw_total else 0.0
    pg_score = pg_hit / pg_total if pg_total else 0.0

    if kw_total and pg_total:
        return 0.7 * kw_score + 0.3 * pg_score
    if kw_total:
        return kw_score
    if pg_total:
        return pg_score
    return 0.0


def run_qa_eval(backend: str, samples: List[QASample]) -> List[QAResult]:
    results: List[QAResult] = []

    for sample in samples:
        t0 = time.perf_counter()
        data = post_json(f"{backend}/chat", {"query": sample.question}, timeout=180)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        answer = str(data.get("answer", ""))
        context = data.get("context", [])
        context_pages = extract_pages_from_context(context)

        kw_hit, kw_total = keyword_hits(answer, sample.expected_keywords)
        pg_hit, pg_total = page_hits(context_pages, sample.must_reference_pages)
        gscore = groundedness_score(answer, sample.expected_keywords, context_pages, sample.must_reference_pages)

        results.append(
            QAResult(
                question=sample.question,
                answer=answer,
                context_pages=context_pages,
                latency_ms=latency_ms,
                keyword_hits=kw_hit,
                keyword_total=kw_total,
                page_hits=pg_hit,
                page_total=pg_total,
                groundedness_score=gscore,
            )
        )

    return results


def run_autocomplete_eval(backend: str, samples: List[AutocompleteSample]) -> List[AutocompleteResult]:
    results: List[AutocompleteResult] = []

    for sample in samples:
        t0 = time.perf_counter()
        data = post_json(f"{backend}/autocomplete", {"prefix": sample.prefix}, timeout=180)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        suggestions = [str(s) for s in data.get("suggestions", []) if str(s).strip()]
        expected = [normalize_text(s) for s in sample.expected_suggestions]

        # Any match = at least one suggestion has substantial overlap / exact match
        hit_count = 0
        for s in suggestions:
            ns = normalize_text(s)
            for exp in expected:
                if exp == ns or exp in ns or ns in exp:
                    hit_count += 1
                    break

        any_match = hit_count > 0
        top1_match = False
        if suggestions and expected:
            top1 = normalize_text(suggestions[0])
            top1_match = any(
                exp == top1 or exp in top1 or top1 in exp for exp in expected
            )

        results.append(
            AutocompleteResult(
                prefix=sample.prefix,
                suggestions=suggestions,
                latency_ms=latency_ms,
                any_match=any_match,
                top1_match=top1_match,
                hit_count=hit_count,
            )
        )

    return results


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    k = (len(values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def mean(values: List[float]) -> float:
    return float(stats.mean(values)) if values else 0.0


def summarize_qa(results: List[QAResult]) -> Dict[str, Any]:
    latencies = [r.latency_ms for r in results]
    gscores = [r.groundedness_score for r in results]

    keyword_rates = [
        (r.keyword_hits / r.keyword_total) if r.keyword_total else 0.0 for r in results
    ]
    page_rates = [
        (r.page_hits / r.page_total) if r.page_total else 0.0 for r in results
    ]

    return {
        "count": len(results),
        "avg_latency_ms": mean(latencies),
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "avg_keyword_recall": mean(keyword_rates),
        "avg_page_recall": mean(page_rates),
        "avg_groundedness": mean(gscores),
    }


def summarize_autocomplete(results: List[AutocompleteResult]) -> Dict[str, Any]:
    latencies = [r.latency_ms for r in results]
    any_match_rate = mean([1.0 if r.any_match else 0.0 for r in results])
    top1_match_rate = mean([1.0 if r.top1_match else 0.0 for r in results])
    hit_rate = mean([float(r.hit_count) for r in results])

    return {
        "count": len(results),
        "avg_latency_ms": mean(latencies),
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "any_match_rate": any_match_rate,
        "top1_match_rate": top1_match_rate,
        "avg_hit_count": hit_rate,
    }


def format_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def write_report(
    out_path: Path,
    pdf_path: Path,
    qa_path: Path,
    ac_path: Path,
    qa_summary: Dict[str, Any],
    ac_summary: Dict[str, Any],
    qa_results: List[QAResult],
    ac_results: List[AutocompleteResult],
) -> None:
    bad_qa = sorted(qa_results, key=lambda r: r.groundedness_score)[:5]
    bad_ac = sorted(ac_results, key=lambda r: (r.top1_match, r.any_match, -r.latency_ms))[:5]

    lines: List[str] = []
    lines.append("# PDF Chatbot Evaluation Report")
    lines.append("")
    lines.append(f"- PDF: `{pdf_path}`")
    lines.append(f"- QA dataset: `{qa_path}`")
    lines.append(f"- Autocomplete dataset: `{ac_path}`")
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
    lines.append(f"- Any-match rate: {format_pct(ac_summary['any_match_rate'])}")
    lines.append(f"- Top-1 match rate: {format_pct(ac_summary['top1_match_rate'])}")
    lines.append(f"- Avg hit count: {ac_summary['avg_hit_count']:.2f}")
    lines.append(f"- Median latency: {ac_summary['p50_latency_ms']:.1f} ms")
    lines.append(f"- P95 latency: {ac_summary['p95_latency_ms']:.1f} ms")
    lines.append("")

    lines.append("## Failure Analysis")
    lines.append("")
    lines.append("### Worst QA examples")
    for r in bad_qa:
        lines.append(f"- Q: {r.question}")
        lines.append(f"  - Groundedness: {r.groundedness_score:.3f}")
        lines.append(f"  - Keyword recall: {(r.keyword_hits / r.keyword_total) if r.keyword_total else 0.0:.3f}")
        lines.append(f"  - Page recall: {(r.page_hits / r.page_total) if r.page_total else 0.0:.3f}")
        lines.append(f"  - Pages seen: {sorted(set(r.context_pages))}")
        lines.append(f"  - Latency: {r.latency_ms:.1f} ms")
    lines.append("")

    lines.append("### Worst autocomplete examples")
    for r in bad_ac:
        lines.append(f"- Prefix: {r.prefix}")
        lines.append(f"  - Any-match: {r.any_match}")
        lines.append(f"  - Top-1 match: {r.top1_match}")
        lines.append(f"  - Hit count: {r.hit_count}")
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
    parser = argparse.ArgumentParser(description="Evaluate a local PDF chatbot backend.")
    parser.add_argument("--pdf", type=Path, default=Path("data/sample_policy.pdf"), help="Path to the PDF to upload.")
    parser.add_argument("--qa", type=Path, default=Path("data/qa_dataset.jsonl"), help="QA dataset JSONL.")
    parser.add_argument("--autocomplete", type=Path, default=Path("data/autocomplete_dataset.jsonl"), help="Autocomplete dataset JSONL.")
    parser.add_argument("--backend", type=str, default="http://127.0.0.1:8000", help="Backend base URL.")

    if not Path("reports").exists():
        Path("reports").mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime('%Y%m%d_%H%M%S')

    parser.add_argument("--out", type=Path, default=Path(f"reports/eval_report_{now}.md"), help="Output markdown report path.")
    parser.add_argument("--skip-upload", action="store_true", help="Skip PDF upload step.")
    args = parser.parse_args()

    if not args.qa.exists():
        print(f"QA dataset not found: {args.qa}", file=sys.stderr)
        return 2
    if not args.autocomplete.exists():
        print(f"Autocomplete dataset not found: {args.autocomplete}", file=sys.stderr)
        return 2
    if not args.skip_upload and not args.pdf.exists():
        print(f"PDF not found: {args.pdf}", file=sys.stderr)
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
        ac_path=args.autocomplete,
        qa_summary=qa_summary,
        ac_summary=ac_summary,
        qa_results=qa_results,
        ac_results=ac_results,
    )

    print(f"Wrote report to: {args.out}")
    print("QA summary:", qa_summary)
    print("Autocomplete summary:", ac_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
