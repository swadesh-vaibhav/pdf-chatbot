#!/usr/bin/env python3
"""
sweep.py — Automated hyperparameter sweep for the PDF chatbot.

Evaluates every combination of chunk_size, chunk_overlap, chat_top_k, and
autocomplete_top_k entirely in-process (no live server required).

Strategy
--------
1. For each (chunk_size, chunk_overlap) pair:
   - Re-chunk the PDF and rebuild the FAISS index once.
2. For each (chat_top_k, ac_top_k) pair on that index:
   - Run in-process QA and autocomplete evaluation.
   - Score with the same metrics used by eval/run_eval.py.
3. Rank all combinations by a composite score and write Markdown + JSON reports.

Composite score
---------------
    composite = 0.6 * avg_groundedness
              + 0.2 * avg_keyword_recall
              + 0.2 * avg_ac_best_similarity

Usage
-----
    python sweep.py \\
        --pdf ../eval/data/sample_policy.pdf \\
        --qa ../eval/data/qa_dataset.jsonl \\
        --autocomplete ../eval/data/autocomplete_dataset.jsonl

    # Custom search space
    python sweep.py \\
        --pdf ../eval/data/sample_policy.pdf \\
        --qa ../eval/data/qa_dataset.jsonl \\
        --autocomplete ../eval/data/autocomplete_dataset.jsonl \\
        --chunk-sizes 512 768 900 1200 \\
        --chunk-overlaps 100 150 200 \\
        --chat-top-ks 3 4 6 \\
        --ac-top-ks 2 3 4
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import faiss
import fitz  # pymupdf
import numpy as np

# ---------------------------------------------------------------------------
# Bootstrap: add the project root and eval folder to the import path so we
# can import backend modules and metric helpers without installing them.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval"
for _p in (str(ROOT), str(EVAL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from chunking import chunk_text  # noqa: E402
from ollama_client import ollama_chat, ollama_embed  # noqa: E402
from metrics import (  # noqa: E402
    autocomplete_match,
    extract_pages_from_text,
    groundedness_score,
    keyword_recall,
    page_recall,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class QASample:
    question: str
    expected_keywords: List[str]
    must_reference_pages: List[int]


@dataclass
class AutocompleteSample:
    prefix: str
    expected_suggestions: List[str]


@dataclass
class HyperParams:
    chunk_size: int
    chunk_overlap: int
    chat_top_k: int
    ac_top_k: int


@dataclass
class SweepResult:
    params: HyperParams
    avg_groundedness: float
    avg_keyword_recall: float
    avg_page_recall: float
    avg_ac_best_similarity: float
    ac_any_match_rate: float
    ac_top1_match_rate: float
    avg_qa_latency_ms: float
    avg_ac_latency_ms: float
    composite_score: float
    num_chunks: int


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {lineno} of {path}: {exc}") from exc
    return rows


def load_qa(path: Path) -> List[QASample]:
    return [
        QASample(
            question=row["question"],
            expected_keywords=list(row.get("expected_keywords", [])),
            must_reference_pages=list(row.get("must_reference_pages", [])),
        )
        for row in read_jsonl(path)
    ]


def load_ac(path: Path) -> List[AutocompleteSample]:
    return [
        AutocompleteSample(
            prefix=row["prefix"],
            expected_suggestions=list(row.get("expected_suggestions", [])),
        )
        for row in read_jsonl(path)
    ]


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def extract_pdf_pages(pdf_path: Path) -> List[Tuple[int, str]]:
    """Return (1-based page number, text) for every non-empty page."""
    doc = fitz.open(str(pdf_path))
    pages = []
    for i in range(len(doc)):
        text = doc[i].get_text("text")
        if text.strip():
            pages.append((i + 1, text))
    return pages


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def build_index(
    pages: List[Tuple[int, str]],
    chunk_size: int,
    chunk_overlap: int,
) -> Tuple[List[Dict[str, Any]], faiss.Index]:
    """Chunk the pages, embed, normalise, and return (chunks, faiss_index)."""
    chunks: List[Dict[str, Any]] = []
    for page_num, text in pages:
        for c in chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap):
            chunks.append({"page": page_num, "text": c})

    embeddings = ollama_embed([c["text"] for c in chunks])
    vectors = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return chunks, index


# ---------------------------------------------------------------------------
# In-process retrieval (no global state, no Redis)
# ---------------------------------------------------------------------------

def retrieve_local(
    query: str,
    chunks: List[Dict[str, Any]],
    index: faiss.Index,
    top_k: int,
) -> List[Dict[str, Any]]:
    q_vec = np.array(ollama_embed([query]), dtype="float32")
    faiss.normalize_L2(q_vec)
    _, ids = index.search(q_vec, top_k)
    return [chunks[i] for i in ids[0] if i != -1]


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------

_QA_SYSTEM = (
    "Answer only from the provided document context. "
    "If the context is insufficient, say you do not know. "
    "Cite page numbers in the answer."
)

_AC_SYSTEM = (
    "Return exactly 3 autocomplete suggestions. "
    "Each suggestion must strictly be grounded in the document context. "
    "Each suggestion must strictly include the prefix. "
    "Each suggestion must strictly be a single sentence. "
    "Return only the suggestions, without any explanation or formatting, in separate lines."
)


def eval_qa_sample(
    sample: QASample,
    chunks: List[Dict[str, Any]],
    index: faiss.Index,
    top_k: int,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    ctx = retrieve_local(sample.question, chunks, index, top_k)
    context = "\n\n".join(f"[page {c['page']}] {c['text']}" for c in ctx)
    messages = [
        {"role": "system", "content": _QA_SYSTEM},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{sample.question}"},
    ]
    answer = ollama_chat(messages)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    found_pages = extract_pages_from_text(answer)
    kw = keyword_recall(answer, sample.expected_keywords)
    pg = page_recall(found_pages, sample.must_reference_pages)
    g = groundedness_score(answer, sample.expected_keywords, found_pages, sample.must_reference_pages)
    return {"latency_ms": latency_ms, "keyword_recall": kw, "page_recall": pg, "groundedness": g}


def eval_ac_sample(
    sample: AutocompleteSample,
    chunks: List[Dict[str, Any]],
    index: faiss.Index,
    top_k: int,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    ctx = retrieve_local(sample.prefix, chunks, index, top_k)
    context = "\n\n".join(f"[page {c['page']}] {c['text']}" for c in ctx)
    messages = [
        {"role": "system", "content": _AC_SYSTEM},
        {"role": "user", "content": f"Prefix: {sample.prefix}\n\nContext:\n{context}"},
    ]
    raw = ollama_chat(messages)
    suggestions = [line.strip("-• \t") for line in raw.splitlines() if line.strip()][:3]
    latency_ms = (time.perf_counter() - t0) * 1000.0

    score = autocomplete_match(suggestions, sample.expected_suggestions)
    return {
        "latency_ms": latency_ms,
        "any_match": score.any_match,
        "top1_match": score.top1_match,
        "best_similarity": score.best_similarity,
    }


def _mean(vals: Sequence[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------

def write_reports(results: List[SweepResult], out_dir: Path) -> Tuple[Path, Path]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = out_dir / f"sweep_{ts}.md"
    json_path = out_dir / f"sweep_{ts}.json"

    ranked = sorted(results, key=lambda r: r.composite_score, reverse=True)
    best = ranked[0]

    # ---- Markdown ----
    lines: List[str] = [
        "# Hyperparameter Sweep Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Combinations evaluated: {len(results)}",
        "",
        "## Composite score formula",
        "",
        "```",
        "composite = 0.6 * avg_groundedness",
        "          + 0.2 * avg_keyword_recall",
        "          + 0.2 * avg_ac_best_similarity",
        "```",
        "",
        "## Ranked results",
        "",
        "| Rank | chunk_size | chunk_overlap | chat_top_k | ac_top_k "
        "| composite | groundedness | kw_recall | ac_similarity | num_chunks |",
        "|------|-----------|---------------|------------|----------"
        "|-----------|--------------|-----------|---------------|------------|",
    ]
    for i, r in enumerate(ranked, 1):
        p = r.params
        lines.append(
            f"| {i} | {p.chunk_size} | {p.chunk_overlap} | {p.chat_top_k} | {p.ac_top_k} "
            f"| {r.composite_score:.4f} | {r.avg_groundedness:.4f} "
            f"| {r.avg_keyword_recall:.4f} | {r.avg_ac_best_similarity:.4f} | {r.num_chunks} |"
        )

    lines += [
        "",
        "## Best configuration",
        "",
        "```python",
        f"CHUNK_SIZE        = {best.params.chunk_size}",
        f"CHUNK_OVERLAP     = {best.params.chunk_overlap}",
        f"CHAT_TOP_K        = {best.params.chat_top_k}",
        f"AUTOCOMPLETE_TOP_K = {best.params.ac_top_k}",
        "```",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Composite score    | {best.composite_score:.4f} |",
        f"| Avg groundedness   | {best.avg_groundedness:.4f} |",
        f"| Avg keyword recall | {best.avg_keyword_recall:.4f} |",
        f"| Avg page recall    | {best.avg_page_recall:.4f} |",
        f"| Avg AC similarity  | {best.avg_ac_best_similarity:.4f} |",
        f"| AC any-match rate  | {best.ac_any_match_rate * 100:.1f}% |",
        f"| AC top-1 match rate| {best.ac_top1_match_rate * 100:.1f}% |",
        f"| Avg QA latency     | {best.avg_qa_latency_ms:.1f} ms |",
        f"| Avg AC latency     | {best.avg_ac_latency_ms:.1f} ms |",
        f"| Num chunks         | {best.num_chunks} |",
        "",
    ]

    md_path.write_text("\n".join(lines), encoding="utf-8")

    # ---- JSON ----
    payload = [
        {
            "rank": i,
            "params": asdict(r.params),
            "metrics": {
                "composite_score": r.composite_score,
                "avg_groundedness": r.avg_groundedness,
                "avg_keyword_recall": r.avg_keyword_recall,
                "avg_page_recall": r.avg_page_recall,
                "avg_ac_best_similarity": r.avg_ac_best_similarity,
                "ac_any_match_rate": r.ac_any_match_rate,
                "ac_top1_match_rate": r.ac_top1_match_rate,
                "avg_qa_latency_ms": r.avg_qa_latency_ms,
                "avg_ac_latency_ms": r.avg_ac_latency_ms,
                "num_chunks": r.num_chunks,
            },
        }
        for i, r in enumerate(ranked, 1)
    ]
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return md_path, json_path


# ---------------------------------------------------------------------------
# Default search space
# ---------------------------------------------------------------------------

_DEFAULT_CHUNK_SIZES = [512, 768, 900, 1200]
_DEFAULT_CHUNK_OVERLAPS = [100, 150]
_DEFAULT_CHAT_TOP_KS = [3, 4, 6]
_DEFAULT_AC_TOP_KS = [2, 3]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hyperparameter sweep for chunk_size, chunk_overlap, and top_k.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pdf", required=True, type=Path, help="Reference PDF for evaluation.")
    parser.add_argument("--qa", required=True, type=Path, help="QA dataset JSONL file.")
    parser.add_argument("--autocomplete", required=True, type=Path, help="Autocomplete dataset JSONL file.")
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).parent / "reports",
        help="Directory to write sweep reports into.",
    )
    parser.add_argument("--chunk-sizes", nargs="+", type=int, default=_DEFAULT_CHUNK_SIZES, metavar="N")
    parser.add_argument("--chunk-overlaps", nargs="+", type=int, default=_DEFAULT_CHUNK_OVERLAPS, metavar="N")
    parser.add_argument("--chat-top-ks", nargs="+", type=int, default=_DEFAULT_CHAT_TOP_KS, metavar="N")
    parser.add_argument("--ac-top-ks", nargs="+", type=int, default=_DEFAULT_AC_TOP_KS, metavar="N")
    args = parser.parse_args()

    for attr, label in (("pdf", "PDF"), ("qa", "QA dataset"), ("autocomplete", "Autocomplete dataset")):
        path: Path = getattr(args, attr)
        if not path.exists():
            print(f"Error: {label} not found: {path}", file=sys.stderr)
            sys.exit(2)

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading datasets ...")
    qa_samples = load_qa(args.qa)
    ac_samples = load_ac(args.autocomplete)
    print(f"  QA samples: {len(qa_samples)},  Autocomplete samples: {len(ac_samples)}")

    print(f"Extracting text from {args.pdf} ...")
    pages = extract_pdf_pages(args.pdf)
    print(f"  Pages with text: {len(pages)}")

    chunk_combos = [
        (cs, co)
        for cs, co in itertools.product(args.chunk_sizes, args.chunk_overlaps)
        if co < cs  # overlap must be strictly less than chunk size
    ]
    top_k_combos = list(itertools.product(args.chat_top_ks, args.ac_top_ks))
    total = len(chunk_combos) * len(top_k_combos)

    print(
        f"\nSweep: {len(chunk_combos)} chunk configs × {len(top_k_combos)} top_k configs "
        f"= {total} combinations\n"
    )

    all_results: List[SweepResult] = []
    run_num = 0

    for chunk_size, chunk_overlap in chunk_combos:
        print(f"Building index: chunk_size={chunk_size}, overlap={chunk_overlap} ...", flush=True)
        chunks, index = build_index(pages, chunk_size, chunk_overlap)
        print(f"  {len(chunks)} chunks", flush=True)

        for chat_top_k, ac_top_k in top_k_combos:
            run_num += 1
            params = HyperParams(chunk_size, chunk_overlap, chat_top_k, ac_top_k)
            print(
                f"  [{run_num}/{total}] chat_top_k={chat_top_k}, ac_top_k={ac_top_k} ",
                end="", flush=True,
            )

            qa_latencies: List[float] = []
            kw_recalls: List[float] = []
            pg_recalls: List[float] = []
            groundednesses: List[float] = []
            for sample in qa_samples:
                r = eval_qa_sample(sample, chunks, index, chat_top_k)
                qa_latencies.append(r["latency_ms"])
                kw_recalls.append(r["keyword_recall"])
                pg_recalls.append(r["page_recall"])
                groundednesses.append(r["groundedness"])
                print(".", end="", flush=True)

            ac_latencies: List[float] = []
            any_matches: List[float] = []
            top1_matches: List[float] = []
            similarities: List[float] = []
            for sample in ac_samples:
                r = eval_ac_sample(sample, chunks, index, ac_top_k)
                ac_latencies.append(r["latency_ms"])
                any_matches.append(1.0 if r["any_match"] else 0.0)
                top1_matches.append(1.0 if r["top1_match"] else 0.0)
                similarities.append(r["best_similarity"])
                print(".", end="", flush=True)

            avg_g = _mean(groundednesses)
            avg_kw = _mean(kw_recalls)
            avg_sim = _mean(similarities)
            composite = 0.6 * avg_g + 0.2 * avg_kw + 0.2 * avg_sim

            all_results.append(
                SweepResult(
                    params=params,
                    avg_groundedness=avg_g,
                    avg_keyword_recall=avg_kw,
                    avg_page_recall=_mean(pg_recalls),
                    avg_ac_best_similarity=avg_sim,
                    ac_any_match_rate=_mean(any_matches),
                    ac_top1_match_rate=_mean(top1_matches),
                    avg_qa_latency_ms=_mean(qa_latencies),
                    avg_ac_latency_ms=_mean(ac_latencies),
                    composite_score=composite,
                    num_chunks=len(chunks),
                )
            )
            print(f" composite={composite:.4f}", flush=True)

    if not all_results:
        print("No valid combinations to evaluate. Check your parameter ranges.", file=sys.stderr)
        sys.exit(1)

    md_path, json_path = write_reports(all_results, args.out)
    best = max(all_results, key=lambda r: r.composite_score)

    print(f"\nSweep complete — {len(all_results)} combinations evaluated.")
    print(
        f"Best: chunk_size={best.params.chunk_size}, overlap={best.params.chunk_overlap}, "
        f"chat_top_k={best.params.chat_top_k}, ac_top_k={best.params.ac_top_k} "
        f"(composite={best.composite_score:.4f})"
    )
    print(f"\nReports written to:\n  {md_path}\n  {json_path}")


if __name__ == "__main__":
    main()
