import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class _Counters:
    requests_total: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    requests_errors: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    requests_in_flight: int = 0

    cache_hits: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    cache_misses: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    llm_calls: int = 0
    llm_latency_ms_total: float = 0.0
    llm_errors: int = 0

    embed_calls: int = 0
    embed_latency_ms_total: float = 0.0
    embed_texts_total: int = 0
    embed_errors: int = 0

    faiss_searches: int = 0
    faiss_latency_ms_total: float = 0.0

    pdfs_uploaded: int = 0
    chunks_indexed: int = 0

    request_latency_ms_total: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    request_count: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


_c = _Counters()
_lock = threading.Lock()


def record_request_start(endpoint: str) -> None:
    with _lock:
        _c.requests_total[endpoint] += 1
        _c.requests_in_flight += 1


def record_request_end(endpoint: str, status_code: int, duration_ms: float) -> None:
    with _lock:
        _c.requests_in_flight = max(0, _c.requests_in_flight - 1)
        _c.request_latency_ms_total[endpoint] += duration_ms
        _c.request_count[endpoint] += 1
        if status_code >= 400:
            _c.requests_errors[endpoint] += 1


def record_cache_hit(cache_type: str) -> None:
    with _lock:
        _c.cache_hits[cache_type] += 1


def record_cache_miss(cache_type: str) -> None:
    with _lock:
        _c.cache_misses[cache_type] += 1


def record_llm_call(latency_ms: float, success: bool = True) -> None:
    with _lock:
        _c.llm_calls += 1
        _c.llm_latency_ms_total += latency_ms
        if not success:
            _c.llm_errors += 1


def record_embed_call(n_texts: int, latency_ms: float, success: bool = True) -> None:
    with _lock:
        _c.embed_calls += 1
        _c.embed_texts_total += n_texts
        _c.embed_latency_ms_total += latency_ms
        if not success:
            _c.embed_errors += 1


def record_faiss_search(latency_ms: float) -> None:
    with _lock:
        _c.faiss_searches += 1
        _c.faiss_latency_ms_total += latency_ms


def record_pdf_upload(n_chunks: int) -> None:
    with _lock:
        _c.pdfs_uploaded += 1
        _c.chunks_indexed += n_chunks


def snapshot() -> dict:
    with _lock:
        endpoint_stats = {}
        all_endpoints = set(list(_c.requests_total.keys()) + list(_c.request_count.keys()))
        for ep in all_endpoints:
            count = _c.request_count[ep]
            endpoint_stats[ep] = {
                "requests_total": _c.requests_total[ep],
                "errors": _c.requests_errors[ep],
                "avg_latency_ms": round(_c.request_latency_ms_total[ep] / count, 2) if count else 0.0,
            }

        cache_types = set(list(_c.cache_hits.keys()) + list(_c.cache_misses.keys()))

        def _hit_rate(k: str) -> float:
            total = _c.cache_hits[k] + _c.cache_misses[k]
            return round(_c.cache_hits[k] / total, 4) if total else 0.0

        return {
            "requests": {
                "in_flight": _c.requests_in_flight,
                "by_endpoint": endpoint_stats,
            },
            "cache": {
                "hits": dict(_c.cache_hits),
                "misses": dict(_c.cache_misses),
                "hit_rates": {k: _hit_rate(k) for k in cache_types},
            },
            "llm": {
                "calls_total": _c.llm_calls,
                "errors": _c.llm_errors,
                "avg_latency_ms": round(_c.llm_latency_ms_total / _c.llm_calls, 2) if _c.llm_calls else 0.0,
            },
            "embeddings": {
                "calls_total": _c.embed_calls,
                "texts_embedded_total": _c.embed_texts_total,
                "errors": _c.embed_errors,
                "avg_latency_ms": round(_c.embed_latency_ms_total / _c.embed_calls, 2) if _c.embed_calls else 0.0,
            },
            "faiss": {
                "searches_total": _c.faiss_searches,
                "avg_latency_ms": round(_c.faiss_latency_ms_total / _c.faiss_searches, 2) if _c.faiss_searches else 0.0,
            },
            "ingestion": {
                "pdfs_uploaded": _c.pdfs_uploaded,
                "chunks_indexed": _c.chunks_indexed,
            },
        }
