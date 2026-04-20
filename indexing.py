"""
indexing.py — FAISS vector index management and semantic retrieval.

Responsibilities:
  - Load/save the FAISS index and chunk metadata to/from disk (build_index, save_index).
  - Build cosine-similarity FAISS indexes from raw embedding vectors (cosine_faiss_index).
  - Retrieve the most relevant text chunks for a query using vector search,
    with Redis-backed caching to avoid redundant embedding calls (retrieve).
"""

import json
import time

import faiss
import numpy as np

import analytics
from chunking import stable_key
from clients import redis_client
from config import CHUNKS_PATH, INDEX_PATH
from logger import get_logger
import state
from ollama_client import ollama_embed

log = get_logger("indexing")


def build_index():
    """Load the persisted FAISS index and chunk list into global state.

    Reads INDEX_PATH (FAISS binary) and CHUNKS_PATH (JSON array of chunk dicts).
    If either file is missing (e.g. first run before any PDF is uploaded),
    global state is reset to empty so the rest of the app handles the
    "no index yet" case gracefully.
    """
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        state.faiss_index = None
        state.chunks = []
        state.embedding_dim = None
        log.info("index.build_skip", extra={"reason": "no persisted index found"})
        return

    state.faiss_index = faiss.read_index(str(INDEX_PATH))
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        state.chunks = json.load(f)
    state.embedding_dim = state.faiss_index.d
    log.info(
        "index.loaded",
        extra={"n_chunks": len(state.chunks), "embedding_dim": state.embedding_dim},
    )


def save_index():
    """Persist the current in-memory FAISS index and chunk list to disk."""
    if state.faiss_index is not None:
        faiss.write_index(state.faiss_index, str(INDEX_PATH))
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(state.chunks, f, ensure_ascii=False)
    log.debug("index.saved", extra={"n_chunks": len(state.chunks)})


def cosine_faiss_index(vectors: np.ndarray) -> faiss.Index:
    """Build a FAISS flat index that computes cosine similarity.

    FAISS's IndexFlatIP performs inner-product (dot-product) search.
    Normalising the vectors to unit length first converts dot-product into
    cosine similarity, so the highest-scoring results are the most similar.

    Args:
        vectors: 2-D float32 array of shape (n_vectors, embedding_dim).
                 Modified in-place by L2 normalisation.

    Returns:
        A populated faiss.IndexFlatIP ready for search.
    """
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def retrieve(query: str, top_k: int = 4) -> list:
    """Find the top-k most relevant chunks for a natural-language query.

    Workflow:
      1. Return early if the index is empty.
      2. Check Redis for a cached result (keyed on a stable hash of the query).
      3. Embed the query with Ollama, normalise, and run FAISS nearest-neighbour search.
      4. Cache the result in Redis for 5 minutes (300 s) to avoid re-embedding.

    Args:
        query:  The user's question or search string.
        top_k:  Maximum number of chunks to return (default 4).

    Returns:
        A list of chunk dicts (up to top_k), each as stored in state.chunks.
        Returns an empty list when the index has not been built yet.
    """
    if state.faiss_index is None or not state.chunks:
        return []

    q_key = f"retrieval:{stable_key(query)}"
    cached = redis_client.get(q_key)
    if cached:
        analytics.record_cache_hit("retrieval")
        log.debug("retrieval.cache_hit", extra={"query_len": len(query), "top_k": top_k})
        return json.loads(cached)

    analytics.record_cache_miss("retrieval")

    q_vec = np.array(ollama_embed([query]), dtype="float32")
    faiss.normalize_L2(q_vec)

    t0 = time.perf_counter()
    _, ids = state.faiss_index.search(q_vec, top_k)
    faiss_ms = (time.perf_counter() - t0) * 1000
    analytics.record_faiss_search(faiss_ms)

    results = []
    for idx in ids[0]:
        if idx == -1:
            continue
        results.append(state.chunks[idx])

    redis_client.setex(q_key, 300, json.dumps(results))
    log.debug(
        "retrieval.complete",
        extra={
            "query_len": len(query),
            "top_k": top_k,
            "n_results": len(results),
            "faiss_latency_ms": round(faiss_ms, 2),
        },
    )
    return results
