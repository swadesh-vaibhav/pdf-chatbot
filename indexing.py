"""
indexing.py — FAISS vector index management and semantic retrieval.

Responsibilities:
  - Load/save the FAISS index and chunk metadata to/from disk (build_index, save_index).
  - Build cosine-similarity FAISS indexes from raw embedding vectors (cosine_faiss_index).
  - Retrieve the most relevant text chunks for a query using vector search,
    with Redis-backed caching to avoid redundant embedding calls (retrieve).
"""

import json

import faiss
import numpy as np

from chunking import stable_key
from clients import redis_client
from config import CHUNKS_PATH, INDEX_PATH
import state
from ollama_client import ollama_embed


def build_index():
    """Load the persisted FAISS index and chunk list into global state.

    Reads INDEX_PATH (FAISS binary) and CHUNKS_PATH (JSON array of chunk dicts).
    If either file is missing (e.g. first run before any PDF is uploaded),
    global state is reset to empty so the rest of the app handles the
    "no index yet" case gracefully.
    """
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        # No persisted data yet — start with an empty index.
        state.faiss_index = None
        state.chunks = []
        state.embedding_dim = None
        return

    state.faiss_index = faiss.read_index(str(INDEX_PATH))
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        state.chunks = json.load(f)
    # Cache the embedding dimension so callers don't need to inspect the index.
    state.embedding_dim = state.faiss_index.d


def save_index():
    """Persist the current in-memory FAISS index and chunk list to disk.

    Called after ingesting a new PDF so that the index survives restarts.
    The FAISS index is only written when it exists; the chunks file is always
    written (to an empty list if needed) to keep the two files in sync.
    """
    if state.faiss_index is not None:
        faiss.write_index(state.faiss_index, str(INDEX_PATH))
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(state.chunks, f, ensure_ascii=False)


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
    # Normalise in-place so inner product equals cosine similarity.
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def retrieve(query: str, top_k: int = 4) -> list[dict]:
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

    # Redis cache key is deterministic for identical queries.
    q_key = f"retrieval:{stable_key(query)}"
    cached = redis_client.get(q_key)
    if cached:
        return json.loads(cached)

    # Embed and normalise the query vector to match the index's cosine metric.
    q_vec = np.array(ollama_embed([query]), dtype="float32")

    # Normalise in-place so inner product equals cosine similarity.
    faiss.normalize_L2(q_vec)

    # FAISS returns parallel arrays of scores and indexes; -1 means "no result".
    scores, ids = state.faiss_index.search(q_vec, top_k)
    results = []
    for idx in ids[0]:
        if idx == -1:
            continue
        results.append(state.chunks[idx])

    # Cache for 5 minutes to reduce Ollama embedding latency on repeated queries.
    redis_client.setex(q_key, 300, json.dumps(results))
    return results
