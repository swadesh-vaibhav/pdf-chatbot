import json

import faiss
import numpy as np

from chunking import stable_key
from clients import redis_client
from config import CHUNKS_PATH, INDEX_PATH
import state
from ollama_client import ollama_embed


def build_index():
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        state.faiss_index = None
        state.chunks = []
        state.embedding_dim = None
        return

    state.faiss_index = faiss.read_index(str(INDEX_PATH))
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        state.chunks = json.load(f)
    state.embedding_dim = state.faiss_index.d


def save_index():
    if state.faiss_index is not None:
        faiss.write_index(state.faiss_index, str(INDEX_PATH))
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(state.chunks, f, ensure_ascii=False)


def cosine_faiss_index(vectors: np.ndarray):
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def retrieve(query: str, top_k: int = 4):
    if state.faiss_index is None or not state.chunks:
        return []

    q_key = f"retrieval:{stable_key(query)}"
    cached = redis_client.get(q_key)
    if cached:
        return json.loads(cached)

    q_vec = np.array(ollama_embed([query]), dtype="float32")
    faiss.normalize_L2(q_vec)

    scores, ids = state.faiss_index.search(q_vec, top_k)
    results = []
    for idx in ids[0]:
        if idx == -1:
            continue
        results.append(state.chunks[idx])

    redis_client.setex(q_key, 300, json.dumps(results))
    return results
