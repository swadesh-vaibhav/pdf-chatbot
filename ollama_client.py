import time

import requests

import analytics
from config import OLLAMA_BASE
from logger import get_logger

log = get_logger("ollama_client")


def ollama_embed(texts):
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{OLLAMA_BASE}/embed",
            json={"model": "embeddinggemma", "input": texts},
            timeout=120,
        )
        resp.raise_for_status()
        embeddings = resp.json()["embeddings"]
        latency_ms = (time.perf_counter() - t0) * 1000
        analytics.record_embed_call(len(texts), latency_ms, success=True)
        log.debug(
            "ollama.embed",
            extra={"n_texts": len(texts), "latency_ms": round(latency_ms, 2)},
        )
        return embeddings
    except Exception:
        latency_ms = (time.perf_counter() - t0) * 1000
        analytics.record_embed_call(len(texts), latency_ms, success=False)
        log.exception(
            "ollama.embed_error",
            extra={"n_texts": len(texts), "latency_ms": round(latency_ms, 2)},
        )
        raise


def ollama_chat(messages):
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{OLLAMA_BASE}/chat",
            json={"model": "gemma3", "messages": messages, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        latency_ms = (time.perf_counter() - t0) * 1000
        analytics.record_llm_call(latency_ms, success=True)
        log.info(
            "ollama.chat",
            extra={
                "model": "gemma3",
                "n_messages": len(messages),
                "response_len": len(content),
                "latency_ms": round(latency_ms, 2),
            },
        )
        return content
    except Exception:
        latency_ms = (time.perf_counter() - t0) * 1000
        analytics.record_llm_call(latency_ms, success=False)
        log.exception(
            "ollama.chat_error",
            extra={"model": "gemma3", "latency_ms": round(latency_ms, 2)},
        )
        raise
