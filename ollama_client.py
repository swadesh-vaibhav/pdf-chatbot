import requests

from config import OLLAMA_BASE


def ollama_embed(texts):
    """
    Uses Ollama's /api/embed endpoint.
    Docs: https://docs.ollama.com/api/embed
    """
    resp = requests.post(
        f"{OLLAMA_BASE}/embed",
        json={"model": "embeddinggemma", "input": texts},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]


def ollama_chat(messages):
    """
    Uses Ollama's /api/chat endpoint.
    """
    resp = requests.post(
        f"{OLLAMA_BASE}/chat",
        json={
            "model": "gemma3",
            "messages": messages,
            "stream": False,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]
