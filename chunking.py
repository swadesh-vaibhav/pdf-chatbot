import hashlib


def stable_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150):
    text = " ".join(text.split())
    out = []
    i = 0
    while i < len(text):
        out.append(text[i : i + chunk_size])
        i += max(1, chunk_size - overlap)
    return [c for c in out if c.strip()]
