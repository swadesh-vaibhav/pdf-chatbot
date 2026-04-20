import os
from pathlib import Path

# ── Model / provider ─────────────────────────────────────────────────────────
# These are the only two values to change when experimenting with models.
# Changing CHAT_MODEL  → Redis answer/autocomplete cache is flushed.
# Changing EMBED_MODEL → Redis cache + vector index are both wiped (embeddings
#                        are model-specific and can't be mixed).
CHAT_PROVIDER  = os.getenv("CHAT_PROVIDER",  "ollama")
CHAT_MODEL     = os.getenv("CHAT_MODEL",     "gemma3")
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "ollama")
EMBED_MODEL    = os.getenv("EMBED_MODEL",    "embeddinggemma")
VECTOR_DB      = os.getenv("VECTOR_DB",      "faiss")

# ── Infrastructure ────────────────────────────────────────────────────────────
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434/api")
REDIS_URL   = os.getenv("REDIS_URL",   "redis://localhost:6379/0")

# ── Chunking & retrieval ──────────────────────────────────────────────────────
CHUNK_SIZE         = int(os.getenv("CHUNK_SIZE",         "900"))
CHUNK_OVERLAP      = int(os.getenv("CHUNK_OVERLAP",      "150"))
CHAT_TOP_K         = int(os.getenv("CHAT_TOP_K",         "4"))
AUTOCOMPLETE_TOP_K = int(os.getenv("AUTOCOMPLETE_TOP_K", "3"))

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR         = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

INDEX_PATH       = DATA_DIR / "faiss.index"
CHUNKS_PATH      = DATA_DIR / "chunks.json"
MODEL_STATE_PATH = DATA_DIR / "model_state.json"
