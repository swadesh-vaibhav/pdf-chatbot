from pathlib import Path

OLLAMA_BASE = "http://localhost:11434/api"
REDIS_URL = "redis://localhost:6379/0"
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Chunking and retrieval tuning knobs.
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
CHAT_TOP_K = 4
AUTOCOMPLETE_TOP_K = 3

INDEX_PATH = DATA_DIR / "faiss.index"
CHUNKS_PATH = DATA_DIR / "chunks.json"
