from pathlib import Path

OLLAMA_BASE = "http://localhost:11434/api"
REDIS_URL = "redis://localhost:6379/0"
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

INDEX_PATH = DATA_DIR / "faiss.index"
CHUNKS_PATH = DATA_DIR / "chunks.json"
