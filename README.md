# PDF Chatbot with Context-Aware Autocomplete

A **retrieval-augmented generation (RAG) system** that enables users to:
- Ask questions over PDF documents
- Receive grounded answers with citations
- Get context-aware autocomplete suggestions

---

## Features

### Document Q&A (RAG)
- Semantic search over PDF content using embeddings
- Context-aware answer generation
- Page-level citation extraction

### Context-Aware Autocomplete
- Suggests query completions grounded in document content
- Improves query formulation and retrieval quality
- Low-latency suggestions using caching

### Streaming Responses
- Real-time token streaming for low-latency user feedback
- Server-Sent Events (SSE) with context metadata
- Graceful handling of partial UTF-8 chunks

### Production-Oriented Design
- FastAPI backend with structured JSON logging and rotating log files
- Redis caching layer with per-endpoint cache hit/miss tracking
- Vector search (FAISS)
- Live metrics endpoint
- Modular evaluation framework

---

## System Design

User → Frontend → Backend → Redis → Vector DB → LLM

---

## Project Structure

```
.
├── app.py                  # FastAPI entry point, request logging middleware
├── routes.py               # API endpoints
├── config.py               # All config via env vars (see Configuration)
├── model_guard.py          # Detects model changes at startup, flushes stale cache/index
├── ollama_client.py        # Embed + chat calls, timed and logged
├── indexing.py             # FAISS index build/load/retrieve
├── chunking.py             # Text chunking utilities
├── clients.py              # Redis client
├── state.py                # Process-local index state
├── schemas.py              # Pydantic request schemas
├── logger.py               # JSON formatter, request ID propagation, rotating file handler
├── analytics.py            # In-memory counters (requests, cache, LLM, FAISS latency)
├── data/                   # Persisted FAISS index, chunks, model state (git-ignored)
├── logs/                   # Rotating log files (git-ignored)
├── eval/
│   ├── metrics.py
│   ├── run_eval.py
│   └── data/
│       ├── qa_dataset.jsonl
│       ├── autocomplete_dataset.jsonl
│       └── sample_policy.pdf
└── frontend/               # Next.js frontend
```

---

## Setup

### Backend
```
pip install -r requirements.txt
uvicorn app:app --reload
```

### Frontend
```
npm install
npm run dev
```

### Redis
```
docker run -d -p 6379:6379 redis
```

### LLM (local)
```
ollama run gemma3
```

---

## Configuration

All settings are read from environment variables. The defaults match the local development stack.

| Variable            | Default            | Description                              |
|---------------------|--------------------|------------------------------------------|
| `CHAT_PROVIDER`     | `ollama`           | Provider for chat completions            |
| `CHAT_MODEL`        | `gemma3`           | Model used for chat and autocomplete     |
| `EMBED_PROVIDER`    | `ollama`           | Provider for embeddings                  |
| `EMBED_MODEL`       | `embeddinggemma`   | Model used to embed chunks and queries   |
| `VECTOR_DB`         | `faiss`            | Vector store backend                     |
| `OLLAMA_BASE`       | `http://localhost:11434/api` | Ollama API base URL           |
| `REDIS_URL`         | `redis://localhost:6379/0`  | Redis connection URL           |
| `CHUNK_SIZE`        | `900`              | Characters per chunk                     |
| `CHUNK_OVERLAP`     | `150`              | Overlap between chunks                   |
| `CHAT_TOP_K`        | `4`                | Chunks retrieved for chat                |
| `AUTOCOMPLETE_TOP_K`| `3`                | Chunks retrieved for autocomplete        |
| `LOG_LEVEL`         | `INFO`             | `DEBUG` adds retrieval-level detail      |

### Changing models

`CHAT_MODEL` and `EMBED_MODEL` are the only knobs needed when experimenting with different models. On startup, the backend compares the current values against the last-seen values stored in `data/model_state.json` and takes corrective action automatically:

| What changed   | Action                                                  |
|----------------|---------------------------------------------------------|
| `CHAT_MODEL`   | Flush answer + autocomplete Redis cache                 |
| `EMBED_MODEL`  | Flush all Redis cache + delete vector index (embeddings are model-specific) |

After clearing stale data the new model state is persisted, so subsequent restarts with the same config are no-ops.

---

## Observability

### Logs

Structured JSON logs are written to both stdout and `logs/app.log`. The file rotates at 10 MB with 5 backups kept.

Every log line includes a `request_id` (UUID, also returned as `X-Request-ID` response header) for end-to-end tracing. Set `LOG_LEVEL=DEBUG` to see retrieval and embedding detail.

### Metrics

`GET /metrics` returns a live snapshot of in-memory counters:

```json
{
  "requests": { "in_flight": 1, "by_endpoint": { "/chat": { "requests_total": 42, "errors": 0, "avg_latency_ms": 1240 } } },
  "cache":    { "hits": { "chat": 30 }, "misses": { "chat": 12 }, "hit_rates": { "chat": 0.714 } },
  "llm":      { "calls_total": 12, "errors": 0, "avg_latency_ms": 3200 },
  "embeddings": { "calls_total": 8, "texts_embedded_total": 312, "avg_latency_ms": 420 },
  "faiss":    { "searches_total": 12, "avg_latency_ms": 1.4 },
  "ingestion": { "pdfs_uploaded": 1, "chunks_indexed": 312 }
}
```

Counters reset on restart. For persistent metrics, push the snapshot to a time-series store.

---

## Hyperparameter Tuning

`tuning/sweep.py` runs an offline grid search over chunking and retrieval parameters. It operates entirely in-process — no running server or Redis needed.

**Search space** (defaults):

| Parameter      | Default values   |
|----------------|------------------|
| `chunk_size`   | 512, 768, 900, 1200 |
| `chunk_overlap`| 100, 150         |
| `chat_top_k`   | 3, 4, 6          |
| `ac_top_k`     | 2, 3             |

Each valid combination (overlap must be < chunk size) rebuilds the FAISS index once, then runs the full QA and autocomplete eval suite. Combinations are ranked by a composite score:

```
composite = 0.6 × avg_groundedness
          + 0.2 × avg_keyword_recall
          + 0.2 × avg_ac_best_similarity
```

**Usage:**
```
python tuning/sweep.py \
    --pdf eval/data/sample_policy.pdf \
    --qa eval/data/qa_dataset.jsonl \
    --autocomplete eval/data/autocomplete_dataset.jsonl
```

Custom search space:
```
python tuning/sweep.py \
    --pdf eval/data/sample_policy.pdf \
    --qa eval/data/qa_dataset.jsonl \
    --autocomplete eval/data/autocomplete_dataset.jsonl \
    --chunk-sizes 512 768 900 \
    --chunk-overlaps 100 150 \
    --chat-top-ks 3 4 \
    --ac-top-ks 2 3
```

Reports are written to `tuning/reports/` as both Markdown and JSON, timestamped per run. The best configuration is printed at the end and shown in full in the report. Winners can be directly used as env vars:

```
CHUNK_SIZE=900 CHUNK_OVERLAP=150 CHAT_TOP_K=4 AUTOCOMPLETE_TOP_K=3 uvicorn app:app
```

---

## Evaluation

Run (macOS/Linux):
```
  python run_eval.py \
    --pdf data/sample_policy.pdf \
    --qa data/qa_dataset.jsonl \
    --autocomplete data/autocomplete_dataset.jsonl \
    --backend http://127.0.0.1:8000 \
    --out eval_report.md
```

Optional machine-readable pipeline output:
```
  python run_eval.py \
    --pdf data/sample_policy.pdf \
    --qa data/qa_dataset.jsonl \
    --autocomplete data/autocomplete_dataset.jsonl \
    --backend http://127.0.0.1:8000 \
    --out eval_report.md \
    --pipeline-out eval_pipeline.json
```

Run (Windows):
```
  python run_eval.py ^
    --pdf data\sample_policy.pdf ^
    --qa data\qa_dataset.jsonl ^
    --autocomplete data\autocomplete_dataset.jsonl ^
    --backend http://127.0.0.1:8000 ^
    --out eval_report.md ^
    --pipeline-out eval_pipeline.json
```

---

## Metrics

- Keyword Recall
- Page Recall
- Groundedness Score
- Autocomplete Match Rate
- Auto-tagged failures (QA + autocomplete)
- Pipeline pass/fail summary JSON

---

## Learnings

- Retrieval quality is the main bottleneck
- Autocomplete improves query formulation
- Evaluation requires semantic scoring, not just keywords

---

## Future Work

- Hybrid search (BM25 + vector)
- Re-ranking
- LLM-as-judge
- Multi-document indexing
