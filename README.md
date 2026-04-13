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
- FastAPI backend
- Redis caching layer
- Vector search (FAISS)
- Modular evaluation framework

---

## System Design

User → Frontend → Backend → Redis → Vector DB → LLM

---

## Project Structure

```
.
├── app/
├── backend/
├── eval/
│   ├── metrics.py
│   ├── run_eval.py
│   ├── data/
│       ├── qa_dataset.jsonl
│       ├── autocomplete_dataset.jsonl
│       ├── sample_policy.pdf
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

---

## Positioning

This is a **retrieval-augmented system with evaluation, caching, and query assistance**, not just a chatbot.
