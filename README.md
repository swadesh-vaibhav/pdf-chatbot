# 📄 PDF Chatbot with Context-Aware Autocomplete

A **retrieval-augmented generation (RAG) system** that enables users to:
- Ask questions over PDF documents
- Receive grounded answers with citations
- Get context-aware autocomplete suggestions

---

## 🚀 Features

### 🔍 Document Q&A (RAG)
- Semantic search over PDF content using embeddings
- Context-aware answer generation
- Page-level citation extraction

### ⚡ Context-Aware Autocomplete (Key Differentiator)
- Suggests query completions grounded in document content
- Improves query formulation and retrieval quality
- Low-latency suggestions using caching

### ⚙️ Production-Oriented Design
- FastAPI backend
- Redis caching layer
- Vector search (FAISS)
- Modular evaluation framework

---

## 🧠 System Architecture

User → Frontend → Backend → Redis → Vector DB → LLM

---

## 📂 Project Structure

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

## 🛠️ Setup

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

## 📊 Evaluation

Run (macOS/Linux):
```
  python run_eval.py \
    --pdf data/sample_policy.pdf \
    --qa data/qa_dataset.jsonl \
    --autocomplete data/autocomplete_dataset.jsonl \
    --backend http://127.0.0.1:8000 \
    --out eval_report.md
```

Run (Windows):
```
  python run_eval.py ^
    --pdf data\sample_policy.pdf ^
    --qa data\qa_dataset.jsonl ^
    --autocomplete data\autocomplete_dataset.jsonl ^
    --backend http://127.0.0.1:8000 ^
    --out eval_report.md
```

---

## 📈 Metrics

- Keyword Recall
- Page Recall
- Groundedness Score
- Autocomplete Match Rate

---

## 🔬 Learnings

- Retrieval quality is the main bottleneck
- Autocomplete improves query formulation
- Evaluation requires semantic scoring, not just keywords

---

## 🔮 Future Work

- Hybrid search (BM25 + vector)
- Re-ranking
- LLM-as-judge
- Streaming responses

---

## 🎯 Positioning

This is a **retrieval-augmented system with evaluation, caching, and query assistance**, not just a chatbot.

---

## 🧑‍💻 Author

Built as a hands-on ML systems project focusing on RAG, evaluation, and system design.
