import json
import time

import fitz  # pymupdf
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

import analytics
from chunking import chunk_text, stable_key
from clients import redis_client
from config import (
    AUTOCOMPLETE_TOP_K,
    CHAT_TOP_K,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    OLLAMA_BASE,
)
from indexing import cosine_faiss_index, retrieve, save_index
from logger import get_logger
from ollama_client import ollama_chat, ollama_embed
from schemas import AutocompleteRequest, ChatRequest
import state
import hashlib
import requests

log = get_logger("routes")
router = APIRouter()


@router.get("/health")
def health():
    try:
        pong = redis_client.ping()
        has_index = state.faiss_index is not None
        if not pong or not has_index:
            log.warning(
                "health.degraded",
                extra={"redis_ok": pong, "has_index": has_index},
            )
        return {"ok": True, "redis": pong, "has_index": has_index}
    except Exception as e:
        log.exception("health.error")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    t0 = time.perf_counter()

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF.")

    pdf_bytes = await file.read()
    file_size_bytes = len(pdf_bytes)

    log.info(
        "upload.start",
        extra={"pdf_filename": file.filename, "file_size_bytes": file_size_bytes},
    )

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    n_pages = len(doc)

    extracted = []
    for page_num in range(n_pages):
        page_text = doc[page_num].get_text("text")
        if page_text.strip():
            extracted.append((page_num + 1, page_text))

    all_chunks = []
    for page_num, text in extracted:
        for c in chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
            all_chunks.append({"page": page_num, "text": c, "source": file.filename})

    if not all_chunks:
        log.warning(
            "upload.no_text",
            extra={"pdf_filename": file.filename, "n_pages": n_pages},
        )
        raise HTTPException(status_code=400, detail="No text found in PDF.")

    embeddings = ollama_embed([c["text"] for c in all_chunks])
    vectors = np.array(embeddings, dtype="float32")

    state.embedding_dim = vectors.shape[1]
    state.faiss_index = cosine_faiss_index(vectors)
    state.chunks = all_chunks

    file_hash = hashlib.sha256(pdf_bytes).hexdigest()
    state.file_hash = file_hash
    save_index()

    analytics.record_pdf_upload(len(state.chunks))
    duration_ms = (time.perf_counter() - t0) * 1000

    log.info(
        "upload.complete",
        extra={
            "pdf_filename": file.filename,
            "file_size_bytes": file_size_bytes,
            "n_pages": n_pages,
            "n_pages_with_text": len(extracted),
            "n_chunks": len(state.chunks),
            "file_hash": file_hash[:12],
            "duration_ms": round(duration_ms, 2),
        },
    )

    return {"ok": True, "chunks": len(state.chunks), "file": file.filename}


@router.post("/chat")
def chat(req: ChatRequest):
    t0 = time.perf_counter()
    cache_key = f"answer:{stable_key(req.query)}:{state.file_hash}"
    cached = redis_client.get(cache_key)
    if cached:
        analytics.record_cache_hit("chat")
        log.info("chat.cache_hit", extra={"query_len": len(req.query)})
        return {"answer": cached, "cached": True, "context": []}

    analytics.record_cache_miss("chat")
    context_chunks = retrieve(req.query, top_k=CHAT_TOP_K)
    context = "\n\n".join(f"[page {c['page']}] {c['text']}" for c in context_chunks)

    messages = [
        {
            "role": "system",
            "content": (
                "Answer only from the provided document context. "
                "If the context is insufficient, you should not answer. "
                "Cite page numbers in the answer."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{req.query}",
        },
    ]

    answer = ollama_chat(messages)
    redis_client.setex(cache_key, 600, answer)

    duration_ms = (time.perf_counter() - t0) * 1000
    log.info(
        "chat.complete",
        extra={
            "query_len": len(req.query),
            "n_context_chunks": len(context_chunks),
            "answer_len": len(answer),
            "duration_ms": round(duration_ms, 2),
        },
    )

    return {"answer": answer, "cached": False, "context": context_chunks}


@router.post("/chat/stream")
def chat_stream(req: ChatRequest):
    t0 = time.perf_counter()
    context_chunks = retrieve(req.query, top_k=CHAT_TOP_K)
    context = "\n\n".join(f"[page {c['page']}] {c['text']}" for c in context_chunks)

    log.info(
        "chat_stream.start",
        extra={"query_len": len(req.query), "n_context_chunks": len(context_chunks)},
    )

    messages = [
        {
            "role": "system",
            "content": (
                "Answer only from the provided document context. "
                "If the context is insufficient, say you do not know. "
                "Cite page numbers in the answer."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{req.query}",
        },
    ]

    def stream():
        yield f"data: {json.dumps({'type': 'meta', 'context': context_chunks})}\n\n"

        resp = requests.post(
            f"{OLLAMA_BASE}/chat",
            json={"model": "gemma3", "messages": messages, "stream": True},
            stream=True,
            timeout=120,
        )
        resp.raise_for_status()

        token_count = 0
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line.decode("utf-8"))
            token = chunk.get("message", {}).get("content", "")
            if token:
                token_count += 1
                yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"

        duration_ms = (time.perf_counter() - t0) * 1000
        log.info(
            "chat_stream.complete",
            extra={
                "query_len": len(req.query),
                "n_tokens_streamed": token_count,
                "duration_ms": round(duration_ms, 2),
            },
        )
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@router.post("/autocomplete")
def autocomplete(req: AutocompleteRequest):
    t0 = time.perf_counter()
    cache_key = f"ac:{stable_key(req.prefix)}:{state.file_hash}"
    cached = redis_client.get(cache_key)
    if cached:
        analytics.record_cache_hit("autocomplete")
        log.info("autocomplete.cache_hit", extra={"prefix_len": len(req.prefix)})
        return {"suggestions": json.loads(cached), "cached": True}

    analytics.record_cache_miss("autocomplete")
    context_chunks = retrieve(req.prefix, top_k=AUTOCOMPLETE_TOP_K)
    context = "\n\n".join(f"[page {c['page']}] {c['text']}" for c in context_chunks)

    messages = [
        {
            "role": "system",
            "content": (
                "Return exactly 3 autocomplete suggestions. "
                "Each suggestion must strictly be grounded in the document context."
                "Each suggestion must strictly include the prefix."
                "Each suggestion must strictly be a single sentence."
                "Return only the suggestions, without any explanation or formatting, in separate lines."
            ),
        },
        {
            "role": "user",
            "content": f"Prefix: {req.prefix}\n\nContext:\n{context}",
        },
    ]

    raw = ollama_chat(messages)
    suggestions = [line.strip("-• \t") for line in raw.splitlines() if line.strip()]
    suggestions = suggestions[:3]

    redis_client.setex(cache_key, 300, json.dumps(suggestions))

    duration_ms = (time.perf_counter() - t0) * 1000
    log.info(
        "autocomplete.complete",
        extra={
            "prefix_len": len(req.prefix),
            "n_suggestions": len(suggestions),
            "duration_ms": round(duration_ms, 2),
        },
    )

    return {"suggestions": suggestions, "cached": False}


@router.get("/metrics")
def metrics():
    return analytics.snapshot()
