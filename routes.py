import json

import fitz  # pymupdf
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

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
from ollama_client import ollama_chat, ollama_embed
from schemas import AutocompleteRequest, ChatRequest
import state
import hashlib
import requests

router = APIRouter()


@router.get("/health")
def health():
    try:
        pong = redis_client.ping()
        return {"ok": True, "redis": pong, "has_index": state.faiss_index is not None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF.")

    pdf_bytes = await file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    extracted = []
    for page_num in range(len(doc)):
        page_text = doc[page_num].get_text("text")
        if page_text.strip():
            extracted.append((page_num + 1, page_text))

    all_chunks = []
    for page_num, text in extracted:
        for c in chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
            all_chunks.append(
                {
                    "page": page_num,
                    "text": c,
                    "source": file.filename,
                }
            )

    if not all_chunks:
        raise HTTPException(status_code=400, detail="No text found in PDF.")

    embeddings = ollama_embed([c["text"] for c in all_chunks])
    vectors = np.array(embeddings, dtype="float32")

    state.embedding_dim = vectors.shape[1]
    state.faiss_index = cosine_faiss_index(vectors)
    state.chunks = all_chunks

    #calculate and save sha256 hash of the file contents
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()
    state.file_hash = file_hash
    save_index()

    return {"ok": True, "chunks": len(state.chunks), "file": file.filename}


@router.post("/chat")
def chat(req: ChatRequest):
    cache_key = f"answer:{stable_key(req.query)}:{state.file_hash}"
    cached = redis_client.get(cache_key)
    if cached:
        return {"answer": cached, "cached": True, "context": []}

    context_chunks = retrieve(req.query, top_k=CHAT_TOP_K)
    context = "\n\n".join(
        f"[page {c['page']}] {c['text']}" for c in context_chunks
    )

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
    return {"answer": answer, "cached": False, "context": context_chunks}

@router.post("/chat/stream")
def chat_stream(req: ChatRequest):
    context_chunks = retrieve(req.query, top_k=CHAT_TOP_K)
    context = "\n\n".join(
        f"[page {c['page']}] {c['text']}" for c in context_chunks
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
        # Send metadata first
        yield f"data: {json.dumps({'type': 'meta', 'context': context_chunks})}\n\n"

        resp = requests.post(
            f"{OLLAMA_BASE}/chat",
            json={
                "model": "gemma3",
                "messages": messages,
                "stream": True,
            },
            stream=True,
            timeout=120,
        )
        resp.raise_for_status()

        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line.decode("utf-8"))
            token = chunk.get("message", {}).get("content", "")
            if token:
                yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")

@router.post("/autocomplete")
def autocomplete(req: AutocompleteRequest):
    cache_key = f"ac:{stable_key(req.prefix)}:{state.file_hash}"
    cached = redis_client.get(cache_key)
    if cached:
        return {"suggestions": json.loads(cached), "cached": True}

    context_chunks = retrieve(req.prefix, top_k=AUTOCOMPLETE_TOP_K)
    context = "\n\n".join(
        f"[page {c['page']}] {c['text']}" for c in context_chunks
    )

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
    return {"suggestions": suggestions, "cached": False}
