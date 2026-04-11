import json

import fitz  # pymupdf
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

from chunking import chunk_text, stable_key
from clients import redis_client
from indexing import cosine_faiss_index, retrieve, save_index
from ollama_client import ollama_chat, ollama_embed
from schemas import AutocompleteRequest, ChatRequest
import state

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
        for c in chunk_text(text):
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
    save_index()

    return {"ok": True, "chunks": len(state.chunks), "file": file.filename}


@router.post("/chat")
def chat(req: ChatRequest):
    cache_key = f"answer:{stable_key(req.query)}"
    cached = redis_client.get(cache_key)
    if cached:
        return {"answer": cached, "cached": True, "context": []}

    context_chunks = retrieve(req.query, top_k=4)
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


@router.post("/autocomplete")
def autocomplete(req: AutocompleteRequest):
    cache_key = f"ac:{stable_key(req.prefix)}"
    cached = redis_client.get(cache_key)
    if cached:
        return {"suggestions": json.loads(cached), "cached": True}

    context_chunks = retrieve(req.prefix, top_k=3)
    context = "\n\n".join(
        f"[page {c['page']}] {c['text']}" for c in context_chunks
    )

    messages = [
        {
            "role": "system",
            "content": (
                "Return exactly 3 autocomplete suggestions. "
                "Each suggestion must be grounded in the document context, "
                "must not add facts not present in the context. "
                "Return plain text, one suggestion per line."
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
