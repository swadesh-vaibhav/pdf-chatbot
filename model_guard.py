"""
model_guard.py — Detect model config changes at startup and take corrective action.

  CHAT_MODEL  changed → flush answer + autocomplete Redis cache
  EMBED_MODEL changed → flush all Redis cache + delete vector index
                        (embeddings from different models are incompatible)
"""

import json

from clients import redis_client
from config import CHAT_MODEL, CHUNKS_PATH, EMBED_MODEL, INDEX_PATH, MODEL_STATE_PATH
from logger import get_logger

log = get_logger("model_guard")

_ANSWER_PATTERNS    = ("answer:*", "ac:*")
_ALL_CACHE_PATTERNS = ("answer:*", "ac:*", "retrieval:*")


def _load_state() -> dict:
    if MODEL_STATE_PATH.exists():
        with open(MODEL_STATE_PATH) as f:
            return json.load(f)
    return {}


def _save_state() -> None:
    with open(MODEL_STATE_PATH, "w") as f:
        json.dump({"chat_model": CHAT_MODEL, "embed_model": EMBED_MODEL}, f)


def _flush_redis(patterns: tuple) -> int:
    deleted = 0
    for pattern in patterns:
        cursor = 0
        while True:
            cursor, keys = redis_client.scan(cursor, match=pattern, count=100)
            if keys:
                redis_client.delete(*keys)
                deleted += len(keys)
            if cursor == 0:
                break
    return deleted


def _delete_index() -> None:
    for path in (INDEX_PATH, CHUNKS_PATH):
        if path.exists():
            path.unlink()


def check_and_apply_model_changes() -> None:
    saved = _load_state()

    if not saved:
        # First run — nothing to flush, just record current config.
        _save_state()
        log.info("model_guard.init", extra={"chat_model": CHAT_MODEL, "embed_model": EMBED_MODEL})
        return

    embed_changed = saved.get("embed_model") != EMBED_MODEL
    chat_changed  = saved.get("chat_model")  != CHAT_MODEL

    if embed_changed:
        deleted = _flush_redis(_ALL_CACHE_PATTERNS)
        _delete_index()
        log.warning(
            "model_guard.embed_model_changed",
            extra={
                "old_embed_model": saved.get("embed_model"),
                "new_embed_model": EMBED_MODEL,
                "redis_keys_deleted": deleted,
                "index_wiped": True,
            },
        )
    elif chat_changed:
        deleted = _flush_redis(_ANSWER_PATTERNS)
        log.warning(
            "model_guard.chat_model_changed",
            extra={
                "old_chat_model": saved.get("chat_model"),
                "new_chat_model": CHAT_MODEL,
                "redis_keys_deleted": deleted,
            },
        )

    if embed_changed or chat_changed:
        _save_state()
