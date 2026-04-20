import json
import logging
import logging.handlers
import os
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

_request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

_STDLIB_ATTRS = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "taskName",
})


def get_request_id() -> str:
    return _request_id_var.get()


def set_request_id(rid: str):
    return _request_id_var.set(rid)


def new_request_id() -> str:
    return str(uuid.uuid4())


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        log = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "request_id": get_request_id(),
            "message": message,
        }
        for key, value in record.__dict__.items():
            if key not in _STDLIB_ATTRS and not key.startswith("_"):
                log[key] = value
        if record.exc_info:
            log["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(log, default=str)


_LOG_DIR = Path(__file__).parent / "logs"


def configure_logging(level: str = "INFO") -> None:
    fmt = JsonFormatter()
    lvl = getattr(logging, level.upper(), logging.INFO)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)

    _LOG_DIR.mkdir(exist_ok=True)
    rotating = logging.handlers.RotatingFileHandler(
        _LOG_DIR / "app.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB per file
        backupCount=5,
        encoding="utf-8",
    )
    rotating.setFormatter(fmt)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(rotating)
    root.setLevel(lvl)
    # Suppress noisy third-party access logs; our middleware owns request logging.
    logging.getLogger("uvicorn.access").propagate = False
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
