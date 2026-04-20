import os
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

import analytics
from indexing import build_index
from logger import configure_logging, get_logger, new_request_id, set_request_id
from routes import router

configure_logging(level=os.getenv("LOG_LEVEL", "INFO"))
log = get_logger("app")

app = FastAPI()


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    rid = new_request_id()
    set_request_id(rid)
    endpoint = request.url.path
    analytics.record_request_start(endpoint)
    start = time.perf_counter()

    log.info(
        "request.start",
        extra={
            "method": request.method,
            "path": endpoint,
            "client_ip": request.client.host if request.client else None,
        },
    )

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000
        analytics.record_request_end(endpoint, 500, duration_ms)
        log.exception(
            "request.unhandled_error",
            extra={"path": endpoint, "duration_ms": round(duration_ms, 2)},
        )
        raise

    duration_ms = (time.perf_counter() - start) * 1000
    analytics.record_request_end(endpoint, response.status_code, duration_ms)

    log.info(
        "request.end",
        extra={
            "method": request.method,
            "path": endpoint,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 2),
        },
    )
    response.headers["X-Request-ID"] = rid
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

build_index()
log.info("app.startup: PDF chatbot backend ready")
