from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from indexing import build_index
from routes import router

app = FastAPI()

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
