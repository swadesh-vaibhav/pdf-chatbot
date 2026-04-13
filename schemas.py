"""Pydantic request schemas used by FastAPI routes."""

from pydantic import BaseModel


class ChatRequest(BaseModel):
    """Request body for chat and chat-stream endpoints."""

    query: str


class AutocompleteRequest(BaseModel):
    """Request body for autocomplete endpoint."""

    prefix: str
