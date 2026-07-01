from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.agent import handle
from app.schemas import ChatRequest, ChatResponse, HealthResponse

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("shl-recommender")

app = FastAPI(title="SHL Assessment Recommender", version="1.0.0")


@app.on_event("startup")
def _warm() -> None:
    try:
        from app.retrieval import get_retriever

        r = get_retriever()
        log.info("Retriever warm: %d catalog items indexed.", len(r.catalog))
    except Exception as e:  # pragma: no cover - defensive
        log.warning("Retriever warm-up skipped: %s", e)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    # Any unexpected error still returns a schema-valid body, never a 500,
    # so a single bad turn can't fail the whole evaluated conversation.
    try:
        return handle(req.messages)
    except Exception:  # pragma: no cover - last-resort safety net
        log.exception("Unhandled error in /chat")
        return ChatResponse(
            reply="Sorry, something went wrong on my side. Could you rephrase "
            "the role you're hiring for?",
            recommendations=[],
            end_of_conversation=False,
        )


