"""
FastAPI app: POST /ask for the question-answering endpoint, GET / for a
minimal chat UI, GET /health for a container healthcheck.

Run locally:   uvicorn src.api:app --reload
Run in Docker: this is the CMD in the Dockerfile.
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.rag_pipeline import answer_question
from src.vector_store import count

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="RAG Knowledge Assistant", version="0.1.0")


class AskRequest(BaseModel):
    question: str


class SourceResponse(BaseModel):
    source_id: str
    source_title: str
    snippet: str
    distance: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "indexed_chunks": count()}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    if count() == 0:
        raise HTTPException(
            status_code=503,
            detail="Vector store is empty. Run `python -m src.build_index --source local` first.",
        )

    try:
        result = answer_question(question)
    except RuntimeError as exc:
        # e.g. missing API key -- a config problem, not a client error
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AskResponse(
        answer=result.text,
        sources=[SourceResponse(**vars(s)) for s in result.sources],
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
