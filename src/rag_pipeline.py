"""
The actual RAG orchestration: embed the question, retrieve nearest chunks,
ask the LLM to answer grounded in those chunks, return the answer plus the
sources it cited -- so a caller (CLI, API, whoever) can show its work rather
than a bare answer.
"""
from dataclasses import dataclass

from src.chunking import Chunk
from src.config import settings
from src.embeddings import embed_query
from src.llm_client import generate_answer
from src.vector_store import query as vector_query


@dataclass(frozen=True)
class Source:
    source_id: str
    source_title: str
    snippet: str
    distance: float


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[Source]


def answer_question(question: str, top_k: int | None = None) -> Answer:
    k = top_k or settings.top_k

    query_vector = embed_query(question)
    hits = vector_query(query_vector, top_k=k)

    chunks = [
        Chunk(
            text=hit["text"],
            chunk_index=0,  # not needed downstream; retrieval doesn't care about position
            source_id=hit["metadata"]["source_id"],
            source_title=hit["metadata"]["source_title"],
        )
        for hit in hits
    ]

    answer_text = generate_answer(question, chunks)

    # Multiple retrieved chunks often come from the same source document --
    # dedupe by source_id for display, keeping the nearest (lowest-distance)
    # occurrence of each. This is purely a display concern: the [n]
    # citations in answer_text refer to numbered passages in the LLM
    # prompt, not to positions in this sources list, so deduping here
    # doesn't break that correspondence.
    seen_source_ids: set[str] = set()
    sources: list[Source] = []
    for hit in hits:
        source_id = hit["metadata"]["source_id"]
        if source_id in seen_source_ids:
            continue
        seen_source_ids.add(source_id)
        sources.append(
            Source(
                source_id=source_id,
                source_title=hit["metadata"]["source_title"],
                snippet=hit["text"][:200] + ("..." if len(hit["text"]) > 200 else ""),
                distance=hit["distance"],
            )
        )

    return Answer(text=answer_text, sources=sources)