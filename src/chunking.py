"""
Splits documents into overlapping text chunks for embedding.

Deliberately dependency-free and pure-function: this is the one piece of the
pipeline that's cheap to get wrong (chunk boundaries that split a sentence
mid-thought hurt retrieval quality) and cheap to test, so it's kept isolated
from the embedding/vector-store/LLM code that needs real infra to run.
"""
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Chunk:
    text: str
    chunk_index: int
    source_id: str
    source_title: str


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Character-based sliding-window chunker with overlap.

    Word-boundary aware: we don't cut a chunk in the middle of a word, we
    back off to the last whitespace before the limit. This is a simple
    heuristic, not sentence-aware splitting (spaCy/nltk would do better),
    but it's zero-dependency and good enough for abstract-length documents.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    text = text.strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        # A previous iteration's hard-cut fallback can land `start` exactly
        # on a whitespace character (see the `next_start = end` fallback
        # below). If we don't skip it here, that leading space gets baked
        # into the window and can push a real word past `end`, truncating
        # it. Always enter the window computation on non-whitespace.
        while start < text_len and text[start].isspace():
            start += 1
        if start >= text_len:
            break

        end = min(start + chunk_size, text_len)

        # back off to the last whitespace so we don't split a word,
        # unless we're at the very end of the text. If no whitespace exists
        # in the window at all (one giant unbroken token longer than
        # chunk_size), we deliberately do NOT back off -- there's nothing to
        # back off to, and refusing to advance here is how infinite loops
        # get born. A rare hard cut through a very long token is an
        # acceptable trade for guaranteed termination.
        if end < text_len:
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= text_len:
            break

        # slide the window forward, keeping `overlap` characters of context
        next_start = end - overlap

        # align forward to the next word boundary so the new chunk doesn't
        # begin mid-word -- but only if that alignment still moves us
        # forward. `end > start` always holds at this point (chunk_size > 0
        # and we only reach here when end < text_len), so falling back to
        # `next_start = end` (i.e. dropping the overlap for this one step)
        # is always safe and always makes progress, unlike patching the
        # word-boundary search in a way that could leave next_start <= start.
        if next_start <= start:
            next_start = end
        elif not text[next_start - 1].isspace():
            next_space = text.find(" ", next_start, end)
            if next_space != -1:
                next_start = next_space + 1
            else:
                # no word boundary between next_start and end (the overlap
                # window landed entirely inside one word) -- drop overlap
                # for this one seam rather than start the next chunk
                # mid-word. `end` is always word-aligned here (it's either
                # text_len, or the whitespace position we backed off to
                # above), so this is always a clean boundary.
                next_start = end

        start = next_start

    return chunks


def chunk_documents(
    documents: Iterable[dict], chunk_size: int, overlap: int
) -> list[Chunk]:
    """
    documents: iterable of {"id": str, "title": str, "text": str}
    Returns a flat list of Chunk objects with stable, traceable IDs
    (source_id + chunk_index) so every retrieved chunk can be cited
    back to its origin document.
    """
    result: list[Chunk] = []
    for doc in documents:
        pieces = chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)
        for i, piece in enumerate(pieces):
            result.append(
                Chunk(
                    text=piece,
                    chunk_index=i,
                    source_id=doc["id"],
                    source_title=doc.get("title", doc["id"]),
                )
            )
    return result
