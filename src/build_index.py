"""
CLI entry point that builds (or rebuilds) the vector index.

Usage:
    python -m src.build_index --source local
    python -m src.build_index --source arxiv --query "retrieval augmented generation" --max-results 40
"""
import argparse
import sys

from src.chunking import chunk_documents
from src.config import settings
from src.embeddings import embed_texts
from src.ingest_arxiv import fetch_with_retry
from src.ingest_local import load_local_documents
from src.vector_store import reset_collection, upsert_chunks


def build_index(source: str, query: str | None, max_results: int) -> None:
    if source == "local":
        documents = load_local_documents()
        print(f"Loaded {len(documents)} local sample documents.")
    elif source == "arxiv":
        if not query:
            print("--query is required when --source arxiv", file=sys.stderr)
            sys.exit(1)
        documents = fetch_with_retry(query, max_results=max_results)
        print(f"Fetched {len(documents)} papers from arXiv for query: {query!r}")
    else:
        print(f"Unknown source: {source!r} (expected 'local' or 'arxiv')", file=sys.stderr)
        sys.exit(1)

    if not documents:
        print("No documents to index. Aborting.", file=sys.stderr)
        sys.exit(1)

    chunks = chunk_documents(
        documents, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap
    )
    print(f"Split into {len(chunks)} chunks (chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap}).")

    print("Computing embeddings locally (first run downloads the model, ~80MB)...")
    embeddings = embed_texts([c.text for c in chunks])

    print("Writing to the vector store...")
    reset_collection()
    upsert_chunks(
        ids=[f"{c.source_id}::{c.chunk_index}" for c in chunks],
        embeddings=embeddings,
        documents=[c.text for c in chunks],
        metadatas=[{"source_id": c.source_id, "source_title": c.source_title} for c in chunks],
    )
    print(f"Done. Indexed {len(chunks)} chunks from {len(documents)} documents.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the RAG vector index.")
    parser.add_argument("--source", choices=["local", "arxiv"], default="local")
    parser.add_argument("--query", type=str, default=None, help="arXiv search query (required if --source arxiv)")
    parser.add_argument("--max-results", type=int, default=40)
    args = parser.parse_args()

    build_index(source=args.source, query=args.query, max_results=args.max_results)


if __name__ == "__main__":
    main()
