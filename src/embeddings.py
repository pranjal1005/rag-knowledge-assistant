"""
Wraps a local sentence-transformers model for embeddings.

Deliberate choice: embeddings run locally (all-MiniLM-L6-v2, ~80MB, CPU-fine)
rather than via a paid embeddings API. Two reasons:
  1. It means the project needs zero API key to build the index -- only the
     final generation step needs one.
  2. Embedding is the highest-volume call in a RAG system (every chunk, every
     rebuild) so it's the place where "free and local" pays off most. The
     generation step, which is low-volume (one call per question), is the
     one place paying for a hosted model is worth it for answer quality.
Tradeoff, stated plainly: MiniLM embeddings are noticeably lower quality than
OpenAI's text-embedding-3 or Voyage's models on nuanced technical text. If
retrieval quality becomes the bottleneck, swapping this module for a hosted
embeddings API is a self-contained change -- nothing else in the codebase
needs to know how embeddings are produced.
"""
from functools import lru_cache

from src.config import settings


@lru_cache(maxsize=1)
def _get_model():
    # Imported lazily + cached: importing sentence-transformers is slow and
    # pulls in torch, so we only pay that cost once, and only in processes
    # that actually embed something (not e.g. a process just building a prompt).
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embeds a batch of texts. Returns one vector (list[float]) per input text."""
    if not texts:
        return []
    model = _get_model()
    vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
