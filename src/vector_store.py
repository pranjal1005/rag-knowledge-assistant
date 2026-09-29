"""
Thin wrapper around Chroma.

If CHROMA_HOST is set (docker-compose sets this so the app container talks
to the chroma container over the network), we use chromadb.HttpClient.
Otherwise we fall back to an embedded PersistentClient writing to local
disk -- so `python -m src.build_index` works with no Docker at all for
quick local iteration, and the same code path works unchanged inside
docker-compose. This is the one piece of infra-swapping logic in the
project; everything that calls this module doesn't know or care which
backend is active.
"""
from functools import lru_cache

from src.config import settings


@lru_cache(maxsize=1)
def _get_client():
    import chromadb

    if settings.chroma_host:
        return chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    return chromadb.PersistentClient(path=settings.chroma_persist_dir)


def _get_collection():
    """
    Deliberately NOT cached (unlike _get_client above). This function used
    to be @lru_cache'd, which caused a real bug: the long-running API
    server process caches a Collection object referencing a specific
    collection ID. `python -m src.build_index` runs as a *separate* OS
    process (via `docker compose exec`) that deletes and recreates the
    collection with a NEW id when rebuilding the index -- but the API
    server's cached reference has no way to know that happened, so its
    next request fails with `InvalidCollectionException: Collection
    <old-id> does not exist`.

    get_or_create_collection() is a cheap HTTP round-trip, not expensive
    enough to justify caching at the cost of this staleness class of bug.
    """
    client = _get_client()
    return client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection() -> None:
    """Drops the collection. Used by build_index for a clean rebuild --
    the next _get_collection() call (from any process) recreates it fresh,
    since that function is no longer cached (see its docstring)."""
    client = _get_client()
    try:
        client.delete_collection(settings.chroma_collection)
    except Exception:
        pass  # collection didn't exist yet -- fine


def upsert_chunks(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict],
) -> None:
    collection = _get_collection()
    batch_size = 500
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.upsert(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )


def query(query_embedding: list[float], top_k: int) -> list[dict]:
    """Returns a list of {id, text, metadata, distance}, nearest first."""
    collection = _get_collection()
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
    hits = []
    ids = result.get("ids", [[]])[0]
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0]
    for i in range(len(ids)):
        hits.append(
            {
                "id": ids[i],
                "text": docs[i],
                "metadata": metas[i],
                "distance": dists[i],
            }
        )
    return hits


def count() -> int:
    """
    Wrapped with a single retry: chromadb's HTTP client has a known flaky
    failure mode on .count() -- RuntimeError('coroutine raised
    StopIteration') from the server side (chroma-core/chroma issue #2213).
    """
    try:
        return _get_collection().count()
    except RuntimeError:
        return _get_collection().count()