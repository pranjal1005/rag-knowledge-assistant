"""
Central configuration, loaded from environment variables (via .env in local dev,
or real env vars in Docker/production). Nothing else in the codebase should
read os.environ directly -- import from here instead, so every setting has
one owner and one place to see the full list of what's configurable.
"""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _get_int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val else default


@dataclass(frozen=True)
class Settings:
    # --- LLM provider ---
    llm_provider: str = os.getenv("LLM_PROVIDER", "anthropic").lower()  # "anthropic" | "openai" | "gemini"
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

    # --- Embeddings (local, no API key needed) ---
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # --- Vector store (Chroma) ---
    # If CHROMA_HOST is set (e.g. in docker-compose), we talk to a Chroma
    # server over HTTP. Otherwise we fall back to an embedded, on-disk
    # PersistentClient -- so the project also runs with zero extra services
    # for local development.
    chroma_host: str = os.getenv("CHROMA_HOST", "")
    chroma_port: int = _get_int("CHROMA_PORT", 8000)
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "knowledge_base")

    # --- Retrieval / chunking ---
    chunk_size: int = _get_int("CHUNK_SIZE", 800)       # characters
    chunk_overlap: int = _get_int("CHUNK_OVERLAP", 150)  # characters
    top_k: int = _get_int("TOP_K", 5)


settings = Settings()
