"""
Loads the bundled sample corpus from data/sample/*.txt.

This exists so the project is runnable end-to-end (build index -> ask
questions) with zero network access and zero API keys for everything except
the final answer-generation call. It's the fastest path to "I can demo this
in an interview right now" and the fastest path to actually testing the
chunking/embedding/retrieval pipeline before pointing it at arXiv.
"""
from pathlib import Path

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"


def load_local_documents() -> list[dict]:
    """Returns [{"id", "title", "text"}] from every .txt file in data/sample/.
    The first line of each file is treated as the title."""
    documents: list[dict] = []
    for path in sorted(SAMPLE_DIR.glob("*.txt")):
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        lines = content.split("\n", 1)
        title = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else title
        documents.append({"id": path.stem, "title": title, "text": body})
    return documents
