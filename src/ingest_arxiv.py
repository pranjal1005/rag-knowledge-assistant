"""
Fetches paper metadata + abstracts from the public arXiv API
(https://export.arxiv.org/api) -- no API key required, but it does need
real network access, so this module (unlike embeddings/vector_store) can't
be exercised inside a network-sandboxed environment. It was written and
syntax-checked but not live-tested against arxiv.org -- run it on your own
machine first and sanity-check a few results before trusting the pipeline
end-to-end.

We fetch abstracts only, not full PDF text. Reasons: abstracts are directly
available as clean text from the API (no PDF parsing/OCR reliability issues),
they're information-dense by design, and for a first version that's a much
better cost/complexity tradeoff than PDF extraction. Swapping to full-text
(via pypdf on downloaded PDFs) is a natural v2 -- the chunking/embedding/
storage code downstream doesn't care where the text came from.
"""
import time
import xml.etree.ElementTree as ET

import requests

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = "{http://www.w3.org/2005/Atom}"


def fetch_arxiv_papers(query: str, max_results: int = 40) -> list[dict]:
    """
    query: an arXiv search query, e.g. "all:retrieval augmented generation"
           or "cat:cs.DC AND all:data pipeline"
    Returns a list of {"id", "title", "text"} dicts, where "text" is the
    abstract and "id" is the arXiv identifier (e.g. "2312.10997").
    """
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    response = requests.get(ARXIV_API_URL, params=params, timeout=30)
    response.raise_for_status()

    root = ET.fromstring(response.text)
    papers: list[dict] = []

    for entry in root.findall(f"{ATOM_NS}entry"):
        arxiv_url = entry.findtext(f"{ATOM_NS}id", default="").strip()
        arxiv_id = arxiv_url.rsplit("/", 1)[-1] if arxiv_url else ""
        title = " ".join(entry.findtext(f"{ATOM_NS}title", default="").split())
        summary = " ".join(entry.findtext(f"{ATOM_NS}summary", default="").split())

        if not arxiv_id or not summary:
            continue

        papers.append({"id": arxiv_id, "title": title, "text": summary})

    return papers


def fetch_with_retry(query: str, max_results: int = 40, retries: int = 3) -> list[dict]:
    """arXiv's API rate-limits aggressively -- back off and retry on failure
    rather than letting one flaky request kill the whole ingestion run."""
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return fetch_arxiv_papers(query, max_results)
        except (requests.RequestException, ET.ParseError) as exc:
            last_error = exc
            time.sleep(2**attempt)  # 1s, 2s, 4s
    raise RuntimeError(f"arXiv fetch failed after {retries} attempts") from last_error
