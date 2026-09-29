# RAG Knowledge Assistant

A citation-grounded question-answering system over a document corpus, built
to demonstrate a real (not tutorial-copied) RAG pipeline: local embeddings,
Chroma as its own service, a swappable LLM backend, and a design that's
honest about what it does and doesn't handle.

Ships with a small bundled sample corpus (data/sample/) covering ETL vs
ELT, how RAG works, vector embeddings, and Docker vs Kubernetes — so it
runs end-to-end with **zero API keys, zero network access, and zero setup**
beyond installing dependencies. Point it at real arXiv papers once you've
seen it work.

## Why this exists

Not a tutorial clone. Built to have a real, defensible answer to "walk me
through a RAG system you've built" — every design decision below is one I
can explain the tradeoff for, not one I copied without understanding.

## Architecture

```
                    ┌─────────────────────┐
   question ───────▶│   embed (local)      │
                    │ sentence-transformers│
                    └──────────┬───────────┘
                               │ query vector
                               ▼
                    ┌─────────────────────┐
                    │   Chroma (vector DB) │◀── indexed chunks
                    │  own container/HTTP  │    (from build_index)
                    └──────────┬───────────┘
                               │ top-k chunks
                               ▼
                    ┌─────────────────────┐
                    │  LLM (Claude/OpenAI) │
                    │  cites chunk numbers │
                    └──────────┬───────────┘
                               │
                               ▼
                     answer + cited sources
```

Two phases, same as any RAG system:
- **Indexing** (`src/build_index.py`): load documents (local sample files or
  live arXiv abstracts) → split into overlapping chunks (`src/chunking.py`)
  → embed each chunk locally (`src/embeddings.py`) → store in Chroma
  (`src/vector_store.py`).
- **Query** (`src/rag_pipeline.py`): embed the question with the same model
  → retrieve the nearest chunks → send them to the LLM with an instruction
  to cite which passage supports each claim (`src/llm_client.py`) → return
  the answer plus the sources it drew from.

## Design decisions and tradeoffs (the part that matters in an interview)

**Local embeddings (sentence-transformers), hosted LLM for generation.**
Embedding is the highest-volume call in the system (every chunk, every
rebuild) — running it locally means zero API cost and zero external
dependency for indexing. Generation is low-volume (one call per question),
so paying for a hosted model there is worth it for answer quality.
Tradeoff: MiniLM embeddings are noticeably weaker than OpenAI's
text-embedding-3 or Voyage's models on nuanced technical text. If retrieval
quality becomes the bottleneck, swapping `embeddings.py` for a hosted API is
a self-contained change — nothing downstream needs to know how embeddings
are produced.

**Chroma, not pgvector or a managed vector DB (Pinecone/Weaviate).**
Chroma needs no schema and runs as a single extra container — the fastest
path to something working. Tradeoff: Chroma doesn't have the operational
maturity (replication, managed scaling) of a hosted vector DB or the
"I already have Postgres" simplicity of pgvector. At real scale, or in an
environment already running Postgres, pgvector would likely be the better
call. `vector_store.py` is the only file that would need to change.

**Character-based chunking with a word-boundary heuristic, not
sentence/semantic chunking.** Zero dependencies, fast, good enough for
abstract-length documents. Tradeoff: it doesn't respect sentence or
paragraph structure, so a chunk boundary can fall in an awkward spot
semantically even though it never splits a word. A production system
ingesting long, structured documents (not just abstracts) would likely
want a proper sentence-aware or recursive splitter (e.g. what LangChain's
`RecursiveCharacterTextSplitter` does, or a dedicated NLP sentence
segmenter).

**Abstracts only from arXiv, not full PDF text.** Abstracts come back as
clean text directly from the API — no PDF parsing/OCR reliability issues,
and they're information-dense by design. Full-text ingestion (via `pypdf`
on downloaded PDFs) is a natural v2; the chunking/embedding/storage code
doesn't care where the text came from.

**Explicit citation instructions in the system prompt, not a separate
verification pass.** The LLM is told to answer only from the provided
passages and to cite a passage number for every claim, and to say "I don't
know" when the passages are insufficient. This meaningfully reduces
hallucination but doesn't eliminate it — the model can still misread or
over-generalize from retrieved context. A stricter system would verify
each cited claim against the source passage after generation; that's not
implemented here.

**Chroma runs as its own container, not embedded in the app process.**
`docker-compose.yml` runs `app` and `chroma` as separate services talking
over the network (see `vector_store.py`'s `CHROMA_HOST` branch), rather
than using Chroma's embedded on-disk mode inside the app container. This is
deliberately closer to how a real deployment would separate a stateful
data service from a stateless app tier — restarting/redeploying the app
doesn't touch the index. For pure local dev without Docker, the same code
falls back to an embedded on-disk store automatically (no `CHROMA_HOST`
set), so you're not forced to run Docker just to iterate on the app code.

**Dual LLM backend (Anthropic/OpenAI/Gemini), switched by one env var.**
`llm_client.py` separates prompt construction (`build_prompt`, pure
function, fully unit-tested with no API key) from the actual network call
(`call_llm`). This is what makes the prompt logic testable in CI without
credentials, and it's what makes switching providers a one-line `.env`
change rather than a code change. Gemini was added specifically because
Google AI Studio's free tier needs no credit card, unlike Anthropic and
OpenAI which both require a minimum paid credit before a key works —
useful to know going in, since it's easy to assume "API key" means "free"
when it usually doesn't.

**Kubernetes: deliberately not used here.** Docker Compose is the right
tool for "two containers on one machine." Reaching for Kubernetes here
would add real operational complexity (a control plane, manifests, pod/
service/ingress concepts) to solve a multi-machine orchestration problem
this project doesn't have. If this needed to scale across nodes with
automated failover, K8s would be the next step — but adopting it before
that point trades simplicity for capability that isn't being used yet.
(Full disclosure, in the interest of not overclaiming: I haven't yet
deployed anything to a real K8s cluster. This project's Docker Compose
setup is real and running; a K8s deployment of it would be genuine future
work, not something to imply is already done.)

## Setup

```bash
cp .env.example .env
# edit .env: set LLM_PROVIDER and the matching API key
```

### Option A — Docker (recommended, matches the architecture above)

```bash
docker compose up --build -d
docker compose exec app python -m src.build_index --source local
# open http://localhost:8000
```

### Option B — Local, no Docker

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m src.build_index --source local
uvicorn src.api:app --reload
# open http://localhost:8000
```

First run downloads the local embedding model (~80MB) — needs network for
that one time only, not for every query afterward.

### Indexing real papers instead of the sample corpus

```bash
python -m src.build_index --source arxiv --query "retrieval augmented generation" --max-results 40
```

`src/ingest_arxiv.py` hits the public arXiv API (no key required). This
needs real network access to run — it was written and syntax-checked but
not live-tested against arxiv.org in the environment this was built in, so
sanity-check the first few results before trusting the pipeline blind.

### Asking questions

- **Web UI**: http://localhost:8000
- **CLI**: `python -m src.cli`
- **API directly**: `curl -X POST localhost:8000/ask -H "Content-Type: application/json" -d '{"question": "What is the difference between ETL and ELT?"}'`

### Running tests

```bash
pip install pytest
pytest tests/ -v
```

`tests/test_chunking.py` and `tests/test_prompt.py` need no API key, no
network, and no heavy ML dependencies — they cover the pure-logic parts of
the pipeline (chunking correctness, prompt construction) and are meant to
run in CI on every commit. Several of these tests are regression guards for
real bugs found while building this (an infinite loop on a very long
unbroken token, two different ways the chunker could split a word) — see
the comments in `test_chunking.py` for specifics.

## Known limitations

- No answer-verification pass — citations are requested, not enforced.
- Chunking is character-based, not sentence-aware (see tradeoffs above).
- `ingest_arxiv.py` is untested against the live API in this repo's build
  environment (no network access there) — verify it works before relying
  on it.
- No auth on the API — fine for a local demo, not fine to expose publicly
  as-is.
- No reranking step after retrieval — top-k by embedding similarity alone,
  which can occasionally surface a topically-similar but not-quite-relevant
  chunk. A cross-encoder reranking pass is the standard fix if this becomes
  a visible problem.

## What I'd do next

Full-text PDF ingestion, a reranking step, sentence-aware chunking, answer
verification against cited sources, and — if this ever needed multi-machine
scaling — moving from Docker Compose to Kubernetes.
