# --- Build stage kept single-stage deliberately: sentence-transformers pulls
# in torch, which is large either way, and a multi-stage split buys little
# here since we're not compiling anything. Not over-engineering for its own
# sake.
FROM python:3.11-slim

WORKDIR /app

# System deps: none needed beyond what python:slim ships with for this
# project (no PDF/OCR libs yet — see ingest_arxiv.py's abstract-only note).

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY static/ ./static/
COPY data/sample/ ./data/sample/

EXPOSE 8000

# Healthcheck hits our own /health endpoint (see src/api.py)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
