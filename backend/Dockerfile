# FIX: Use slim base — the original may have included extras that inflated image size.
# python:3.11-slim has no pre-installed torch or ML libs.
FROM python:3.11-slim

WORKDIR /app

# System deps: only what's needed for ONNX Runtime (fastembed) + SQLite
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (cached layer)
COPY requirements.txt .
# FIX: --no-cache-dir reduces image size; no torch wheels needed
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model into the image so the first request
# doesn't trigger a cold download on the free-tier instance.
# fastembed downloads to ~/.cache/fastembed by default.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/paraphrase-MiniLM-L3-v2')"

COPY . .

EXPOSE 8000

# FIX: single worker on free tier — multiple workers multiply RAM usage.
# --timeout-keep-alive 75 keeps Render's health-check connections alive.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--timeout-keep-alive", "75"]
