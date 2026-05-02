FROM python:3.11-slim

WORKDIR /app

# ── System build dependencies ──────────────────────────────────────────────────
# gcc/g++ required for numpy and faiss-cpu compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ────────────────────────────────────────────────────────
# Install CPU-only torch FIRST (prevents pulling the 2 GB CUDA wheel when
# sentence-transformers later requests torch as a dependency)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements before source code so Docker can cache this layer
COPY demo/requirements.txt /tmp/demo-req.txt
COPY rag/requirements.txt  /tmp/rag-req.txt
RUN pip install --no-cache-dir -r /tmp/demo-req.txt && \
    pip install --no-cache-dir -r /tmp/rag-req.txt

# ── Pre-download BGE embedding model ──────────────────────────────────────────
# Bake the model into the image so startup is fast on HF Spaces (no network wait).
# Store in /app/.cache/huggingface so it survives the non-root user switch below.
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-base-en-v1.5')" \
    && chmod -R 755 /app/.cache

# ── Application code ───────────────────────────────────────────────────────────
# .dockerignore excludes .env, data/parsed/, paper/, scripts/, etc.
COPY . .

# ── Non-root user (HF Spaces requirement) ─────────────────────────────────────
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/app/.cache/huggingface \
    PORT=7860

EXPOSE 7860

# ── Start server ───────────────────────────────────────────────────────────────
# Run from /app (repo root) so both `demo` and `rag` are importable as packages.
# 1 worker keeps SQLite writes safe; 4 threads handle concurrent requests.
CMD exec gunicorn \
    --bind "0.0.0.0:${PORT}" \
    --workers 1 \
    --threads 4 \
    --timeout 120 \
    demo.app:app
