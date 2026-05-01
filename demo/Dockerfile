FROM python:3.11-slim

WORKDIR /app

# System deps (numpy/torch need gcc; keep layer small with --no-install-recommends)
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only torch first (keeps image ~1.5 GB smaller than default)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Cloud Run injects PORT; default to 8080
ENV PORT=8080

EXPOSE 8080

# Gunicorn: 1 worker keeps SQLite writes safe; 4 threads handle concurrent requests
CMD exec gunicorn --bind "0.0.0.0:${PORT}" --workers 1 --threads 4 --timeout 120 app:app
