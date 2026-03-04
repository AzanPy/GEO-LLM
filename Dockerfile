FROM python:3.11-slim

WORKDIR /app

# Minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python deps (cache-friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Create cache dirs (optional)
RUN mkdir -p /app/.cache/bm25 /app/.cache/meta

EXPOSE 8000

# Default worker count (override via env)
ENV WEB_CONCURRENCY=2

CMD ["sh", "-c", "gunicorn -w ${WEB_CONCURRENCY} -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:8000 --timeout 120"]