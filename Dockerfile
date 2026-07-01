FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf_cache \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


RUN python -c "from fastembed import TextEmbedding; \
    TextEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2')"

COPY . .


ENV CATALOG_PATH=data/catalog.json

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]