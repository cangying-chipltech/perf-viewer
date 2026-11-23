FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PERF_VIEWER_DATA_DIR=/app/data

WORKDIR /app

COPY pyproject.toml README.md ./
COPY app ./app
COPY .kilo ./.kilo

RUN pip install --no-cache-dir . \
    && useradd -r -u 10001 -m appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8600
VOLUME ["/app/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8600/healthz')" || exit 1

CMD ["python", "-m", "app.cli", "serve", "--host", "0.0.0.0", "--port", "8600"]
