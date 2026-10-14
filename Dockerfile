# Multi-stage production Dockerfile for ModelTrace single-origin localhost container
# Stage 1: Build React 18 + TypeScript + Vite frontend into /build/frontend/dist
FROM node:20-bookworm-slim AS frontend-builder

WORKDIR /build/frontend

COPY frontend/package.json ./
COPY frontend/package-lock.json ./

# A release always uses the committed, npm-generated dependency lock.
RUN npm ci --no-audit --no-fund

COPY frontend/index.html frontend/tsconfig.json frontend/vite.config.ts ./
COPY frontend/public ./public
COPY frontend/src ./src

# Run real TypeScript typecheck (tsc --noEmit) and Vite production build (vite build)
RUN npm run typecheck && npm run build && \
    test -s /build/frontend/dist/index.html && \
    test -s /build/frontend/dist/assets/app.js && \
    test -s /build/frontend/dist/assets/styles.css && \
    test ! -f /build/frontend/dist/UNSUITABLE_FOR_RELEASE.json

# Stage 2: Python 3.11 runtime with compatible pinned dependencies and compiled frontend assets
FROM python:3.11-slim-bookworm AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODELTRACE_HOST=0.0.0.0 \
    MODELTRACE_PORT=8000 \
    MODELTRACE_DB_PATH=/app/data/modeltrace.db \
    MODELTRACE_EXPORTS_DIR=/app/exports \
    MODELTRACE_LOCAL_MODELS_DIR=/app/data/local_models

COPY pyproject.toml requirements.lock requirements.txt README.md ./
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.lock

COPY modeltrace ./modeltrace
COPY scripts ./scripts
COPY frontend/public ./frontend/public
COPY --from=frontend-builder /build/frontend/dist ./frontend/dist

RUN python3 -m pip install --no-cache-dir --no-deps -e . && \
    useradd --create-home --shell /bin/bash modeltrace && \
    mkdir -p /app/data /app/exports /app/data/local_models && \
    chown -R modeltrace:modeltrace /app

USER modeltrace
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)" || exit 1

CMD ["modeltrace", "serve", "--host", "0.0.0.0", "--port", "8000"]
