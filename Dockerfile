FROM python:3.11-slim AS base

COPY --from=ghcr.io/astral-sh/uv:0.7 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev --no-install-project

COPY agingclaw/ agingclaw/
COPY CLOCKSdata/ CLOCKSdata/
COPY data/ data/

RUN uv sync --no-dev

RUN mkdir -p data/uploads

EXPOSE 8765

CMD ["uv", "run", "agingclaw-web", "--host", "0.0.0.0", "--port", "8765"]
