FROM python:3.13-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

COPY web_app.py ./
COPY web ./web

EXPOSE 7777

CMD ["/app/.venv/bin/uvicorn", "web_app:app", "--host", "0.0.0.0", "--port", "7777"]
