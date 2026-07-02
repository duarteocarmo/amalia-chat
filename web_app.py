import json
import os
import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

CONFIG = {
    "api_url": "https://duarteocarmo--amalia-vllm-gguf-api-serve.modal.run",
    "api_key_env_var": "VLLM_API_KEY",
    "model": "amalia",
    "request_limit": 15,
    "rate_limit_window_seconds": 60 * 60,
    "max_messages": 50,
    "max_message_chars": 8_000,
    "max_total_chars": 32_000,
    "default_temperature": 0.7,
    "default_max_tokens": 512,
    "request_timeout_seconds": 120,
}

app = FastAPI(title="AMALIA Chat")
request_log: dict[str, deque[float]] = defaultdict(deque)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None


@app.get("/")
async def index() -> HTMLResponse:
    html = Path("web/index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat/completions")
async def chat_completions(
    request: Request, chat_request: ChatRequest
) -> StreamingResponse:
    check_rate_limit(ip_address=ip_address_for(request=request))
    validate_chat_request(chat_request=chat_request)
    return StreamingResponse(
        content=stream_completion(chat_request=chat_request),
        media_type="text/event-stream",
    )


def ip_address_for(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def check_rate_limit(ip_address: str) -> None:
    now = time.time()
    cutoff = now - CONFIG["rate_limit_window_seconds"]
    timestamps = request_log[ip_address]

    while timestamps and timestamps[0] < cutoff:
        timestamps.popleft()

    if len(timestamps) >= CONFIG["request_limit"]:
        raise HTTPException(
            status_code=429,
            detail="Calma pá. Espera uma hora (429)",
        )

    timestamps.append(now)


def validate_chat_request(chat_request: ChatRequest) -> None:
    if not chat_request.messages:
        raise HTTPException(status_code=400, detail="At least one message is required.")
    if len(chat_request.messages) > CONFIG["max_messages"]:
        raise HTTPException(status_code=400, detail="Too many messages.")

    total_chars = 0
    for message in chat_request.messages:
        if message.role not in {"user", "assistant", "system"}:
            raise HTTPException(status_code=400, detail="Invalid message role.")
        if not message.content.strip():
            raise HTTPException(
                status_code=400, detail="Empty messages are not allowed."
            )
        if len(message.content) > CONFIG["max_message_chars"]:
            raise HTTPException(status_code=400, detail="Message is too long.")
        total_chars += len(message.content)

    if total_chars > CONFIG["max_total_chars"]:
        raise HTTPException(status_code=400, detail="Conversation is too long.")


async def stream_completion(chat_request: ChatRequest) -> AsyncIterator[str]:
    api_key = os.environ.get(CONFIG["api_key_env_var"])
    if not api_key:
        yield sse_event(data={"error": "Server missing VLLM_API_KEY."})
        return

    payload = {
        "model": CONFIG["model"],
        "messages": [message.model_dump() for message in chat_request.messages],
        "stream": True,
        "temperature": chat_request.temperature or CONFIG["default_temperature"],
        "max_tokens": chat_request.max_tokens or CONFIG["default_max_tokens"],
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    timeout = httpx.Timeout(timeout=CONFIG["request_timeout_seconds"])

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            method="POST",
            url=f"{CONFIG['api_url']}/v1/chat/completions",
            headers=headers,
            json=payload,
        ) as response:
            if response.status_code != 200:
                text = await response.aread()
                yield sse_event(
                    data={
                        "error": f"Upstream error {response.status_code}: {text.decode()}"
                    }
                )
                return

            async for line in response.aiter_lines():
                if not line or line == "data: [DONE]":
                    continue
                if not line.startswith("data: "):
                    continue

                chunk = json.loads(line[len("data: ") :])
                delta = chunk["choices"][0]["delta"]
                token = delta.get("content")
                if token:
                    yield sse_event(data={"token": token})

    yield sse_event(data={"done": True})


def sse_event(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
