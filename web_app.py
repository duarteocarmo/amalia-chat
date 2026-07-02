import hashlib
import json
import logging
import os
import secrets
import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import weave
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

CONFIG = {
    "api_url": "https://duarteocarmo--amalia-vllm-gguf-api-serve.modal.run",
    "api_key_env_var": "VLLM_API_KEY",
    "model": "amalia",
    "request_limit": 100,
    "rate_limit_window_seconds": 60 * 60,
    "max_messages": 50,
    "max_message_chars": 8_000,
    "max_total_chars": 32_000,
    "default_temperature": 0.7,
    "default_max_tokens": 512,
    "request_timeout_seconds": 120,
    "weave_project": "duarteocarmo/amalia-chat",
    "wandb_api_key_env_var": "WANDB_API_KEY",
    "session_cookie_name": "amalia_session",
    "session_cookie_max_age_seconds": 60 * 60 * 24 * 30,
}

if os.environ.get(CONFIG["wandb_api_key_env_var"]):
    weave.init(CONFIG["weave_project"])

app = FastAPI(title="AMALIA Chat")
request_log: dict[str, deque[float]] = defaultdict(deque)

logger = logging.getLogger("uvicorn.error")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None


@app.get("/")
async def index(request: Request) -> HTMLResponse:
    html = Path("web/index.html").read_text(encoding="utf-8")
    response = HTMLResponse(content=html)
    if not session_id_from(request=request):
        session_id = new_session_id()
        set_session_cookie(
            response=response,
            request=request,
            session_id=session_id,
        )
        logger.info(
            "session_cookie_issued path=/ key=%s",
            fingerprint_for(value=f"session:{session_id}"),
        )
    return response


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat/completions")
async def chat_completions(
    request: Request, chat_request: ChatRequest
) -> StreamingResponse:
    rate_limit_key, session_id = rate_limit_key_for(request=request)
    check_rate_limit(
        rate_limit_key=rate_limit_key,
        ip_address=ip_address_for(request=request),
    )
    validate_chat_request(chat_request=chat_request)
    response = StreamingResponse(
        content=stream_completion(chat_request=chat_request),
        media_type="text/event-stream",
    )
    if session_id:
        set_session_cookie(response=response, request=request, session_id=session_id)
        logger.info(
            "session_cookie_issued path=/chat/completions key=%s",
            fingerprint_for(value=f"session:{session_id}"),
        )
    return response


def rate_limit_key_for(request: Request) -> tuple[str, str | None]:
    session_id = session_id_from(request=request)
    if session_id:
        return f"session:{session_id}", None

    session_id = new_session_id()
    return f"session:{session_id}", session_id


def session_id_from(request: Request) -> str | None:
    session_id = request.cookies.get(CONFIG["session_cookie_name"])
    if session_id and 20 <= len(session_id) <= 200:
        return session_id
    return None


def new_session_id() -> str:
    return secrets.token_urlsafe(32)


def set_session_cookie(response: Response, request: Request, session_id: str) -> None:
    response.set_cookie(
        key=CONFIG["session_cookie_name"],
        value=session_id,
        max_age=CONFIG["session_cookie_max_age_seconds"],
        httponly=True,
        secure=is_https_request(request=request),
        samesite="lax",
    )


def is_https_request(request: Request) -> bool:
    forwarded_proto = request.headers.get("x-forwarded-proto", "")
    return (
        request.url.scheme == "https"
        or forwarded_proto.split(",")[0].strip() == "https"
    )


def ip_address_for(request: Request) -> str:
    cloudflare_ip = request.headers.get("cf-connecting-ip")
    if cloudflare_ip:
        return cloudflare_ip.strip()

    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()

    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()

    if request.client:
        return request.client.host
    return "unknown"


def check_rate_limit(rate_limit_key: str, ip_address: str) -> None:
    now = time.time()
    cutoff = now - CONFIG["rate_limit_window_seconds"]
    timestamps = request_log[rate_limit_key]

    expired_count = 0
    while timestamps and timestamps[0] < cutoff:
        timestamps.popleft()
        expired_count += 1

    key_fingerprint = fingerprint_for(value=rate_limit_key)
    if len(timestamps) >= CONFIG["request_limit"]:
        logger.warning(
            "rate_limit_blocked key=%s ip=%s bucket_size=%s limit=%s expired=%s",
            key_fingerprint,
            ip_address,
            len(timestamps),
            CONFIG["request_limit"],
            expired_count,
        )
        raise HTTPException(
            status_code=429,
            detail="Calma pá. Espera uma hora (429)",
        )

    timestamps.append(now)
    logger.info(
        "rate_limit_allowed key=%s ip=%s bucket_size=%s limit=%s expired=%s",
        key_fingerprint,
        ip_address,
        len(timestamps),
        CONFIG["request_limit"],
        expired_count,
    )


def fingerprint_for(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:12]


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

    started_at = time.perf_counter()
    response_text = ""
    status = "success"
    error = None
    messages = [message.model_dump() for message in chat_request.messages]

    payload = {
        "model": CONFIG["model"],
        "messages": messages,
        "stream": True,
        "temperature": chat_request.temperature or CONFIG["default_temperature"],
        "max_tokens": chat_request.max_tokens or CONFIG["default_max_tokens"],
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    timeout = httpx.Timeout(timeout=CONFIG["request_timeout_seconds"])

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                method="POST",
                url=f"{CONFIG['api_url']}/v1/chat/completions",
                headers=headers,
                json=payload,
            ) as response:
                if response.status_code != 200:
                    text = await response.aread()
                    status = "error"
                    error = f"Upstream error {response.status_code}: {text.decode()}"
                    yield sse_event(data={"error": error})
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
                        response_text += token
                        yield sse_event(data={"token": token})

        yield sse_event(data={"done": True})
    except Exception as exc:
        status = "error"
        error = str(exc)
        raise
    finally:
        trace_conversation(
            messages=messages,
            response=response_text,
            latency_seconds=round(time.perf_counter() - started_at, 4),
            status=status,
            error=error,
        )


@weave.op(name="chat_conversation", kind="llm")
def trace_conversation(
    messages: list[dict[str, str]],
    response: str,
    latency_seconds: float,
    status: str,
    error: str | None,
) -> dict:
    return {
        "messages": messages,
        "response": response,
        "latency_seconds": latency_seconds,
        "status": status,
        "error": error,
    }


def sse_event(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
