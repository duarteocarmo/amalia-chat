import json
from typing import Any

import aiohttp
import modal

CONFIG = {
    "app_name": "amalia-vllm-gguf-api",  # Modal app name.
    "model_repo": "duarteocarmo/AMALIA-9B-0626-SFT-GGUF",  # Hugging Face GGUF repo.
    "model_quant": "Q4_K_M",  # GGUF quantization to serve.
    "tokenizer": "amalia-llm/AMALIA-9B-0626-SFT",  # Base tokenizer/config for vLLM.
    "served_model_name": "amalia",  # Model name exposed by the OpenAI-compatible API.
    "cuda_image": "nvidia/cuda:12.9.0-devel-ubuntu22.04",  # CUDA base image for Modal.
    "python_version": "3.12",  # Python version in the Modal image.
    "vllm_version": "0.21.0",  # vLLM version to install.
    "gguf_plugin_package": "vllm-gguf-plugin",  # vLLM plugin required for GGUF.
    "huggingface_cache_volume": "amalia-huggingface-cache",  # HF cache volume name.
    "vllm_cache_volume": "amalia-vllm-cache",  # vLLM compilation/cache volume name.
    "api_key_env_var": "VLLM_API_KEY",  # Local env var injected into Modal for API auth.
    "hf_token_env_var": "HF_TOKEN",  # Local env var injected into Modal for HF downloads.
    "vllm_port": 8000,  # Port exposed by the vLLM server.
    "minutes": 60,  # Seconds in one minute for timeout math.
    "gpu_type": "L40S",  # Modal GPU type to request.
    "n_gpu": 1,  # Number of GPUs per replica.
    "fast_boot": False,  # Use eager mode for faster cold starts but slower inference.
    "max_model_len": 32768,  # AMALIA context length.
    "gpu_memory_utilization": "0.90",  # Fraction of GPU memory vLLM can use.
    "scaledown_window_minutes": 15,  # Minutes to keep idle replicas alive.
    "function_timeout_minutes": 20,  # Max Modal function startup/runtime wait.
    "startup_timeout_minutes": 20,  # Max web server startup wait.
    "max_inputs": 50,  # Concurrent inputs per replica.
    "test_timeout_minutes": 5,  # Smoke test timeout.
    "test_timeout_buffer_minutes": 1,  # Buffer subtracted from healthcheck timeout.
    "default_prompt": "Explica em português europeu o que é a aprendizagem automática.",  # Smoke test prompt.
    "temperature": 0.0,  # Sampling temperature for smoke test.
    "top_p": 0.9,  # Nucleus sampling value for smoke test.
    "max_tokens": 1024,  # Max generated tokens for smoke test.
    "host": "0.0.0.0",  # Host vLLM binds to inside the container.
    "uvicorn_log_level": "info",  # vLLM API server log level.
    "hf_xet_high_performance": "1",  # Faster Hugging Face model transfers.
    "vllm_log_stats_interval": "1",  # More frequent vLLM stats logs.
}

MODEL = f"{CONFIG['model_repo']}:{CONFIG['model_quant']}"

vllm_image = (
    modal.Image.from_registry(
        tag=CONFIG["cuda_image"],
        add_python=CONFIG["python_version"],
    )
    .entrypoint([])
    .uv_pip_install(
        f"vllm=={CONFIG['vllm_version']}",
        CONFIG["gguf_plugin_package"],
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": CONFIG["hf_xet_high_performance"],
            "VLLM_LOG_STATS_INTERVAL": CONFIG["vllm_log_stats_interval"],
        }
    )
)

hf_cache_vol = modal.Volume.from_name(
    name=CONFIG["huggingface_cache_volume"],
    create_if_missing=True,
)
vllm_cache_vol = modal.Volume.from_name(
    name=CONFIG["vllm_cache_volume"],
    create_if_missing=True,
)
env_secret = modal.Secret.from_local_environ(
    env_keys=[
        CONFIG["api_key_env_var"],
        CONFIG["hf_token_env_var"],
    ]
)

app = modal.App(name=CONFIG["app_name"])


@app.function(
    image=vllm_image,
    gpu=f"{CONFIG['gpu_type']}:{CONFIG['n_gpu']}",
    scaledown_window=CONFIG["scaledown_window_minutes"] * CONFIG["minutes"],
    timeout=CONFIG["function_timeout_minutes"] * CONFIG["minutes"],
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[env_secret],
)
@modal.concurrent(max_inputs=CONFIG["max_inputs"])
@modal.web_server(
    port=CONFIG["vllm_port"],
    startup_timeout=CONFIG["startup_timeout_minutes"] * CONFIG["minutes"],
)
def serve() -> None:
    import os
    import subprocess

    cmd = [
        "vllm",
        "serve",
        MODEL,
        "--tokenizer",
        CONFIG["tokenizer"],
        "--hf-config-path",
        CONFIG["tokenizer"],
        "--served-model-name",
        CONFIG["served_model_name"],
        "--host",
        CONFIG["host"],
        "--port",
        str(CONFIG["vllm_port"]),
        "--uvicorn-log-level",
        CONFIG["uvicorn_log_level"],
        "--tensor-parallel-size",
        str(CONFIG["n_gpu"]),
        "--max-model-len",
        str(CONFIG["max_model_len"]),
        "--gpu-memory-utilization",
        CONFIG["gpu_memory_utilization"],
        "--api-key",
        os.environ[CONFIG["api_key_env_var"]],
    ]

    cmd.append("--enforce-eager" if CONFIG["fast_boot"] else "--no-enforce-eager")

    subprocess.Popen(args=cmd)


@app.local_entrypoint()
async def test(
    api_key: str,
    test_timeout: int = CONFIG["test_timeout_minutes"] * CONFIG["minutes"],
    prompt: str = CONFIG["default_prompt"],
) -> None:
    url = await serve.get_web_url.aio()
    messages = [{"role": "user", "content": prompt}]

    async with aiohttp.ClientSession(base_url=url) as session:
        print(f"Running health check for server at {url}")
        timeout = (
            test_timeout - CONFIG["test_timeout_buffer_minutes"] * CONFIG["minutes"]
        )
        async with session.get(url="/health", timeout=timeout) as resp:
            assert resp.status == 200, f"Failed health check for server at {url}"
        print(f"Successful health check for server at {url}")

        print(f"Sending prompt to {url}: {prompt}")
        await send_chat_request(session=session, messages=messages, api_key=api_key)


async def send_chat_request(
    session: aiohttp.ClientSession,
    messages: list[dict[str, str]],
    api_key: str,
) -> None:
    payload: dict[str, Any] = {
        "model": CONFIG["served_model_name"],
        "messages": messages,
        "stream": True,
        "temperature": CONFIG["temperature"],
        "top_p": CONFIG["top_p"],
        "max_tokens": CONFIG["max_tokens"],
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "Authorization": f"Bearer {api_key}",
    }

    async with session.post(
        url="/v1/chat/completions",
        json=payload,
        headers=headers,
    ) as resp:
        async for raw in resp.content:
            resp.raise_for_status()
            line = raw.decode().strip()
            if not line or line == "data: [DONE]":
                continue
            if line.startswith("data: "):
                line = line[len("data: ") :]

            chunk = json.loads(line)
            content = chunk["choices"][0]["delta"].get("content")
            if content:
                print(content, end="")
    print()
