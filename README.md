# AMALIA vLLM Modal API

OpenAI-compatible Modal API for `duarteocarmo/AMALIA-9B-0626-SFT-GGUF:Q4_K_M`.

Notes from the vLLM docs:

- GGUF support is experimental.
- Current vLLM GGUF support requires `vllm-gguf-plugin`.
- vLLM can load Hugging Face GGUF quants with `repo_id:quant_type`.
- vLLM uses `HF_TOKEN` for Hugging Face Hub downloads.
- vLLM recommends using the base HF tokenizer/config instead of converting the GGUF tokenizer.

## Environment

Modal auth uses:

```bash
export MODAL_TOKEN_ID="..."
export MODAL_TOKEN_SECRET="..."
```

Deployment injects these local env vars into the Modal app programmatically with `modal.Secret.from_local_environ`:

```bash
export VLLM_API_KEY="..." # bearer token for the deployed API
export HF_TOKEN="..." # Hugging Face token for model/tokenizer downloads
```

## Deploy

```bash
make deploy
```

## Test

```bash
make test
```

## Local chat UI

Run the small FastAPI proxy and nanochat-style frontend:

```bash
make chat
```

Open <http://localhost:8080>. The proxy reads `VLLM_API_KEY` from your environment and rate-limits chat requests to 15 per IP per hour.

## Docker

```bash
docker build -t amalia-chat .
docker run --rm -p 8080:8080 -e VLLM_API_KEY="$VLLM_API_KEY" amalia-chat
```

The deployed API serves OpenAI-compatible chat completions at:

```text
/v1/chat/completions
```

Use model name `amalia`.
