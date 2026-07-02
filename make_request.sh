#!/usr/bin/env bash
set -euo pipefail

API_URL="${API_URL:-https://duarteocarmo--amalia-vllm-gguf-api-serve.modal.run}"
MODEL="${MODEL:-amalia}"
PROMPT="${1:-Capital de Portugal?}"

if [[ -z "${VLLM_API_KEY:-}" ]]; then
  echo "Set VLLM_API_KEY in your environment" >&2
  exit 1
fi

curl -fsS -L "${API_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${VLLM_API_KEY}" \
  -d @- <<EOF
{
  "model": "${MODEL}",
  "messages": [
    {
      "role": "user",
      "content": "${PROMPT}"
    }
  ],
  "temperature": 0.2,
  "max_tokens": 256
}
EOF

echo
