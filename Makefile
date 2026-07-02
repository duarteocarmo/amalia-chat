default: help

.PHONY: help
help: # Show help for each of the Makefile recipes.
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | sort | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done

.PHONY: install
install: # Install dependencies with uv
	uv sync

.PHONY: format
format: # Format the codebase with ruff
	uv run ruff check . --fix
	uv run ruff format .

.PHONY: check
check: # Run linting and checks
	uv lock --check
	uv run ruff check .
	uv run ruff format --check .
	uv run python -m py_compile amalia_vllm_modal.py web_app.py

.PHONY: lint
lint: check

.PHONY: clean
clean: # Clean up temporary files
	@rm -rf .ipynb_checkpoints
	@rm -rf **/.ipynb_checkpoints
	@rm -rf .pytest_cache
	@rm -rf **/.pytest_cache
	@rm -rf __pycache__
	@rm -rf **/__pycache__
	@rm -rf .ruff_cache
	@rm -rf build
	@rm -rf dist

.PHONY: deploy
deploy: # Deploy the Modal API with VLLM_API_KEY and HF_TOKEN from env vars
	@test -n "$${VLLM_API_KEY}" || (echo "Set VLLM_API_KEY in your environment"; exit 1)
	@test -n "$${HF_TOKEN}" || (echo "Set HF_TOKEN in your environment"; exit 1)
	uv run modal deploy amalia_vllm_modal.py

.PHONY: chat
chat: # Run the local FastAPI chat UI
	@test -n "$${VLLM_API_KEY}" || (echo "Set VLLM_API_KEY in your environment"; exit 1)
	uv run uvicorn web_app:app --host 0.0.0.0 --port 8080 --reload

.PHONY: test
test: # Run Modal smoke test using VLLM_API_KEY env var
	@test -n "$${VLLM_API_KEY}" || (echo "Set VLLM_API_KEY in your environment"; exit 1)
	uv run modal run amalia_vllm_modal.py --api-key "$${VLLM_API_KEY}"
