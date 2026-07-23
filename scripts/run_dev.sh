#!/usr/bin/env bash
set -euo pipefail

export INSIGHT_LLM_PROVIDER="${INSIGHT_LLM_PROVIDER:-ollama}"
export OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2:3b}"
export OLLAMA_PLANNER_MODEL="${OLLAMA_PLANNER_MODEL:-llama3.2:3b}"

exec python3 -m uvicorn functions.api:app --host 127.0.0.1 --port 8000
