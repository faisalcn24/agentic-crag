@echo off
echo Starting Document Analysis RAG (local Ollama mode)...
echo.

REM Use the local Ollama model for answer generation (overrides .env for this session)
set INSIGHT_LLM_PROVIDER=ollama
set OLLAMA_MODEL=llama3.2:3b
set OLLAMA_PLANNER_MODEL=llama3.2:3b

REM Start Ollama in the background if it is not already running
start "" /B ollama serve >nul 2>&1

REM Give Ollama a moment to come up
timeout /t 3 /nobreak >nul

REM Run the API in this window
py -3.13 -m uvicorn functions.api:app --host 127.0.0.1 --port 8000
