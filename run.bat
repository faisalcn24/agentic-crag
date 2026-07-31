@echo off
setlocal
echo Starting Document Analysis RAG...
echo.

REM Prefer the repository virtual environment created by the setup instructions.
pushd "%~dp0"
if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" -m uvicorn functions.api:app --host 127.0.0.1 --port 8000
) else (
    py -3.13 -m uvicorn functions.api:app --host 127.0.0.1 --port 8000
)
set exit_code=%errorlevel%
popd
exit /b %exit_code%
