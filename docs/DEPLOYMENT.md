# AWS deployment

The API can run on one Ubuntu EC2 instance. FastAPI, document parsing,
embeddings, persisted vector indexes, retrieval, and the default Ollama
`llama3.2:3b` model all run on the instance. Groq remains optional; when
enabled, questions and retrieved excerpts leave the instance.

## Architecture

```text
public :80 -> Nginx -> FastAPI 127.0.0.1:8000
                         |
                         +-> Ollama 127.0.0.1:11434
                         +-> /opt/agentic-crag/data
```

The deployment uses `agentic-crag-api.service`, Nginx, and Ollama. It does not install a web UI or automatic retention service.

## Before deploying

- Open SSH port `22` only to your IP and HTTP port `80` as narrowly as your use case permits.
- Use an instance with enough RAM for Ollama, the embedding model, and the re-ranker.
- Set an AWS billing alert and stop the instance when it is not needed.
- Use only public/demo documents if you enable Groq.
- Do not expose this minimal API to untrusted users without adding authentication and rate limiting.

## Scripted setup

```bash
sudo apt update
sudo apt install -y git
sudo mkdir -p /opt/agentic-crag
sudo chown -R ubuntu:ubuntu /opt/agentic-crag
git clone https://github.com/faisalcn24/agentic-crag.git /opt/agentic-crag/app
cd /opt/agentic-crag/app
bash deploy/setup_ec2.sh
```

The script installs Python, Nginx, Git, Ollama, and `llama3.2:3b`; creates the virtual environment; installs dependencies; and enables the API and Nginx services.

Review `/opt/agentic-crag/.env` after setup:

```env
AGENTIC_CRAG_LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:3b
OLLAMA_PLANNER_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CONTEXT_WINDOW=8192
AGENTIC_CRAG_STORAGE_DIR=/opt/agentic-crag/data
AGENTIC_CRAG_MAX_UPLOAD_MB=10
```

To use Groq, change `AGENTIC_CRAG_LLM_PROVIDER` to `groq`, set `GROQ_API_KEY`, and restart `agentic-crag-api`.

## Manual service installation

After cloning the repository, creating `venv`, installing `requirements.txt`, and creating `/opt/agentic-crag/.env`:

```bash
sudo cp deploy/agentic-crag-api.service /etc/systemd/system/
sudo cp deploy/nginx-agentic-crag.conf /etc/nginx/sites-available/agentic-crag
sudo ln -sf /etc/nginx/sites-available/agentic-crag /etc/nginx/sites-enabled/agentic-crag
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl daemon-reload
sudo systemctl enable --now ollama agentic-crag-api nginx
```

## Smoke test

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1/health
```

Open `http://<ec2-public-ip>/` for the graphical chat page or
`http://<ec2-public-ip>/docs` for FastAPI's interactive API documentation.

## Operations

```bash
sudo systemctl status ollama agentic-crag-api nginx
sudo journalctl -u agentic-crag-api -f
sudo systemctl restart agentic-crag-api
cd /opt/agentic-crag/app
bash deploy/update_app.sh
```

## API endpoints

- `GET /health`
- `GET /indexes`
- `POST /indexes` with multipart `files` and optional `index_id`
- `DELETE /indexes/{index_id}`
- `POST /chat` with `index_id`, `message`, and optional `history`
- `POST /agent` with `index_id`, `message`, and optional `history`
- `POST /retrieve` with `index_id`, `query`, and `top_k`
- `GET /metrics`

Uploads are restricted to PDF, DOCX, XLSX, PNG, JPEG, TIFF, BMP, and WebP with a
10 MB combined request limit by default. OCR and hybrid BM25/vector retrieval
run locally on the instance. Uploaded files and indexes persist until explicitly
deleted through the API or removed by an operator.
