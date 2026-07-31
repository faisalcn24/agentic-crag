# Document Analysis RAG on AWS

The API can run on one Ubuntu EC2 instance. FastAPI, document parsing, embeddings, persisted vector indexes, retrieval, and the default Ollama `llama3.2:3b` model all run on the instance. Groq remains an optional provider; when enabled, retrieved excerpts leave the instance.

## Architecture

```text
public :80 -> Nginx -> FastAPI 127.0.0.1:8000
                         |
                         +-> Ollama 127.0.0.1:11434
                         +-> /opt/insight-ai/data
```

The deployment uses `insight-api.service`, Nginx, and Ollama. It does not install a web UI or automatic retention service.

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
sudo mkdir -p /opt/insight-ai
sudo chown -R ubuntu:ubuntu /opt/insight-ai
git clone https://github.com/faisalcn24/insight-ai-2.git /opt/insight-ai/app
cd /opt/insight-ai/app
bash deploy/setup_ec2.sh
```

The script installs Python, Nginx, Git, Ollama, and `llama3.2:3b`; creates the virtual environment; installs dependencies; and enables the API and Nginx services.

Review `/opt/insight-ai/.env` after setup:

```env
INSIGHT_LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:3b
OLLAMA_PLANNER_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CONTEXT_WINDOW=8192
INSIGHT_STORAGE_DIR=/opt/insight-ai/data
INSIGHT_MAX_UPLOAD_MB=10
```

To use Groq, change `INSIGHT_LLM_PROVIDER` to `groq`, set `GROQ_API_KEY`, and restart `insight-api`.

## Manual service installation

After cloning the repository, creating `venv`, installing `requirements.txt`, and creating `/opt/insight-ai/.env`:

```bash
sudo cp deploy/insight-api.service /etc/systemd/system/
sudo cp deploy/nginx-insight-ai.conf /etc/nginx/sites-available/insight-ai
sudo ln -sf /etc/nginx/sites-available/insight-ai /etc/nginx/sites-enabled/insight-ai
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl daemon-reload
sudo systemctl enable --now ollama insight-api nginx
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
sudo systemctl status ollama insight-api nginx
sudo journalctl -u insight-api -f
sudo systemctl restart insight-api
cd /opt/insight-ai/app
bash deploy/update_app.sh
```

The update script also disables and removes legacy `insight-ui` and `insight-retention` unit files from older deployments.

## API endpoints

- `GET /health`
- `GET /indexes`
- `POST /indexes` with multipart `files` and optional `index_id`
- `DELETE /indexes/{index_id}`
- `POST /chat` with `index_id`, `message`, and optional `history`
- `POST /agent` with `index_id`, `message`, and optional `history`
- `POST /retrieve` with `index_id`, `query`, and `top_k`
- `GET /metrics`

Uploads are restricted to PDF, DOCX, and XLSX with a 10 MB combined request limit by default. Uploaded files and indexes persist until explicitly deleted through the API or removed by an operator.
