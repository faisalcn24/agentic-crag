#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="${APP_ROOT:-/opt/agentic-crag}"
APP_DIR="${APP_DIR:-$APP_ROOT/app}"
DATA_DIR="${DATA_DIR:-$APP_ROOT/data}"
ENV_FILE="${ENV_FILE:-$APP_ROOT/.env}"
REPO_URL="${REPO_URL:-https://github.com/faisalcn24/agentic-crag.git}"
APP_USER="${APP_USER:-ubuntu}"
SWAP_FILE="${SWAP_FILE:-/swapfile}"
SWAP_SIZE="${SWAP_SIZE:-2G}"

if [[ "$(id -u)" -eq 0 ]]; then
  echo "Run this script as the ubuntu user, not root. It will use sudo where needed."
  exit 1
fi

echo "Installing system packages..."
sudo apt update
sudo apt install -y python3 python3-venv nginx git curl

if ! command -v ollama >/dev/null 2>&1; then
  echo "Installing Ollama for the default local model..."
  curl -fsSL https://ollama.com/install.sh | sh
fi
sudo systemctl enable --now ollama
ollama pull "${OLLAMA_MODEL:-llama3.2:3b}"

if [[ ! -f "$SWAP_FILE" ]]; then
  echo "Creating $SWAP_SIZE swap file at $SWAP_FILE..."
  sudo fallocate -l "$SWAP_SIZE" "$SWAP_FILE"
  sudo chmod 600 "$SWAP_FILE"
  sudo mkswap "$SWAP_FILE"
  sudo swapon "$SWAP_FILE"
  echo "$SWAP_FILE none swap sw 0 0" | sudo tee -a /etc/fstab >/dev/null
else
  echo "Swap file already exists at $SWAP_FILE; leaving it unchanged."
fi

echo "Preparing app directories..."
sudo mkdir -p "$DATA_DIR" "$APP_DIR"
sudo chown -R "$APP_USER:$APP_USER" "$APP_ROOT"

if [[ -d "$APP_DIR/.git" ]]; then
  echo "Repository already exists. Pulling latest main..."
  git -C "$APP_DIR" pull --ff-only
else
  if [[ -n "$(find "$APP_DIR" -mindepth 1 -maxdepth 1 2>/dev/null)" ]]; then
    echo "$APP_DIR is not empty and is not a Git repo. Move or empty it before running setup."
    exit 1
  fi
  echo "Cloning $REPO_URL..."
  git clone "$REPO_URL" "$APP_DIR"
fi

cd "$APP_DIR"

echo "Creating Python virtual environment..."
python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Creating $ENV_FILE from .env.example..."
  cp .env.example "$ENV_FILE"
  sed -i "s|^AGENTIC_CRAG_STORAGE_DIR=.*|AGENTIC_CRAG_STORAGE_DIR=$DATA_DIR|" "$ENV_FILE"
  chmod 600 "$ENV_FILE"
  echo "Edit $ENV_FILE if you want to change the default local Ollama configuration."
else
  echo "$ENV_FILE already exists; leaving it unchanged."
fi

echo "Installing systemd and Nginx config..."
sudo cp deploy/agentic-crag-api.service /etc/systemd/system/
sudo cp deploy/nginx-agentic-crag.conf /etc/nginx/sites-available/agentic-crag
sudo ln -sf /etc/nginx/sites-available/agentic-crag /etc/nginx/sites-enabled/agentic-crag
sudo rm -f /etc/nginx/sites-enabled/default

sudo nginx -t
sudo systemctl daemon-reload
sudo systemctl enable --now agentic-crag-api nginx
sudo systemctl restart agentic-crag-api
sudo systemctl reload nginx

echo "Setup complete."
echo "Check health with:"
echo "  curl http://127.0.0.1:8000/health"
echo "  curl http://127.0.0.1/health"
echo "If chat fails, confirm Ollama is running and the model settings in $ENV_FILE are correct, then run:"
echo "  sudo systemctl restart agentic-crag-api"
