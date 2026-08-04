#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/agentic-crag/app}"

if [[ ! -d "$APP_DIR/.git" ]]; then
  echo "$APP_DIR is not a Git checkout."
  exit 1
fi

cd "$APP_DIR"

echo "Pulling latest code..."
git pull --ff-only

echo "Updating Python dependencies..."
./venv/bin/pip install -r requirements.txt

echo "Refreshing service and Nginx templates..."
sudo cp deploy/agentic-crag-api.service /etc/systemd/system/
sudo cp deploy/nginx-agentic-crag.conf /etc/nginx/sites-available/agentic-crag
sudo ln -sf /etc/nginx/sites-available/agentic-crag /etc/nginx/sites-enabled/agentic-crag
sudo rm -f /etc/nginx/sites-enabled/default

sudo nginx -t
sudo systemctl daemon-reload
sudo systemctl restart agentic-crag-api
sudo systemctl reload nginx

echo "Update complete."
echo "Check health with:"
echo "  curl http://127.0.0.1:8000/health"
echo "  curl http://127.0.0.1/health"
