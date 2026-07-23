#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/insight-ai/app}"

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
sudo cp deploy/insight-api.service /etc/systemd/system/
sudo cp deploy/nginx-insight-ai.conf /etc/nginx/sites-available/insight-ai
sudo ln -sf /etc/nginx/sites-available/insight-ai /etc/nginx/sites-enabled/insight-ai
sudo rm -f /etc/nginx/sites-enabled/default

sudo nginx -t
sudo systemctl disable --now insight-ui.service insight-retention.timer 2>/dev/null || true
sudo rm -f /etc/systemd/system/insight-ui.service /etc/systemd/system/insight-retention.service /etc/systemd/system/insight-retention.timer
sudo systemctl daemon-reload
sudo systemctl restart insight-api
sudo systemctl reload nginx

echo "Update complete."
echo "Check health with:"
echo "  curl http://127.0.0.1:8000/health"
echo "  curl http://127.0.0.1/health"
