#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================================="
echo "  Setting up Merlin AI Offline Assistant & Coder"
echo "  Target: Debian / Ubuntu Linux Server (x86_64 / ARM64)"
echo "=========================================================="

echo "[1/5] Installing system prerequisites..."
if command -v sudo &> /dev/null; then
    sudo apt update
    sudo apt install -y git curl zstd python3-pip python3-venv
else
    apt update
    apt install -y git curl zstd python3-pip python3-venv
fi

echo "[2/5] Setting up Python virtual environment..."
python3 -m venv "$DIR/venv"
source "$DIR/venv/bin/activate"
pip install --upgrade pip
pip install -r "$DIR/requirements.txt"

echo "[3/5] Initializing SQLite memory database..."
python3 "$DIR/database.py"

echo "[4/5] Installing Ollama & pulling Qwen2.5-Coder-1.5B..."
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
fi
ollama pull qwen2.5-coder:1.5b || true

echo "[5/5] Installing and starting systemd service..."
chmod +x "$DIR/download_model.sh" "$DIR/download_dataset.sh" "$DIR/start_llama_server.sh" "$DIR/cron_worker.py" "$DIR/import_dataset.py"

SERVICE_FILE="/etc/systemd/system/offline-assistant.service"
cat <<EOF > "$SERVICE_FILE"
[Unit]
Description=Merlin AI Offline Personal Assistant & Coder
After=network.target ollama.service

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$DIR
ExecStart=$DIR/venv/bin/python $DIR/main.py
Restart=always
RestartSec=5
Environment=HOST=0.0.0.0
Environment=PORT=5000
Environment=PYTHONUNBUFFERED=1
Environment=N_CTX=2048
Environment=CONTEXT_BUDGET=1000
Environment=MAX_TOKENS=512
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable offline-assistant.service
systemctl restart offline-assistant.service

echo ""
echo "=========================================================="
echo "  Setup Complete! Merlin AI is LIVE on port 5000."
echo "=========================================================="
echo "Check status:  systemctl status offline-assistant.service"
echo "Health check:  curl http://127.0.0.1:5000/health"
echo "=========================================================="

