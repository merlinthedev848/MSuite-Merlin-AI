#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================================="
echo "  Setting up Aether Offline AI Assistant & Coder"
echo "  Target: Raspberry Pi 5 / Linux Server (ARM64 / x86_64)"
echo "=========================================================="

echo "[1/4] Installing system build prerequisites..."
sudo apt update
sudo apt install -y build-essential cmake git curl python3-pip python3-venv

echo "[2/4] Setting up Python virtual environment..."
python3 -m venv "$DIR/venv"
source "$DIR/venv/bin/activate"
pip install --upgrade pip

echo "[3/4] Initializing memory database..."
python3 "$DIR/memory.py"

echo "[4/4] Making helper scripts executable..."
chmod +x "$DIR/download_model.sh" "$DIR/start_llama_server.sh" "$DIR/cron_worker.py"

echo ""
echo "=========================================================="
echo "  Setup Complete!"
echo "=========================================================="
echo "Next steps:"
echo "1. Download the offline model weights (~1GB):"
echo "   ./download_model.sh"
echo ""
echo "2. Start the local network assistant server:"
echo "   ./venv/bin/python server.py"
echo ""
echo "3. Open your browser on any phone or laptop on your Wi-Fi:"
echo "   http://<YOUR_PI_IP>:5000"
echo "=========================================================="
