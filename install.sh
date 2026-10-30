#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUDO=""
if [ "$(id -u)" -ne 0 ] && command -v sudo &> /dev/null; then
    SUDO="sudo"
fi

echo "=========================================================="
echo "  Merlin AI — Full System Bootstrap, Intel iGPU & AI Setup"
echo "  Target: Debian 12/13 (Proxmox LXC / VM with Intel GPU)"
echo "=========================================================="

echo "[1/6] Running full system update & installing core default tools..."
export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get update -y
$SUDO apt-get full-upgrade -y
$SUDO apt-get install -y \
    git curl wget ca-certificates gnupg lsb-release apt-transport-https \
    build-essential cmake pkg-config jq unzip zip zstd nano vim htop \
    net-tools iproute2 dnsutils lsof pciutils usbutils \
    python3 python3-pip python3-venv python3-dev

echo "[2/6] Installing Intel iGPU (DRI / Vulkan / OpenCL / VAAPI) drivers & tools..."
$SUDO apt-get install -y \
    mesa-vulkan-drivers \
    vulkan-tools \
    intel-opencl-icd \
    clinfo \
    vainfo \
     mesa-va-drivers || true

# Ensure render/video groups exist and current user is a member for /dev/dri access
$SUDO groupadd -f render || true
$SUDO groupadd -f video || true
$SUDO usermod -aG render,video "$(whoami)" || true

if [ -e /dev/dri/renderD128 ]; then
    echo "  [✓] Detected GPU render node: /dev/dri/renderD128"
    $SUDO chmod 666 /dev/dri/renderD128 /dev/dri/card* 2>/dev/null || true
else
    echo "  [!] Note: /dev/dri/renderD128 not detected yet. Pass /dev/dri through in Proxmox LXC Resources -> Add -> Device Passthrough."
fi

echo "[3/6] Setting up Python virtual environment & AI dependencies..."
python3 -m venv "$DIR/venv"
source "$DIR/venv/bin/activate"
pip install --upgrade pip wheel setuptools
pip install -r "$DIR/requirements.txt"

echo "[4/6] Initializing SQLite FTS5 memory database..."
python3 "$DIR/database.py"

echo "[5/6] Installing Ollama with Intel iGPU (Vulkan) acceleration..."
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
fi

# Configure Ollama systemd override to enable Intel iGPU Vulkan acceleration and keep model warm in memory
$SUDO mkdir -p /etc/systemd/system/ollama.service.d
cat << 'EOF' | $SUDO tee /etc/systemd/system/ollama.service.d/intel-gpu.conf > /dev/null
[Service]
Environment="OLLAMA_HOST=127.0.0.1:11434"
Environment="OLLAMA_VULKAN=1"
Environment="OLLAMA_KEEP_ALIVE=24h"
Environment="OLLAMA_NUM_PARALLEL=2"
Environment="OLLAMA_FLASH_ATTENTION=1"
SupplementaryGroups=video render
EOF

$SUDO systemctl daemon-reload
$SUDO systemctl enable ollama
$SUDO systemctl restart ollama

# Wait for Ollama daemon to accept connections
echo "  [*] Waiting for Ollama daemon to start..."
for i in {1..15}; do
    if curl -s http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

echo "  [*] Pulling Qwen2.5-Coder-1.5B model..."
ollama pull qwen2.5-coder:1.5b

echo "[6/6] Installing and starting Merlin AI systemd service..."
chmod +x "$DIR/download_model.sh" "$DIR/download_dataset.sh" "$DIR/start_llama_server.sh" "$DIR/cron_worker.py" "$DIR/import_dataset.py"

SERVICE_FILE="/etc/systemd/system/offline-assistant.service"
cat <<EOF | $SUDO tee "$SERVICE_FILE" > /dev/null
[Unit]
Description=Merlin AI Offline Personal Assistant & Coder
After=network.target ollama.service
Wants=ollama.service

[Service]
Type=simple
User=$(whoami)
SupplementaryGroups=video render
WorkingDirectory=$DIR
ExecStart=$DIR/venv/bin/python $DIR/main.py
Restart=always
RestartSec=5
Environment=HOST=0.0.0.0
Environment=PORT=5000
Environment=PYTHONUNBUFFERED=1
Environment=OLLAMA_KEEP_ALIVE=24h
Environment=N_CTX=4096
Environment=CONTEXT_BUDGET=1500
Environment=MAX_TOKENS=1024
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
EOF

$SUDO systemctl daemon-reload
$SUDO systemctl enable offline-assistant.service
$SUDO systemctl restart offline-assistant.service

echo ""
echo "=========================================================="
echo "  Setup Complete! Merlin AI + Intel GPU Acceleration Ready"
echo "=========================================================="
echo "  GPU Render Devices : $(ls /dev/dri/ 2>/dev/null | tr '\n' ' ' || echo 'None')"
echo "  Service Status     : systemctl status offline-assistant.service"
echo "  Ollama GPU Status  : ollama ps"
echo "  Health Check       : curl http://127.0.0.1:5000/health"
echo "=========================================================="


