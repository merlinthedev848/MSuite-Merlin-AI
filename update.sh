#!/bin/bash
set -e
echo "=========================================="
echo "Merlin AI Autonomous Engine - Update Suite"
echo "=========================================="
ARCH=$(uname -m)
echo "[1/3] Detected architecture: $ARCH"

if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    echo "[2/3] Installing native ARM64 daemon binary..."
    cp -f merlin-server-linux-arm64 merlin-server
    chmod +x merlin-server
elif [ "$ARCH" = "x86_64" ] || [ "$ARCH" = "amd64" ]; then
    echo "[2/3] Installing native AMD64 daemon binary..."
    cp -f merlin-server-linux-amd64 merlin-server
    chmod +x merlin-server
else
    echo "[2/3] Architecture $ARCH. Compiling from source..."
    if command -v go >/dev/null 2>&1; then
        go build -o merlin-server .
    fi
fi

echo "[3/3] Restarting daemon..."
if systemctl is-active --quiet merlin 2>/dev/null; then
    sudo systemctl restart merlin
    echo "Merlin systemd service successfully restarted."
elif systemctl is-active --quiet offline-assistant 2>/dev/null; then
    sudo systemctl restart offline-assistant
    echo "Offline-assistant systemd service successfully restarted."
else
    echo "Please restart your service or run ./merlin-server"
fi
echo "=========================================="
echo "Update complete! Live assets active."
echo "=========================================="
