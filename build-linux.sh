#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "  Merlin AI — High-Performance Native Linux Server Build  "
echo "=========================================================="

CGO_ENABLED=0 go build -ldflags="-s -w" -o merlin-server .
chmod +x merlin-server

echo "✅ Compiled standalone binary: ./merlin-server"
echo "To start immediately: ./merlin-server"
echo "To install systemd service:"
echo "  sudo cp offline-assistant.service /etc/systemd/system/"
echo "  sudo systemctl daemon-reload"
echo "  sudo systemctl restart offline-assistant"
