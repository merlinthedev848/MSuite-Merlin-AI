#!/usr/bin/env bash
# ==============================================================================
# Merlin AI — Automated Clean Uninstallation & Native Server Installation Script
# ==============================================================================
set -e

echo "=================================================================="
echo "  Merlin AI: Clean System Wipe & Native Linux Server Install      "
echo "=================================================================="

# 1. Require Root / Sudo privileges
if [ "$EUID" -ne 0 ]; then
  echo "❌ Error: Please run this script with sudo or as root:"
  echo "   sudo ./install-clean.sh"
  exit 1
fi

TARGET_USER="${SUDO_USER:-$USER}"
INSTALL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo "▶ Target User: $TARGET_USER"
echo "▶ Working Directory: $INSTALL_DIR"

# 2. Stop and disable any running instances
echo ""
echo "▶ 1. Stopping any existing services and lingering processes..."
systemctl stop offline-assistant 2>/dev/null || true
systemctl disable offline-assistant 2>/dev/null || true

# Kill any processes still bound to port 5000
fuser -k 5000/tcp 2>/dev/null || true
sleep 1

# 3. Clean up old Python virtual environments, build artifacts and bytecode
echo "▶ 2. Cleaning legacy Python caches and virtual environments..."
rm -rf "$INSTALL_DIR/venv" "$INSTALL_DIR/.venv" "$INSTALL_DIR/__pycache__"
find "$INSTALL_DIR" -name "*.pyc" -delete 2>/dev/null || true

# 4. Detect Architecture and Select Binary
echo "▶ 3. Detecting Linux hardware architecture..."
ARCH=$(uname -m)
BINARY_SOURCE=""

if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ] || [ "$ARCH" = "armv7l" ]; then
  echo "   Detected ARM64 / Raspberry Pi architecture ($ARCH)"
  BINARY_SOURCE="$INSTALL_DIR/merlin-server-linux-arm64"
else
  echo "   Detected x86_64 / AMD64 architecture ($ARCH)"
  BINARY_SOURCE="$INSTALL_DIR/merlin-server-linux-amd64"
fi

if [ ! -f "$BINARY_SOURCE" ]; then
  echo "   Pre-compiled binary not found, attempting local Go build..."
  if command -v go &>/dev/null; then
    CGO_ENABLED=0 go build -ldflags="-s -w" -o "$INSTALL_DIR/merlin-server" "$INSTALL_DIR"
    BINARY_SOURCE="$INSTALL_DIR/merlin-server"
  else
    echo "❌ Error: Binary $BINARY_SOURCE not found and 'go' compiler is not installed."
    exit 1
  fi
fi

# 5. Install standalone binary to /usr/local/bin
echo "▶ 4. Deploying compiled binary to /usr/local/bin/merlin-server..."
cp -f "$BINARY_SOURCE" /usr/local/bin/merlin-server
chmod +x /usr/local/bin/merlin-server

# 6. Configure systemd unit file
echo "▶ 5. Installing systemd service unit..."
SERVICE_FILE="/etc/systemd/system/offline-assistant.service"

cat <<EOF > "$SERVICE_FILE"
[Unit]
Description=Merlin AI High-Performance Native Linux Daemon
After=network.target

[Service]
Type=simple
User=$TARGET_USER
WorkingDirectory=$INSTALL_DIR
ExecStart=/usr/local/bin/merlin-server
Restart=always
RestartSec=3
Environment=HOST=0.0.0.0
Environment=PORT=5000
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
EOF

# 7. Reload systemd and launch
echo "▶ 6. Reloading systemd daemon and enabling service..."
systemctl daemon-reload
systemctl enable offline-assistant
systemctl restart offline-assistant

sleep 2

# 8. Verification
echo ""
echo "=================================================================="
if systemctl is-active --quiet offline-assistant; then
  echo "✅ Merlin AI is ONLINE and running at native speed!"
  echo "   Dashboard: http://$(hostname -I | awk '{print $1}'):5000"
  echo "   Service Status: active (running)"
  echo "   View live logs: journalctl -u offline-assistant -f"
else
  echo "⚠️ Service did not report active status. Checking journal logs:"
  journalctl -u offline-assistant -n 20 --no-pager
fi
echo "=================================================================="
