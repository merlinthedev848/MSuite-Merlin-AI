#!/usr/bin/env bash
# ==============================================================================
# Merlin AI — Complete Clean Wipe & Full Native System Installer
# One-command script: Removes all old files/processes, installs dependencies,
# and configures the ultra-fast native Linux server.
# ==============================================================================
set -e

# Color helpers
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${CYAN}${BOLD}"
echo "=================================================================="
echo "    MERLIN AI — COMPLETE CLEAN WIPE & NATIVE LINUX INSTALLER     "
echo "=================================================================="
echo -e "${NC}"

# Check for root/sudo
if [ "$EUID" -ne 0 ]; then
  echo -e "${RED}❌ Error: This setup script must be run as root or with sudo.${NC}"
  echo -e "   Please run: ${YELLOW}sudo ./setup.sh${NC}\n"
  exit 1
fi

REAL_USER="${SUDO_USER:-$USER}"
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "▶ Running as user: ${BOLD}$REAL_USER${NC}"
echo -e "▶ Installation directory: ${BOLD}$INSTALL_DIR${NC}\n"

# ==============================================================================
# PHASE 1: COMPLETELY REMOVE EVERYTHING OLD FIRST
# ==============================================================================
echo -e "${YELLOW}------------------------------------------------------------------${NC}"
echo -e "${YELLOW}PHASE 1: COMPLETELY REMOVING ALL LEGACY FILES & OLD SERVICES...${NC}"
echo -e "${YELLOW}------------------------------------------------------------------${NC}"

# 1. Stop and disable all old systemd services
echo "1. Stopping and disabling legacy services..."
systemctl stop offline-assistant 2>/dev/null || true
systemctl disable offline-assistant 2>/dev/null || true
systemctl stop merlin 2>/dev/null || true
systemctl disable merlin 2>/dev/null || true

# 2. Kill any processes bound to port 5000
echo "2. Killing any lingering processes on port 5000..."
if command -v fuser &>/dev/null; then
  fuser -k 5000/tcp 2>/dev/null || true
elif command -v lsof &>/dev/null; then
  lsof -ti:5000 | xargs kill -9 2>/dev/null || true
fi
sleep 1

# 3. Clean up legacy Python environments, caches and bytecode
echo "3. Removing old virtual environments, pip packages, and Python bytecode..."
rm -rf "$INSTALL_DIR/venv" "$INSTALL_DIR/.venv" "$INSTALL_DIR/__pycache__"
find "$INSTALL_DIR" -name "*.pyc" -delete 2>/dev/null || true
find "$INSTALL_DIR" -name "*.pyo" -delete 2>/dev/null || true

# 4. Remove old systemd unit files
echo "4. Removing old service units..."
rm -f /etc/systemd/system/offline-assistant.service
rm -f /etc/systemd/system/merlin.service
rm -f /usr/local/bin/merlin-server
systemctl daemon-reload
echo -e "${GREEN}✓ Complete clean wipe successful!${NC}\n"

# ==============================================================================
# PHASE 2: INSTALL ALL SYSTEM DEPENDENCIES
# ==============================================================================
echo -e "${YELLOW}------------------------------------------------------------------${NC}"
echo -e "${YELLOW}PHASE 2: INSTALLING SYSTEM DEPENDENCIES & AI RUNTIME...${NC}"
echo -e "${YELLOW}------------------------------------------------------------------${NC}"

if command -v apt-get &>/dev/null; then
  echo "Updating package repositories and installing core utilities..."
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y
  apt-get install -y curl git psmisc ca-certificates jq iproute2
elif command -v dnf &>/dev/null; then
  dnf install -y curl git psmisc ca-certificates jq iproute
elif command -v yum &>/dev/null; then
  yum install -y curl git psmisc ca-certificates jq iproute
elif command -v pacman &>/dev/null; then
  pacman -Sy --noconfirm curl git psmisc ca-certificates jq iproute2
fi

# Check for Ollama (The high-speed C++/Go AI engine)
if ! command -v ollama &>/dev/null; then
  echo "Installing Ollama AI engine..."
  curl -fsSL https://ollama.com/install.sh | sh
else
  echo -e "${GREEN}✓ Ollama AI runtime already installed.${NC}"
fi

# Ensure Ollama daemon is running
systemctl enable --now ollama 2>/dev/null || true
sleep 2

# Background pull of lightweight model so it's ready immediately
echo "Ensuring AI model (qwen2.5-coder:1.5b) is available..."
(ollama pull qwen2.5-coder:1.5b >/dev/null 2>&1 &) || true

echo -e "${GREEN}✓ All dependencies installed and active.${NC}\n"

# ==============================================================================
# PHASE 3: DEPLOY THE NEW NATIVE COMPILED SERVER
# ==============================================================================
echo -e "${YELLOW}------------------------------------------------------------------${NC}"
echo -e "${YELLOW}PHASE 3: DEPLOYING NEW NATIVE LINUX SERVER BINARY...${NC}"
echo -e "${YELLOW}------------------------------------------------------------------${NC}"

ARCH=$(uname -m)
echo "Detected Linux CPU Architecture: $ARCH"

CHOSEN_BINARY=""
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ] || [ "$ARCH" = "armv7l" ]; then
  CHOSEN_BINARY="$INSTALL_DIR/merlin-server-linux-arm64"
else
  CHOSEN_BINARY="$INSTALL_DIR/merlin-server-linux-amd64"
fi

if [ -f "$CHOSEN_BINARY" ]; then
  echo "Deploying pre-compiled static binary: $(basename "$CHOSEN_BINARY")..."
  cp -f "$CHOSEN_BINARY" /usr/local/bin/merlin-server
else
  echo "Pre-compiled binary not found for $ARCH, compiling locally with Go..."
  if ! command -v go &>/dev/null; then
    echo "Installing Go compiler..."
    apt-get install -y golang || true
  fi
  CGO_ENABLED=0 go build -ldflags="-s -w" -o /usr/local/bin/merlin-server "$INSTALL_DIR"
fi

chmod +x /usr/local/bin/merlin-server
echo -e "${GREEN}✓ Native binary installed to /usr/local/bin/merlin-server${NC}\n"

# ==============================================================================
# PHASE 4: CONFIGURE & LAUNCH SYSTEMD DAEMON
# ==============================================================================
echo -e "${YELLOW}------------------------------------------------------------------${NC}"
echo -e "${YELLOW}PHASE 4: CONFIGURING & STARTING LINUX DAEMON...${NC}"
echo -e "${YELLOW}------------------------------------------------------------------${NC}"

cat <<EOF > /etc/systemd/system/merlin.service
[Unit]
Description=Merlin AI High-Performance Native Linux Daemon
After=network.target ollama.service

[Service]
Type=simple
User=$REAL_USER
WorkingDirectory=$INSTALL_DIR
ExecStart=/usr/local/bin/merlin-server
Restart=always
RestartSec=3
Environment=HOST=0.0.0.0
Environment=PORT=5000
Environment=DATABASE_PATH=$INSTALL_DIR/assistant_memory.db
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
EOF

# Symlink legacy name so both commands work seamlessly
ln -sf /etc/systemd/system/merlin.service /etc/systemd/system/offline-assistant.service

systemctl daemon-reload
systemctl enable merlin
systemctl restart merlin

# ==============================================================================
# PHASE 5: VERIFICATION
# ==============================================================================
echo -e "Waiting for Merlin server to initialize..."
HEALTHY=false
for i in {1..10}; do
  if curl -s http://127.0.0.1:5000/api/health | grep -q "healthy"; then
    HEALTHY=true
    break
  fi
  sleep 1
done

IP_ADDR=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "127.0.0.1")

echo ""
echo -e "${GREEN}${BOLD}==================================================================${NC}"
if [ "$HEALTHY" = true ]; then
  echo -e "${GREEN}${BOLD}  🚀 MERLIN AI IS ONLINE AND FULLY OPERATIONAL!                  ${NC}"
  echo -e "${GREEN}${BOLD}==================================================================${NC}"
  echo -e "  ▶ Browser URL:      ${CYAN}${BOLD}http://${IP_ADDR}:5000${NC}"
  echo -e "  ▶ Localhost:        ${CYAN}http://localhost:5000${NC}"
  echo -e "  ▶ Service Name:     ${BOLD}merlin${NC} (and offline-assistant)"
  echo -e "  ▶ Check Status:     ${YELLOW}sudo systemctl status merlin${NC}"
  echo -e "  ▶ Stream Live Logs: ${YELLOW}sudo journalctl -u merlin -f${NC}"
else
  echo -e "${YELLOW}  ⚠️ Service started. Check status if the dashboard doesn't open:${NC}"
  echo -e "  ▶ Check Status:     ${YELLOW}sudo systemctl status merlin${NC}"
  echo -e "  ▶ Stream Live Logs: ${YELLOW}sudo journalctl -u merlin -f${NC}"
fi
echo -e "${GREEN}${BOLD}==================================================================${NC}\n"
