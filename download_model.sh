#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$DIR/models"
mkdir -p "$MODELS_DIR"

echo "=========================================================="
echo "  Downloading Lightweight Offline Coding LLM for Pi 5"
echo "=========================================================="
echo "1) Qwen2.5-Coder-1.5B-Instruct (Q4_K_M) - ~1.0 GB [RECOMMENDED for Pi 5 & Server]"
echo "2) Qwen2.5-Coder-0.5B-Instruct (Q4_K_M) - ~380 MB [Ultra-lightweight / Fast]"
echo "3) Qwen2.5-Coder-3B-Instruct (Q4_K_M)   - ~2.1 GB [High reasoning, 4GB/8GB Pi]"
echo ""
read -p "Choose model (default: 1): " choice
choice=${choice:-1}

if [ "$choice" == "2" ]; then
    URL="https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"
    OUTFILE="$MODELS_DIR/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"
elif [ "$choice" == "3" ]; then
    URL="https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF/resolve/main/qwen2.5-coder-3b-instruct-q4_k_m.gguf"
    OUTFILE="$MODELS_DIR/qwen2.5-coder-3b-instruct-q4_k_m.gguf"
else
    URL="https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
    OUTFILE="$MODELS_DIR/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
fi

echo "[*] Downloading weights to $OUTFILE..."
curl -L -C - -o "$OUTFILE" "$URL"

echo ""
echo "[✓] Model downloaded successfully!"
echo "You can now start the assistant offline."
