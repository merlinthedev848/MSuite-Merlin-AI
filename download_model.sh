#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$DIR/models"
mkdir -p "$MODELS_DIR"

echo "=========================================================="
echo "  Merlin AI — Download Offline Models (GGUF Q4_K_M)"
echo "=========================================================="
echo "--- 💻 CODING SPECIALISTS (Qwen2.5-Coder) ---"
echo "1) Qwen2.5-Coder-1.5B-Instruct   (~1.0 GB) [Recommended balance for Pi 5 & CPU]"
echo "2) Qwen2.5-Coder-0.5B-Instruct   (~380 MB) [FASTEST - 30+ tokens/sec on CPU]"
echo "3) Qwen2.5-Coder-7B-Instruct     (~4.7 GB) [Flagship Coder, 8GB+ RAM / GPU]"
echo "4) Qwen2.5-Coder-14B-Instruct    (~9.0 GB) [Frontier Coder - Rivals GPT-4o]"
echo ""
echo "--- 🧠 DEEP REASONING & THINKING (DeepSeek-R1) ---"
echo "5) DeepSeek-R1-Distill-Qwen-1.5B (~1.1 GB) [Fast reasoning & chain-of-thought]"
echo "6) DeepSeek-R1-Distill-Qwen-7B   (~4.7 GB) [Deep reasoning & complex math/logic]"
echo "7) DeepSeek-R1-Distill-Qwen-14B  (~9.0 GB) [Elite reasoning benchmark leader]"
echo ""
echo "--- 🤖 GENERAL ASSISTANTS (Meta Llama & Phi-4) ---"
echo "8) Llama-3.2-3B-Instruct         (~2.0 GB) [Meta's fast lightweight assistant]"
echo "9) Meta-Llama-3.1-8B-Instruct    (~4.9 GB) [Meta's industry standard 8B powerhouse]"
echo "10) Microsoft Phi-4-14B          (~9.1 GB) [Complex analytical reasoning]"
echo ""
read -p "Choose model (default: 1): " choice
choice=${choice:-1}

if [ "$choice" == "2" ]; then
    URL="https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"
    OUTFILE="$MODELS_DIR/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"
elif [ "$choice" == "3" ]; then
    URL="https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf"
    OUTFILE="$MODELS_DIR/qwen2.5-coder-7b-instruct-q4_k_m.gguf"
elif [ "$choice" == "4" ]; then
    URL="https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct-GGUF/resolve/main/qwen2.5-coder-14b-instruct-q4_k_m.gguf"
    OUTFILE="$MODELS_DIR/qwen2.5-coder-14b-instruct-q4_k_m.gguf"
elif [ "$choice" == "5" ]; then
    URL="https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"
    OUTFILE="$MODELS_DIR/deepseek-r1-distill-qwen-1.5b-q4_k_m.gguf"
elif [ "$choice" == "6" ]; then
    URL="https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"
    OUTFILE="$MODELS_DIR/deepseek-r1-distill-qwen-7b-q4_k_m.gguf"
elif [ "$choice" == "7" ]; then
    URL="https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf"
    OUTFILE="$MODELS_DIR/deepseek-r1-distill-qwen-14b-q4_k_m.gguf"
elif [ "$choice" == "8" ]; then
    URL="https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    OUTFILE="$MODELS_DIR/llama-3.2-3b-instruct-q4_k_m.gguf"
elif [ "$choice" == "9" ]; then
    URL="https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    OUTFILE="$MODELS_DIR/meta-llama-3.1-8b-instruct-q4_k_m.gguf"
elif [ "$choice" == "10" ]; then
    URL="https://huggingface.co/unsloth/phi-4-GGUF/resolve/main/phi-4-Q4_K_M.gguf"
    OUTFILE="$MODELS_DIR/phi-4-14b-q4_k_m.gguf"
else
    URL="https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
    OUTFILE="$MODELS_DIR/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
fi

echo "[*] Downloading weights to $OUTFILE..."
curl -L -C - --retry 3 --retry-delay 2 --progress-bar -o "$OUTFILE" "$URL"

echo ""
echo "[✓] Model downloaded successfully!"
echo "You can now start the assistant offline."
