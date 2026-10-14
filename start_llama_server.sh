#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL=$(find "$DIR/models" -name "*.gguf" | head -n 1)

if [ -z "$MODEL" ]; then
    echo "[!] No .gguf model found in $DIR/models/"
    echo "    Please run ./download_model.sh first."
    exit 1
fi

echo "[*] Found model: $MODEL"

# Find llama-server binary in common local locations or PATH
SERVER_BIN=""
if command -v llama-server &> /dev/null; then
    SERVER_BIN="llama-server"
elif [ -f "$DIR/llama.cpp/build/bin/llama-server" ]; then
    SERVER_BIN="$DIR/llama.cpp/build/bin/llama-server"
elif [ -f "$HOME/llama.cpp/build/bin/llama-server" ]; then
    SERVER_BIN="$HOME/llama.cpp/build/bin/llama-server"
fi

if [ -z "$SERVER_BIN" ]; then
    echo "[!] llama-server binary not found."
    echo "    Building llama.cpp from source (fast ARM NEON compilation)..."
    git clone https://github.com/ggerganov/llama.cpp.git "$DIR/llama.cpp"
    cd "$DIR/llama.cpp"
    cmake -B build -DGGML_NATIVE=ON
    cmake --build build --config Release -j4
    SERVER_BIN="$DIR/llama.cpp/build/bin/llama-server"
    cd "$DIR"
fi

echo "[*] Launching C++ llama-server on port 8080 (4 threads)..."
exec "$SERVER_BIN" \
    -m "$MODEL" \
    --host 127.0.0.1 \
    --port 8080 \
    -c 4096 \
    -t 4
