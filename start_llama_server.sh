#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "$1" ]; then
    if [ -f "$1" ]; then
        MODEL="$1"
    elif [ -f "$DIR/models/$1" ]; then
        MODEL="$DIR/models/$1"
    else
        MODEL=$(find "$DIR/models" -iname "*$1*.gguf" | head -n 1)
    fi
else
    MODEL=$(find "$DIR/models" -name "*.gguf" | head -n 1)
fi

if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo "[!] No matching .gguf model found in $DIR/models/"
    echo "    Please run ./download_model.sh first."
    exit 1
fi

echo "[*] Active model: $(basename "$MODEL")"

# Find llama-server binary in common local locations or PATH
SERVER_BIN=""
if command -v llama-server &> /dev/null; then
    SERVER_BIN="llama-server"
elif [ -f "$DIR/llama.cpp/build/bin/llama-server" ]; then
    SERVER_BIN="$DIR/llama.cpp/build/bin/llama-server"
elif [ -f "$HOME/llama.cpp/build/bin/llama-server" ]; then
    SERVER_BIN="$HOME/llama.cpp/build/bin/llama-server"
fi

NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
# Optimal threads: use physical cores (NPROC/2 on hyperthreaded CPUs), capped at 6 to prevent L3 cache thrashing
if [ -z "$THREADS" ]; then
    if [ "$NPROC" -gt 4 ]; then
        THREADS=$((NPROC / 2))
        [ "$THREADS" -gt 6 ] && THREADS=6
    else
        THREADS=$NPROC
    fi
fi
CTX_SIZE=${CTX_SIZE:-2048}
N_GPU_LAYERS=${N_GPU_LAYERS:-0}

if [ "$N_GPU_LAYERS" == "0" ] && command -v nvidia-smi &> /dev/null; then
    N_GPU_LAYERS=99
fi

if [ -z "$SERVER_BIN" ]; then
    echo "[!] llama-server binary not found."
    if [ -d "$DIR/llama.cpp" ]; then
        echo "    Updating existing llama.cpp repository..."
        git -C "$DIR/llama.cpp" pull --ff-only || true
    else
        echo "    Cloning latest stable llama.cpp..."
        git clone https://github.com/ggerganov/llama.cpp.git "$DIR/llama.cpp"
    fi
    echo "    Building llama.cpp from source (fast native compilation)..."
    cd "$DIR/llama.cpp"
    cmake -B build
    cmake --build build --config Release -j"$NPROC"
    SERVER_BIN="$DIR/llama.cpp/build/bin/llama-server"
    cd "$DIR"
fi

echo "[*] Launching C++ llama-server on port 8080 ($THREADS threads, ctx $CTX_SIZE, GPU layers: $N_GPU_LAYERS)..."
exec "$SERVER_BIN" \
    -m "$MODEL" \
    --host 127.0.0.1 \
    --port 8080 \
    -c "$CTX_SIZE" \
    -t "$THREADS" \
    -b 512 \
    -ub 256 \
    -ngl "$N_GPU_LAYERS" \
    --cont-batching
