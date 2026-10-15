#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$DIR/venv/bin/python"

if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

if [ -n "$1" ]; then
    DATASET="$1"
    SAMPLES="${2:-200}"
else
    echo "=========================================================="
    echo "  Merlin AI — Local Hugging Face Dataset Downloader"
    echo "=========================================================="
    echo "Choose dataset to download and cache locally:"
    echo "1) openbmb/UltraData-Stack-v3-L2 (Default - High-Yield Data Stack)"
    echo "2) Anthropic/hh-rlhf             (Helpful & Harmless Alignment)"
    echo "3) sahil2801/CodeAlpaca-20k       (20k Coding Tasks & Solutions)"
    echo "4) iamtarun/python_code_instructions_18k_alpaca (18k Python Problems)"
    echo "5) Custom Repository ID"
    echo ""
    read -p "Select option (default: 1): " choice
    choice=${choice:-1}

    if [ "$choice" == "2" ]; then
        DATASET="Anthropic/hh-rlhf"
    elif [ "$choice" == "3" ]; then
        DATASET="sahil2801/CodeAlpaca-20k"
    elif [ "$choice" == "4" ]; then
        DATASET="iamtarun/python_code_instructions_18k_alpaca"
    elif [ "$choice" == "5" ]; then
        read -p "Enter Hugging Face repo ID: " DATASET
    else
        DATASET="openbmb/UltraData-Stack-v3-L2"
    fi

    read -p "Number of samples to cache locally (default: 200): " SAMPLES
    SAMPLES=${SAMPLES:-200}
fi

echo ""
echo "=========================================================="
echo "[*] Dataset:      $DATASET"
echo "[*] Samples:      $SAMPLES"
echo "[*] Local Disk:   $DIR/data/datasets"
echo "=========================================================="

"$PYTHON_BIN" "$DIR/import_dataset.py" "$DATASET" --limit "$SAMPLES" --download-local

echo ""
echo "=========================================================="
echo "[✓] Dataset assimilated and saved locally to disk."
echo "    Merlin can now load and query this dataset 100% offline!"
echo "=========================================================="
