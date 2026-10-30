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
    echo "--- 💻 CODING & SOFTWARE ENGINEERING ---"
    echo "1) openbmb/UltraData-Stack-v3-L2 (Default - Curated Code Stack)"
    echo "2) openbmb/UltraData-SFT-Agent-2609:Code-Agent (Agentic Coding & Tool Use)"
    echo "3) sahil2801/CodeAlpaca-20k (20k Coding Tasks & Solutions)"
    echo "4) iamtarun/python_code_instructions_18k_alpaca (18k Python Problems)"
    echo ""
    echo "--- 🤖 PERSONAL ASSISTANT (PA) & TASKS ---"
    echo "5) databricks/databricks-dolly-15k (Human PA: Scheduling, QA, Brainstorming)"
    echo "6) teknium/OpenHermes-2.5 (Gold-Standard General Assistant)"
    echo "7) openbmb/UltraData-SFT-Agent-2609:General-Agent (Multi-Step Task Planning)"
    echo ""
    echo "--- 📄 DOCUMENTS, SUMMARIZATION & KNOWLEDGE ---"
    echo "8) knkarthick/dialogsum (Document, Meeting & Dialogue Summarization)"
    echo "9) samsum (Notes, Action Items & Message Summarization)"
    echo "10) openbmb/UltraData-SFT-Agent-2609:Search-Agent (Document Search & Retrieval)"
    echo ""
    echo "--- 💬 NATURAL CONVERSATION & ALIGNMENT ---"
    echo "11) HuggingFaceH4/ultrachat_200k (High-Quality Multi-Turn Conversation)"
    echo "12) Anthropic/hh-rlhf (Helpful & Harmless Dialogue Alignment)"
    echo ""
    echo "--- 🌟 COMPLETE PACKAGES ---"
    echo "13) All-in-One General AI Starter Pack (Downloads curated samples from all 4 pillars)"
    echo "14) Custom Hugging Face Repository"
    echo ""
    read -p "Select option (default: 1): " choice
    choice=${choice:-1}

    if [ "$choice" == "2" ]; then
        DATASET="openbmb/UltraData-SFT-Agent-2609:Code-Agent"
    elif [ "$choice" == "3" ]; then
        DATASET="sahil2801/CodeAlpaca-20k"
    elif [ "$choice" == "4" ]; then
        DATASET="iamtarun/python_code_instructions_18k_alpaca"
    elif [ "$choice" == "5" ]; then
        DATASET="databricks/databricks-dolly-15k"
    elif [ "$choice" == "6" ]; then
        DATASET="teknium/OpenHermes-2.5"
    elif [ "$choice" == "7" ]; then
        DATASET="openbmb/UltraData-SFT-Agent-2609:General-Agent"
    elif [ "$choice" == "8" ]; then
        DATASET="knkarthick/dialogsum"
    elif [ "$choice" == "9" ]; then
        DATASET="samsum"
    elif [ "$choice" == "10" ]; then
        DATASET="openbmb/UltraData-SFT-Agent-2609:Search-Agent"
    elif [ "$choice" == "11" ]; then
        DATASET="HuggingFaceH4/ultrachat_200k"
    elif [ "$choice" == "12" ]; then
        DATASET="Anthropic/hh-rlhf"
    elif [ "$choice" == "13" ]; then
        DATASET="all_in_one"
    elif [ "$choice" == "14" ]; then
        read -p "Enter Hugging Face repo ID: " DATASET
    else
        DATASET="openbmb/UltraData-Stack-v3-L2"
    fi

    if [ "$DATASET" != "all_in_one" ]; then
        read -p "Number of samples to cache locally (0 = Unlimited / Full Dataset, default: 0): " SAMPLES
        SAMPLES=${SAMPLES:-0}
    fi
fi

if [ "$DATASET" == "all_in_one" ] || [ "$DATASET" == "__all__" ] || [ "$DATASET" == "all" ]; then
    echo ""
    echo "=========================================================="
    echo "[*] Downloading & Assimilating ALL Curated Datasets..."
    echo "=========================================================="
    "$PYTHON_BIN" "$DIR/import_dataset.py" "__all__" --limit "${SAMPLES:-0}" --download-local
    echo ""
    echo "[✓] All Curated Datasets downloaded and indexed into SQLite memory!"
    exit 0
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
