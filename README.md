# Merlin AI — Offline AI Assistant & Self-Learning Coder

An air-gapped, ultra-lightweight AI Personal Assistant (PA) and coding assistant engineered specifically for **Raspberry Pi 5** and low-resource **Linux servers**.

- **100% Offline & Private**: Zero cloud dependencies, zero external telemetry.
- **Minimal Footprint**: Operates on CPU (ARM NEON / AVX2) using ~1.0 GB to 1.5 GB RAM.
- **Local Network Access**: Connect from any device on your Wi-Fi/LAN (phone, tablet, laptop) via a fast, self-contained web app.
- **Personal Assistant (PA)**: Persistent To-Dos, quick technical notes, and live hardware health diagnostics (temperature, CPU load, memory).
- **Self-Learning Distillation Engine**: When connected to the internet, it interviews a commercial "Teacher" AI (OpenAI, Gemini, OpenRouter) to answer questions it struggled with, validates the code, and permanently indexes the lessons in its local SQLite database.

---

## Architecture Overview

```
 ┌──────────────────────────────────────────────────────────────┐
 │             YOUR LOCAL NETWORK (WI-FI / LAN)                 │
 │                                                              │
 │   📱 Phone / 💻 Laptop Browser    VS Code (Continue ext)     │
 │       http://<pi_ip>:5000         http://<pi_ip>:8080        │
 └──────────────────────┬──────────────────────▲────────────────┘
                        │                      │
 ┌──────────────────────▼──────────────────────┴────────────────┐
 │                 RASPBERRY PI 5 / LINUX SERVER                │
 │                                                              │
 │   ┌──────────────────────────────────────────────────────┐   │
 │   │ server.py (Lightweight Web Server + SSE Streaming)   │   │
 │   └──────────────────────────┬───────────────────────────┘   │
 │                              │                               │
 │   ┌──────────────────────────┴───────────────────────────┐   │
 │   │ engine.py (llama-server / llama-cpp-python)          │   │
 │   │ Model: Qwen2.5-Coder-1.5B (GGUF Q4_K_M)              │   │
 │   └──────────▲───────────────────────────────▲───────────┘   │
 │              │                               │               │
 │   ┌──────────▼───────────┐       ┌───────────▼───────────┐   │
 │   │ assistant_memory.db  │       │ sync_teacher.py       │   │
 │   │ - Learned Skills     │◄──────┤ (Interviews Cloud AI) │   │
 │   │ - Knowledge Gaps     │       │                       │   │
 │   │ - To-Dos & Notes     │       │ Only when plugged into│   │
 │   └──────────────────────┘       │ Commercial AI API key │   │
 └──────────────────────────────────┴───────────┬───────────┘   │
                                                │               │
 ┌──────────────────────────────────────────────┴───────────────┐
 │ COMMERCIAL AI TEACHER (OpenAI / Claude / Gemini / OpenRouter)│
 └──────────────────────────────────────────────────────────────┘
```

---

## 🚀 1-Line Automated Install (Debian 12, Ubuntu, Raspberry Pi OS)

Run this single command on your server to install dependencies, clone the repo, download the coding model, and start the systemd daemon automatically:

```bash
sudo systemctl stop offline-assistant.service 2>/dev/null || true; cd ~ && rm -rf ~/merlin-ai && sudo apt update && sudo apt install -y git curl build-essential cmake python3 python3-pip python3-venv && git clone https://github.com/merlinthedev848/merlin-ai.git ~/merlin-ai && cd ~/merlin-ai && chmod +x *.sh cron_worker.py && ./install.sh && (echo 1 | ./download_model.sh) && sudo sed -i "s|User=pi|User=$USER|g; s|/home/pi/offline-ai-assistant|$PWD|g" offline-assistant.service && sudo cp offline-assistant.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable --now offline-assistant.service && echo -e "\n\n[SUCCESS] Merlin AI is running! Access it at: http://$(hostname -I | awk '{print $1}'):5000\n"
```

---

## Manual Step-by-Step Installation

### 1. Transfer or Clone to Pi / Server
Copy this folder to your Pi or Linux machine:
```bash
scp -r merlin-ai pi@<PI_IP>:~/merlin-ai
```
Or run directly inside the directory:
```bash
cd merlin-ai
```

### 2. Run the Installer
```bash
chmod +x install.sh
./install.sh
```

### 3. Download the 1.5B Coding Model
```bash
./download_model.sh
```
*(Select option 1 for Qwen2.5-Coder-1.5B-Instruct, which takes ~1.0 GB of storage and runs at ~12–18 tokens/sec on Pi 5).*

### 4. Start the Server
```bash
./venv/bin/python main.py
```
Open your browser on any phone, tablet, or PC on your local Wi-Fi:
```
http://<YOUR_PI_IP>:5000
```
*(If testing on the same machine: `http://localhost:5000`)*

---

## ⚡ Model Speed & Performance Tuning (Fixing Slow Response Times)

Offline LLM inference speed on CPU is strictly governed by **RAM memory bandwidth** ($GB/s$) and **prompt prefill length**. Here is how to achieve instant, responsive generation:

### 1. Select the Right Model for Your Hardware
Run `./download_model.sh` and select the appropriate model:
- **`Qwen2.5-Coder-0.5B-Instruct` (~380 MB)**: **FASTEST (35–50+ tokens/sec on CPU)**. Responds near-instantly with virtually zero prefill latency. Highly capable at scripts, syntax, terminal commands, and quick queries.
- **`Qwen2.5-Coder-1.5B-Instruct` (~1.0 GB)**: **BALANCED (15–22 tokens/sec on CPU)**. Recommended default for Raspberry Pi 5 and quad-core CPUs.
- **`7B` / `14B` Models**: **2–4 tokens/sec on CPU**. A 7B model requires reading ~4 GB of weights from RAM for *every single token*. On CPU DDR4/LPDDR4 RAM, this is memory-bus bottlenecked and will feel slow without a dedicated GPU.
- **`DeepSeek-R1`**: Spends 300–800 hidden tokens generating `<think>...</think>` chain-of-thought logic before outputting its answer. On a CPU at 10 tok/s, 500 thinking tokens take **50 seconds** before the first word appears.

### 2. Enable Native C++ Acceleration (`llama-server`)
Running the compiled C++ `llama-server` is **2x–4x faster** than in-process Python bindings:
```bash
./start_llama_server.sh
```
This script automatically compiles `llama.cpp` natively with `-DGGML_NATIVE=ON`, enables Flash-Attention (`--flash-attn`), activates 8-bit quantized KV cache (`-ctk q8_0 -ctv q8_0`), and binds to physical CPU cores. Merlin AI detects and routes traffic to it automatically.

### 3. CPU Context Window Budgeting
By default, Merlin AI tunes context prefill:
- `CONTEXT_BUDGET=1500`: Keeps conversational context around 1500 tokens so Time-To-First-Token (TTFT) stays under 1.5 seconds on CPU.
- `N_CTX=4096`: Limits total KV cache size to prevent RAM bandwidth saturation.

---

## 🌐 OpenAI-Compatible REST API

Merlin AI exposes a drop-in replacement for OpenAI endpoints (`POST /v1/chat/completions` and `GET /v1/models`), allowing external IDEs, extensions (Continue.dev, Cursor, Neovim), scripts, and workflows to connect offline with zero cloud dependency:

### 1. Streaming Chat Completion (`stream: true`)
```bash
curl http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "merlin-ai",
    "messages": [
      {"role": "system", "content": "You are an expert software engineer."},
      {"role": "user", "content": "Write an async Python socket server."}
    ],
    "stream": true
  }'
```

### 2. List Available Local Models
```bash
curl http://localhost:5000/v1/models
```

### 3. VS Code / Continue.dev Configuration
In `~/.continue/config.json`:
```json
{
  "models": [
    {
      "title": "Merlin AI (Offline)",
      "provider": "openai",
      "model": "merlin-ai",
      "apiBase": "http://<YOUR_SERVER_IP>:5000/v1"
    }
  ]
}
```

---

## Operating as a Background Service (`systemd`)

To run the assistant automatically on boot:

1. Copy the service unit:
   ```bash
   sudo cp offline-assistant.service /etc/systemd/system/
   ```
2. Adjust `User` and `WorkingDirectory` in `/etc/systemd/system/offline-assistant.service` if your username is not `pi`.
3. Enable and start:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable --now offline-assistant.service
   ```
The web dashboard will always be alive at `http://<YOUR_PI_IP>:5000`.

### Autonomous Background Worker (`systemd timer`)

To have the assistant periodically audit system health, prune stale history, and automatically resolve pending knowledge gaps whenever an internet connection is detected:

```bash
sudo cp offline-cron.service offline-cron.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now offline-cron.timer
```
*(Runs `cron_worker.py` every 15 minutes with atomic lock concurrency protection).*

---

## How the Self-Learning Loop Works

1. **Daily Offline Use**:
   - While coding or asking questions, if you notice the assistant lacks a concept, click **"Record a Knowledge Gap"** in the sidebar (or type `:gap <question>` in CLI).
   - The gap is recorded in `assistant_memory.db`.
2. **Teaching Merlin with Google Gemini (Free Tier)**:
   - Obtain a free API key from Google AI Studio: [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
   - **Via Web Dashboard**:
     - Click **"⚡ Sync Teacher"** in the Memory tab.
     - Select **Google Gemini** as Provider.
     - Choose your model (`gemini-2.0-flash` [Recommended], `gemini-1.5-flash`, or `gemini-1.5-pro`).
     - Paste your `AIza...` key (or leave empty if `GEMINI_API_KEY` is exported).
     - Enter an optional topic to distill immediately, or let it resolve all pending knowledge gaps.
     - Click **"Start Distillation"**.
   - **Via Terminal / CLI**:
     ```bash
     export GEMINI_API_KEY="AIzaSyYourGeminiApiKeyHere"
     python sync_teacher.py --topic "Building an async WebSocket server in Python"
     ```
     Or interactively in CLI mode:
     ```bash
     python cli.py
     # Type: :teach Building high throughput TCP servers
     ```
3. **Local Hugging Face Datasets (`openbmb/UltraData-Stack-v3-L2` by Default)**:
   - Merlin uses **`openbmb/UltraData-Stack-v3-L2`** by default for high-yield pre-training and instruction knowledge, cached locally on disk in `data/datasets/` for 100% offline access.
   - **Download & Cache Locally (One Command)**:
     ```bash
     ./download_dataset.sh
     ```
     *(Streams and saves `openbmb/UltraData-Stack-v3-L2` to local disk at `data/datasets/openbmb_UltraData-Stack-v3-L2`)*
   - **In Python Code**:
     ```python
     from import_dataset import load_local_dataset

     # 1. Default AGI Pretraining Stack
     ds_ultra = load_local_dataset("openbmb/UltraData-Stack-v3-L2")

     # 2. Agentic SFT Subsets (Pass HF_TOKEN if gated or authenticate via huggingface-cli login)
     ds_code_agent    = load_local_dataset("openbmb/UltraData-SFT-Agent-2609", subset="Code-Agent")
     ds_general_agent = load_local_dataset("openbmb/UltraData-SFT-Agent-2609", subset="General-Agent")
     ds_search_agent  = load_local_dataset("openbmb/UltraData-SFT-Agent-2609", subset="Search-Agent")

     # 3. Anthropic RLHF Alignment
     ds_rlhf = load_local_dataset("Anthropic/hh-rlhf")
     ```
   - **Via Web Dashboard**:
     - Click **"🤗 Import Dataset"** in the Memory & Skills tab.
     - Select any dataset or subset (e.g., `openbmb/UltraData-SFT-Agent-2609 [Code-Agent]`).
     - Enter your optional Hugging Face access token (`hf_...`) if accessing gated datasets.
     - Click **"Import & Assimilate"** to stream and save to disk.
   - **Via Interactive CLI**:
     ```bash
     python cli.py
     # Type: :dataset openbmb/UltraData-SFT-Agent-2609:Code-Agent 50
     ```
4. **Automatic Recall**:
   - The next time you are offline and ask about that topic, Merlin automatically retrieves the distilled lesson from its local database and incorporates it into your answer.

---

## CLI Mode

For SSH terminal sessions without a browser:
```bash
./venv/bin/python cli.py
```
Commands:
- `:teach <topic>` — Distill a topic directly from the Gemini Teacher
- `:dataset <repo> [N]` — Stream and assimilate Hugging Face datasets
- `:tasks` — View active to-do list
- `:addtask <task>` — Add a new task
- `:skills` — Inspect learned knowledge
- `:gap <question>` — Record a knowledge gap
- `:sys` — View live CPU temp, RAM, and load
- `:toggle` — Switch between Coder and PA mode
