# Aether — Offline AI Assistant & Self-Learning Coder

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
sudo apt update && sudo apt install -y git curl build-essential cmake python3 python3-pip python3-venv && git clone https://github.com/merlinthedev848/merlin-ai.git && cd merlin-ai && chmod +x install.sh download_model.sh start_llama_server.sh cron_worker.py && ./install.sh && ./venv/bin/pip install llama-cpp-python requests && mkdir -p models && curl -L -C - -o models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf && sudo sed -i "s|User=pi|User=$USER|g; s|/home/pi/offline-ai-assistant|$PWD|g" offline-assistant.service && sudo cp offline-assistant.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable --now offline-assistant.service && echo -e "\n\n[SUCCESS] Merlin-AI is running! Access it at: http://$(hostname -I | awk '{print $1}'):5000\n"
```

---

## Manual Step-by-Step Installation

### 1. Transfer or Clone to Pi / Server
Copy this folder to your Pi or Linux machine:
```bash
scp -r offline-ai-assistant pi@<PI_IP>:~/offline-ai-assistant
```
Or run directly inside the directory:
```bash
cd offline-ai-assistant
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
./venv/bin/python server.py
```
Open your browser on any phone, tablet, or PC on your local Wi-Fi:
```
http://<YOUR_PI_IP>:5000
```
*(If testing on the same machine: `http://localhost:5000`)*

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
2. **Plugging into Commercial AI**:
   - When you have internet access, click **"⚡ Sync Teacher"** in the web UI (or run `python sync_teacher.py <API_KEY>`).
   - The assistant contacts the commercial AI (OpenAI, Gemini, OpenRouter), asks its queued questions, validates the syntax of the returned code, and saves the verified lesson into the `knowledge` table.
   - It also appends the training pair to `training_dataset.jsonl`.
3. **Automatic Recall**:
   - The next time you are offline and ask about that topic, the assistant retrieves the distilled lesson from its local database and applies it to your solution.

---

## CLI Mode

For SSH terminal sessions without a browser:
```bash
./venv/bin/python cli.py
```
Commands:
- `:tasks` — View active to-do list
- `:addtask <task>` — Add a new task
- `:skills` — Inspect learned knowledge
- `:gap <question>` — Record a knowledge gap
- `:sys` — View live CPU temp, RAM, and load
- `:toggle` — Switch between Coder and PA mode
