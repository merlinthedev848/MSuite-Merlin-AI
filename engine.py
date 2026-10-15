import os
import json
import re
import urllib.request
import urllib.error
import threading
from memory import search_knowledge, get_recent_chat, save_chat_message, log_event

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DEFAULT_LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080")

def get_optimal_threads():
    if "THREADS" in os.environ:
        try:
            return int(os.environ["THREADS"])
        except ValueError:
            pass
    cpu_cores = os.cpu_count() or 4
    # On systems with <= 4 cores (such as Raspberry Pi 4/5 or quad-core cloud VMs),
    # use all cores so llama.cpp runs at full hardware capacity.
    if cpu_cores <= 4:
        return max(1, cpu_cores)
    return min(8, max(2, cpu_cores - 1))

class InferenceEngine:
    def __init__(self):
        self.llama_cpp_model = None
        self.backend = None
        self.active_model_name = "None"
        # Concurrency Lock: Prevents simultaneous access to the CPU inference engine
        self._inference_lock = threading.Lock()
        self._active_abort_events = {} # session_id -> threading.Event()
        self.refresh_backend()

    def refresh_backend(self):
        """Checks for llama-server or local GGUF models."""
        # 1. Probe C++ llama-server
        try:
            req = urllib.request.Request(f"{DEFAULT_LLAMA_SERVER_URL}/health", headers={"User-Agent": "Merlin-AI/1.0"})
            with urllib.request.urlopen(req, timeout=1.0) as res:
                if res.status == 200:
                    self.backend = "llama-server"
                    self.active_model_name = "Qwen2.5-Coder via llama-server"
                    log_event("engine_init", f"Connected to C++ llama-server at {DEFAULT_LLAMA_SERVER_URL}", "success")
                    return
        except Exception:
            pass

        # 2. Check for local GGUF and llama_cpp
        if os.path.exists(MODELS_DIR):
            models = [f for f in os.listdir(MODELS_DIR) if f.endswith(".gguf")]
            if models:
                self.active_model_name = models[0]
                try:
                    from llama_cpp import Llama
                    model_path = os.path.join(MODELS_DIR, models[0])
                    threads = get_optimal_threads()
                    n_ctx = int(os.getenv("N_CTX", "2048"))
                    n_gpu = int(os.getenv("N_GPU_LAYERS", "0"))
                    self.llama_cpp_model = Llama(
                        model_path=model_path,
                        n_ctx=n_ctx,
                        n_threads=threads,
                        n_threads_batch=threads,
                        n_batch=512,
                        n_gpu_layers=n_gpu,
                        verbose=False
                    )
                    self.backend = "llama-cpp-python"
                    log_event("engine_init", f"Loaded local model {models[0]} with {threads} threads, ctx {n_ctx}", "success")
                    return
                except ImportError:
                    pass
                except Exception as e:
                    log_event("engine_error", f"Failed loading GGUF: {e}", "danger")

        self.backend = "offline_standby"
        self.active_model_name = "Waiting for model download or llama-server"

    def get_available_models(self):
        """Returns a list of all downloaded models with metadata."""
        models = []
        if os.path.exists(MODELS_DIR):
            for f in sorted(os.listdir(MODELS_DIR)):
                if f.endswith(".gguf"):
                    path = os.path.join(MODELS_DIR, f)
                    size_mb = round(os.path.getsize(path) / (1024 * 1024), 1)
                    models.append({
                        "filename": f,
                        "name": f.replace(".gguf", ""),
                        "size_mb": size_mb,
                        "is_active": (f == self.active_model_name)
                    })
        return models

    def set_active_model(self, model_filename):
        """Switches the active GGUF model in memory."""
        with self._inference_lock:
            model_path = os.path.join(MODELS_DIR, model_filename)
            if not os.path.exists(model_path):
                return False, f"Model file '{model_filename}' not found."
            
            try:
                from llama_cpp import Llama
                threads = get_optimal_threads()
                n_ctx = int(os.getenv("N_CTX", "2048"))
                n_gpu = int(os.getenv("N_GPU_LAYERS", "0"))
                self.llama_cpp_model = None
                self.llama_cpp_model = Llama(
                    model_path=model_path,
                    n_ctx=n_ctx,
                    n_threads=threads,
                    n_threads_batch=threads,
                    n_batch=512,
                    n_gpu_layers=n_gpu,
                    verbose=False
                )
                self.active_model_name = model_filename
                self.backend = "llama-cpp-python"
                log_event("model_switch", f"Switched active model to {model_filename}", "success")
                return True, f"Active model switched to {model_filename}"
            except Exception as e:
                log_event("model_switch_error", f"Failed switching to {model_filename}: {e}", "danger")
                return False, str(e)

    def stop_generation(self, session_id):
        """Signals an active stream to abort immediately."""
        if session_id in self._active_abort_events:
            self._active_abort_events[session_id].set()
            return True
        return False

    def build_system_prompt(self, mode="assistant"):
        if mode in ("agent", "antigravity"):
            return (
                "You are Merlin (Merlin AI), an autonomous agentic pair-programming assistant inspired by Google DeepMind's Antigravity. "
                "You reason deeply step-by-step through technical problems, plan carefully before executing, and write clean, modular, production-ready code. "
                "When addressing tasks: break down complex requirements, think through edge cases and architecture, provide clear terminal commands or diffs, "
                "and explain your reasoning with precision."
            )
        elif mode == "coder":
            return (
                "You are Merlin (Merlin AI), an expert offline software engineer and coding assistant running locally on a private server. "
                "Your name is Merlin. You react and respond naturally when addressed as Merlin or Merlin AI. "
                "Provide clean, efficient, production-ready code. Always consider memory limits, edge cases, and algorithmic complexity. "
                "Keep prose explanations concise and accurate."
            )
        else:
            return (
                "You are Merlin (Merlin AI), an intelligent offline Personal Assistant (PA) and coding companion running locally on the user's private network. "
                "Your name is Merlin. You react and respond naturally when addressed as Merlin or Merlin AI. "
                "You assist with daily tasks, server diagnostics, scheduling, and programming questions. "
                "Be helpful, direct, and concise."
            )

    def _stream_from_llama_server(self, payload_dict, abort_event, timeout=120):
        """
        Streams SSE tokens from llama-server unbuffered so each token reaches the client immediately.
        """
        if HAS_REQUESTS:
            try:
                with requests.post(
                    f"{DEFAULT_LLAMA_SERVER_URL}/v1/chat/completions",
                    json=payload_dict,
                    headers={"Content-Type": "application/json", "User-Agent": "Merlin-AI/1.0"},
                    stream=True,
                    timeout=timeout
                ) as r:
                    r.raise_for_status()
                    for line in r.iter_lines(decode_unicode=True, chunk_size=512):
                        if abort_event.is_set():
                            break
                        if not line:
                            continue
                        if line.startswith("data: "):
                            data_content = line[6:].strip()
                            if data_content == "[DONE]":
                                break
                            try:
                                chunk_json = json.loads(data_content)
                                delta = chunk_json["choices"][0]["delta"].get("content", "")
                                if delta:
                                    yield delta
                            except Exception:
                                continue
                return
            except Exception as e:
                if not abort_event.is_set():
                    yield f"[Error querying llama-server via requests: {e}]"
                return

        # Fallback to standard library urllib with unbuffered chunk parsing (avoids 8KB buffer delay)
        payload = json.dumps(payload_dict).encode("utf-8")
        req = urllib.request.Request(
            f"{DEFAULT_LLAMA_SERVER_URL}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "Merlin-AI/1.0"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                buf = b""
                while not abort_event.is_set():
                    chunk = resp.read(64)
                    if not chunk:
                        if buf:
                            line_str = buf.decode("utf-8", errors="replace").strip()
                            if line_str.startswith("data: ") and line_str[6:].strip() != "[DONE]":
                                try:
                                    chunk_json = json.loads(line_str[6:].strip())
                                    delta = chunk_json["choices"][0]["delta"].get("content", "")
                                    if delta:
                                        yield delta
                                except Exception:
                                    pass
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        line_str = line.decode("utf-8", errors="replace").strip()
                        if not line_str:
                            continue
                        if line_str.startswith("data: "):
                            data_content = line_str[6:].strip()
                            if data_content == "[DONE]":
                                return
                            try:
                                chunk_json = json.loads(data_content)
                                delta = chunk_json["choices"][0]["delta"].get("content", "")
                                if delta:
                                    yield delta
                            except Exception:
                                continue
        except Exception as e:
            if not abort_event.is_set():
                yield f"[Error querying llama-server: {e}]"

    def generate_stream(self, prompt, session_id="default", mode="assistant"):
        """
        Thread-safe token generator.
        Injects past learned knowledge and recent history into the prompt.
        Respects abort signals if user cancels request.
        """
        abort_event = threading.Event()
        self._active_abort_events[session_id] = abort_event

        # Save user message to persistent history
        save_chat_message(session_id, "user", prompt)

        # Detect greetings and short conversational cues
        cleaned = prompt.strip().lower()
        is_greeting = cleaned in {
            "hello", "hi", "hey", "merlin", "hello merlin", "hi merlin", "hey merlin",
            "good morning", "good afternoon", "good evening", "how are you", "who are you",
            "yo", "sup", "what's up", "whats up"
        } or (len(cleaned.split()) <= 2 and any(w in cleaned for w in ["hi", "hello", "hey", "merlin", "yo", "sup"]))

        # Retrieve relevant learned skills from SQLite (skip on greetings to eliminate latency and prefill bloat)
        learned_context = [] if is_greeting else search_knowledge(prompt, limit=2)
        memory_prompt = ""
        if learned_context:
            memory_prompt = "\n[Verified lessons learned from past teacher sessions]:\n"
            for k in learned_context:
                memory_prompt += f"- Topic: {k['topic']}\n  Lesson: {k['lesson']}\n"
                if k['code_example']:
                    memory_prompt += f"  Reference Code:\n{k['code_example']}\n"

        # Assemble conversation context (strip prior internal <think>...</think> from history to keep prefill instantaneous)
        history = get_recent_chat(session_id, limit=6)
        messages = [{"role": "system", "content": self.build_system_prompt(mode) + memory_prompt}]
        for h in history[:-1]:
            content = h["content"]
            if h["role"] == "assistant":
                content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
            if content:
                messages.append({"role": h["role"], "content": content})
        messages.append({"role": "user", "content": prompt})

        full_response = ""
        max_tokens = 90 if is_greeting else int(os.getenv("MAX_TOKENS", "1024"))
        stop_tokens = ["<|im_end|>", "<|endoftext|>", "</s>"]

        # Acquire lock to ensure only one thread generates on the CPU at a time
        acquired = self._inference_lock.acquire(timeout=25.0)
        if not acquired:
            msg = "[Server Busy: Another generation is currently executing. Please wait a moment.]"
            yield msg
            self._active_abort_events.pop(session_id, None)
            return

        try:
            # Re-probe if in standby mode
            if self.backend == "offline_standby":
                self.refresh_backend()

            if self.backend == "llama-server":
                payload_dict = {
                    "messages": messages,
                    "temperature": 0.2 if mode == "coder" else 0.7,
                    "stream": True,
                    "max_tokens": max_tokens,
                    "stop": stop_tokens
                }
                for chunk in self._stream_from_llama_server(payload_dict, abort_event):
                    if abort_event.is_set():
                        break
                    full_response += chunk
                    yield chunk

            elif self.backend == "llama-cpp-python" and self.llama_cpp_model:
                try:
                    stream = self.llama_cpp_model.create_chat_completion(
                        messages=messages,
                        temperature=0.2 if mode == "coder" else 0.7,
                        max_tokens=max_tokens,
                        stream=True,
                        stop=stop_tokens
                    )
                    for chunk in stream:
                        if abort_event.is_set():
                            break
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            full_response += delta
                            yield delta
                except Exception as e:
                    err = f"[Inference error: {e}]"
                    full_response += err
                    yield err

            else:
                if is_greeting:
                    msg = (
                        "Hello! I am **Merlin** (Merlin AI), your offline companion and private software assistant.\n\n"
                        "I am running locally on your private network. Currently, no local GGUF model is loaded into memory.\n\n"
                        "**To activate local AI generation:**\n"
                        "1. Run `./download_model.sh` to download Qwen2.5-Coder or DeepSeek-R1.\n"
                        "2. Start `./start_llama_server.sh` or install `llama-cpp-python`.\n\n"
                        "Once started, I will automatically detect the model!"
                    )
                else:
                    msg = (
                        "⚠️ **Offline Intelligence Standby**\n\n"
                        "The local web dashboard and SQLite memory system are fully active.\n\n"
                        "**To activate the local AI model:**\n"
                        "1. Run `./download_model.sh` to download Qwen2.5-Coder (GGUF format).\n"
                        "2. Start `./start_llama_server.sh` OR install `pip install llama-cpp-python`.\n"
                        "Once started, the engine automatically detects the model with zero restart needed."
                    )
                full_response = msg
                yield msg

        finally:
            self._inference_lock.release()
            self._active_abort_events.pop(session_id, None)
            if full_response:
                save_chat_message(session_id, "assistant", full_response)
