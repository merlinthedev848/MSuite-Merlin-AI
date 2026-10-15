import os
import json
import re
import urllib.request
import urllib.error
import threading
import time
import shutil
from memory import search_knowledge, get_recent_chat, save_chat_message, log_event, get_context_window

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
    # For hyperthreaded or multi-core CPUs (>4), running on physical cores prevents L1/L2/L3 cache line thrashing.
    physical_cores = max(2, cpu_cores // 2)
    return min(6, physical_cores)

def load_llama_cpp_instance(model_path, threads, n_ctx, n_gpu):
    """Instantiates Llama with Flash-Attention and 8-bit quantized KV cache for low RAM bandwidth."""
    from llama_cpp import Llama
    kwargs = {
        "model_path": model_path,
        "n_ctx": n_ctx,
        "n_threads": threads,
        "n_threads_batch": threads,
        "n_batch": 512,
        "n_gpu_layers": n_gpu,
        "verbose": False
    }
    try:
        kwargs["flash_attn"] = True
        kwargs["type_k"] = 2  # LLAMA_FTYPE_MOSTLY_Q8_0
        kwargs["type_v"] = 2  # LLAMA_FTYPE_MOSTLY_Q8_0
        return Llama(**kwargs)
    except Exception:
        kwargs.pop("flash_attn", None)
        kwargs.pop("type_k", None)
        kwargs.pop("type_v", None)
        return Llama(**kwargs)

def fit_prompt_to_context(prompt, max_chars=16000):
    """
    Prevents '[Inference error: Requested tokens exceed context window]'
    when large documents or files are attached by safely trimming large attachments
    while keeping instructions and code intact.
    """
    if not prompt or len(prompt) <= max_chars:
        return prompt

    if "--- [ATTACHED FILE:" in prompt and "--- [END OF" in prompt:
        try:
            prefix, rest = prompt.split("--- [ATTACHED FILE:", 1)
            file_header, rest2 = rest.split("] ---", 1)
            file_body, suffix = rest2.split("--- [END OF", 1)
            suffix = "--- [END OF" + suffix

            half_keep = max_chars // 2
            trimmed_body = (
                file_body[:half_keep]
                + f"\n\n... [Note: Truncated {len(file_body) - max_chars} characters to fit local memory context] ...\n\n"
                + file_body[-(max_chars // 4):]
            )
            return prefix + "--- [ATTACHED FILE:" + file_header + "] ---" + trimmed_body + suffix
        except Exception:
            pass

    half = max_chars // 2
    return prompt[:half] + f"\n\n... [Prompt trimmed to fit local context buffer] ...\n\n" + prompt[-half:]

class InferenceEngine:
    def __init__(self):
        self.llama_cpp_model = None
        self.backend = None
        self.active_model_name = "None"
        # Concurrency Lock: Prevents simultaneous access to the CPU inference engine
        self._inference_lock = threading.Lock()
        self._active_abort_events = {} # session_id -> threading.Event()
        self.refresh_backend()

    def _try_autostart_llama_server(self):
        """Attempts to auto-launch C++ llama-server if binary exists."""
        if not os.path.exists(MODELS_DIR):
            return False
        models = [f for f in os.listdir(MODELS_DIR) if f.endswith(".gguf")]
        if not models:
            return False
        model_path = os.path.join(MODELS_DIR, models[0])

        server_bin = None
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(base_dir, "llama.cpp", "build", "bin", "llama-server"),
            os.path.join(base_dir, "llama.cpp", "build", "bin", "llama-server.exe"),
            shutil.which("llama-server"),
            shutil.which("llama-server.exe")
        ]
        for c in candidates:
            if c and os.path.exists(c) and os.path.isfile(c):
                server_bin = c
                break

        if not server_bin:
            return False

        threads = get_optimal_threads()
        n_ctx = int(os.getenv("N_CTX", "4096"))
        n_gpu = int(os.getenv("N_GPU_LAYERS", "0"))
        cmd = [
            server_bin,
            "-m", model_path,
            "--host", "127.0.0.1",
            "--port", "8080",
            "-c", str(n_ctx),
            "-t", str(threads),
            "-b", "512",
            "-ub", "256",
            "--flash-attn",
            "-ctk", "q8_0",
            "-ctv", "q8_0",
            "--cont-batching"
        ]
        if n_gpu > 0:
            cmd.extend(["-ngl", str(n_gpu)])

        try:
            import subprocess
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for _ in range(8):
                time.sleep(0.5)
                try:
                    req = urllib.request.Request(f"{DEFAULT_LLAMA_SERVER_URL}/health", headers={"User-Agent": "Merlin-AI/1.0"})
                    with urllib.request.urlopen(req, timeout=0.5) as res:
                        if res.status == 200:
                            return True
                except Exception:
                    pass
        except Exception as e:
            log_event("autostart_error", f"Could not auto-start llama-server: {e}", "warning")
        return False

    def refresh_backend(self):
        """Checks for llama-server or local GGUF models."""
        # 1. Probe C++ llama-server (fastest, unbuffered native binary)
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

        # 2. If llama-server binary exists on disk, attempt auto-launch
        if self._try_autostart_llama_server():
            self.backend = "llama-server"
            self.active_model_name = "Qwen2.5-Coder via llama-server (autostarted)"
            log_event("engine_init", "Autostarted native C++ llama-server on port 8080", "success")
            return

        # 3. Fallback: Local GGUF via llama-cpp-python
        if os.path.exists(MODELS_DIR):
            models = [f for f in os.listdir(MODELS_DIR) if f.endswith(".gguf")]
            if models:
                self.active_model_name = models[0]
                try:
                    model_path = os.path.join(MODELS_DIR, models[0])
                    threads = get_optimal_threads()
                    n_ctx = int(os.getenv("N_CTX", "4096"))
                    n_gpu = int(os.getenv("N_GPU_LAYERS", "0"))
                    self.llama_cpp_model = load_llama_cpp_instance(model_path, threads, n_ctx, n_gpu)
                    self.backend = "llama-cpp-python"
                    log_event("engine_init", f"Loaded local model {models[0]} with {threads} threads, ctx {n_ctx} (Flash-Attn enabled)", "success")
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
                threads = get_optimal_threads()
                n_ctx = int(os.getenv("N_CTX", "4096"))
                n_gpu = int(os.getenv("N_GPU_LAYERS", "0"))
                self.llama_cpp_model = None
                self.llama_cpp_model = load_llama_cpp_instance(model_path, threads, n_ctx, n_gpu)
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

        # Assemble conversation context via intelligent sliding-window context manager
        # Capping context budget on CPU keeps prefill latency snappy (< 1-2s) instead of stalling for minutes
        context_budget = int(os.getenv("CONTEXT_BUDGET", "1500"))
        context_history = get_context_window(session_id, max_tokens=context_budget)
        messages = [{"role": "system", "content": self.build_system_prompt(mode) + memory_prompt}]
        for h in context_history[:-1]:
            content = h["content"]
            if h["role"] == "assistant":
                content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
            if content:
                messages.append({"role": h["role"], "content": content})
        safe_prompt = fit_prompt_to_context(prompt)
        messages.append({"role": "user", "content": safe_prompt})

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

    def generate_chat_completion_stream(self, messages, temperature=0.7, max_tokens=1024, stop=None):
        """
        OpenAI-compatible streaming generator for /v1/chat/completions.
        """
        abort_event = threading.Event()
        stop_tokens = stop if isinstance(stop, list) else (["<|im_end|>", "<|endoftext|>", "</s>"] if not stop else [stop])

        acquired = self._inference_lock.acquire(timeout=30.0)
        if not acquired:
            yield "[Error: Server is currently busy executing another request.]"
            return

        try:
            if self.backend == "offline_standby":
                self.refresh_backend()

            if self.backend == "llama-server":
                payload_dict = {
                    "messages": messages,
                    "temperature": temperature,
                    "stream": True,
                    "max_tokens": max_tokens,
                    "stop": stop_tokens
                }
                for chunk in self._stream_from_llama_server(payload_dict, abort_event):
                    yield chunk

            elif self.backend == "llama-cpp-python" and self.llama_cpp_model:
                res = self.llama_cpp_model.create_chat_completion(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                    stop=stop_tokens
                )
                for chunk in res:
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
            else:
                yield (
                    "Merlin AI is in offline standby mode. No local model weights (.gguf) are currently loaded. "
                    "Run `./download_model.sh` to load a model."
                )

        except Exception as e:
            yield f"[Inference error: {e}]"
        finally:
            self._inference_lock.release()

    def generate_chat_completion(self, messages, temperature=0.7, max_tokens=1024, stream=False, stop=None):
        """
        OpenAI-compatible handler for /v1/chat/completions.
        Returns generator if stream=True, or complete string if stream=False.
        """
        if stream:
            return self.generate_chat_completion_stream(messages, temperature=temperature, max_tokens=max_tokens, stop=stop)

        chunks = []
        for delta in self.generate_chat_completion_stream(messages, temperature=temperature, max_tokens=max_tokens, stop=stop):
            chunks.append(delta)
        return "".join(chunks)


