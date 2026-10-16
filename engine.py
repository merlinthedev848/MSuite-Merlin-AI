import os
import json
import re
import urllib.request
import urllib.error
import threading
import time
import shutil
import atexit
from memory import search_knowledge, get_recent_chat, save_chat_message, log_event, get_context_window

try:
    import requests
    from requests.adapters import HTTPAdapter
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
        self._server_process = None
        self._http_session = None
        self._last_probe_time = 0.0
        if HAS_REQUESTS:
            self._http_session = requests.Session()
            adapter = HTTPAdapter(pool_connections=10, pool_maxsize=20, max_retries=1)
            self._http_session.mount("http://", adapter)
        # Concurrency Lock: Prevents simultaneous access to the CPU inference engine
        self._inference_lock = threading.Lock()
        self._active_abort_events = {} # session_id -> threading.Event()
        atexit.register(self.cleanup)
        self.refresh_backend(fast=False)

    def cleanup(self):
        """Gracefully terminates background llama-server child processes and HTTP sessions on exit."""
        if self._http_session:
            try:
                self._http_session.close()
            except Exception:
                pass
            self._http_session = None
        if self._server_process:
            try:
                self._server_process.terminate()
                self._server_process.wait(timeout=2.0)
            except Exception:
                try:
                    self._server_process.kill()
                except Exception:
                    pass
            self._server_process = None

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
            self._server_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Fast, non-blocking startup check (max 1.5s)
            for _ in range(5):
                time.sleep(0.3)
                try:
                    req = urllib.request.Request(f"{DEFAULT_LLAMA_SERVER_URL}/health", headers={"User-Agent": "Merlin-AI/1.0"})
                    with urllib.request.urlopen(req, timeout=0.3) as res:
                        if res.status == 200:
                            return True
                except Exception:
                    pass
        except Exception as e:
            log_event("autostart_error", f"Could not auto-start llama-server: {e}", "warning")
        return False

    def refresh_backend(self, fast: bool = False):
        """
        Checks for llama-server or local GGUF models.
        When fast=True, skips heavy autostart and disk loading if probed recently.
        """
        now = time.time()
        if fast and (now - self._last_probe_time < 5.0):
            return
        self._last_probe_time = now

        # 1. Quick probe of C++ llama-server (fastest, unbuffered native binary)
        try:
            req = urllib.request.Request(f"{DEFAULT_LLAMA_SERVER_URL}/health", headers={"User-Agent": "Merlin-AI/1.0"})
            with urllib.request.urlopen(req, timeout=0.25) as res:
                if res.status == 200:
                    self.backend = "llama-server"
                    self.active_model_name = "Qwen2.5-Coder via llama-server"
                    return
        except Exception:
            pass

        # If already loaded with llama-cpp-python, no need to re-scan
        if self.backend == "llama-cpp-python" and self.llama_cpp_model:
            return

        # 2. Only attempt autostart or full GGUF load if not in fast-probe mode
        if not fast:
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

    def detect_conversation_intent(self, prompt: str) -> str:
        """
        Automatically classifies the conversation domain (coder, reasoning, assistant)
        so the single text window automatically provides the optimal persona and model behavior.
        """
        text = prompt.lower()

        # Exact code phrases & symbols
        code_phrases = [
            "def ", "class ", "import ", "const ", "```", "c++",
            "write code", "how to code", "unit test", "stack trace", "select * from"
        ]
        if any(phrase in text for phrase in code_phrases):
            return "coder"

        # Word-boundary technical keywords (prevents 'api' matching 'capital' or 'git' matching 'digital')
        code_words = (
            r"\b(?:function|python|javascript|typescript|rust|bash|sql|regex|docker|git|"
            r"api|json|endpoint|bug|error|refactor|compile|script|linux|systemd|algorithm|"
            r"curl|ssh|html|css|npm|pip|gcc|cmake|dockerfile|pytest|fastapi|sqlite)\b"
        )
        if re.search(code_words, text):
            return "coder"

        # Complex Reasoning & Architectural Signals
        reasoning_signals = [
            "step by step", "architecture", "tradeoffs", "design pattern", "analyze",
            "compare", "optimize", "benchmark", "pros and cons", "deep dive", "proof"
        ]
        if any(sig in text for sig in reasoning_signals):
            return "reasoning"

        return "assistant"

    def auto_select_optimal_model(self, intent: str):
        """
        Automatically ensures the optimal available local model is active
        without requiring the user to manually switch models.
        """
        if not os.path.exists(MODELS_DIR):
            return
        models = [f for f in os.listdir(MODELS_DIR) if f.endswith(".gguf")]
        if len(models) <= 1:
            return

        target_keyword = "coder" if intent == "coder" else ("r1" if intent == "reasoning" else "")
        best_candidate = None

        if target_keyword:
            for m in models:
                if target_keyword in m.lower():
                    best_candidate = m
                    break

        if not best_candidate:
            best_candidate = models[0]

        if best_candidate and best_candidate != self.active_model_name:
            self.set_active_model(best_candidate)

    def build_system_prompt(self, mode="auto", detected_intent="assistant"):
        intent = mode if mode in ("coder", "reasoning", "assistant") else detected_intent

        if intent in ("reasoning", "agent"):
            return (
                "You are Merlin (Merlin AI), an elite autonomous software engineering partner and systems architect. "
                "You reason deeply step-by-step through technical challenges, plan thoroughly before execution, and construct clean, modular, production-grade solutions. "
                "When addressing tasks: break down complex requirements, consider edge cases, provide precise shell commands or code diffs, "
                "and explain your design decisions clearly."
            )
        elif intent == "coder":
            return (
                "You are Merlin (Merlin AI), an expert offline software engineer and coding assistant running locally on a private server. "
                "Your name is Merlin. You react and respond naturally when addressed as Merlin or Merlin AI. "
                "Provide clean, efficient, idiomatic, production-ready code. Always consider memory limits, edge cases, and algorithmic complexity. "
                "Keep prose concise and focused on the code."
            )
        else:
            return (
                "You are Merlin (Merlin AI), an intelligent offline Personal Assistant and software companion running locally on the user's private network. "
                "Your name is Merlin. You react and respond naturally when addressed as Merlin or Merlin AI. "
                "You assist with daily tasks, server diagnostics, Linux administration, scheduling, and programming questions. "
                "Be direct, helpful, and concise."
            )

    def _stream_from_llama_server(self, payload_dict, abort_event, timeout=120):
        """
        Streams SSE tokens from llama-server unbuffered so each token reaches the client immediately.
        Actively terminates remote generation upon abort event by closing the HTTP stream.
        """
        if HAS_REQUESTS and self._http_session:
            r = None
            try:
                r = self._http_session.post(
                    f"{DEFAULT_LLAMA_SERVER_URL}/v1/chat/completions",
                    json=payload_dict,
                    headers={"Content-Type": "application/json", "User-Agent": "Merlin-AI/1.0"},
                    stream=True,
                    timeout=timeout
                )
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True, chunk_size=1):
                    if abort_event.is_set():
                        r.close()
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
            finally:
                if r is not None:
                    try:
                        r.close()
                    except Exception:
                        pass

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

    def _is_conversational_greeting(self, prompt: str) -> bool:
        """
        Determines if a prompt is a pure greeting or simple identity check.
        These are answered in <30ms with instant token streaming, bypassing CPU prefill stalls.
        """
        cleaned = re.sub(r"[^\w\s]", "", prompt.strip().lower())
        words = cleaned.split()
        if not words:
            return True

        exact_matches = {
            "hello", "hi", "hey", "merlin", "hello merlin", "hi merlin", "hey merlin",
            "good morning", "good afternoon", "good evening", "how are you", "how are you doing",
            "who are you", "what are you", "what is your name", "who made you", "what can you do",
            "help", "yo", "sup", "whats up", "ping", "test", "greetings"
        }
        if cleaned in exact_matches:
            return True

        if len(words) <= 3 and any(w in ["hi", "hello", "hey", "merlin", "yo", "sup", "greetings"] for w in words):
            action_words = {
                "write", "create", "make", "fix", "code", "run", "install", "how", "why",
                "what", "where", "python", "script", "explain", "debug", "test", "benchmark", "git"
            }
            if not any(w in action_words for w in words):
                return True

        return False

    def _get_conversational_response(self, prompt: str) -> str:
        """Generates an immediate, warm Merlin response for common greetings."""
        cleaned = re.sub(r"[^\w\s]", "", prompt.strip().lower())

        if any(q in cleaned for q in ["who are you", "what are you", "what is your name", "who made you"]):
            return (
                "I am **Merlin** (Merlin AI), your autonomous offline Personal Assistant and expert coding companion. "
                "I run completely locally on your private network, ensuring zero data leaves your server.\n\n"
                "I can write and debug code, manage Linux services, automate workflows, and maintain persistent memory. "
                "How can I assist you right now?"
            )

        if any(q in cleaned for q in ["what can you do", "help", "capabilities"]):
            return (
                "Here is what I can do for you 100% offline:\n\n"
                "- 💻 **Software Engineering & Coding:** Architecture, Python, C/C++, JavaScript, Bash, SQL, and debugging.\n"
                "- ⚙️ **System Administration:** Linux troubleshooting, systemd services, network diagnostics, and shell tools.\n"
                "- 🧠 **Persistent Knowledge Engine:** Storing lessons and coding patterns across conversations in SQLite.\n"
                "- 🔌 **OpenAI Compatible API:** Exposing `/v1/chat/completions` for direct integration into VS Code, Continue, or custom tools.\n\n"
                "What would you like to build or inspect today?"
            )

        if any(q in cleaned for q in ["how are you", "how are you doing"]):
            return (
                "I'm operating at peak performance and ready to assist! "
                "All systems are green. What would you like to work on today?"
            )

        # Dynamic greetings
        greetings = [
            "Hello! I'm **Merlin**, your offline AI assistant and coding partner. How can I help you today?",
            "Hey there! Merlin at your service. What are we building or troubleshooting today?",
            "Greetings! I'm Merlin, running 100% privately and offline on your local server. What can I do for you?",
            "Hi! Merlin here, ready to assist with code, server tasks, or anything else you need. What's on your mind?"
        ]
        import random
        return random.choice(greetings)

    def generate_stream(self, prompt, session_id="default", mode="assistant"):
        """
        Thread-safe token generator.
        Provides sub-second latency on greetings and cached prompt prefill for LLM inference.
        """
        abort_event = threading.Event()
        self._active_abort_events[session_id] = abort_event

        # Save user message to persistent history
        save_chat_message(session_id, "user", prompt)

        # --- 1. Sub-Second Instant Greeting Circuit ---
        if self._is_conversational_greeting(prompt):
            greeting_text = self._get_conversational_response(prompt)
            accumulated = ""
            # Stream words with micro-delays for realistic, instantaneous delivery (< 0.2s total)
            words = greeting_text.split(" ")
            for i, word in enumerate(words):
                if abort_event.is_set():
                    break
                chunk = word + (" " if i < len(words) - 1 else "")
                accumulated += chunk
                yield chunk
                time.sleep(0.012)

            self._active_abort_events.pop(session_id, None)
            if accumulated:
                save_chat_message(session_id, "assistant", accumulated)
            return

        # --- 2. Heavy LLM Inference Circuit (Automated Intent & Model Routing) ---
        intent = self.detect_conversation_intent(prompt) if mode in ("auto", "assistant", "") else mode
        self.auto_select_optimal_model(intent)

        learned_context = search_knowledge(prompt, limit=2)
        memory_prompt = ""
        if learned_context:
            memory_prompt = "[Verified Lessons from Past Teacher Sessions]:\n"
            for k in learned_context:
                memory_prompt += f"- Topic: {k['topic']}\n  Lesson: {k['lesson']}\n"
                if k.get("code_example"):
                    memory_prompt += f"  Code:\n{k['code_example']}\n"

        # Static system prompt enables 100% KV-cache prefix hit in llama-server
        system_content = self.build_system_prompt(mode=mode, detected_intent=intent)
        messages = [{"role": "system", "content": system_content}]

        # Inject memory context as a separate system turn if available
        if memory_prompt:
            messages.append({"role": "system", "content": memory_prompt})

        # Assemble compact context history to avoid CPU prefill stalls
        context_budget = int(os.getenv("CONTEXT_BUDGET", "1000"))
        context_history = get_context_window(session_id, max_tokens=context_budget)
        for h in context_history[:-1]:
            content = h["content"]
            if h["role"] == "assistant":
                content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
            if content:
                messages.append({"role": h["role"], "content": content})

        safe_prompt = fit_prompt_to_context(prompt)
        messages.append({"role": "user", "content": safe_prompt})

        full_response = ""
        max_tokens = int(os.getenv("MAX_TOKENS", "512"))
        stop_tokens = ["<|im_end|>", "<|endoftext|>", "</s>"]
        temperature = 0.2 if intent in ("coder", "reasoning") else 0.7

        # Acquire lock to ensure only one thread generates on the CPU at a time
        acquired = self._inference_lock.acquire(timeout=15.0)
        if not acquired:
            msg = "[Server Busy: Another generation is currently executing. Please wait a moment.]"
            yield msg
            self._active_abort_events.pop(session_id, None)
            return

        try:
            # Fast check if in standby mode
            if self.backend == "offline_standby":
                self.refresh_backend(fast=True)

            if self.backend == "llama-server":
                payload_dict = {
                    "messages": messages,
                    "temperature": temperature,
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
                        temperature=temperature,
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
                msg = (
                    "⚠️ **Offline Intelligence Standby**\n\n"
                    "The local web dashboard and SQLite memory system are fully active.\n\n"
                    "**To activate the local AI model:**\n"
                    "1. Run `./download_model.sh` to download Qwen2.5-Coder (GGUF format).\n"
                    "2. Start `./start_llama_server.sh` OR install `pip install llama-cpp-python`.\n"
                    "Once started, the engine automatically detects the model with zero restart needed."
                )
                full_response = msg
                words = msg.split(" ")
                for i, word in enumerate(words):
                    if abort_event.is_set():
                        break
                    yield word + (" " if i < len(words) - 1 else "")
                    time.sleep(0.008)

        finally:
            self._inference_lock.release()
            self._active_abort_events.pop(session_id, None)
            if full_response:
                save_chat_message(session_id, "assistant", full_response)

    # Dual alias for chat_stream / generate_stream compatibility
    chat_stream = generate_stream

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


