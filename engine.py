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

# Endpoints configuration with standard environment variables
DEFAULT_LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080").rstrip("/")

def get_ollama_url():
    url = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "http://" + url
    return url.rstrip("/")

DEFAULT_OLLAMA_URL = get_ollama_url()

def get_optimal_threads():
    if "THREADS" in os.environ:
        try:
            return int(os.environ["THREADS"])
        except ValueError:
            pass
    cpu_cores = os.cpu_count() or 4
    if cpu_cores <= 4:
        return max(1, cpu_cores)
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

def sanitize_prompt(prompt: str) -> str:
    """Strips raw ChatML / special delimiter tokens to prevent prompt injection."""
    if not prompt:
        return ""
    # Strip dangerous ChatML/Llama delimiter control sequences
    return re.sub(r"<\|(?:im_start|im_end|endoftext|system|user|assistant)\|>", "", prompt)

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
        self.ollama_url = DEFAULT_OLLAMA_URL
        self.llama_server_url = DEFAULT_LLAMA_SERVER_URL

        if HAS_REQUESTS:
            self._http_session = requests.Session()
            adapter = HTTPAdapter(pool_connections=20, pool_maxsize=50, max_retries=1)
            self._http_session.mount("http://", adapter)
            self._http_session.mount("https://", adapter)

        # In-process CPU Lock: Used strictly for serializing non-threadsafe llama-cpp-python instances.
        # Remote daemons (Ollama / llama-server) handle concurrent streams via continuous batching.
        self._inference_lock = threading.Lock()
        self._active_abort_events = {}  # session_id -> threading.Event()
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

    def _probe_ollama(self, timeout=0.8):
        """Probes Ollama daemon and retrieves installed models."""
        url = f"{self.ollama_url}/api/tags"
        try:
            if HAS_REQUESTS and self._http_session:
                resp = self._http_session.get(url, timeout=timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    models = [m.get("name") for m in data.get("models", []) if m.get("name")]
                    return True, models
            else:
                req = urllib.request.Request(url, headers={"User-Agent": "Merlin-AI/1.0"})
                with urllib.request.urlopen(req, timeout=timeout) as res:
                    if res.status == 200:
                        data = json.loads(res.read().decode("utf-8"))
                        models = [m.get("name") for m in data.get("models", []) if m.get("name")]
                        return True, models
        except Exception:
            pass
        return False, []

    def _probe_llama_server(self, timeout=0.5):
        """Probes C++ llama-server health endpoint."""
        url = f"{self.llama_server_url}/health"
        try:
            if HAS_REQUESTS and self._http_session:
                resp = self._http_session.get(url, timeout=timeout)
                return resp.status_code == 200
            else:
                req = urllib.request.Request(url, headers={"User-Agent": "Merlin-AI/1.0"})
                with urllib.request.urlopen(req, timeout=timeout) as res:
                    return res.status == 200
        except Exception:
            return False

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
            for _ in range(5):
                time.sleep(0.3)
                if self._probe_llama_server(timeout=0.3):
                    return True
        except Exception as e:
            log_event("autostart_error", f"Could not auto-start llama-server: {e}", "warning")
        return False

    def refresh_backend(self, fast: bool = False):
        """
        Detects available inference backends following priority order:
        1. Ollama daemon (standard port 11434 or OLLAMA_HOST)
        2. C++ llama-server (port 8080)
        3. Autostarted llama-server
        4. Local GGUF via llama-cpp-python
        5. Offline Standby
        """
        now = time.time()
        if fast and (now - self._last_probe_time < 5.0):
            return
        self._last_probe_time = now

        # 1. Probe Ollama Daemon (Top Priority for local / container deployments)
        ollama_ok, ollama_models = self._probe_ollama(timeout=0.6)
        if ollama_ok:
            self.backend = "ollama"
            if ollama_models:
                # Keep active if still valid, or pick top candidate
                if self.active_model_name not in ollama_models:
                    # Prefer coder/qwen if present
                    preferred = [m for m in ollama_models if "coder" in m.lower() or "qwen" in m.lower()]
                    self.active_model_name = preferred[0] if preferred else ollama_models[0]
            else:
                self.active_model_name = "ollama (no models installed - run 'ollama pull qwen2.5-coder:1.5b')"
            return

        # 2. Probe native C++ llama-server
        if self._probe_llama_server(timeout=0.3):
            self.backend = "llama-server"
            self.active_model_name = "Qwen2.5-Coder via llama-server"
            return

        # If already loaded with llama-cpp-python, preserve instance
        if self.backend == "llama-cpp-python" and self.llama_cpp_model:
            return

        # 3. Only attempt heavy autostart or full GGUF load if not in fast-probe mode
        if not fast:
            if self._try_autostart_llama_server():
                self.backend = "llama-server"
                self.active_model_name = "Qwen2.5-Coder via llama-server (autostarted)"
                log_event("engine_init", "Autostarted native C++ llama-server on port 8080", "success")
                return

            # 4. Fallback: Local GGUF via llama-cpp-python
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
        self.active_model_name = "Waiting for Ollama, model download, or llama-server"

    def get_available_models(self):
        """Returns a unified registry of all available models across Ollama and local GGUFs."""
        models = []

        # 1. Fetch Ollama models if daemon is available
        ollama_ok, ollama_models = self._probe_ollama(timeout=0.5)
        if ollama_ok:
            for m in ollama_models:
                models.append({
                    "filename": m,
                    "name": m,
                    "backend": "ollama",
                    "size_mb": "N/A",
                    "is_active": (m == self.active_model_name and self.backend == "ollama")
                })

        # 2. Fetch local .gguf models
        if os.path.exists(MODELS_DIR):
            for f in sorted(os.listdir(MODELS_DIR)):
                if f.endswith(".gguf"):
                    path = os.path.join(MODELS_DIR, f)
                    try:
                        size_mb = round(os.path.getsize(path) / (1024 * 1024), 1)
                    except Exception:
                        size_mb = 0.0
                    models.append({
                        "filename": f,
                        "name": f.replace(".gguf", ""),
                        "backend": "llama-cpp",
                        "size_mb": size_mb,
                        "is_active": (f == self.active_model_name and self.backend != "ollama")
                    })

        return models

    def set_active_model(self, model_name):
        """Switches the active model across Ollama or local GGUFs."""
        clean_name = model_name.strip()

        # Check if model belongs to Ollama
        ollama_ok, ollama_models = self._probe_ollama(timeout=0.8)
        if ollama_ok and clean_name in ollama_models:
            self.backend = "ollama"
            self.active_model_name = clean_name
            log_event("model_switch", f"Switched active Ollama model to {clean_name}", "success")
            return True, f"Active model set to Ollama model '{clean_name}'"

        # Check local GGUF models
        with self._inference_lock:
            model_path = os.path.join(MODELS_DIR, clean_name)
            if not os.path.exists(model_path) and not clean_name.endswith(".gguf"):
                model_path = os.path.join(MODELS_DIR, clean_name + ".gguf")
                clean_name += ".gguf"

            if os.path.exists(model_path):
                try:
                    threads = get_optimal_threads()
                    n_ctx = int(os.getenv("N_CTX", "4096"))
                    n_gpu = int(os.getenv("N_GPU_LAYERS", "0"))
                    self.llama_cpp_model = None
                    self.llama_cpp_model = load_llama_cpp_instance(model_path, threads, n_ctx, n_gpu)
                    self.active_model_name = clean_name
                    self.backend = "llama-cpp-python"
                    log_event("model_switch", f"Switched active local model to {clean_name}", "success")
                    return True, f"Active model switched to {clean_name}"
                except Exception as e:
                    log_event("model_switch_error", f"Failed switching to {clean_name}: {e}", "danger")
                    return False, str(e)

        return False, f"Model '{clean_name}' not found in Ollama or local models directory."

    def stop_generation(self, session_id):
        """Signals an active stream to abort immediately."""
        if session_id in self._active_abort_events:
            self._active_abort_events[session_id].set()
            return True
        return False

    def detect_conversation_intent(self, prompt: str) -> str:
        """Classifies the conversation domain (coder, reasoning, assistant)."""
        text = prompt.lower()
        code_phrases = [
            "def ", "class ", "import ", "const ", "```", "c++",
            "write code", "how to code", "unit test", "stack trace", "select * from"
        ]
        if any(phrase in text for phrase in code_phrases):
            return "coder"

        code_words = (
            r"\b(?:function|python|javascript|typescript|rust|bash|sql|regex|docker|git|"
            r"api|json|endpoint|bug|error|refactor|compile|script|linux|systemd|algorithm|"
            r"curl|ssh|html|css|npm|pip|gcc|cmake|dockerfile|pytest|fastapi|sqlite)\b"
        )
        if re.search(code_words, text):
            return "coder"

        reasoning_signals = [
            "step by step", "architecture", "tradeoffs", "design pattern", "analyze",
            "compare", "optimize", "benchmark", "pros and cons", "deep dive", "proof"
        ]
        if any(sig in text for sig in reasoning_signals):
            return "reasoning"

        return "assistant"

    def auto_select_optimal_model(self, intent: str):
        """Ensures the optimal available model is active based on detected intent."""
        if self.backend == "ollama":
            # If Ollama has specialized coder or reasoning models, select them
            ollama_ok, models = self._probe_ollama(timeout=0.3)
            if ollama_ok and len(models) > 1:
                target = "coder" if intent == "coder" else ("r1" if intent == "reasoning" else "")
                if target:
                    matches = [m for m in models if target in m.lower()]
                    if matches and matches[0] != self.active_model_name:
                        self.set_active_model(matches[0])
            return

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

    def build_system_prompt(self, mode="auto", detected_intent="general"):
        """
        Unified, powerhouse system prompt for Merlin AI.
        Seamlessly integrates elite software engineering, architecture, Linux systems administration,
        and proactive personal assistant (PA) workflows into a single powerful persona without mode-swapping.
        """
        return (
            "You are Merlin (Merlin AI), an elite autonomous software engineer, systems architect, and intelligent Personal Assistant. "
            "You run completely privately and locally on the user's private infrastructure with zero external telemetry.\n\n"
            "Core Directives & Dual Mastery:\n"
            "1. Software Engineering & Architecture:\n"
            "   - Write complete, robust, type-annotated, idiomatic, and production-grade code (Python, JavaScript/TypeScript, C/C++, Rust, Go, Bash, SQL, Docker).\n"
            "   - Never provide lazy placeholders (such as '# TODO' or '// rest of code'); always output complete, working implementations.\n"
            "   - Thoroughly address memory efficiency, algorithmic complexity, error handling, edge cases, and concurrency safety.\n"
            "   - When debugging or refactoring, explain root causes with high technical precision and provide drop-in code diffs.\n\n"
            "2. Executive Personal Assistant (PA) & Operations:\n"
            "   - Be organized, proactive, and concise. Maintain workflow continuity across conversation turns.\n"
            "   - Structure technical tasks, organize to-dos with priorities, draft clear technical notes, and synthesize complex documents into high-yield summaries.\n"
            "   - When answering system administration or Linux queries, provide precise shell commands with necessary flags, and explain critical parameters.\n\n"
            "3. Response Structure & Style:\n"
            "   - Communicate authoritatively, directly, and helpfully as a senior technical partner.\n"
            "   - Use clean GitHub-flavored markdown: bold headers, tables, bullet points, and syntax-highlighted code blocks with language identifiers.\n"
            "   - Eliminate conversational filler and introductory fluff. Deliver immediate, high-impact answers."
        )

    # -----------------------------------------------------------------
    # Ollama Streaming Adapter
    # -----------------------------------------------------------------
    def _stream_from_ollama(self, messages, temperature=0.7, max_tokens=1024, stop=None, abort_event=None, timeout=(3.0, 120.0)):
        """
        Streams NDJSON token chunks from Ollama's native /api/chat endpoint.
        Applies keep_alive, context budget bounds, and immediate socket teardown upon cancellation.
        """
        if abort_event is None:
            abort_event = threading.Event()

        stop_list = stop if isinstance(stop, list) else ([stop] if stop else ["<|im_end|>", "<|endoftext|>", "</s>"])
        keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "5m")
        num_ctx = int(os.getenv("N_CTX", "4096"))

        payload_dict = {
            "model": self.active_model_name,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": num_ctx,
                "stop": stop_list
            },
            "keep_alive": keep_alive
        }

        url = f"{self.ollama_url}/api/chat"

        if HAS_REQUESTS and self._http_session:
            r = None
            try:
                r = self._http_session.post(
                    url,
                    json=payload_dict,
                    headers={"Content-Type": "application/json", "User-Agent": "Merlin-AI/1.0"},
                    stream=True,
                    timeout=timeout
                )
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if abort_event.is_set():
                        r.close()
                        break
                    if not line:
                        continue
                    try:
                        chunk_json = json.loads(line)
                        delta = chunk_json.get("message", {}).get("content", "")
                        if delta:
                            yield delta
                        if chunk_json.get("done", False):
                            break
                    except Exception:
                        continue
                return
            except Exception as e:
                if not abort_event.is_set():
                    yield f"[Error querying Ollama daemon: {e}]"
                return
            finally:
                if r is not None:
                    try:
                        r.close()
                    except Exception:
                        pass

        # Standard library urllib fallback for streaming NDJSON
        payload = json.dumps(payload_dict).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "Merlin-AI/1.0"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout[1]) as resp:
                buf = b""
                while not abort_event.is_set():
                    chunk = resp.read(256)
                    if not chunk:
                        if buf:
                            try:
                                chunk_json = json.loads(buf.decode("utf-8", errors="replace").strip())
                                delta = chunk_json.get("message", {}).get("content", "")
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
                        try:
                            chunk_json = json.loads(line_str)
                            delta = chunk_json.get("message", {}).get("content", "")
                            if delta:
                                yield delta
                            if chunk_json.get("done", False):
                                return
                        except Exception:
                            continue
        except Exception as e:
            if not abort_event.is_set():
                yield f"[Error querying Ollama daemon via urllib: {e}]"

    # -----------------------------------------------------------------
    # llama-server SSE Streaming Adapter
    # -----------------------------------------------------------------
    def _stream_from_llama_server(self, payload_dict, abort_event, timeout=(3.0, 120.0)):
        """
        Streams SSE tokens from llama-server with buffered line parsing (avoids single-byte syscall overhead).
        """
        url = f"{self.llama_server_url}/v1/chat/completions"

        if HAS_REQUESTS and self._http_session:
            r = None
            try:
                r = self._http_session.post(
                    url,
                    json=payload_dict,
                    headers={"Content-Type": "application/json", "User-Agent": "Merlin-AI/1.0"},
                    stream=True,
                    timeout=timeout
                )
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
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

        # Standard library urllib with chunk-buffered parsing
        payload = json.dumps(payload_dict).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "Merlin-AI/1.0"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout[1]) as resp:
                buf = b""
                while not abort_event.is_set():
                    chunk = resp.read(256)
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
        Provides sub-second latency on greetings and efficient cached prefill for LLM inference.
        """
        abort_event = threading.Event()
        self._active_abort_events[session_id] = abort_event

        # Save user message to persistent history
        save_chat_message(session_id, "user", prompt)

        # --- 1. Sub-Second Instant Greeting Circuit ---
        if self._is_conversational_greeting(prompt):
            greeting_text = self._get_conversational_response(prompt)
            buffer = []
            words = greeting_text.split(" ")
            for i, word in enumerate(words):
                if abort_event.is_set():
                    break
                chunk = word + (" " if i < len(words) - 1 else "")
                buffer.append(chunk)
                yield chunk
                time.sleep(0.012)

            self._active_abort_events.pop(session_id, None)
            if buffer:
                save_chat_message(session_id, "assistant", "".join(buffer))
            return

        # --- 2. Heavy LLM Inference Circuit ---
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

        system_content = self.build_system_prompt(mode=mode, detected_intent=intent)
        messages = [{"role": "system", "content": system_content}]

        if memory_prompt:
            messages.append({"role": "system", "content": memory_prompt})

        context_budget = int(os.getenv("CONTEXT_BUDGET", "1000"))
        context_history = get_context_window(session_id, max_tokens=context_budget)
        for h in context_history[:-1]:
            content = h["content"]
            if h["role"] == "assistant":
                content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
            if content:
                messages.append({"role": h["role"], "content": content})

        # Sanitize prompt against delimiter injection and fit to context window
        safe_prompt = sanitize_prompt(fit_prompt_to_context(prompt))
        messages.append({"role": "user", "content": safe_prompt})

        max_tokens = int(os.getenv("MAX_TOKENS", "512"))
        stop_tokens = ["<|im_end|>", "<|endoftext|>", "</s>"]
        temperature = 0.2 if intent in ("coder", "reasoning") else 0.7

        # In-process CPU Lock: Acquire ONLY when running llama-cpp-python
        needs_lock = (self.backend == "llama-cpp-python")
        acquired = False
        if needs_lock:
            acquired = self._inference_lock.acquire(timeout=15.0)
            if not acquired:
                msg = "[Server Busy: Another CPU generation is currently executing. Please wait a moment.]"
                yield msg
                self._active_abort_events.pop(session_id, None)
                return

        response_buffer = []
        t_start = time.time()

        try:
            if self.backend == "offline_standby":
                self.refresh_backend(fast=True)

            if self.backend == "ollama":
                for chunk in self._stream_from_ollama(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop_tokens,
                    abort_event=abort_event
                ):
                    if abort_event.is_set():
                        break
                    response_buffer.append(chunk)
                    yield chunk

            elif self.backend == "llama-server":
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
                    response_buffer.append(chunk)
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
                            response_buffer.append(delta)
                            yield delta
                except Exception as e:
                    err = f"[Inference error: {e}]"
                    response_buffer.append(err)
                    yield err

            else:
                msg = (
                    "⚠️ **Offline Intelligence Standby**\n\n"
                    "The local web dashboard and SQLite memory system are fully active.\n\n"
                    "**To activate the local AI model:**\n"
                    "1. **Option A (Ollama - Recommended):** Install Ollama and run `ollama run qwen2.5-coder:1.5b`.\n"
                    "2. **Option B (Native llama.cpp):** Run `./download_model.sh` and `./start_llama_server.sh`.\n"
                    "Once started, Merlin AI automatically detects the engine with zero restart needed."
                )
                words = msg.split(" ")
                for i, word in enumerate(words):
                    if abort_event.is_set():
                        break
                    chunk = word + (" " if i < len(words) - 1 else "")
                    response_buffer.append(chunk)
                    yield chunk
                    time.sleep(0.008)

        finally:
            if needs_lock and acquired:
                self._inference_lock.release()
            self._active_abort_events.pop(session_id, None)

            full_response = "".join(response_buffer)
            if full_response:
                save_chat_message(session_id, "assistant", full_response)
                elapsed = time.time() - t_start
                token_count = len(response_buffer)
                tps = round(token_count / max(elapsed, 0.001), 1)
                log_event("inference_stream", f"Completed generation: {token_count} chunks ({tps} chunks/s) via {self.backend}", "info")

    chat_stream = generate_stream

    def generate_chat_completion_stream(self, messages, temperature=0.7, max_tokens=1024, stop=None):
        """OpenAI-compatible streaming generator for /v1/chat/completions."""
        abort_event = threading.Event()
        stop_tokens = stop if isinstance(stop, list) else (["<|im_end|>", "<|endoftext|>", "</s>"] if not stop else [stop])

        if self.backend == "offline_standby":
            self.refresh_backend(fast=True)

        if self.backend == "ollama":
            for chunk in self._stream_from_ollama(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_tokens,
                abort_event=abort_event
            ):
                yield chunk
            return

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
            return

        if self.backend == "llama-cpp-python" and self.llama_cpp_model:
            acquired = self._inference_lock.acquire(timeout=30.0)
            if not acquired:
                yield "[Error: Server is currently busy executing another request.]"
                return
            try:
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
            except Exception as e:
                yield f"[Inference error: {e}]"
            finally:
                self._inference_lock.release()
            return

        yield (
            "Merlin AI is in offline standby mode. No local model weights or Ollama instances are detected. "
            "Please start Ollama or run `./download_model.sh`."
        )

    def generate_chat_completion(self, messages, temperature=0.7, max_tokens=1024, stream=False, stop=None):
        """OpenAI-compatible handler for /v1/chat/completions."""
        if stream:
            return self.generate_chat_completion_stream(messages, temperature=temperature, max_tokens=max_tokens, stop=stop)

        chunks = []
        for delta in self.generate_chat_completion_stream(messages, temperature=temperature, max_tokens=max_tokens, stop=stop):
            chunks.append(delta)
        return "".join(chunks)
