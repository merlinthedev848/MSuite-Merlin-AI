import atexit
import json
import os
import re
import shutil
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

from database import (
    add_gap,
    get_context_window,
    get_recent_chat,  # noqa: F401
    log_event,
    save_chat_message,
    search_knowledge,
)
from sync_teacher import run_sync

try:
    import requests
    from requests.adapters import HTTPAdapter
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DEFAULT_LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080").rstrip("/")

# Pre-compiled regular expressions
_CHATML_TOKEN_RE = re.compile(r"<\|(?:im_start|im_end|endoftext|system|user|assistant)\|>")
_CODE_WORDS_RE = re.compile(
    r"\b(?:function|python|javascript|typescript|rust|bash|sql|regex|docker|git|"
    r"api|json|endpoint|bug|error|refactor|compile|script|linux|systemd|algorithm|"
    r"curl|ssh|html|css|npm|pip|gcc|cmake|dockerfile|pytest|fastapi|sqlite)\b"
)
_NON_WORD_RE = re.compile(r"[^\w\s]")
_THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>")


def get_ollama_url() -> str:
    url = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "http://" + url
    return url.rstrip("/")


DEFAULT_OLLAMA_URL = get_ollama_url()


def get_optimal_threads() -> int:
    if "THREADS" in os.environ:
        try:
            return max(1, int(os.environ["THREADS"]))
        except ValueError:
            pass
    # Respect Linux cgroup v1/v2 CPU affinity inside LXC / Docker containers first
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity_cores = len(os.sched_getaffinity(0))
            if affinity_cores > 0:
                return max(1, min(8, affinity_cores))
        except OSError:
            pass
    cpu_cores = os.cpu_count() or 4
    if cpu_cores <= 4:
        return max(1, cpu_cores)
    physical_cores = max(2, cpu_cores // 2)
    return min(6, physical_cores)


def load_llama_cpp_instance(model_path: str, threads: int, n_ctx: int, n_gpu: int):
    """Instantiates Llama with FP16 KV cache (quantized KV cache without flash_attn corrupts logits on CPU)."""
    from llama_cpp import Llama
    kwargs: dict[str, Any] = {
        "model_path": model_path,
        "n_ctx": n_ctx,
        "n_threads": max(1, threads - 1) if threads > 2 else threads,
        "n_threads_batch": threads,
        "n_batch": 512,
        "n_gpu_layers": n_gpu,
        "use_mmap": True,
        "verbose": False,
    }
    return Llama(**kwargs)


def sanitize_prompt(prompt: str) -> str:
    """Strips raw ChatML / special delimiter tokens to prevent prompt injection."""
    if not prompt:
        return ""
    return _CHATML_TOKEN_RE.sub("", prompt)


def fit_prompt_to_context(prompt: str, max_chars: int = 16000) -> str:
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
        except ValueError:
            pass

    half = max_chars // 2
    return prompt[:half] + "\n\n... [Prompt trimmed to fit local context buffer] ...\n\n" + prompt[-half:]


class InferenceEngine:
    def __init__(self, lazy_init: bool = False):
        self.llama_cpp_model = None
        self.backend: str = "loading" if lazy_init else "offline_standby"
        self.active_model_name: str = "Initializing model engine..." if lazy_init else "None"
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
        self._inference_lock = threading.Lock()
        self._abort_lock = threading.Lock()
        self._active_abort_events: dict[str, threading.Event] = {}
        self._init_done_event = threading.Event()
        atexit.register(self.cleanup)

        if lazy_init:
            def _bg_init():
                try:
                    self.refresh_backend(fast=False)
                finally:
                    self._init_done_event.set()
            threading.Thread(target=_bg_init, daemon=True, name="merlin-engine-init").start()
        else:
            self.refresh_backend(fast=False)
            self._init_done_event.set()

    def cleanup(self) -> None:
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

    def _probe_ollama(self, timeout: float = 0.8) -> tuple[bool, list[str]]:
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

    def _probe_llama_server(self, timeout: float = 0.5) -> bool:
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

    def _try_autostart_llama_server(self) -> bool:
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
            shutil.which("llama-server.exe"),
        ]
        for c in candidates:
            if c and os.path.exists(c) and os.path.isfile(c):
                server_bin = c
                break

        if not server_bin:
            return False

        threads = get_optimal_threads()
        n_ctx = int(os.getenv("N_CTX", "2048"))
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
            "--cont-batching",
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

    def refresh_backend(self, fast: bool = False) -> None:
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

        # 1. Probe Ollama Daemon
        ollama_ok, ollama_models = self._probe_ollama(timeout=0.6)
        if ollama_ok:
            self.backend = "ollama"
            if ollama_models:
                if self.active_model_name not in ollama_models:
                    preferred_15b = [m for m in ollama_models if "1.5b" in m.lower() or "7b" in m.lower() or "3b" in m.lower()]
                    preferred_coder = [m for m in ollama_models if "coder" in m.lower() or "qwen" in m.lower()]
                    if preferred_15b:
                        self.active_model_name = preferred_15b[0]
                    elif preferred_coder:
                        self.active_model_name = preferred_coder[0]
                    else:
                        self.active_model_name = ollama_models[0]
            else:
                self.active_model_name = "ollama (no models installed - run 'ollama pull qwen2.5-coder:1.5b')"
            return

        # 2. Probe native C++ llama-server
        if self._probe_llama_server(timeout=0.3):
            self.backend = "llama-server"
            self.active_model_name = "Qwen2.5-Coder via llama-server"
            return

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
                    with self._inference_lock:
                        self.active_model_name = models[0]
                        try:
                            model_path = os.path.join(MODELS_DIR, models[0])
                            threads = get_optimal_threads()
                            n_ctx = int(os.getenv("N_CTX", "2048"))
                            n_gpu = int(os.getenv("N_GPU_LAYERS", "0"))
                            self.llama_cpp_model = load_llama_cpp_instance(model_path, threads, n_ctx, n_gpu)
                            self.backend = "llama-cpp-python"
                            log_event("engine_init", f"Loaded local model {models[0]} with {threads} threads, ctx {n_ctx}", "success")
                            return
                        except ImportError:
                            pass
                        except Exception as e:
                            log_event("engine_error", f"Failed loading GGUF: {e}", "danger")

        self.backend = "offline_standby"
        self.active_model_name = "Waiting for Ollama, model download, or llama-server"

    def get_available_models(self) -> list[dict[str, Any]]:
        """Returns a unified registry of all available models across Ollama and local GGUFs."""
        models: list[dict[str, Any]] = []

        ollama_ok, ollama_models = self._probe_ollama(timeout=0.5)
        if ollama_ok:
            for m in ollama_models:
                models.append({
                    "filename": m,
                    "name": m,
                    "backend": "ollama",
                    "size_mb": "N/A",
                    "is_active": (m == self.active_model_name and self.backend == "ollama"),
                })

        if os.path.exists(MODELS_DIR):
            for f in sorted(os.listdir(MODELS_DIR)):
                if f.endswith(".gguf"):
                    path = os.path.join(MODELS_DIR, f)
                    try:
                        size_mb = round(os.path.getsize(path) / (1024 * 1024), 1)
                    except OSError:
                        size_mb = 0.0
                    models.append({
                        "filename": f,
                        "name": f.replace(".gguf", ""),
                        "backend": "llama-cpp",
                        "size_mb": size_mb,
                        "is_active": (f == self.active_model_name and self.backend != "ollama"),
                    })

        return models

    def set_active_model(self, model_name: str) -> tuple[bool, str]:
        """Switches the active model across Ollama or local GGUFs."""
        clean_name = os.path.basename(model_name.strip()) if ":" not in model_name else model_name.strip()

        ollama_ok, ollama_models = self._probe_ollama(timeout=0.8)
        if ollama_ok and clean_name in ollama_models:
            self.backend = "ollama"
            self.active_model_name = clean_name
            log_event("model_switch", f"Switched active Ollama model to {clean_name}", "success")
            return True, f"Active model set to Ollama model '{clean_name}'"

        safe_filename = os.path.basename(clean_name)
        with self._inference_lock:
            model_path = os.path.join(MODELS_DIR, safe_filename)
            if not os.path.exists(model_path) and not safe_filename.endswith(".gguf"):
                model_path = os.path.join(MODELS_DIR, safe_filename + ".gguf")
                safe_filename += ".gguf"

            if os.path.exists(model_path) and os.path.isfile(model_path):
                try:
                    threads = get_optimal_threads()
                    n_ctx = int(os.getenv("N_CTX", "2048"))
                    n_gpu = int(os.getenv("N_GPU_LAYERS", "0"))
                    self.llama_cpp_model = None
                    self.llama_cpp_model = load_llama_cpp_instance(model_path, threads, n_ctx, n_gpu)
                    self.active_model_name = safe_filename
                    self.backend = "llama-cpp-python"
                    log_event("model_switch", f"Switched active local model to {safe_filename}", "success")
                    return True, f"Active model switched to {safe_filename}"
                except Exception as e:
                    log_event("model_switch_error", f"Failed switching to {safe_filename}: {e}", "danger")
                    return False, str(e)

        return False, f"Model '{clean_name}' not found in Ollama or local models directory."

    def _register_abort_event(self, session_id: str) -> threading.Event:
        """Atomically cancels any in-flight stream on the same session_id and registers a fresh Event."""
        new_event = threading.Event()
        with self._abort_lock:
            prev = self._active_abort_events.get(session_id)
            if prev is not None:
                prev.set()
            self._active_abort_events[session_id] = new_event
        return new_event

    def _unregister_abort_event(self, session_id: str, event: threading.Event) -> None:
        """Only removes the abort event if it is still the active one for session_id (prevents race)."""
        with self._abort_lock:
            if self._active_abort_events.get(session_id) is event:
                self._active_abort_events.pop(session_id, None)

    def stop_generation(self, session_id: str) -> bool:
        """Signals an active stream to abort immediately."""
        with self._abort_lock:
            ev = self._active_abort_events.get(session_id)
            if ev is not None:
                ev.set()
                return True
        return False

    def detect_conversation_intent(self, prompt: str) -> str:
        """Classifies the conversation domain (coder, reasoning, assistant)."""
        text = prompt.lower()
        code_phrases = (
            "def ", "class ", "import ", "const ", "```", "c++",
            "write code", "how to code", "unit test", "stack trace", "select * from",
        )
        if any(phrase in text for phrase in code_phrases):
            return "coder"

        if _CODE_WORDS_RE.search(text):
            return "coder"

        reasoning_signals = (
            "step by step", "architecture", "tradeoffs", "design pattern", "analyze",
            "compare", "optimize", "benchmark", "pros and cons", "deep dive", "proof",
        )
        if any(sig in text for sig in reasoning_signals):
            return "reasoning"

        return "assistant"

    def auto_select_optimal_model(self, intent: str) -> None:
        """Ensures the optimal available model is active based on detected intent."""
        if self.backend == "ollama":
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

    def build_system_prompt(self, mode: str = "auto", detected_intent: str = "general") -> str:
        """
        Concise, high-signal system prompt optimized for fast CPU prefill latency.
        """
        if detected_intent in ("coder", "reasoning"):
            return (
                "You are Merlin, an expert offline software engineer and systems architect. "
                "Write complete, production-ready, bug-free code and clear technical explanations. "
                "Use markdown code blocks with language tags and never use placeholders."
            )
        return (
            "You are Merlin, an intelligent offline AI assistant and software engineer running locally. "
            "Be clear, direct, helpful, and concise."
        )

    def _stream_from_ollama(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: Any | None = None,
        abort_event: threading.Event | None = None,
        timeout: tuple[float, float] = (3.0, 120.0),
    ) -> Iterator[str]:
        """Streams NDJSON token chunks from Ollama's native /api/chat endpoint."""
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
                "num_thread": get_optimal_threads(),
                "stop": stop_list,
            },
            "keep_alive": keep_alive,
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
                    timeout=timeout,
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
                    except ValueError:
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

        payload = json.dumps(payload_dict).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "Merlin-AI/1.0"},
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
                            except ValueError:
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
                        except ValueError:
                            continue
        except Exception as e:
            if not abort_event.is_set():
                yield f"[Error querying Ollama daemon via urllib: {e}]"

    def _stream_from_llama_server(
        self,
        payload_dict: dict[str, Any],
        abort_event: threading.Event,
        timeout: tuple[float, float] = (3.0, 120.0),
    ) -> Iterator[str]:
        """Streams SSE tokens from llama-server with buffered line parsing."""
        url = f"{self.llama_server_url}/v1/chat/completions"

        if HAS_REQUESTS and self._http_session:
            r = None
            try:
                r = self._http_session.post(
                    url,
                    json=payload_dict,
                    headers={"Content-Type": "application/json", "User-Agent": "Merlin-AI/1.0"},
                    stream=True,
                    timeout=timeout,
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
                        except (ValueError, KeyError, IndexError):
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

        payload = json.dumps(payload_dict).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "Merlin-AI/1.0"},
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
                                except (ValueError, KeyError, IndexError):
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
                            except (ValueError, KeyError, IndexError):
                                continue
        except Exception as e:
            if not abort_event.is_set():
                yield f"[Error querying llama-server: {e}]"

    def generate_stream(self, prompt: str, session_id: str = "default", mode: str = "assistant") -> Iterator[str]:
        """
        Thread-safe live LLM token generator.
        Every prompt goes directly to the active AI inference backend (zero static/canned responses).
        """
        abort_event = self._register_abort_event(session_id)
        save_chat_message(session_id, "user", prompt)

        # Wait if lazy background model initialization is still in progress
        if not self._init_done_event.is_set():
            self._init_done_event.wait(timeout=30.0)

        # --- Live LLM Inference ---
        intent = self.detect_conversation_intent(prompt) if mode in ("auto", "assistant", "") else mode
        self.auto_select_optimal_model(intent)

        # Retrieve distilled lessons from Teacher LLMs / Hugging Face datasets in SQLite FTS5 memory
        learned_context = search_knowledge(prompt, limit=2) if len(prompt.strip()) > 12 else []
        memory_parts: list[str] = []
        if learned_context:
            memory_parts.append("[Distilled Knowledge from Teacher Models]:\n")
            for k in learned_context:
                memory_parts.append(f"- Topic: {k['topic']}\n  Lesson: {k['lesson'][:600]}\n")
                if k.get("code_example"):
                    memory_parts.append(f"  Reference Code:\n{k['code_example'][:800]}\n")
        memory_prompt = "".join(memory_parts)

        system_content = self.build_system_prompt(mode=mode, detected_intent=intent)
        messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]

        if memory_prompt:
            messages.append({"role": "system", "content": memory_prompt})

        context_budget = int(os.getenv("CONTEXT_BUDGET", "1000"))
        context_history = get_context_window(session_id, max_tokens=context_budget)
        for h in context_history[:-1]:
            content = h["content"]
            if h["role"] == "assistant":
                content = _THINK_BLOCK_RE.sub("", content).strip()
            if content:
                messages.append({"role": h["role"], "content": content})

        safe_prompt = sanitize_prompt(fit_prompt_to_context(prompt))
        messages.append({"role": "user", "content": safe_prompt})

        max_tokens = int(os.getenv("MAX_TOKENS", "512"))
        stop_tokens = ["<|im_end|>", "<|endoftext|>", "</s>"]
        temperature = 0.2 if intent in ("coder", "reasoning") else 0.7

        if self.backend == "offline_standby":
            self.refresh_backend(fast=True)

        # Acquire in-process lock AFTER refresh_backend so backend transitions are respected
        needs_lock = (self.backend == "llama-cpp-python")
        acquired = False
        if needs_lock:
            acquired = self._inference_lock.acquire(timeout=15.0)
            if not acquired:
                msg = "[Server Busy: Another CPU generation is currently executing. Please wait a moment.]"
                yield msg
                self._unregister_abort_event(session_id, abort_event)
                return

        response_buffer: list[str] = []
        t_start = time.time()

        try:
            if self.backend == "ollama":
                for chunk in self._stream_from_ollama(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop_tokens,
                    abort_event=abort_event,
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
                    "stop": stop_tokens,
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
                        top_p=0.9,
                        top_k=40,
                        repeat_penalty=1.15,
                        max_tokens=max_tokens,
                        stream=True,
                        stop=stop_tokens,
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
            self._unregister_abort_event(session_id, abort_event)

            full_response = "".join(response_buffer)
            if full_response:
                save_chat_message(session_id, "assistant", full_response)
                elapsed = time.time() - t_start
                token_count = len(response_buffer)
                tps = round(token_count / max(elapsed, 0.001), 1)
                log_event("inference_stream", f"Completed generation: {token_count} chunks ({tps} chunks/s) via {self.backend}", "info")

                # Autonomous Self-Improvement Loop:
                # If this was a substantive technical/reasoning query not yet in FTS5 memory,
                # record a knowledge gap and trigger background Teacher LLM distillation if an API key is configured.
                if not learned_context and intent in ("coder", "reasoning") and len(safe_prompt) >= 20:
                    uncertainty_markers = (
                        "i'm not sure", "i am not sure", "i don't know", "i do not know",
                        "cannot verify", "as a small", "beyond my knowledge",
                    )
                    should_learn = any(m in full_response.lower() for m in uncertainty_markers) or len(safe_prompt) >= 35
                    if should_learn:
                        try:
                            add_gap(safe_prompt[:300], context=f"Auto-captured from {intent} session ({self.active_model_name})")
                            teacher_key = (
                                os.getenv("TEACHER_API_KEY")
                                or os.getenv("GEMINI_API_KEY")
                                or os.getenv("GOOGLE_API_KEY")
                                or os.getenv("GROQ_API_KEY")
                                or os.getenv("OPENAI_API_KEY")
                            )
                            if teacher_key:
                                threading.Thread(
                                    target=run_sync,
                                    kwargs={"api_key": teacher_key},
                                    daemon=True,
                                ).start()
                        except Exception:
                            pass

    chat_stream = generate_stream

    def generate_chat_completion_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: Any | None = None,
    ) -> Iterator[str]:
        """OpenAI-compatible streaming generator for /v1/chat/completions."""
        if not self._init_done_event.is_set():
            self._init_done_event.wait(timeout=30.0)

        abort_event = threading.Event()
        stop_tokens = stop if isinstance(stop, list) else (["<|im_end|>", "<|endoftext|>", "</s>"] if not stop else [stop])

        if self.backend == "offline_standby":
            self.refresh_backend(fast=True)

        if self.backend == "ollama":
            yield from self._stream_from_ollama(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_tokens,
                abort_event=abort_event,
            )
            return

        if self.backend == "llama-server":
            payload_dict = {
                "messages": messages,
                "temperature": temperature,
                "stream": True,
                "max_tokens": max_tokens,
                "stop": stop_tokens,
            }
            yield from self._stream_from_llama_server(payload_dict, abort_event)
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
                    top_p=0.9,
                    top_k=40,
                    repeat_penalty=1.15,
                    max_tokens=max_tokens,
                    stream=True,
                    stop=stop_tokens,
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

    def generate_chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stream: bool = False,
        stop: Any | None = None,
    ):
        """OpenAI-compatible handler for /v1/chat/completions."""
        if stream:
            return self.generate_chat_completion_stream(messages, temperature=temperature, max_tokens=max_tokens, stop=stop)

        return "".join(self.generate_chat_completion_stream(messages, temperature=temperature, max_tokens=max_tokens, stop=stop))
