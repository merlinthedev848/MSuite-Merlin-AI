import os
import json
import urllib.request
import urllib.error
import threading
from memory import search_knowledge, get_recent_chat, save_chat_message, log_event

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DEFAULT_LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080")

class InferenceEngine:
    def __init__(self):
        self.llama_cpp_model = None
        self.backend = None
        self.active_model_name = "None"
        # Concurrency Lock: Prevents simultaneous access to the C++ inference engine
        self._inference_lock = threading.Lock()
        self._active_abort_events = {} # session_id -> threading.Event()
        self.refresh_backend()

    def refresh_backend(self):
        """Checks for llama-server or local GGUF models."""
        # 1. Probe C++ llama-server
        try:
            req = urllib.request.Request(f"{DEFAULT_LLAMA_SERVER_URL}/health", headers={"User-Agent": "Aether/1.0"})
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
                    threads = int(os.getenv("THREADS", "4"))
                    self.llama_cpp_model = Llama(
                        model_path=model_path,
                        n_ctx=4096,
                        n_threads=threads,
                        verbose=False
                    )
                    self.backend = "llama-cpp-python"
                    log_event("engine_init", f"Loaded local model {models[0]} with {threads} threads", "success")
                    return
                except ImportError:
                    pass
                except Exception as e:
                    log_event("engine_error", f"Failed loading GGUF: {e}", "danger")

        self.backend = "offline_standby"
        self.active_model_name = "Waiting for model download or llama-server"

    def stop_generation(self, session_id):
        """Signals an active stream to abort immediately."""
        if session_id in self._active_abort_events:
            self._active_abort_events[session_id].set()
            return True
        return False

    def build_system_prompt(self, mode="assistant"):
        if mode == "coder":
            return (
                "You are an expert, offline software engineer and coding assistant running locally on a private server. "
                "Provide clean, efficient, production-ready code. Always consider memory limits, edge cases, and algorithmic complexity. "
                "Keep prose explanations concise and accurate."
            )
        else:
            return (
                "You are an intelligent, offline Personal Assistant (PA) and coding companion running locally on the user's private network. "
                "You assist with daily tasks, server diagnostics, scheduling, and programming questions. "
                "Be helpful, direct, and concise."
            )

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

        # Retrieve relevant learned skills from SQLite
        learned_context = search_knowledge(prompt, limit=2)
        memory_prompt = ""
        if learned_context:
            memory_prompt = "\n[Verified lessons learned from past teacher sessions]:\n"
            for k in learned_context:
                memory_prompt += f"- Topic: {k['topic']}\n  Lesson: {k['lesson']}\n"
                if k['code_example']:
                    memory_prompt += f"  Reference Code:\n{k['code_example']}\n"

        # Assemble conversation context
        history = get_recent_chat(session_id, limit=6)
        messages = [{"role": "system", "content": self.build_system_prompt(mode) + memory_prompt}]
        for h in history[:-1]:
            messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": prompt})

        full_response = ""

        # Acquire lock to ensure only one thread generates on the CPU at a time
        acquired = self._inference_lock.acquire(timeout=45.0)
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
                payload = json.dumps({
                    "messages": messages,
                    "temperature": 0.2 if mode == "coder" else 0.7,
                    "stream": True,
                    "max_tokens": 2048
                }).encode("utf-8")

                req = urllib.request.Request(
                    f"{DEFAULT_LLAMA_SERVER_URL}/v1/chat/completions",
                    data=payload,
                    headers={"Content-Type": "application/json", "User-Agent": "Aether/1.0"}
                )
                try:
                    with urllib.request.urlopen(req, timeout=120) as resp:
                        for line in resp:
                            if abort_event.is_set():
                                break
                            line_str = line.decode("utf-8").strip()
                            if line_str.startswith("data: "):
                                data_content = line_str[6:]
                                if data_content == "[DONE]":
                                    break
                                try:
                                    chunk_json = json.loads(data_content)
                                    delta = chunk_json["choices"][0]["delta"].get("content", "")
                                    if delta:
                                        full_response += delta
                                        yield delta
                                except Exception:
                                    continue
                except Exception as e:
                    err = f"[Error querying llama-server: {e}]"
                    full_response += err
                    yield err

            elif self.backend == "llama-cpp-python" and self.llama_cpp_model:
                stream = self.llama_cpp_model.create_chat_completion(
                    messages=messages,
                    temperature=0.2 if mode == "coder" else 0.7,
                    max_tokens=2048,
                    stream=True
                )
                for chunk in stream:
                    if abort_event.is_set():
                        break
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        full_response += delta
                        yield delta

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
