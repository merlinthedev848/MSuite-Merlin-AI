import os
import json
import urllib.request
import urllib.error
from memory import search_knowledge, get_recent_chat, save_chat_message

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DEFAULT_LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080")

class InferenceEngine:
    def __init__(self):
        self.llama_cpp_model = None
        self.backend = None
        self._init_backend()

    def _init_backend(self):
        # 1. Check if external llama-server is responding
        try:
            req = urllib.request.Request(f"{DEFAULT_LLAMA_SERVER_URL}/health", headers={"User-Agent": "Aether/1.0"})
            with urllib.request.urlopen(req, timeout=1.0) as res:
                if res.status == 200:
                    self.backend = "llama-server"
                    print(f"[*] Connected to C++ llama-server at {DEFAULT_LLAMA_SERVER_URL}")
                    return
        except Exception:
            pass

        # 2. Check for llama-cpp-python
        try:
            from llama_cpp import Llama
            # Find first .gguf in models/
            if os.path.exists(MODELS_DIR):
                models = [f for f in os.listdir(MODELS_DIR) if f.endswith(".gguf")]
                if models:
                    model_path = os.path.join(MODELS_DIR, models[0])
                    print(f"[*] Loading model in-process with llama-cpp-python: {model_path}...")
                    self.llama_cpp_model = Llama(
                        model_path=model_path,
                        n_ctx=4096,
                        n_threads=int(os.getenv("THREADS", "4")),
                        verbose=False
                    )
                    self.backend = "llama-cpp-python"
                    return
        except ImportError:
            pass
        except Exception as e:
            print(f"[!] Warning loading local model: {e}")

        self.backend = "standalone_api"
        print("[*] Running in standalone API mode. Connect llama-server or install llama-cpp-python to enable inference.")

    def build_system_prompt(self, mode="assistant"):
        if mode == "coder":
            return (
                "You are an expert, offline software engineer and coding assistant running locally on a private server. "
                "Provide clean, efficient, well-documented code. Explain architectural decisions concisely. "
                "Always check for edge cases and prefer modern best practices."
            )
        else:
            return (
                "You are an intelligent, offline Personal Assistant (PA) and coding companion running locally on the user's private network. "
                "You are concise, helpful, capable, and respectful of privacy. You assist with daily tasks, server diagnostics, "
                "and programming solutions."
            )

    def generate_stream(self, prompt, session_id="default", mode="assistant"):
        """
        Yields text chunks as they are generated.
        Augments prompt with retrieved long-term memory and conversation history.
        """
        # Save user message
        save_chat_message(session_id, "user", prompt)

        # Retrieve relevant learned skills
        learned_context = search_knowledge(prompt, limit=2)
        memory_prompt = ""
        if learned_context:
            memory_prompt = "\n[Relevant learned knowledge from past teacher syncs]:\n"
            for k in learned_context:
                memory_prompt += f"- Topic: {k['topic']}\n  Lesson: {k['lesson']}\n"
                if k['code_example']:
                    memory_prompt += f"  Code Reference:\n{k['code_example']}\n"

        # Build messages with recent history
        history = get_recent_chat(session_id, limit=6)
        messages = [{"role": "system", "content": self.build_system_prompt(mode) + memory_prompt}]
        for h in history[:-1]: # exclude latest user prompt which is passed below
            messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": prompt})

        full_response = ""

        if self.backend == "llama-server":
            # Stream from llama-server (OpenAI-compatible /v1/chat/completions)
            payload = json.dumps({
                "messages": messages,
                "temperature": 0.3 if mode == "coder" else 0.7,
                "stream": True,
                "max_tokens": 2048
            }).encode("utf-8")

            req = urllib.request.Request(
                f"{DEFAULT_LLAMA_SERVER_URL}/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json", "User-Agent": "Aether/1.0"}
            )
            try:
                with urllib.request.urlopen(req) as resp:
                    for line in resp:
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
                err_msg = f"[Error communicating with llama-server: {e}]"
                yield err_msg
                full_response = err_msg

        elif self.backend == "llama-cpp-python" and self.llama_cpp_model:
            stream = self.llama_cpp_model.create_chat_completion(
                messages=messages,
                temperature=0.3 if mode == "coder" else 0.7,
                max_tokens=2048,
                stream=True
            )
            for chunk in stream:
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    full_response += delta
                    yield delta

        else:
            msg = (
                "⚠️ **Assistant Offline Engine Notice**\n\n"
                "The server interface and memory database are fully operational! However, no local LLM engine is running yet.\n\n"
                "**To activate 100% offline intelligence:**\n"
                "1. Run `./download_model.sh` to download Qwen2.5-Coder-1.5B (GGUF).\n"
                "2. Start `llama-server` (`./start_llama_server.sh`) OR install `pip install llama-cpp-python`.\n\n"
                "All your tasks, notes, telemetry, and teacher sync will remain persistently stored."
            )
            yield msg
            full_response = msg

        # Save assistant message to memory history
        save_chat_message(session_id, "assistant", full_response)
