import os
import json
import urllib.parse
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import threading
import time
import base64
import re
import struct

from memory import (
    init_db,
    create_conversation, get_conversations, get_conversation, update_conversation, delete_conversation,
    save_message, get_conversation_messages, clear_conversation_messages,
    add_todo, get_todos, toggle_todo, delete_todo,
    add_note, get_notes, update_note, delete_note,
    get_all_knowledge, add_knowledge, delete_knowledge,
    get_unresolved_gaps, add_gap, delete_gap,
    get_system_telemetry, get_recent_audit_logs, log_event
)
from engine import InferenceEngine
from sync_teacher import run_sync, sync_state, query_gemini_multimodal, fetch_live_models_for_provider
from import_dataset import import_hf_dataset, import_state, POPULAR_DATASETS

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
UPLOADS_DIR = os.path.join(WEB_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "5000"))
API_KEY_SECRET = os.getenv("MERLIN_API_KEY") or os.getenv("API_KEY")

engine = None
server_start_time = time.time()

def format_bytes(size):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}" if unit != 'B' else f"{size} B"
        size /= 1024.0
    return f"{size:.1f} TB"

def check_auth(headers):
    """Optional bearer token authentication for OpenAI-compatible and management endpoints."""
    if not API_KEY_SECRET:
        return True
    auth = headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:].strip()
        return token == API_KEY_SECRET
    return False

class AssistantRequestHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # -------------------------------------------------------------
        # OpenAI-Compatible Endpoints: GET /v1/models & /v1/models/<id>
        # -------------------------------------------------------------
        if path == "/v1/models":
            available = engine.get_available_models()
            model_items = [
                {
                    "id": "merlin-ai",
                    "object": "model",
                    "created": int(server_start_time),
                    "owned_by": "merlin-ai",
                    "permission": [],
                    "root": "merlin-ai",
                    "parent": None
                }
            ]
            for m in available:
                model_items.append({
                    "id": m["name"],
                    "object": "model",
                    "created": int(server_start_time),
                    "owned_by": "local",
                    "permission": [],
                    "root": m["filename"],
                    "parent": None
                })
            self.send_json({"object": "list", "data": model_items})
            return

        elif path.startswith("/v1/models/"):
            model_id = path.split("/v1/models/")[1].strip()
            self.send_json({
                "id": model_id,
                "object": "model",
                "created": int(server_start_time),
                "owned_by": "local"
            })
            return

        # -------------------------------------------------------------
        # Conversational State Engine: GET /api/conversations
        # -------------------------------------------------------------
        elif path == "/api/conversations":
            self.send_json(get_conversations())
            return

        elif path.startswith("/api/conversations/"):
            conv_id = path.split("/api/conversations/")[1].strip().split("/")[0]
            conv = get_conversation(conv_id)
            if not conv:
                self.send_json({"error": "Conversation not found"}, status=404)
                return
            messages = get_conversation_messages(conv_id)
            self.send_json({"conversation": conv, "messages": messages})
            return

        # -------------------------------------------------------------
        # Native SSE Real-Time Streaming Chat: /api/chat/stream
        # -------------------------------------------------------------
        elif path == "/api/chat/stream":
            query = urllib.parse.parse_qs(parsed.query)
            prompt = query.get("prompt", [""])[0]
            mode = query.get("mode", ["assistant"])[0]
            session_id = query.get("session_id", ["default"])[0]

            if not prompt.strip():
                self.send_json({"error": "Prompt cannot be empty"}, status=400)
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-transform")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            # Immediate initial comment ping so the browser/proxy establishes stream with 0 latency
            try:
                self.wfile.write(b": connected\n\n")
                self.wfile.flush()
            except Exception:
                return

            try:
                for chunk in engine.generate_stream(prompt, session_id=session_id, mode=mode):
                    payload = f"data: {json.dumps({'chunk': chunk})}\n\n"
                    self.wfile.write(payload.encode("utf-8"))
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                engine.stop_generation(session_id)
            except Exception as e:
                log_event("stream_error", f"Stream exception: {e}", "danger")
            return

        # -------------------------------------------------------------
        # Status, System & Knowledge Endpoints
        # -------------------------------------------------------------
        elif path == "/api/status":
            uptime_sec = int(time.time() - server_start_time)
            status_data = {
                "backend": engine.backend,
                "model": engine.active_model_name,
                "uptime_seconds": uptime_sec,
                "knowledge_count": len(get_all_knowledge()),
                "pending_gaps": len(get_unresolved_gaps())
            }
            self.send_json(status_data)
            return

        elif path == "/api/system":
            self.send_json(get_system_telemetry())
            return

        elif path == "/api/todos":
            self.send_json(get_todos())
            return

        elif path == "/api/notes":
            self.send_json(get_notes())
            return

        elif path == "/api/knowledge":
            self.send_json(get_all_knowledge())
            return

        elif path == "/api/gaps":
            self.send_json(get_unresolved_gaps())
            return

        elif path == "/api/sync":
            self.send_json(sync_state)
            return

        elif path == "/api/teacher/models":
            query = urllib.parse.parse_qs(parsed.query)
            provider = query.get("provider", ["gemini"])[0]
            api_key = query.get("api_key", [""])[0]
            models = fetch_live_models_for_provider(provider, api_key)
            self.send_json({"provider": provider, "models": models})
            return

        elif path == "/api/audit":
            self.send_json(get_recent_audit_logs())
            return

        elif path == "/api/models":
            self.send_json(engine.get_available_models())
            return

        elif path == "/api/dataset/status":
            self.send_json(import_state)
            return

        elif path == "/api/dataset/popular":
            self.send_json(POPULAR_DATASETS)
            return

        # Static Web App
        if path in ["", "/"]:
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        try:
            self._handle_post()
        except Exception as e:
            log_event("server_post_error", f"Unhandled error in do_POST: {e}", "danger")
            try:
                self.send_json({"error": f"Internal Server Error: {e}"}, status=500)
            except Exception:
                pass

    def do_DELETE(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path.startswith("/api/conversations/"):
            conv_id = path.split("/api/conversations/")[1].strip().split("/")[0]
            success = delete_conversation(conv_id)
            self.send_json({"success": success})
            return
        self.send_json({"error": "Not found"}, status=404)

    def _handle_post(self):
        def safe_str(val):
            return str(val).strip() if val is not None else ""

        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", 0))

        # Payload safety limit (max 35MB for files/audio/images)
        if length > 35 * 1024 * 1024:
            self.send_json({"error": "Payload exceeds 35MB limit"}, status=413)
            return

        body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        try:
            data = json.loads(body)
        except Exception:
            data = {}

        # -------------------------------------------------------------
        # OpenAI-Compatible REST API: POST /v1/chat/completions
        # -------------------------------------------------------------
        if path == "/v1/chat/completions":
            if not check_auth(self.headers):
                self.send_json({"error": {"message": "Invalid API key provided", "type": "invalid_request_error"}}, status=401)
                return

            messages = data.get("messages", [])
            if not messages or not isinstance(messages, list):
                self.send_json({"error": {"message": "'messages' array is required", "type": "invalid_request_error"}}, status=400)
                return

            model = safe_str(data.get("model")) or engine.active_model_name or "merlin-ai"
            stream = bool(data.get("stream", False))
            temperature = float(data.get("temperature", 0.7))
            max_tokens = int(data.get("max_tokens", 1024))
            stop = data.get("stop", None)

            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache, no-transform")
                self.send_header("Connection", "keep-alive")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                chat_id = f"chatcmpl-{int(time.time()*1000)}"
                created_ts = int(time.time())

                try:
                    for delta_chunk in engine.generate_chat_completion(
                        messages, temperature=temperature, max_tokens=max_tokens, stream=True, stop=stop
                    ):
                        chunk_obj = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": delta_chunk},
                                    "finish_reason": None
                                }
                            ]
                        }
                        self.wfile.write(f"data: {json.dumps(chunk_obj)}\n\n".encode("utf-8"))
                        self.wfile.flush()

                    # Final finish chunk
                    final_chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop"
                            }
                        ]
                    }
                    self.wfile.write(f"data: {json.dumps(final_chunk)}\n\n".encode("utf-8"))
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except Exception as e:
                    log_event("v1_stream_error", f"Error in /v1/chat/completions stream: {e}", "danger")
                return
            else:
                completion_text = engine.generate_chat_completion(
                    messages, temperature=temperature, max_tokens=max_tokens, stream=False, stop=stop
                )
                prompt_tokens = sum(len(m.get("content", "")) // 4 for m in messages)
                completion_tokens = len(str(completion_text)) // 4
                self.send_json({
                    "id": f"chatcmpl-{int(time.time()*1000)}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": str(completion_text)
                            },
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    }
                })
                return

        # -------------------------------------------------------------
        # Conversational State Engine: POST /api/conversations
        # -------------------------------------------------------------
        elif path == "/api/conversations":
            title = safe_str(data.get("title")) or "New Chat"
            mode = safe_str(data.get("mode")) or "assistant"
            conv_id = safe_str(data.get("id")) or None
            conv = create_conversation(title=title, mode=mode, conv_id=conv_id)
            self.send_json(conv)
            return

        elif path.startswith("/api/conversations/") and path.endswith("/clear"):
            conv_id = path.split("/api/conversations/")[1].split("/clear")[0].strip()
            clear_conversation_messages(conv_id)
            self.send_json({"success": True})
            return

        elif path.startswith("/api/conversations/") and path.endswith("/delete"):
            conv_id = path.split("/api/conversations/")[1].split("/delete")[0].strip()
            success = delete_conversation(conv_id)
            self.send_json({"success": success})
            return

        elif path.startswith("/api/conversations/") and path.endswith("/messages"):
            conv_id = path.split("/api/conversations/")[1].split("/messages")[0].strip()
            role = safe_str(data.get("role")) or "user"
            content = safe_str(data.get("content"))
            meta = data.get("metadata", {})
            mid = save_message(conv_id, role, content, metadata=meta)
            self.send_json({"success": True, "id": mid})
            return

        elif path.startswith("/api/conversations/"):
            conv_id = path.split("/api/conversations/")[1].strip()
            title = data.get("title")
            mode = data.get("mode")
            pinned = data.get("pinned")
            success = update_conversation(conv_id, title=title, mode=mode, pinned=pinned)
            self.send_json({"success": success})
            return

        # -------------------------------------------------------------
        # File Upload (Audio, Images, Code, Documents)
        # -------------------------------------------------------------
        elif path == "/api/upload":
            name = safe_str(data.get("name")) or "upload.bin"
            clean_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', os.path.basename(name))
            file_ext = os.path.splitext(clean_name)[1].lower()
            unique_id = f"{int(time.time())}_{clean_name}"
            file_path = os.path.join(UPLOADS_DIR, unique_id)

            raw_b64 = data.get("data", "")
            if "," in raw_b64:
                raw_b64 = raw_b64.split(",", 1)[1]
            try:
                file_bytes = base64.b64decode(raw_b64)
            except Exception as e:
                self.send_json({"error": f"Invalid base64 payload: {e}"}, status=400)
                return

            with open(file_path, "wb") as f:
                f.write(file_bytes)

            text_exts = {
                ".txt", ".py", ".js", ".jsx", ".ts", ".tsx", ".html", ".css", ".json",
                ".csv", ".md", ".sh", ".c", ".cpp", ".h", ".hpp", ".rs", ".go", ".sql",
                ".yaml", ".yml", ".xml", ".log", ".ini", ".env", ".toml", ".conf"
            }
            image_exts = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg", ".bmp"}
            audio_exts = {".wav", ".mp3", ".ogg", ".m4a", ".webm", ".flac", ".aac"}
            doc_exts = {".pdf", ".docx", ".doc"}

            category = "other"
            extracted_text = ""
            dimensions = None

            if file_ext in text_exts:
                category = "text"
                try:
                    extracted_text = file_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    extracted_text = file_bytes.decode("latin-1", errors="replace")
            elif file_ext in doc_exts:
                category = "document"
                if file_ext == ".pdf":
                    try:
                        import pypdf
                        reader = pypdf.PdfReader(file_path)
                        extracted_text = "\n".join([page.extract_text() or "" for page in reader.pages])
                    except Exception:
                        try:
                            raw_str = file_bytes.decode("latin-1", errors="ignore")
                            text_pieces = re.findall(r'\(([^\(\)\\]{2,})\)', raw_str)
                            if text_pieces:
                                extracted_text = " ".join(text_pieces[:500])
                        except Exception:
                            pass
            elif file_ext in image_exts:
                category = "image"
                try:
                    if file_bytes.startswith(b'\x89PNG\r\n\x1a\n') and len(file_bytes) >= 24:
                        w, h = struct.unpack(">II", file_bytes[16:24])
                        dimensions = f"{w}x{h}"
                except Exception:
                    pass
            elif file_ext in audio_exts:
                category = "audio"

            log_event("file_upload", f"Uploaded {clean_name} ({category}, {format_bytes(len(file_bytes))})", "info")
            self.send_json({
                "success": True,
                "file": {
                    "id": unique_id,
                    "name": clean_name,
                    "url": f"/uploads/{unique_id}",
                    "category": category,
                    "size": len(file_bytes),
                    "size_str": format_bytes(len(file_bytes)),
                    "extracted_text": extracted_text[:40000] if extracted_text else "",
                    "lines": len(extracted_text.splitlines()) if extracted_text else 0,
                    "dimensions": dimensions
                }
            })
            return

        # -------------------------------------------------------------
        # Multimodal AI Analysis (Images, Audio, Docs)
        # -------------------------------------------------------------
        elif path == "/api/analyze/multimodal":
            file_id = safe_str(data.get("file_id"))
            prompt = safe_str(data.get("prompt")) or "Analyze this file thoroughly and describe all details, insights, or text."
            api_key = safe_str(data.get("api_key")) or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("TEACHER_API_KEY")

            if not api_key:
                self.send_json({"error": "Google Gemini API key required for deep multimodal vision/audio analysis."}, status=400)
                return

            file_path = os.path.join(UPLOADS_DIR, file_id)
            if not os.path.exists(file_path):
                self.send_json({"error": "File not found"}, status=404)
                return

            with open(file_path, "rb") as f:
                raw_data = base64.b64encode(f.read()).decode("utf-8")

            ext = os.path.splitext(file_id)[1].lower()
            mime_map = {
                ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".webp": "image/webp", ".gif": "image/gif",
                ".mp3": "audio/mp3", ".wav": "audio/wav", ".ogg": "audio/ogg", ".m4a": "audio/m4a"
            }
            mime_type = mime_map.get(ext, "application/octet-stream")

            try:
                analysis = query_gemini_multimodal(prompt, raw_data, mime_type, api_key)
                self.send_json({"success": True, "analysis": analysis})
            except Exception as e:
                self.send_json({"error": str(e)}, status=500)
            return

        # Abort streaming generation
        elif path == "/api/chat/stop":
            session_id = safe_str(data.get("session_id")) or "default"
            stopped = engine.stop_generation(session_id)
            self.send_json({"success": stopped})
            return

        # Switch Active Model
        elif path == "/api/models/active":
            model_name = safe_str(data.get("model"))
            if not model_name:
                self.send_json({"error": "Model filename required"}, status=400)
                return
            success, msg = engine.set_active_model(model_name)
            self.send_json({"success": success, "message": msg, "active_model": engine.active_model_name}, status=200 if success else 400)
            return

        # To-Dos CRUD
        elif path == "/api/todos":
            task = safe_str(data.get("task"))
            priority = safe_str(data.get("priority")) or "normal"
            if task:
                tid = add_todo(task, priority)
                self.send_json({"success": True, "id": tid})
            else:
                self.send_json({"error": "Task text required"}, status=400)
            return

        elif path == "/api/todos/toggle":
            tid = data.get("id")
            if tid:
                success = toggle_todo(tid)
                self.send_json({"success": success})
            else:
                self.send_json({"error": "ID required"}, status=400)
            return

        elif path == "/api/todos/delete":
            tid = data.get("id")
            if tid:
                delete_todo(tid)
                self.send_json({"success": True})
            else:
                self.send_json({"error": "ID required"}, status=400)
            return

        # Notes CRUD
        elif path == "/api/notes":
            title = safe_str(data.get("title"))
            content = safe_str(data.get("content"))
            tags = safe_str(data.get("tags"))
            if content:
                nid = add_note(title or "Untitled Note", content, tags)
                self.send_json({"success": True, "id": nid})
            else:
                self.send_json({"error": "Note content required"}, status=400)
            return

        elif path == "/api/notes/update":
            nid = data.get("id")
            title = safe_str(data.get("title"))
            content = safe_str(data.get("content"))
            tags = safe_str(data.get("tags"))
            if nid and content:
                update_note(nid, title or "Untitled Note", content, tags)
                self.send_json({"success": True})
            else:
                self.send_json({"error": "ID and content required"}, status=400)
            return

        elif path == "/api/notes/delete":
            nid = data.get("id")
            if nid:
                delete_note(nid)
                self.send_json({"success": True})
            else:
                self.send_json({"error": "ID required"}, status=400)
            return

        # Knowledge CRUD
        elif path == "/api/knowledge/delete":
            kid = data.get("id")
            if kid:
                delete_knowledge(kid)
                self.send_json({"success": True})
            else:
                self.send_json({"error": "ID required"}, status=400)
            return

        # Gaps CRUD
        elif path == "/api/gaps":
            question = safe_str(data.get("question"))
            context = safe_str(data.get("context"))
            if question:
                gid = add_gap(question, context)
                self.send_json({"success": True, "id": gid})
            else:
                self.send_json({"error": "Question required"}, status=400)
            return

        elif path == "/api/gaps/delete":
            gid = data.get("id")
            if gid:
                delete_gap(gid)
                self.send_json({"success": True})
            else:
                self.send_json({"error": "ID required"}, status=400)
            return

        # Trigger Teacher Sync
        elif path == "/api/sync":
            api_key = (
                safe_str(data.get("api_key"))
                or os.getenv("GEMINI_API_KEY", "")
                or os.getenv("GOOGLE_API_KEY", "")
                or os.getenv("TEACHER_API_KEY", "")
                or os.getenv("GROQ_API_KEY", "")
                or os.getenv("OPENAI_API_KEY", "")
            )
            topic = safe_str(data.get("topic"))
            endpoint = safe_str(data.get("endpoint")) or None
            model = safe_str(data.get("model")) or None
            if not api_key:
                self.send_json({"error": "API Key is required to communicate with teacher AI."}, status=400)
                return

            if sync_state.get("is_syncing"):
                self.send_json({"error": "Sync is already in progress"}, status=409)
                return

            def bg_sync():
                run_sync(api_key, endpoint=endpoint, model=model, topic_to_explore=topic or None)

            t = threading.Thread(target=bg_sync, daemon=True)
            t.start()
            self.send_json({"success": True, "message": "Teacher synchronization started."})
            return

        # Trigger Hugging Face Dataset Import
        elif path == "/api/dataset/import":
            repo_id = safe_str(data.get("dataset")) or "openbmb/UltraData-Stack-v3-L2"
            subset = safe_str(data.get("subset")) or None
            limit = int(data.get("limit") or 50)
            save_local = bool(data.get("save_local", True))
            token = safe_str(data.get("token")) or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or None

            if import_state.get("is_importing"):
                self.send_json({"error": "Another dataset import is already in progress"}, status=409)
                return

            def bg_import():
                import_hf_dataset(repo_id, subset=subset, limit=limit, save_local=save_local, token=token)

            t = threading.Thread(target=bg_import, daemon=True)
            t.start()
            self.send_json({"success": True, "message": f"Started streaming import from Hugging Face: {repo_id}"})
            return

        self.send_json({"error": f"Endpoint '{path}' not found"}, status=404)

def run():
    global engine
    init_db()
    engine = InferenceEngine()
    server = ThreadingHTTPServer((HOST, PORT), AssistantRequestHandler)
    print("=" * 66)
    print("  MERLIN AI — GEMINI-GRADE ASSISTANT & CODER (PRODUCTION READY)")
    print(f"  Web UI:            http://localhost:{PORT}")
    print(f"  OpenAI REST API:   http://localhost:{PORT}/v1/chat/completions")
    print(f"  Network Access:    http://0.0.0.0:{PORT} (Accessible across LAN)")
    print(f"  Model Engine:      {engine.backend} ({engine.active_model_name})")
    print("=" * 66)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()

if __name__ == "__main__":
    run()
