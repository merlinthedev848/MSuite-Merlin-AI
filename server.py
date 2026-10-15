import os
import json
import urllib.parse
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import threading
import time

from memory import (
    init_db,
    add_todo, get_todos, toggle_todo, delete_todo,
    add_note, get_notes, update_note, delete_note,
    get_all_knowledge, add_knowledge, delete_knowledge,
    get_unresolved_gaps, add_gap, delete_gap,
    get_system_telemetry, get_recent_audit_logs, log_event
)
from engine import InferenceEngine
from sync_teacher import run_sync, sync_state
from import_dataset import import_hf_dataset, import_state, POPULAR_DATASETS

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "5000"))

engine = None
server_start_time = time.time()

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
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # 1. SSE Real-Time Streaming Chat
        if path == "/api/chat/stream":
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

            # Immediate initial comment ping so the browser/proxy establishes the stream with 0 latency
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

        # 2. Status & Model Health
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

        # 3. System Hardware Telemetry
        elif path == "/api/system":
            self.send_json(get_system_telemetry())
            return

        # 4. To-Dos
        elif path == "/api/todos":
            self.send_json(get_todos())
            return

        # 5. Notes
        elif path == "/api/notes":
            self.send_json(get_notes())
            return

        # 6. Knowledge Base
        elif path == "/api/knowledge":
            self.send_json(get_all_knowledge())
            return

        # 7. Knowledge Gaps
        elif path == "/api/gaps":
            self.send_json(get_unresolved_gaps())
            return

        # 8. Teacher Sync Status
        elif path == "/api/sync":
            self.send_json(sync_state)
            return

        # 9. Audit Logs
        elif path == "/api/audit":
            self.send_json(get_recent_audit_logs())
            return

        # 10. Available Downloaded Models
        elif path == "/api/models":
            self.send_json(engine.get_available_models())
            return

        # 11. Hugging Face Dataset Status & Curated Repos
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
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", 0))

        # Payload safety limit (max 1MB)
        if length > 1024 * 1024:
            self.send_json({"error": "Payload exceeds 1MB limit"}, status=413)
            return

        body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        try:
            data = json.loads(body)
        except Exception:
            data = {}

        # Abort streaming generation
        if path == "/api/chat/stop":
            session_id = data.get("session_id", "default")
            stopped = engine.stop_generation(session_id)
            self.send_json({"success": stopped})
            return

        # Switch Active Model
        elif path == "/api/models/active":
            model_name = data.get("model", "").strip()
            if not model_name:
                self.send_json({"error": "Model filename required"}, status=400)
                return
            success, msg = engine.set_active_model(model_name)
            self.send_json({"success": success, "message": msg, "active_model": engine.active_model_name}, status=200 if success else 400)
            return

        # To-Dos CRUD
        elif path == "/api/todos":
            task = data.get("task", "").strip()
            priority = data.get("priority", "normal")
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
            title = data.get("title", "").strip()
            content = data.get("content", "").strip()
            tags = data.get("tags", "").strip()
            if content:
                nid = add_note(title or "Untitled Note", content, tags)
                self.send_json({"success": True, "id": nid})
            else:
                self.send_json({"error": "Note content required"}, status=400)
            return

        elif path == "/api/notes/update":
            nid = data.get("id")
            title = data.get("title", "").strip()
            content = data.get("content", "").strip()
            tags = data.get("tags", "").strip()
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
            question = data.get("question", "").strip()
            context = data.get("context", "").strip()
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
                data.get("api_key", "").strip()
                or os.getenv("GEMINI_API_KEY", "")
                or os.getenv("GOOGLE_API_KEY", "")
                or os.getenv("TEACHER_API_KEY", "")
                or os.getenv("GROQ_API_KEY", "")
                or os.getenv("OPENAI_API_KEY", "")
            )
            topic = data.get("topic", "").strip()
            endpoint = data.get("endpoint", "").strip() or None
            model = data.get("model", "").strip() or None
            if not api_key:
                self.send_json({"error": "API Key is required to communicate with the teacher AI. Enter your Gemini API key or set GEMINI_API_KEY."}, status=400)
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
            repo_id = data.get("dataset", "").strip() or "openbmb/UltraData-Stack-v3-L2"
            subset = data.get("subset", "").strip() or None
            limit = int(data.get("limit", 50))
            save_local = bool(data.get("save_local", True))
            token = data.get("token", "").strip() or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or None

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
    print("  MERLIN AI — OFFLINE ASSISTANT & CODER (PRODUCTION READY)")
    print(f"  Local Access:      http://localhost:{PORT}")
    print(f"  Network Access:    http://0.0.0.0:{PORT} (Open from any LAN device)")
    print(f"  Model Engine:      {engine.backend} ({engine.active_model_name})")
    print("=" * 66)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()

if __name__ == "__main__":
    run()
