import os
import json
import urllib.parse
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import threading

from memory import (
    init_db,
    add_todo, get_todos, toggle_todo, delete_todo,
    add_note, get_notes, delete_note,
    get_all_knowledge, add_knowledge,
    get_unresolved_gaps, add_gap,
    get_system_telemetry
)
from engine import InferenceEngine
from sync_teacher import run_sync

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "5000"))

engine = None

class AssistantRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # SSE Streaming Chat
        if path == "/api/chat/stream":
            query = urllib.parse.parse_qs(parsed.query)
            prompt = query.get("prompt", [""])[0]
            mode = query.get("mode", ["assistant"])[0]
            session_id = query.get("session_id", ["default"])[0]

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            try:
                for chunk in engine.generate_stream(prompt, session_id=session_id, mode=mode):
                    payload = f"data: {json.dumps({'chunk': chunk})}\n\n"
                    self.wfile.write(payload.encode("utf-8"))
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except Exception as e:
                pass
            return

        # System Telemetry
        elif path == "/api/system":
            self.send_json(get_system_telemetry())
            return

        # To-Dos
        elif path == "/api/todos":
            self.send_json(get_todos())
            return

        # Notes
        elif path == "/api/notes":
            self.send_json(get_notes())
            return

        # Knowledge Base
        elif path == "/api/knowledge":
            self.send_json(get_all_knowledge())
            return

        # Knowledge Gaps
        elif path == "/api/gaps":
            self.send_json(get_unresolved_gaps())
            return

        # Serve static web files
        if path == "/" or path == "":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        try:
            data = json.loads(body)
        except Exception:
            data = {}

        if path == "/api/todos":
            task = data.get("task", "").strip()
            if task:
                tid = add_todo(task)
                self.send_json({"success": True, "id": tid})
            else:
                self.send_json({"error": "Task required"}, status=400)
            return

        elif path == "/api/todos/toggle":
            tid = data.get("id")
            if tid:
                toggle_todo(tid)
                self.send_json({"success": True})
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

        elif path == "/api/notes":
            title = data.get("title", "").strip()
            content = data.get("content", "").strip()
            if title or content:
                nid = add_note(title or "Untitled Note", content)
                self.send_json({"success": True, "id": nid})
            else:
                self.send_json({"error": "Content required"}, status=400)
            return

        elif path == "/api/notes/delete":
            nid = data.get("id")
            if nid:
                delete_note(nid)
                self.send_json({"success": True})
            else:
                self.send_json({"error": "ID required"}, status=400)
            return

        elif path == "/api/gaps":
            question = data.get("question", "").strip()
            context = data.get("context", "").strip()
            if question:
                gid = add_gap(question, context)
                self.send_json({"success": True, "id": gid})
            else:
                self.send_json({"error": "Question required"}, status=400)
            return

        elif path == "/api/sync":
            api_key = data.get("api_key", "").strip()
            topic = data.get("topic", "").strip()
            if not api_key:
                self.send_json({"error": "API Key is required to sync with commercial AI"}, status=400)
                return

            def bg_sync():
                run_sync(api_key, topic_to_explore=topic or None)

            t = threading.Thread(target=bg_sync, daemon=True)
            t.start()
            self.send_json({"success": True, "message": "Teacher synchronization started in background."})
            return

        self.send_json({"error": "Endpoint not found"}, status=404)

def run():
    global engine
    init_db()
    engine = InferenceEngine()
    server = ThreadingHTTPServer((HOST, PORT), AssistantRequestHandler)
    print(f"\n==================================================================")
    print(f"  AETHER OFFLINE AI ASSISTANT & CODER")
    print(f"  Local Access:      http://localhost:{PORT}")
    print(f"  Network Access:    http://0.0.0.0:{PORT} (Open from any LAN device)")
    print(f"  Model Backend:     {engine.backend}")
    print(f"==================================================================\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()

if __name__ == "__main__":
    run()
