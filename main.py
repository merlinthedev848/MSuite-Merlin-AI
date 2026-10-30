import asyncio
import base64
import json
import os
import re
import struct
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from database import (
    add_gap,
    add_knowledge,
    add_note,
    add_todo,
    clear_session_messages,
    create_session,
    delete_gap,
    delete_knowledge,
    delete_note,
    delete_session,
    delete_todo,
    get_all_knowledge,
    get_counts,
    get_last_session_message,
    get_notes,
    get_recent_audit_logs,
    get_session,
    get_session_messages,
    get_sessions,
    get_system_telemetry,
    get_todos,
    get_unresolved_gaps,
    init_db,
    log_event,
    save_message,
    search_knowledge,
    toggle_todo,
    update_knowledge,
    update_note,
    update_session,
)
from engine import InferenceEngine
from import_dataset import POPULAR_DATASETS, import_hf_dataset, import_state
from sync_teacher import (
    fetch_live_models_for_provider,
    load_saved_api_key,
    query_gemini_argon,
    query_gemini_multimodal,
    run_sync,
    save_persistent_api_key,
    sync_state,
)

# Path configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")
UPLOADS_DIR = os.path.join(WEB_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "5000"))
API_KEY_SECRET = os.getenv("MERLIN_API_KEY") or os.getenv("API_KEY")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))  # 50 MB default

_SAFE_FILENAME_RE = re.compile(r"[^a-zA-Z0-9_.-]")
_SAFE_FILE_ID_RE = re.compile(r"^[a-zA-Z0-9_.-]+$")

server_start_time = time.time()
engine: InferenceEngine | None = None
_engine_lock = threading.Lock()


def get_engine(lazy_init: bool = False) -> InferenceEngine:
    global engine
    if engine is None:
        with _engine_lock:
            if engine is None:
                init_db()
                engine = InferenceEngine(lazy_init=lazy_init)
    return engine


_autonomous_stop_event = threading.Event()


def _autonomous_gemini_learning_worker() -> None:
    """
    Background self-improvement daemon:
    Whenever a Gemini / Teacher API key is configured, periodically resolves any
    captured Knowledge Gaps and studies the next unlearned curriculum topic from Gemini.
    """
    # Wait 25s after server boot before starting background learning
    if _autonomous_stop_event.wait(25.0):
        return
    while not _autonomous_stop_event.is_set():
        try:
            key = load_saved_api_key()
            if key and not sync_state.get("is_syncing"):
                run_sync(key)
        except Exception:
            pass
        # Study a new topic / resolve gaps every 5 minutes without hitting rate limits
        if _autonomous_stop_event.wait(300.0):
            break


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    # Load any persisted Gemini/Teacher API key from .env on startup
    saved_key = load_saved_api_key()
    # Initialize in lazy background mode so Uvicorn binds port 5000 in < 50ms
    # and never triggers 502 Bad Gateway while loading multi-GB GGUF weights.
    eng = get_engine(lazy_init=True)
    threading.Thread(target=_autonomous_gemini_learning_worker, daemon=True).start()
    print("=" * 66)
    print("  MERLIN AI — FASTAPI CORE RUNTIME (PRODUCTION READY)")
    print(f"  Web Dashboard:     http://localhost:{PORT}")
    print(f"  OpenAI REST API:   http://localhost:{PORT}/v1/chat/completions")
    print(f"  Network Host:      http://{HOST}:{PORT}")
    print(f"  Model Engine:      {eng.backend} ({eng.active_model_name})")
    print(f"  Gemini Learning:   {'ACTIVE (Autonomous 24/7)' if saved_key else 'Ready (Add API Key in UI)'}")
    print("=" * 66)
    yield
    _autonomous_stop_event.set()
    if engine is not None:
        engine.cleanup()


app = FastAPI(
    title="Merlin AI",
    description="High-performance, offline AI Assistant & Coder with modern reactive UX and OpenAI-compatible API.",
    version="2.0.0",
    lifespan=lifespan,
)

# Hardened CORS Middleware: strictly compliant with W3C spec (no wildcard origins with credentials)
CORS_ORIGINS = [orig.strip() for orig in os.getenv("CORS_ORIGINS", "*").split(",") if orig.strip()]
if "*" in CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def format_bytes(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{size:.1f} TB"


def verify_api_key(authorization: str | None = Header(None)) -> bool:
    """Validates optional Bearer token if API_KEY_SECRET is configured."""
    if not API_KEY_SECRET:
        return True
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"message": "Missing or invalid Bearer token", "type": "invalid_request_error"},
        )
    token = authorization[7:].strip()
    if token != API_KEY_SECRET:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"message": "Invalid API key provided", "type": "invalid_request_error"},
        )
    return True


# =====================================================================
# Strict Pydantic Request Contracts
# =====================================================================

class _BaseContract(BaseModel):
    model_config = ConfigDict(extra="ignore")


class OpenAIMessage(_BaseContract):
    role: str
    content: str


class OpenAIChatRequest(_BaseContract):
    model: str | None = "merlin-ai"
    messages: list[OpenAIMessage]
    stream: bool | None = False
    temperature: float | None = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=1024, ge=1, le=32768)
    stop: Any | None = None


class ChatStopRequest(_BaseContract):
    session_id: str = "default"


class SessionCreateRequest(_BaseContract):
    title: str | None = "New Chat"
    mode: str | None = "assistant"
    id: str | None = None


class SessionUpdateRequest(_BaseContract):
    title: str | None = None
    mode: str | None = None
    pinned: bool | int | None = None


class MessageAppendRequest(_BaseContract):
    role: str = "user"
    content: str = ""
    metadata: dict[str, Any] | None = None


class ModelSelectRequest(_BaseContract):
    model: str = ""


class TodoCreateRequest(_BaseContract):
    task: str = ""
    priority: str = "normal"


class IdRequest(_BaseContract):
    id: int | str | None = None


class NoteCreateRequest(_BaseContract):
    title: str = ""
    content: str = ""
    tags: str = ""


class NoteUpdateRequest(_BaseContract):
    id: int | str | None = None
    title: str = ""
    content: str = ""
    tags: str = ""


class KnowledgeCreateRequest(_BaseContract):
    topic: str = ""
    lesson: str = ""
    code_example: str = ""
    tags: str = ""
    source: str = "manual"


class KnowledgeUpdateRequest(_BaseContract):
    id: int | str | None = None
    topic: str = ""
    lesson: str = ""
    code_example: str = ""
    tags: str = ""


class GapCreateRequest(_BaseContract):
    question: str = ""
    context: str = ""


class FileUploadRequest(_BaseContract):
    name: str = "upload.bin"
    data: str = ""


class MultimodalAnalyzeRequest(_BaseContract):
    file_id: str = ""
    prompt: str = "Analyze this file thoroughly and describe all details."
    api_key: str = ""


class TeacherSyncRequest(_BaseContract):
    api_key: str = ""
    topic: str = ""
    endpoint: str | None = None
    model: str | None = None


class ArgonQueryRequest(_BaseContract):
    prompt: str = ""
    api_key: str = ""
    model: str = "gemini-4-argon"
    reasoning_effort: str = "high"
    thinking_budget: int | str | None = 2048
    system_instruction: str | None = None


class ArgonAssimilateRequest(_BaseContract):
    topic: str = ""
    lesson: str = ""
    code_example: str = ""
    tags: str = "gemini-4-argon,reasoning,distilled"


class DatasetImportRequest(_BaseContract):
    dataset: str = "openbmb/UltraData-Stack-v3-L2"
    subset: str | None = None
    limit: int | str | None = 50
    save_local: bool = True
    token: str | None = None


# =====================================================================
# OpenAI-Compatible REST API Layer
# =====================================================================

@app.get("/v1/models")
def list_v1_models():
    """Returns OpenAI-compatible model registry."""
    eng = get_engine()
    available = eng.get_available_models()
    data = [
        {
            "id": "merlin-ai",
            "object": "model",
            "created": int(server_start_time),
            "owned_by": "merlin-ai",
            "permission": [],
            "root": "merlin-ai",
            "parent": None,
        }
    ]
    for m in available:
        data.append({
            "id": m["name"],
            "object": "model",
            "created": int(server_start_time),
            "owned_by": "local",
            "permission": [],
            "root": m["filename"],
            "parent": None,
        })
    return {"object": "list", "data": data}


@app.get("/v1/models/{model_id}")
def get_v1_model(model_id: str):
    return {
        "id": model_id,
        "object": "model",
        "created": int(server_start_time),
        "owned_by": "local",
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: OpenAIChatRequest, request: Request, auth: bool = Depends(verify_api_key)):
    """Standard OpenAI POST /v1/chat/completions endpoint with async disconnect handling."""
    eng = get_engine()
    model_name = req.model or eng.active_model_name or "merlin-ai"
    messages_dicts = [{"role": m.role, "content": m.content} for m in req.messages]
    temp = req.temperature if req.temperature is not None else 0.7
    max_tok = req.max_tokens if req.max_tokens is not None else 1024

    if req.stream:
        chat_id = f"chatcmpl-{int(time.time() * 1000)}"
        created_ts = int(time.time())
        # Unbounded queue with producer-side backpressure prevents QueueFull on stop_sentinel
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        stop_sentinel = object()
        abort_event = threading.Event()

        def openai_worker():
            try:
                for delta_chunk in eng.generate_chat_completion_stream(
                    messages_dicts,
                    temperature=temp,
                    max_tokens=max_tok,
                    stop=req.stop,
                ):
                    if abort_event.is_set():
                        break
                    while queue.qsize() > 400 and not abort_event.is_set():
                        time.sleep(0.01)
                    loop.call_soon_threadsafe(queue.put_nowait, delta_chunk)
            except Exception as e:
                log_event("v1_worker_error", f"OpenAI stream worker error: {e}", "danger")
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, stop_sentinel)

        threading.Thread(target=openai_worker, daemon=True).start()

        async def sse_generator():
            try:
                while True:
                    if await request.is_disconnected():
                        abort_event.set()
                        break

                    try:
                        delta = await asyncio.wait_for(queue.get(), timeout=0.08)
                    except TimeoutError:
                        continue

                    if delta is stop_sentinel:
                        break

                    chunk_obj = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": model_name,
                        "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk_obj)}\n\n"

                final_chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                log_event("v1_stream_error", f"Stream error: {e}", "danger")
            finally:
                abort_event.set()

        return StreamingResponse(
            sse_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Offload synchronous generation to worker thread so event loop stays responsive
    completion_text = await asyncio.to_thread(
        eng.generate_chat_completion,
        messages_dicts,
        temp,
        max_tok,
        False,
        req.stop,
    )
    prompt_tokens = sum(len(m.content) // 4 for m in req.messages)
    completion_tokens = len(str(completion_text)) // 4
    return {
        "id": f"chatcmpl-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": str(completion_text)},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


# =====================================================================
# Native Web UI Real-Time Token Streaming
# =====================================================================

@app.get("/api/chat/stream")
async def chat_stream(request: Request, prompt: str, mode: str = "assistant", session_id: str = "default"):
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    eng = get_engine()
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    stop_sentinel = object()
    abort_event = threading.Event()

    def stream_worker():
        try:
            for chunk in eng.generate_stream(prompt, session_id=session_id, mode=mode):
                if abort_event.is_set():
                    break
                while queue.qsize() > 400 and not abort_event.is_set():
                    time.sleep(0.01)
                loop.call_soon_threadsafe(queue.put_nowait, chunk)
        except Exception as e:
            log_event("stream_worker_error", f"Stream worker exception: {e}", "danger")
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, stop_sentinel)

    threading.Thread(target=stream_worker, daemon=True).start()

    async def sse_event_stream():
        yield ": connected\n\n"
        try:
            while True:
                if await request.is_disconnected():
                    abort_event.set()
                    eng.stop_generation(session_id)
                    break

                try:
                    chunk = await asyncio.wait_for(queue.get(), timeout=0.08)
                except TimeoutError:
                    continue

                if chunk is stop_sentinel:
                    break

                yield f"data: {json.dumps({'chunk': chunk})}\n\n"

            yield "data: [DONE]\n\n"
        except (BrokenPipeError, ConnectionResetError):
            abort_event.set()
            eng.stop_generation(session_id)
        except Exception as e:
            log_event("stream_error", f"Stream exception: {e}", "danger")
        finally:
            abort_event.set()
            eng.stop_generation(session_id)

    return StreamingResponse(
        sse_event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/chat/stop")
def chat_stop(data: ChatStopRequest):
    sid = data.session_id.strip() or "default"
    stopped = get_engine().stop_generation(sid)
    return {"success": stopped}


# =====================================================================
# Sessions / Conversations Endpoints
# =====================================================================

@app.get("/api/conversations")
@app.get("/api/sessions")
def list_conversations():
    return get_sessions()


@app.post("/api/conversations")
@app.post("/api/sessions")
def create_new_conversation(data: SessionCreateRequest):
    title = (data.title or "New Chat").strip() or "New Chat"
    mode = (data.mode or "assistant").strip() or "assistant"
    cid = (data.id or "").strip() or None
    return create_session(title=title, mode=mode, session_id=cid)


@app.get("/api/conversations/{conv_id}")
@app.get("/api/sessions/{conv_id}")
def get_conversation_details(conv_id: str):
    conv = get_session(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    messages = get_session_messages(conv_id)
    return {"conversation": conv, "messages": messages}


@app.post("/api/conversations/{conv_id}")
@app.post("/api/sessions/{conv_id}")
def update_conversation_details(conv_id: str, data: SessionUpdateRequest):
    pinned_bool = bool(data.pinned) if data.pinned is not None else None
    success = update_session(conv_id, title=data.title, mode=data.mode, pinned=pinned_bool)
    return {"success": success}


@app.delete("/api/conversations/{conv_id}")
@app.delete("/api/sessions/{conv_id}")
@app.post("/api/conversations/{conv_id}/delete")
@app.post("/api/sessions/{conv_id}/delete")
def delete_conversation_route(conv_id: str):
    return {"success": delete_session(conv_id)}


@app.post("/api/conversations/{conv_id}/clear")
@app.post("/api/sessions/{conv_id}/clear")
def clear_conversation_route(conv_id: str):
    return {"success": clear_session_messages(conv_id)}


@app.post("/api/conversations/{conv_id}/messages")
@app.post("/api/sessions/{conv_id}/messages")
def append_conversation_message(conv_id: str, data: MessageAppendRequest):
    role = data.role.strip() or "user"
    content = data.content
    metadata = data.metadata or {}
    last_msg = get_last_session_message(conv_id)
    if last_msg and last_msg["role"] == role and last_msg["content"] == content:
        return {"success": True, "id": last_msg["id"], "deduplicated": True}
    mid = save_message(conv_id, role, content, metadata=metadata)
    return {"success": True, "id": mid}


# =====================================================================
# Status, Telemetry & Diagnostics
# =====================================================================

@app.api_route("/health", methods=["GET", "HEAD"])
@app.api_route("/api/health", methods=["GET", "HEAD"])
def health_check():
    eng = get_engine()
    return {
        "status": "ok",
        "backend": eng.backend,
        "model": eng.active_model_name or "None",
    }


@app.api_route("/api/status", methods=["GET", "HEAD"])
def get_status():
    uptime_sec = int(time.time() - server_start_time)
    eng = get_engine()
    counts = get_counts()
    return {
        "backend": eng.backend,
        "model": eng.active_model_name or "None",
        "uptime_seconds": uptime_sec,
        "knowledge_count": counts["knowledge_count"],
        "pending_gaps": counts["pending_gaps"],
    }


@app.get("/api/system")
def get_system():
    return get_system_telemetry()


@app.get("/api/models")
def get_models():
    return get_engine().get_available_models()


@app.post("/api/models/active")
def set_active_model(data: ModelSelectRequest):
    model_name = data.model.strip()
    if not model_name:
        raise HTTPException(status_code=400, detail="Model filename required")
    eng = get_engine()
    success, msg = eng.set_active_model(model_name)
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"success": True, "message": msg, "active_model": eng.active_model_name}


@app.get("/api/audit")
def get_audit():
    return get_recent_audit_logs()


# =====================================================================
# Personal Assistant (To-Dos & Notes)
# =====================================================================

@app.get("/api/todos")
def list_todos():
    return get_todos()


@app.post("/api/todos")
def create_todo(data: TodoCreateRequest):
    task = data.task.strip()
    priority = data.priority.strip() or "normal"
    if not task:
        raise HTTPException(status_code=400, detail="Task text required")
    tid = add_todo(task, priority)
    return {"success": True, "id": tid}


def _parse_int_id(raw_id: int | str | None) -> int:
    if raw_id is None or raw_id == "":
        raise HTTPException(status_code=400, detail="ID required")
    try:
        return int(raw_id)
    except (ValueError, TypeError) as err:
        raise HTTPException(status_code=400, detail="Invalid integer ID") from err


@app.post("/api/todos/toggle")
def toggle_todo_route(data: IdRequest):
    tid = _parse_int_id(data.id)
    return {"success": toggle_todo(tid)}


@app.post("/api/todos/delete")
def delete_todo_route(data: IdRequest):
    tid = _parse_int_id(data.id)
    return {"success": delete_todo(tid)}


@app.get("/api/notes")
def list_notes():
    return get_notes()


@app.post("/api/notes")
def create_note(data: NoteCreateRequest):
    title = data.title.strip()
    content = data.content.strip()
    tags = data.tags.strip()
    if not content:
        raise HTTPException(status_code=400, detail="Note content required")
    nid = add_note(title or "Untitled Note", content, tags)
    return {"success": True, "id": nid}


@app.post("/api/notes/update")
def update_note_route(data: NoteUpdateRequest):
    nid = _parse_int_id(data.id)
    title = data.title.strip()
    content = data.content.strip()
    tags = data.tags.strip()
    if not content:
        raise HTTPException(status_code=400, detail="ID and content required")
    return {"success": update_note(nid, title or "Untitled Note", content, tags)}


@app.post("/api/notes/delete")
def delete_note_route(data: IdRequest):
    nid = _parse_int_id(data.id)
    return {"success": delete_note(nid)}


# =====================================================================
# Knowledge & Gaps
# =====================================================================

@app.get("/api/knowledge")
def list_knowledge():
    return get_all_knowledge()


@app.get("/api/knowledge/search")
def search_knowledge_route(q: str = "", limit: int = 10):
    safe_limit = max(1, min(limit, 200))
    if not q.strip():
        return get_all_knowledge(limit=safe_limit)
    return search_knowledge(q, limit=safe_limit)


@app.post("/api/knowledge")
def create_knowledge_route(data: KnowledgeCreateRequest):
    topic = data.topic.strip()
    lesson = data.lesson.strip()
    code_example = data.code_example.strip()
    tags = data.tags.strip()
    source = data.source.strip() or "manual"
    if not topic or not lesson:
        raise HTTPException(status_code=400, detail="Topic and lesson are required")
    kid = add_knowledge(topic, lesson, code_example=code_example, tags=tags, source=source)
    return {"success": True, "id": kid}


@app.post("/api/knowledge/update")
def update_knowledge_route(data: KnowledgeUpdateRequest):
    kid = _parse_int_id(data.id)
    topic = data.topic.strip()
    lesson = data.lesson.strip()
    code_example = data.code_example.strip()
    tags = data.tags.strip()
    if not topic or not lesson:
        raise HTTPException(status_code=400, detail="ID, topic, and lesson are required")
    return {"success": update_knowledge(kid, topic, lesson, code_example=code_example, tags=tags)}


@app.post("/api/knowledge/delete")
def delete_knowledge_route(data: IdRequest):
    kid = _parse_int_id(data.id)
    return {"success": delete_knowledge(kid)}


@app.get("/api/gaps")
def list_gaps():
    return get_unresolved_gaps()


@app.post("/api/gaps")
def create_gap(data: GapCreateRequest):
    q = data.question.strip()
    ctx = data.context.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question required")
    gid = add_gap(q, ctx)
    return {"success": True, "id": gid}


@app.post("/api/gaps/delete")
def delete_gap_route(data: IdRequest):
    gid = _parse_int_id(data.id)
    return {"success": delete_gap(gid)}


# =====================================================================
# File Upload & Multimodal Analysis
# =====================================================================

@app.post("/api/upload")
def upload_file(data: FileUploadRequest):
    name = data.name.strip() or "upload.bin"
    clean_name = _SAFE_FILENAME_RE.sub("_", os.path.basename(name)) or "upload.bin"
    file_ext = os.path.splitext(clean_name)[1].lower()
    unique_id = f"{int(time.time())}_{clean_name}"
    file_path = os.path.join(UPLOADS_DIR, unique_id)

    raw_b64 = data.data
    if "," in raw_b64:
        raw_b64 = raw_b64.split(",", 1)[1]

    # Enforce strict upload size ceiling before allocating decoded buffer
    if len(raw_b64) > int(MAX_UPLOAD_BYTES * 1.38):
        raise HTTPException(status_code=413, detail=f"File exceeds maximum allowed size ({format_bytes(MAX_UPLOAD_BYTES)})")

    try:
        file_bytes = base64.b64decode(raw_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 payload: {e}") from e

    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds maximum allowed size ({format_bytes(MAX_UPLOAD_BYTES)})")

    with open(file_path, "wb") as f:
        f.write(file_bytes)

    text_exts = {
        ".txt", ".py", ".js", ".jsx", ".ts", ".tsx", ".html", ".css", ".json",
        ".csv", ".md", ".sh", ".c", ".cpp", ".h", ".hpp", ".rs", ".go", ".sql",
        ".yaml", ".yml", ".xml", ".log", ".ini", ".env", ".toml", ".conf",
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
                pass
    elif file_ext in image_exts:
        category = "image"
        try:
            if file_bytes.startswith(b"\x89PNG\r\n\x1a\n") and len(file_bytes) >= 24:
                w, h = struct.unpack(">II", file_bytes[16:24])
                dimensions = f"{w}x{h}"
        except struct.error:
            pass
    elif file_ext in audio_exts:
        category = "audio"

    log_event("file_upload", f"Uploaded {clean_name} ({category}, {format_bytes(len(file_bytes))})", "info")
    return {
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
            "dimensions": dimensions,
        },
    }


@app.post("/api/analyze/multimodal")
def analyze_multimodal(data: MultimodalAnalyzeRequest):
    file_id = os.path.basename(data.file_id.strip())
    prompt = data.prompt.strip() or "Analyze this file thoroughly and describe all details."
    api_key = (
        data.api_key.strip()
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("TEACHER_API_KEY")
    )

    if not api_key:
        raise HTTPException(status_code=400, detail="Teacher API key required for multimodal vision/audio analysis.")

    if not file_id or not _SAFE_FILE_ID_RE.match(file_id):
        raise HTTPException(status_code=400, detail="Invalid or unsafe file_id parameter")

    real_uploads = os.path.realpath(UPLOADS_DIR)
    file_path = os.path.realpath(os.path.join(real_uploads, file_id))
    if os.path.commonpath([file_path, real_uploads]) != real_uploads or not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    with open(file_path, "rb") as f:
        raw_data = base64.b64encode(f.read()).decode("utf-8")

    ext = os.path.splitext(file_id)[1].lower()
    mime_map = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".webp": "image/webp", ".gif": "image/gif",
        ".mp3": "audio/mp3", ".wav": "audio/wav", ".ogg": "audio/ogg", ".m4a": "audio/m4a",
    }
    mime_type = mime_map.get(ext, "application/octet-stream")

    try:
        analysis = query_gemini_multimodal(prompt, raw_data, mime_type, api_key)
        return {"success": True, "analysis": analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# =====================================================================
# Teacher Distillation & Datasets
# =====================================================================

@app.get("/api/sync")
def get_teacher_sync_status():
    return sync_state


@app.post("/api/sync")
def start_teacher_sync(data: TeacherSyncRequest):
    api_key = (
        data.api_key.strip()
        or os.getenv("GEMINI_API_KEY", "")
        or os.getenv("GOOGLE_API_KEY", "")
        or os.getenv("TEACHER_API_KEY", "")
        or os.getenv("GROQ_API_KEY", "")
        or os.getenv("OPENAI_API_KEY", "")
    )
    topic = data.topic.strip()
    endpoint = (data.endpoint or "").strip() or None
    model = (data.model or "").strip() or None

    if not api_key:
        raise HTTPException(status_code=400, detail="API Key is required to communicate with teacher AI.")

    # Persist the Teacher API key to .env so autonomous 24/7 Gemini learning survives reboots
    save_persistent_api_key(api_key)

    if sync_state.get("is_syncing"):
        raise HTTPException(status_code=409, detail="Sync is already in progress")

    def bg_sync():
        run_sync(api_key, endpoint=endpoint, model=model, topic_to_explore=topic or None)

    threading.Thread(target=bg_sync, daemon=True).start()
    return {"success": True, "message": "Teacher synchronization started."}


@app.get("/api/teacher/models")
def get_teacher_models(provider: str = "gemini", api_key: str = ""):
    models = fetch_live_models_for_provider(provider, api_key)
    return {"provider": provider, "models": models}


@app.post("/api/teacher/argon/query")
def argon_query_route(data: ArgonQueryRequest):
    prompt = data.prompt.strip()
    api_key = (
        data.api_key.strip()
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("TEACHER_API_KEY")
    )
    model = data.model.strip() or "gemini-4-argon"
    reasoning_effort = data.reasoning_effort.strip() or "high"
    try:
        thinking_budget = int(data.thinking_budget or 2048)
    except (ValueError, TypeError):
        thinking_budget = 2048
    system_instruction = (data.system_instruction or "").strip() or None

    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required for Gemini 4 Argon reasoning query.")

    if api_key:
        save_persistent_api_key(api_key)

    try:
        result = query_gemini_argon(
            prompt=prompt,
            api_key=api_key,
            model=model,
            reasoning_effort=reasoning_effort,
            thinking_budget=thinking_budget,
            system_instruction=system_instruction,
        )
        log_event("argon_query", f"Gemini 4 Argon reasoning completed via {result.get('model', model)}", "info")
        return result
    except Exception as e:
        log_event("argon_error", f"Gemini 4 Argon query failed: {e}", "danger")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/teacher/argon/assimilate")
def argon_assimilate_route(data: ArgonAssimilateRequest):
    topic = data.topic.strip()
    lesson = data.lesson.strip()
    code_example = data.code_example.strip()
    tags = data.tags.strip() or "gemini-4-argon,reasoning,distilled"
    if not topic or not lesson:
        raise HTTPException(status_code=400, detail="Topic and lesson are required for assimilation.")
    kid = add_knowledge(topic, lesson, code_example=code_example, tags=tags, source="gemini-4-argon")
    log_event("argon_assimilate", f"Assimilated reasoning lesson #{kid}: {topic}", "success")
    return {"success": True, "id": kid, "message": f"Successfully assimilated '{topic}' into local SQLite memory."}


@app.get("/api/dataset/status")
def get_dataset_status():
    return import_state


@app.get("/api/dataset/popular")
def get_dataset_popular():
    return POPULAR_DATASETS


@app.post("/api/dataset/import")
def start_dataset_import_route(data: DatasetImportRequest):
    repo_id = data.dataset.strip() or "openbmb/UltraData-Stack-v3-L2"
    subset = (data.subset or "").strip() or None
    try:
        limit = max(1, min(int(data.limit or 50), 2000))
    except (ValueError, TypeError):
        limit = 50
    save_local = bool(data.save_local)
    token = (data.token or "").strip() or os.getenv("HF_TOKEN") or None

    if import_state.get("is_importing"):
        raise HTTPException(status_code=409, detail="Another dataset import is already in progress")

    def bg_import():
        import_hf_dataset(repo_id, subset=subset, limit=limit, save_local=save_local, token=token)

    threading.Thread(target=bg_import, daemon=True).start()
    return {"success": True, "message": f"Started streaming import from Hugging Face: {repo_id}"}


# =====================================================================
# Static Web Files & SPA Route
# =====================================================================

app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.api_route("/", methods=["GET", "HEAD"])
def serve_index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


@app.api_route("/styles.css", methods=["GET", "HEAD"])
def serve_css():
    css_path = os.path.join(WEB_DIR, "styles.css")
    if os.path.exists(css_path):
        return FileResponse(css_path)
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


@app.api_route("/script.js", methods=["GET", "HEAD"])
def serve_js():
    js_path = os.path.join(WEB_DIR, "script.js")
    if os.path.exists(js_path):
        return FileResponse(js_path)
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


def run():
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False, log_level="info")


if __name__ == "__main__":
    run()
