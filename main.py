import os
import time
import json
import re
import base64
import struct
import threading
import asyncio
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Request, HTTPException, Depends, Header, status
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import uvicorn

from database import (
    init_db,
    create_session, get_sessions, get_session, update_session, delete_session,
    save_message, get_session_messages, clear_session_messages,
    add_todo, get_todos, toggle_todo, delete_todo,
    add_note, get_notes, update_note, delete_note,
    get_all_knowledge, add_knowledge, update_knowledge, delete_knowledge, search_knowledge,
    get_unresolved_gaps, add_gap, delete_gap,
    get_system_telemetry, get_recent_audit_logs, log_event
)
from engine import InferenceEngine
from sync_teacher import run_sync, sync_state, query_gemini_multimodal, fetch_live_models_for_provider, query_gemini_argon
from import_dataset import import_hf_dataset, import_state, POPULAR_DATASETS

# Path configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")
UPLOADS_DIR = os.path.join(WEB_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "5000"))
API_KEY_SECRET = os.getenv("MERLIN_API_KEY") or os.getenv("API_KEY")

server_start_time = time.time()
engine: Optional[InferenceEngine] = None

def get_engine() -> InferenceEngine:
    global engine
    if engine is None:
        init_db()
        engine = InferenceEngine()
    return engine

@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    eng = get_engine()
    print("=" * 66)
    print("  MERLIN AI — FASTAPI CORE RUNTIME (PRODUCTION READY)")
    print(f"  Web Dashboard:     http://localhost:{PORT}")
    print(f"  OpenAI REST API:   http://localhost:{PORT}/v1/chat/completions")
    print(f"  Network Host:      http://{HOST}:{PORT}")
    print(f"  Model Engine:      {eng.backend} ({eng.active_model_name})")
    print("=" * 66)
    yield
    if engine is not None:
        engine.cleanup()

# FastAPI Application
app = FastAPI(
    title="Merlin AI",
    description="High-performance, offline AI Assistant & Coder with modern reactive UX and OpenAI-compatible API.",
    version="2.0.0",
    lifespan=lifespan
)

# Standard CORS Middleware for third-party tools & extensions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def format_bytes(size: float) -> str:
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}" if unit != 'B' else f"{size} B"
        size /= 1024.0
    return f"{size:.1f} TB"

def verify_api_key(authorization: Optional[str] = Header(None)):
    """Validates optional Bearer token if API_KEY_SECRET is configured."""
    if not API_KEY_SECRET:
        return True
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"message": "Missing or invalid Bearer token", "type": "invalid_request_error"}
        )
    token = authorization[7:].strip()
    if token != API_KEY_SECRET:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"message": "Invalid API key provided", "type": "invalid_request_error"}
        )
    return True

# =====================================================================
# OpenAI-Compatible REST API Layer
# =====================================================================

class OpenAIMessage(BaseModel):
    role: str
    content: str

class OpenAIChatRequest(BaseModel):
    model: Optional[str] = "merlin-ai"
    messages: List[OpenAIMessage]
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024
    stop: Optional[Any] = None

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
            "parent": None
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
            "parent": None
        })
    return {"object": "list", "data": data}

@app.get("/v1/models/{model_id}")
def get_v1_model(model_id: str):
    return {
        "id": model_id,
        "object": "model",
        "created": int(server_start_time),
        "owned_by": "local"
    }

@app.post("/v1/chat/completions")
async def chat_completions(req: OpenAIChatRequest, request: Request, auth: bool = Depends(verify_api_key)):
    """Standard OpenAI POST /v1/chat/completions endpoint with async disconnect handling."""
    eng = get_engine()
    model_name = req.model or eng.active_model_name or "merlin-ai"
    messages_dicts = [{"role": m.role, "content": m.content} for m in req.messages]

    if req.stream:
        chat_id = f"chatcmpl-{int(time.time()*1000)}"
        created_ts = int(time.time())
        queue = asyncio.Queue(maxsize=500)
        loop = asyncio.get_running_loop()
        stop_sentinel = object()
        abort_event = threading.Event()

        def openai_worker():
            try:
                for delta_chunk in eng.generate_chat_completion_stream(
                    messages_dicts,
                    temperature=req.temperature,
                    max_tokens=req.max_tokens,
                    stop=req.stop
                ):
                    if abort_event.is_set():
                        break
                    asyncio.run_coroutine_threadsafe(queue.put(delta_chunk), loop).result()
            except Exception as e:
                log_event("v1_worker_error", f"OpenAI stream worker error: {e}", "danger")
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(stop_sentinel), loop)

        threading.Thread(target=openai_worker, daemon=True).start()

        async def sse_generator():
            try:
                while True:
                    if await request.is_disconnected():
                        abort_event.set()
                        break

                    try:
                        delta = await asyncio.wait_for(queue.get(), timeout=0.08)
                    except asyncio.TimeoutError:
                        continue

                    if delta is stop_sentinel:
                        break

                    chunk_obj = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": model_name,
                        "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(chunk_obj)}\n\n"

                final_chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
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
                "X-Accel-Buffering": "no"
            }
        )
    else:
        completion_text = eng.generate_chat_completion(
            messages_dicts,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            stream=False,
            stop=req.stop
        )
        prompt_tokens = sum(len(m.content) // 4 for m in req.messages)
        completion_tokens = len(str(completion_text)) // 4
        return {
            "id": f"chatcmpl-{int(time.time()*1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": str(completion_text)},
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }

# =====================================================================
# Native Web UI Real-Time Token Streaming
# =====================================================================

@app.get("/api/chat/stream")
async def chat_stream(request: Request, prompt: str, mode: str = "assistant", session_id: str = "default"):
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    eng = get_engine()
    queue = asyncio.Queue(maxsize=500)
    loop = asyncio.get_running_loop()
    stop_sentinel = object()
    abort_event = threading.Event()

    def stream_worker():
        try:
            for chunk in eng.generate_stream(prompt, session_id=session_id, mode=mode):
                if abort_event.is_set():
                    break
                asyncio.run_coroutine_threadsafe(queue.put(chunk), loop).result()
        except Exception as e:
            log_event("stream_worker_error", f"Stream worker exception: {e}", "danger")
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(stop_sentinel), loop)

    t = threading.Thread(target=stream_worker, daemon=True)
    t.start()

    async def sse_event_stream():
        # Flush initial comment immediately to establish connection (< 5ms)
        yield ": connected\n\n"
        try:
            while True:
                if await request.is_disconnected():
                    abort_event.set()
                    eng.stop_generation(session_id)
                    break

                try:
                    chunk = await asyncio.wait_for(queue.get(), timeout=0.08)
                except asyncio.TimeoutError:
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
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/api/chat/stop")
def chat_stop(data: dict):
    sid = str(data.get("session_id", "default")).strip()
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
def create_new_conversation(data: dict):
    title = str(data.get("title", "New Chat")).strip() or "New Chat"
    mode = str(data.get("mode", "assistant")).strip() or "assistant"
    cid = str(data.get("id", "")).strip() or None
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
def update_conversation_details(conv_id: str, data: dict):
    title = data.get("title")
    mode = data.get("mode")
    pinned = data.get("pinned")
    success = update_session(conv_id, title=title, mode=mode, pinned=pinned)
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
def append_conversation_message(conv_id: str, data: dict):
    role = str(data.get("role", "user")).strip()
    content = str(data.get("content", ""))
    metadata = data.get("metadata", {})
    existing_msgs = get_session_messages(conv_id, limit=200)
    if existing_msgs and existing_msgs[-1]["role"] == role and existing_msgs[-1]["content"] == content:
        return {"success": True, "id": existing_msgs[-1]["id"], "deduplicated": True}
    mid = save_message(conv_id, role, content, metadata=metadata)
    return {"success": True, "id": mid}

# =====================================================================
# Status, Telemetry & Diagnostics
# =====================================================================

@app.get("/health")
@app.get("/api/health")
def health_check():
    eng = get_engine()
    return {
        "status": "ok",
        "backend": eng.backend,
        "model": eng.active_model_name or "None"
    }

@app.get("/api/status")
def get_status():
    uptime_sec = int(time.time() - server_start_time)
    eng = get_engine()
    return {
        "backend": eng.backend,
        "model": eng.active_model_name or "None",
        "uptime_seconds": uptime_sec,
        "knowledge_count": len(get_all_knowledge()),
        "pending_gaps": len(get_unresolved_gaps())
    }

@app.get("/api/system")
def get_system():
    return get_system_telemetry()

@app.get("/api/models")
def get_models():
    return get_engine().get_available_models()

@app.post("/api/models/active")
def set_active_model(data: dict):
    model_name = str(data.get("model", "")).strip()
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
def create_todo(data: dict):
    task = str(data.get("task", "")).strip()
    priority = str(data.get("priority", "normal")).strip()
    if not task:
        raise HTTPException(status_code=400, detail="Task text required")
    tid = add_todo(task, priority)
    return {"success": True, "id": tid}

@app.post("/api/todos/toggle")
def toggle_todo_route(data: dict):
    tid = data.get("id")
    if not tid:
        raise HTTPException(status_code=400, detail="ID required")
    return {"success": toggle_todo(tid)}

@app.post("/api/todos/delete")
def delete_todo_route(data: dict):
    tid = data.get("id")
    if not tid:
        raise HTTPException(status_code=400, detail="ID required")
    return {"success": delete_todo(tid)}

@app.get("/api/notes")
def list_notes():
    return get_notes()

@app.post("/api/notes")
def create_note(data: dict):
    title = str(data.get("title", "")).strip()
    content = str(data.get("content", "")).strip()
    tags = str(data.get("tags", "")).strip()
    if not content:
        raise HTTPException(status_code=400, detail="Note content required")
    nid = add_note(title or "Untitled Note", content, tags)
    return {"success": True, "id": nid}

@app.post("/api/notes/update")
def update_note_route(data: dict):
    nid = data.get("id")
    title = str(data.get("title", "")).strip()
    content = str(data.get("content", "")).strip()
    tags = str(data.get("tags", "")).strip()
    if not nid or not content:
        raise HTTPException(status_code=400, detail="ID and content required")
    return {"success": update_note(nid, title or "Untitled Note", content, tags)}

@app.post("/api/notes/delete")
def delete_note_route(data: dict):
    nid = data.get("id")
    if not nid:
        raise HTTPException(status_code=400, detail="ID required")
    return {"success": delete_note(nid)}

# =====================================================================
# Knowledge & Gaps
# =====================================================================

@app.get("/api/knowledge")
def list_knowledge():
    return get_all_knowledge()

@app.get("/api/knowledge/search")
def search_knowledge_route(q: str = "", limit: int = 10):
    if not q.strip():
        return get_all_knowledge(limit=limit)
    return search_knowledge(q, limit=limit)

@app.post("/api/knowledge")
def create_knowledge_route(data: dict):
    topic = str(data.get("topic", "")).strip()
    lesson = str(data.get("lesson", "")).strip()
    code_example = str(data.get("code_example", "")).strip()
    tags = str(data.get("tags", "")).strip()
    source = str(data.get("source", "manual")).strip() or "manual"
    if not topic or not lesson:
        raise HTTPException(status_code=400, detail="Topic and lesson are required")
    kid = add_knowledge(topic, lesson, code_example=code_example, tags=tags, source=source)
    return {"success": True, "id": kid}

@app.post("/api/knowledge/update")
def update_knowledge_route(data: dict):
    kid = data.get("id")
    topic = str(data.get("topic", "")).strip()
    lesson = str(data.get("lesson", "")).strip()
    code_example = str(data.get("code_example", "")).strip()
    tags = str(data.get("tags", "")).strip()
    if not kid or not topic or not lesson:
        raise HTTPException(status_code=400, detail="ID, topic, and lesson are required")
    return {"success": update_knowledge(int(kid), topic, lesson, code_example=code_example, tags=tags)}

@app.post("/api/knowledge/delete")
def delete_knowledge_route(data: dict):
    kid = data.get("id")
    if not kid:
        raise HTTPException(status_code=400, detail="ID required")
    return {"success": delete_knowledge(kid)}

@app.get("/api/gaps")
def list_gaps():
    return get_unresolved_gaps()

@app.post("/api/gaps")
def create_gap(data: dict):
    q = str(data.get("question", "")).strip()
    ctx = str(data.get("context", "")).strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question required")
    gid = add_gap(q, ctx)
    return {"success": True, "id": gid}

@app.post("/api/gaps/delete")
def delete_gap_route(data: dict):
    gid = data.get("id")
    if not gid:
        raise HTTPException(status_code=400, detail="ID required")
    return {"success": delete_gap(gid)}

# =====================================================================
# File Upload & Multimodal Analysis
# =====================================================================

@app.post("/api/upload")
def upload_file(data: dict):
    name = str(data.get("name", "upload.bin")).strip()
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
        raise HTTPException(status_code=400, detail=f"Invalid base64 payload: {e}")

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
            "dimensions": dimensions
        }
    }

@app.post("/api/analyze/multimodal")
def analyze_multimodal(data: dict):
    file_id = os.path.basename(str(data.get("file_id", "")).strip())
    prompt = str(data.get("prompt", "Analyze this file thoroughly and describe all details.")).strip()
    api_key = str(data.get("api_key", "")).strip() or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("TEACHER_API_KEY")

    if not api_key:
        raise HTTPException(status_code=400, detail="Teacher API key required for multimodal vision/audio analysis.")

    if not file_id:
        raise HTTPException(status_code=400, detail="file_id is required")

    file_path = os.path.join(UPLOADS_DIR, file_id)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

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
        return {"success": True, "analysis": analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================================
# Teacher Distillation & Datasets
# =====================================================================

@app.get("/api/sync")
def get_teacher_sync_status():
    return sync_state

@app.post("/api/sync")
def start_teacher_sync(data: dict):
    api_key = (
        str(data.get("api_key", "")).strip()
        or os.getenv("GEMINI_API_KEY", "")
        or os.getenv("GOOGLE_API_KEY", "")
        or os.getenv("TEACHER_API_KEY", "")
        or os.getenv("GROQ_API_KEY", "")
        or os.getenv("OPENAI_API_KEY", "")
    )
    topic = str(data.get("topic", "")).strip()
    endpoint = str(data.get("endpoint", "")).strip() or None
    model = str(data.get("model", "")).strip() or None

    if not api_key:
        raise HTTPException(status_code=400, detail="API Key is required to communicate with teacher AI.")

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
def argon_query_route(data: dict):
    prompt = str(data.get("prompt", "")).strip()
    api_key = str(data.get("api_key", "")).strip() or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("TEACHER_API_KEY")
    model = str(data.get("model", "gemini-4-argon")).strip() or "gemini-4-argon"
    reasoning_effort = str(data.get("reasoning_effort", "high")).strip()
    thinking_budget = int(data.get("thinking_budget") or 2048)
    system_instruction = str(data.get("system_instruction", "")).strip() or None

    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required for Gemini 4 Argon reasoning query.")

    try:
        result = query_gemini_argon(
            prompt=prompt,
            api_key=api_key,
            model=model,
            reasoning_effort=reasoning_effort,
            thinking_budget=thinking_budget,
            system_instruction=system_instruction
        )
        log_event("argon_query", f"Gemini 4 Argon reasoning completed via {result.get('model', model)}", "info")
        return result
    except Exception as e:
        log_event("argon_error", f"Gemini 4 Argon query failed: {e}", "danger")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/teacher/argon/assimilate")
def argon_assimilate_route(data: dict):
    topic = str(data.get("topic", "")).strip()
    lesson = str(data.get("lesson", "")).strip()
    code_example = str(data.get("code_example", "")).strip()
    tags = str(data.get("tags", "gemini-4-argon,reasoning,distilled")).strip()
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
def start_dataset_import_route(data: dict):
    repo_id = str(data.get("dataset", "openbmb/UltraData-Stack-v3-L2")).strip()
    subset = str(data.get("subset", "")).strip() or None
    limit = int(data.get("limit") or 50)
    save_local = bool(data.get("save_local", True))
    token = str(data.get("token", "")).strip() or os.getenv("HF_TOKEN") or None

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

@app.get("/")
def serve_index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))

@app.get("/styles.css")
def serve_css():
    css_path = os.path.join(WEB_DIR, "styles.css")
    if os.path.exists(css_path):
        return FileResponse(css_path)
    return FileResponse(os.path.join(WEB_DIR, "index.html"))

@app.get("/script.js")
def serve_js():
    js_path = os.path.join(WEB_DIR, "script.js")
    if os.path.exists(js_path):
        return FileResponse(js_path)
    return FileResponse(os.path.join(WEB_DIR, "index.html"))

def run():
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False, log_level="info")

if __name__ == "__main__":
    run()
