import os
import sqlite3
import datetime
import platform
import shutil
import time
import json
import uuid
import threading
import re
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assistant_memory.db")
_local = threading.local()

def get_connection():
    """Returns a thread-local cached SQLite connection tuned for high performance."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(DB_PATH, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # Execute performance PRAGMAs once per connection
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            conn.execute("PRAGMA cache_size = -64000;")  # 64MB memory page cache
            conn.execute("PRAGMA temp_store = MEMORY;")
            conn.execute("PRAGMA mmap_size = 268435456;") # 256MB memory mapped I/O
            conn.execute("PRAGMA busy_timeout = 30000;")
        except Exception:
            pass
        _local.conn = conn
    return conn

@contextmanager
def get_db():
    """
    High-performance thread-safe context manager. Reuses connections
    and guarantees automatic transaction commit on success or rollback on error.
    """
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise

def estimate_tokens(text: str) -> int:
    """Accurate token estimator (~3.8 characters per token for English & code)."""
    if not text:
        return 0
    return max(1, int(len(text) / 3.8))

def init_db():
    """Initializes tables, indices, and auto-migrates existing schemas."""
    with get_db() as conn:
        c = conn.cursor()
        
        # 1. Sessions / Conversations
        c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            mode TEXT DEFAULT 'assistant',
            pinned INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC);")
        c.execute("CREATE INDEX IF NOT EXISTS idx_sessions_pinned ON sessions(pinned DESC, updated_at DESC);")

        # Create conversations view/alias for backward compatibility
        c.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            mode TEXT DEFAULT 'assistant',
            pinned INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # 2. Messages (Structured multi-turn history)
        c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            token_count INTEGER DEFAULT 0,
            metadata TEXT DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        # Auto-migration: check if session_id or conversation_id columns exist from prior migrations
        c.execute("PRAGMA table_info(messages);")
        msg_cols = [col[1] for col in c.fetchall()]
        if "session_id" not in msg_cols:
            c.execute("ALTER TABLE messages ADD COLUMN session_id TEXT;")
            if "conversation_id" in msg_cols:
                c.execute("UPDATE messages SET session_id = conversation_id WHERE session_id IS NULL;")
        if "conversation_id" not in msg_cols:
            c.execute("ALTER TABLE messages ADD COLUMN conversation_id TEXT;")
            if "session_id" in msg_cols:
                c.execute("UPDATE messages SET conversation_id = session_id WHERE conversation_id IS NULL;")

        c.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id ASC);")
        c.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, id ASC);")

        # 3. Knowledge Base
        c.execute("""
        CREATE TABLE IF NOT EXISTS knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT NOT NULL,
            lesson TEXT NOT NULL,
            code_example TEXT,
            tags TEXT DEFAULT '',
            source TEXT DEFAULT 'distilled',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        # Auto-migration for knowledge table
        c.execute("PRAGMA table_info(knowledge);")
        k_cols = [col[1] for col in c.fetchall()]
        if "source" not in k_cols:
            c.execute("ALTER TABLE knowledge ADD COLUMN source TEXT DEFAULT 'distilled';")
        if "code_example" not in k_cols:
            c.execute("ALTER TABLE knowledge ADD COLUMN code_example TEXT DEFAULT '';")
        if "tags" not in k_cols:
            c.execute("ALTER TABLE knowledge ADD COLUMN tags TEXT DEFAULT '';")

        c.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_topic ON knowledge(topic);")
        c.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_tags ON knowledge(tags);")
        try:
            c.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
                topic,
                lesson,
                tags,
                code_example,
                content='knowledge',
                content_rowid='id'
            );
            """)
            c.execute("""
            CREATE TRIGGER IF NOT EXISTS trg_knowledge_ai AFTER INSERT ON knowledge BEGIN
                INSERT INTO knowledge_fts(rowid, topic, lesson, tags, code_example)
                VALUES (new.id, new.topic, new.lesson, new.tags, new.code_example);
            END;
            """)
            c.execute("""
            CREATE TRIGGER IF NOT EXISTS trg_knowledge_ad AFTER DELETE ON knowledge BEGIN
                INSERT INTO knowledge_fts(knowledge_fts, rowid, topic, lesson, tags, code_example)
                VALUES('delete', old.id, old.topic, old.lesson, old.tags, old.code_example);
            END;
            """)
        except Exception:
            pass

        # 4. Knowledge Gaps (State Machine: pending -> processing -> resolved / failed)
        c.execute("""
        CREATE TABLE IF NOT EXISTS gaps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            context TEXT DEFAULT '',
            status TEXT DEFAULT 'pending',
            attempts INTEGER DEFAULT 0,
            last_error TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resolved_at TIMESTAMP
        );
        """)

        # Auto-migration for gaps
        c.execute("PRAGMA table_info(gaps);")
        gap_cols = [col[1] for col in c.fetchall()]
        for col_name, col_type in [("status", "TEXT DEFAULT 'pending'"), 
                                   ("attempts", "INTEGER DEFAULT 0"),
                                   ("last_error", "TEXT DEFAULT ''"),
                                   ("resolved_at", "TIMESTAMP")]:
            if col_name not in gap_cols:
                c.execute(f"ALTER TABLE gaps ADD COLUMN {col_name} {col_type};")

        c.execute("CREATE INDEX IF NOT EXISTS idx_gaps_status ON gaps(status);")

        # 5. Personal Assistant: To-Dos
        c.execute("""
        CREATE TABLE IF NOT EXISTS todos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task TEXT NOT NULL,
            completed INTEGER DEFAULT 0,
            priority TEXT DEFAULT 'normal',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP
        );
        """)

        # 6. Personal Assistant: Notes
        c.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            tags TEXT DEFAULT '',
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # 7. Legacy chat_history table for backward compatibility
        c.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_history(session_id, id);")

        # 8. Audit Logging
        c.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            details TEXT NOT NULL,
            status TEXT DEFAULT 'info',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

# --- Sessions / Conversations Management ---

def create_session(title: str = "New Chat", mode: str = "assistant", session_id: str = None) -> dict:
    sid = session_id or str(uuid.uuid4())[:8]
    now = datetime.datetime.now().isoformat()
    with get_db() as conn:
        conn.execute("""
            INSERT INTO sessions (id, title, mode, pinned, created_at, updated_at)
            VALUES (?, ?, ?, 0, ?, ?)
        """, (sid, title.strip() or "New Chat", mode, now, now))
        conn.execute("""
            INSERT OR REPLACE INTO conversations (id, title, mode, pinned, created_at, updated_at)
            VALUES (?, ?, ?, 0, ?, ?)
        """, (sid, title.strip() or "New Chat", mode, now, now))
    return {"id": sid, "title": title, "mode": mode, "pinned": 0, "created_at": now, "updated_at": now}

def get_sessions(limit: int = 100) -> list:
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""
            SELECT s.id, s.title, s.mode, s.pinned, s.created_at, s.updated_at,
                   COUNT(m.id) as message_count,
                   (SELECT content FROM messages WHERE session_id = s.id OR conversation_id = s.id ORDER BY id DESC LIMIT 1) as last_message
            FROM sessions s
            LEFT JOIN messages m ON (m.session_id = s.id OR m.conversation_id = s.id)
            GROUP BY s.id
            ORDER BY s.pinned DESC, s.updated_at DESC
            LIMIT ?
        """, (limit,))
        rows = [dict(r) for r in c.fetchall()]
        if not rows:
            # Fallback to conversations table if sessions was empty
            c.execute("""
                SELECT c.id, c.title, c.mode, c.pinned, c.created_at, c.updated_at,
                       COUNT(m.id) as message_count,
                       (SELECT content FROM messages WHERE conversation_id = c.id OR session_id = c.id ORDER BY id DESC LIMIT 1) as last_message
                FROM conversations c
                LEFT JOIN messages m ON (m.conversation_id = c.id OR m.session_id = c.id)
                GROUP BY c.id
                ORDER BY c.pinned DESC, c.updated_at DESC
                LIMIT ?
            """, (limit,))
            rows = [dict(r) for r in c.fetchall()]
        return rows

def get_session(session_id: str) -> dict:
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT id, title, mode, pinned, created_at, updated_at FROM sessions WHERE id = ?", (session_id,))
        row = c.fetchone()
        if not row:
            c.execute("SELECT id, title, mode, pinned, created_at, updated_at FROM conversations WHERE id = ?", (session_id,))
            row = c.fetchone()
        return dict(row) if row else None

def update_session(session_id: str, title: str = None, mode: str = None, pinned: bool = None) -> bool:
    with get_db() as conn:
        c = conn.cursor()
        updates = []
        params = []
        if title is not None:
            updates.append("title = ?")
            params.append(title.strip())
        if mode is not None:
            updates.append("mode = ?")
            params.append(mode.strip())
        if pinned is not None:
            updates.append("pinned = ?")
            params.append(1 if pinned else 0)
        
        if not updates:
            return False

        updates.append("updated_at = CURRENT_TIMESTAMP")
        params.append(session_id)

        sql = f"UPDATE sessions SET {', '.join(updates)} WHERE id = ?"
        c.execute(sql, params)
        c.execute(f"UPDATE conversations SET {', '.join(updates)} WHERE id = ?", params)
        return True

def delete_session(session_id: str) -> bool:
    with get_db() as conn:
        c = conn.cursor()
        c.execute("DELETE FROM messages WHERE session_id = ? OR conversation_id = ?", (session_id, session_id))
        c.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
        c.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        c.execute("DELETE FROM conversations WHERE id = ?", (session_id,))
        return True

# Aliases for conversation-based nomenclature
create_conversation = create_session
get_conversations = get_sessions
get_conversation = get_session
update_conversation = update_session
delete_conversation = delete_session

# --- Messages & Context Management ---

def save_message(session_id: str, role: str, content: str, metadata: dict = None) -> int:
    token_count = estimate_tokens(content)
    meta_json = json.dumps(metadata or {})
    
    with get_db() as conn:
        c = conn.cursor()
        # Auto-create session if not present
        c.execute("SELECT id FROM sessions WHERE id = ?", (session_id,))
        if not c.fetchone():
            c.execute("SELECT id FROM conversations WHERE id = ?", (session_id,))
            if not c.fetchone():
                title = content[:32] + "..." if len(content) > 32 else content
                title = title.replace("\n", " ").strip() or "New Chat"
                now = datetime.datetime.now().isoformat()
                c.execute("""
                    INSERT INTO sessions (id, title, mode, pinned, created_at, updated_at)
                    VALUES (?, ?, 'assistant', 0, ?, ?)
                """, (session_id, title, now, now))
                c.execute("""
                    INSERT INTO conversations (id, title, mode, pinned, created_at, updated_at)
                    VALUES (?, ?, 'assistant', 0, ?, ?)
                """, (session_id, title, now, now))
        else:
            c.execute("UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (session_id,))
            c.execute("UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (session_id,))

        c.execute("""
            INSERT INTO messages (session_id, conversation_id, role, content, token_count, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (session_id, session_id, role, content, token_count, meta_json))
        msg_id = c.lastrowid

        # Mirror to legacy chat_history
        c.execute("INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
                  (session_id, role, content))

        return msg_id

def get_session_messages(session_id: str, limit: int = 200) -> list:
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""
            SELECT id, session_id, role, content, token_count, metadata, created_at
            FROM messages
            WHERE session_id = ? OR conversation_id = ?
            ORDER BY id ASC
            LIMIT ?
        """, (session_id, session_id, limit))
        return [dict(r) for r in c.fetchall()]

get_conversation_messages = get_session_messages

def clear_session_messages(session_id: str) -> bool:
    with get_db() as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ? OR conversation_id = ?", (session_id, session_id))
        conn.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
        return True

clear_conversation_messages = clear_session_messages

def get_context_window(session_id: str, max_tokens: int = 4200) -> list:
    """
    Sliding-window context manager:
    Packs historical messages under max_tokens budget while preserving the initial user intent and most recent turns.
    """
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""
            SELECT role, content, token_count FROM messages
            WHERE session_id = ? OR conversation_id = ?
            ORDER BY id ASC
        """, (session_id, session_id))
        msgs = [dict(r) for r in c.fetchall()]

    if not msgs:
        return get_recent_chat(session_id, limit=8)

    total_tokens = sum(m.get("token_count") or estimate_tokens(m["content"]) for m in msgs)
    if total_tokens <= max_tokens:
        return [{"role": m["role"], "content": m["content"]} for m in msgs]

    first_msg = msgs[0]
    first_tokens = first_msg.get("token_count") or estimate_tokens(first_msg["content"])
    remaining_budget = max_tokens - first_tokens

    recent = []
    accumulated = 0
    for m in reversed(msgs[1:]):
        tok = m.get("token_count") or estimate_tokens(m["content"])
        if accumulated + tok > remaining_budget:
            break
        recent.append({"role": m["role"], "content": m["content"]})
        accumulated += tok

    recent.reverse()
    result = [{"role": first_msg["role"], "content": first_msg["content"]}]
    if len(msgs) > len(recent) + 1:
        result.append({
            "role": "system",
            "content": f"[Note: {len(msgs) - len(recent) - 1} earlier turns truncated to fit context budget]"
        })
    result.extend(recent)
    return result

# Legacy aliases
save_chat_message = save_message

def get_recent_chat(session_id: str, limit: int = 8) -> list:
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""
            SELECT role, content FROM (
                SELECT id, role, content FROM chat_history 
                WHERE session_id = ? 
                ORDER BY id DESC LIMIT ?
            ) ORDER BY id ASC
        """, (session_id, limit))
        return [dict(r) for r in c.fetchall()]

def clear_chat_history(session_id: str = None):
    with get_db() as conn:
        if session_id:
            conn.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM messages WHERE session_id = ? OR conversation_id = ?", (session_id, session_id))
        else:
            conn.execute("DELETE FROM chat_history")
            conn.execute("DELETE FROM messages")

# --- Audit & Logging ---

def log_event(event_type: str, details: str, status: str = "info"):
    with get_db() as conn:
        conn.execute("INSERT INTO audit_log (event_type, details, status) VALUES (?, ?, ?)",
                     (event_type, str(details), status))

def get_recent_audit_logs(limit: int = 25) -> list:
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT id, event_type, details, status, created_at FROM audit_log ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(r) for r in c.fetchall()]

# --- Knowledge & Memory Retrieval ---

def add_knowledge(topic: str, lesson: str, code_example: str = "", tags: str = "", source: str = "distilled") -> int:
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""
            INSERT INTO knowledge (topic, lesson, code_example, tags, source)
            VALUES (?, ?, ?, ?, ?)
        """, (topic.strip(), lesson.strip(), code_example.strip() if code_example else "", tags.strip(), source))
        log_event("knowledge_added", f"Assimilated: {topic}", "success")
        return c.lastrowid

def delete_knowledge(knowledge_id: int) -> bool:
    with get_db() as conn:
        conn.execute("DELETE FROM knowledge WHERE id = ?", (knowledge_id,))
        log_event("knowledge_deleted", f"Deleted knowledge ID: {knowledge_id}", "info")
        return True

def search_knowledge(query: str, limit: int = 3) -> list:
    if not query or not query.strip():
        return []

    tokens = [t.lower() for t in re.findall(r'\b\w{3,}\b', query)]
    if not tokens:
        return []

    with get_db() as conn:
        c = conn.cursor()
        # 1. High-speed FTS5 index search (< 0.2ms)
        try:
            fts_query = " OR ".join(f'"{t}"*' for t in tokens[:6])
            c.execute("""
                SELECT k.id, k.topic, k.lesson, k.code_example, k.tags
                FROM knowledge_fts fts
                JOIN knowledge k ON k.id = fts.rowid
                WHERE knowledge_fts MATCH ?
                LIMIT ?
            """, (fts_query, limit))
            results = [dict(r) for r in c.fetchall()]
            if results:
                return results
        except Exception:
            pass

        # 2. Fast indexed SQL LIKE fallback with strict limit
        clauses = []
        params = []
        for tok in tokens[:4]:
            clauses.append("(topic LIKE ? OR tags LIKE ?)")
            params.extend([f"%{tok}%", f"%{tok}%"])
        
        sql = f"SELECT id, topic, lesson, code_example, tags FROM knowledge WHERE {' OR '.join(clauses)} ORDER BY id DESC LIMIT ?"
        params.append(limit)
        try:
            c.execute(sql, params)
            return [dict(r) for r in c.fetchall()]
        except Exception:
            return []

def get_all_knowledge(limit: int = 100) -> list:
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""
            SELECT id, topic, lesson, code_example, tags, source, created_at
            FROM knowledge ORDER BY id DESC LIMIT ?
        """, (limit,))
        return [dict(r) for r in c.fetchall()]

# --- Gaps State Machine ---

def add_gap(question: str, context: str = "") -> int:
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT id FROM gaps WHERE question = ? AND status = 'pending'", (question.strip(),))
        existing = c.fetchone()
        if existing:
            return existing["id"]

        c.execute("INSERT INTO gaps (question, context, status) VALUES (?, ?, 'pending')",
                  (question.strip(), context.strip()))
        gid = c.lastrowid
        log_event("gap_recorded", f"Recorded gap #{gid}: {question}", "warning")
        return gid

def get_unresolved_gaps(limit: int = 50) -> list:
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""
            SELECT id, question, context, status, attempts, last_error, created_at
            FROM gaps WHERE status IN ('pending', 'failed')
            ORDER BY id ASC LIMIT ?
        """, (limit,))
        return [dict(r) for r in c.fetchall()]

def set_gap_status(gap_id: int, status: str, error_msg: str = ""):
    with get_db() as conn:
        now = datetime.datetime.now().isoformat() if status == 'resolved' else None
        conn.execute("""
            UPDATE gaps 
            SET status = ?, last_error = ?, attempts = attempts + 1, resolved_at = ?
            WHERE id = ?
        """, (status, error_msg, now, gap_id))

def delete_gap(gap_id: int) -> bool:
    with get_db() as conn:
        conn.execute("DELETE FROM gaps WHERE id = ?", (gap_id,))
        return True

# --- Personal Assistant: Tasks & Notes ---

def add_todo(task: str, priority: str = "normal") -> int:
    with get_db() as conn:
        c = conn.cursor()
        c.execute("INSERT INTO todos (task, priority) VALUES (?, ?)", (task.strip(), priority))
        return c.lastrowid

def get_todos() -> list:
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT id, task, completed, priority, created_at, completed_at FROM todos ORDER BY completed ASC, id DESC")
        return [dict(r) for r in c.fetchall()]

def toggle_todo(todo_id: int) -> bool:
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT completed FROM todos WHERE id = ?", (todo_id,))
        row = c.fetchone()
        if not row:
            return False
        new_val = 0 if row["completed"] else 1
        now = datetime.datetime.now().isoformat() if new_val == 1 else None
        c.execute("UPDATE todos SET completed = ?, completed_at = ? WHERE id = ?", (new_val, now, todo_id))
        return True

def delete_todo(todo_id: int) -> bool:
    with get_db() as conn:
        conn.execute("DELETE FROM todos WHERE id = ?", (todo_id,))
        return True

def add_note(title: str, content: str, tags: str = "") -> int:
    with get_db() as conn:
        c = conn.cursor()
        c.execute("INSERT INTO notes (title, content, tags) VALUES (?, ?, ?)",
                  (title.strip(), content.strip(), tags.strip()))
        return c.lastrowid

def update_note(note_id: int, title: str, content: str, tags: str = "") -> bool:
    with get_db() as conn:
        conn.execute("""
            UPDATE notes 
            SET title = ?, content = ?, tags = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (title.strip(), content.strip(), tags.strip(), note_id))
        return True

def get_notes() -> list:
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT id, title, content, tags, updated_at FROM notes ORDER BY id DESC")
        return [dict(r) for r in c.fetchall()]

def delete_note(note_id: int) -> bool:
    with get_db() as conn:
        conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        return True

# --- Hardware Diagnostics & Telemetry ---

def get_system_telemetry() -> dict:
    stats = {
        "os": f"{platform.system()} {platform.release()}",
        "arch": platform.machine(),
        "temperature": "N/A",
        "cpu_usage": "N/A",
        "memory": "N/A",
        "disk": "N/A",
        "uptime": "N/A"
    }

    # Pi 5 / Linux Temperature Sensor
    for zone in ["/sys/class/thermal/thermal_zone0/temp", "/sys/devices/virtual/thermal/thermal_zone0/temp"]:
        if os.path.exists(zone):
            try:
                with open(zone, "r") as f:
                    temp_c = int(f.read().strip()) / 1000.0
                    stats["temperature"] = f"{temp_c:.1f}°C"
                    break
            except Exception:
                pass

    # Linux /proc/loadavg
    if os.path.exists("/proc/loadavg"):
        try:
            with open("/proc/loadavg", "r") as f:
                parts = f.read().split()[:3]
                stats["cpu_usage"] = f"Load (1/5/15m): {', '.join(parts)}"
        except Exception:
            pass

    # Linux /proc/meminfo
    if os.path.exists("/proc/meminfo"):
        try:
            mem = {}
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        mem[parts[0].strip()] = int(parts[1].strip().split()[0])
            total_mb = mem.get("MemTotal", 0) // 1024
            avail_mb = mem.get("MemAvailable", 0) // 1024
            used_mb = total_mb - avail_mb
            pct = int((used_mb / max(total_mb, 1)) * 100)
            stats["memory"] = f"{used_mb}MB / {total_mb}MB ({pct}%)"
        except Exception:
            pass

    # Linux /proc/uptime
    if os.path.exists("/proc/uptime"):
        try:
            with open("/proc/uptime", "r") as f:
                uptime_sec = float(f.read().split()[0])
                days = int(uptime_sec // 86400)
                hours = int((uptime_sec % 86400) // 3600)
                mins = int((uptime_sec % 3600) // 60)
                stats["uptime"] = f"{days}d {hours}h {mins}m"
        except Exception:
            pass

    # Disk usage
    try:
        root_dir = (os.path.splitdrive(os.path.abspath(__file__))[0] + "\\") if platform.system() == "Windows" else "/"
        total, used, free = shutil.disk_usage(root_dir)
        total_gb = total // (2**30)
        used_gb = used // (2**30)
        pct = int((used / max(total, 1)) * 100)
        stats["disk"] = f"{used_gb}GB / {total_gb}GB ({pct}%)"
    except Exception:
        pass

    return stats

# Initialize on import
init_db()
