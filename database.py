import datetime
import json
import os
import platform
import re
import shutil
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assistant_memory.db")
_local = threading.local()
_init_lock = threading.Lock()
_initialized_db_path: str | None = None

# Pre-compiled regular expressions (eliminates per-call regex compilation overhead)
_END_OF_ATTACHMENT_RE = re.compile(r"---\s*\[END OF[^\]]*\]\s*---")
_ATTACHED_FILE_RE = re.compile(r"---\s*\[ATTACHED FILE:\s*([^\]]+)\]")
_WORD_TOKEN_RE = re.compile(r"\b\w{3,}\b")


def get_connection() -> sqlite3.Connection:
    """Returns a thread-local cached SQLite connection tuned for WAL concurrency."""
    conn = getattr(_local, "conn", None)
    cached_path = getattr(_local, "db_path", None)
    if conn is None or cached_path != DB_PATH:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        conn = sqlite3.connect(DB_PATH, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            conn.execute("PRAGMA cache_size = -64000;")   # 64MB page cache
            conn.execute("PRAGMA temp_store = MEMORY;")
            conn.execute("PRAGMA mmap_size = 268435456;")  # 256MB mmap I/O
            conn.execute("PRAGMA busy_timeout = 30000;")
        except sqlite3.Error:
            pass
        _local.conn = conn
        _local.db_path = DB_PATH
    return conn


@contextmanager
def get_db(max_retries: int = 4, retry_delay: float = 0.04):
    """
    Thread-safe transactional context manager with exponential backoff on SQLite lock contention.
    Supports re-entrant calls on the same thread without premature commits.
    """
    conn = get_connection()
    depth = getattr(_local, "tx_depth", 0)
    if depth > 0:
        _local.tx_depth = depth + 1
        try:
            yield conn
        finally:
            _local.tx_depth = depth
        return

    _local.tx_depth = 1
    retries = 0
    try:
        while True:
            try:
                yield conn
                conn.commit()
                break
            except sqlite3.OperationalError as e:
                conn.rollback()
                if "locked" in str(e).lower() and retries < max_retries:
                    retries += 1
                    time.sleep(retry_delay * (2 ** retries))
                    continue
                raise
            except Exception:
                conn.rollback()
                raise
    finally:
        _local.tx_depth = 0


def estimate_tokens(text: str) -> int:
    """Fast token estimator (~3.8 chars/token for English & code)."""
    if not text:
        return 0
    return max(1, int(len(text) / 3.8))


def init_db(force: bool = False) -> None:
    """Initializes tables, indices, FTS5 triggers, and schema migrations idempotently."""
    global _initialized_db_path
    if not force and _initialized_db_path == DB_PATH and os.path.exists(DB_PATH):
        return

    with _init_lock:
        if not force and _initialized_db_path == DB_PATH and os.path.exists(DB_PATH):
            return

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
                conversation_id TEXT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                token_count INTEGER DEFAULT 0,
                metadata TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            c.execute("PRAGMA table_info(messages);")
            msg_cols = {col[1] for col in c.fetchall()}
            if "session_id" not in msg_cols:
                c.execute("ALTER TABLE messages ADD COLUMN session_id TEXT;")
                if "conversation_id" in msg_cols:
                    c.execute("UPDATE messages SET session_id = conversation_id WHERE session_id IS NULL;")
            if "conversation_id" not in msg_cols:
                c.execute("ALTER TABLE messages ADD COLUMN conversation_id TEXT;")
            c.execute("UPDATE messages SET session_id = conversation_id WHERE session_id IS NULL AND conversation_id IS NOT NULL;")
            c.execute("UPDATE messages SET conversation_id = session_id WHERE conversation_id IS NULL AND session_id IS NOT NULL;")

            c.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id ASC);")
            c.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, id ASC);")

            # 3. Knowledge Base + FTS5
            c.execute("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                lesson TEXT NOT NULL,
                code_example TEXT DEFAULT '',
                tags TEXT DEFAULT '',
                source TEXT DEFAULT 'distilled',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            c.execute("PRAGMA table_info(knowledge);")
            k_cols = {col[1] for col in c.fetchall()}
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
                c.execute("""
                CREATE TRIGGER IF NOT EXISTS trg_knowledge_au AFTER UPDATE ON knowledge BEGIN
                    INSERT INTO knowledge_fts(knowledge_fts, rowid, topic, lesson, tags, code_example)
                    VALUES('delete', old.id, old.topic, old.lesson, old.tags, old.code_example);
                    INSERT INTO knowledge_fts(rowid, topic, lesson, tags, code_example)
                    VALUES (new.id, new.topic, new.lesson, new.tags, new.code_example);
                END;
                """)
            except sqlite3.Error:
                pass

            # 4. Knowledge Gaps
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
            c.execute("PRAGMA table_info(gaps);")
            gap_cols = {col[1] for col in c.fetchall()}
            for col_name, col_type in [
                ("status", "TEXT DEFAULT 'pending'"),
                ("attempts", "INTEGER DEFAULT 0"),
                ("last_error", "TEXT DEFAULT ''"),
                ("resolved_at", "TIMESTAMP"),
            ]:
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

        _initialized_db_path = DB_PATH


# --- Sessions / Conversations Management ---

def _derive_session_title(content: str) -> str:
    """Derives a clean, concise conversation title from a user prompt."""
    if not content:
        return "New Chat"
    cleaned = content
    if "--- [ATTACHED FILE:" in cleaned and "--- [END OF" in cleaned:
        parts = _END_OF_ATTACHMENT_RE.split(cleaned, maxsplit=1)
        tail = parts[1].strip() if len(parts) > 1 else ""
        if tail:
            cleaned = tail
        else:
            m = _ATTACHED_FILE_RE.search(cleaned)
            cleaned = f"File: {m.group(1).strip()}" if m else "Attached File"
    cleaned = cleaned.replace("\n", " ").strip()
    if not cleaned:
        return "New Chat"
    return (cleaned[:36] + "...") if len(cleaned) > 36 else cleaned


def create_session(
    title: str = "New Chat",
    mode: str = "assistant",
    session_id: str | None = None,
    conv_id: str | None = None
) -> dict[str, Any]:
    sid = session_id or conv_id or str(uuid.uuid4())[:8]
    clean_title = (title or "New Chat").strip() or "New Chat"
    now = datetime.datetime.now().isoformat()
    with get_db() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO sessions (id, title, mode, pinned, created_at, updated_at)
            VALUES (?, ?, ?, 0, ?, ?)
        """, (sid, clean_title, mode, now, now))
        conn.execute("""
            INSERT OR REPLACE INTO conversations (id, title, mode, pinned, created_at, updated_at)
            VALUES (?, ?, ?, 0, ?, ?)
        """, (sid, clean_title, mode, now, now))
    return {"id": sid, "title": clean_title, "mode": mode, "pinned": 0, "created_at": now, "updated_at": now}


def get_sessions(limit: int = 100) -> list[dict[str, Any]]:
    """Fetches sessions using indexed session_id lookup (O(S log M) instead of O(S*M) OR-join)."""
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""
            SELECT s.id, s.title, s.mode, s.pinned, s.created_at, s.updated_at,
                   (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) as message_count,
                   (SELECT content FROM messages m WHERE m.session_id = s.id ORDER BY m.id DESC LIMIT 1) as last_message
            FROM sessions s
            ORDER BY s.pinned DESC, s.updated_at DESC
            LIMIT ?
        """, (limit,))
        rows = [dict(r) for r in c.fetchall()]
        if not rows:
            c.execute("""
                SELECT c.id, c.title, c.mode, c.pinned, c.created_at, c.updated_at,
                       (SELECT COUNT(*) FROM messages m WHERE m.session_id = c.id) as message_count,
                       (SELECT content FROM messages m WHERE m.session_id = c.id ORDER BY m.id DESC LIMIT 1) as last_message
                FROM conversations c
                ORDER BY c.pinned DESC, c.updated_at DESC
                LIMIT ?
            """, (limit,))
            rows = [dict(r) for r in c.fetchall()]
        return rows


def get_session(session_id: str) -> dict[str, Any] | None:
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT id, title, mode, pinned, created_at, updated_at FROM sessions WHERE id = ?", (session_id,))
        row = c.fetchone()
        if not row:
            c.execute("SELECT id, title, mode, pinned, created_at, updated_at FROM conversations WHERE id = ?", (session_id,))
            row = c.fetchone()
        return dict(row) if row else None


def update_session(
    session_id: str,
    title: str | None = None,
    mode: str | None = None,
    pinned: bool | None = None
) -> bool:
    with get_db() as conn:
        c = conn.cursor()
        updates = []
        params: list[Any] = []
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


create_conversation = create_session
get_conversations = get_sessions
get_conversation = get_session
update_conversation = update_session
delete_conversation = delete_session


# --- Messages & Context Management ---

def save_message(session_id: str, role: str, content: str, metadata: dict | None = None) -> int:
    token_count = estimate_tokens(content)
    meta_json = json.dumps(metadata or {})

    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT id, title FROM sessions WHERE id = ?", (session_id,))
        existing_row = c.fetchone()
        if not existing_row:
            c.execute("SELECT id, title FROM conversations WHERE id = ?", (session_id,))
            existing_row = c.fetchone()

        if not existing_row:
            title = _derive_session_title(content) if role == "user" else "New Chat"
            now = datetime.datetime.now().isoformat()
            c.execute("""
                INSERT OR REPLACE INTO sessions (id, title, mode, pinned, created_at, updated_at)
                VALUES (?, ?, 'assistant', 0, ?, ?)
            """, (session_id, title, now, now))
            c.execute("""
                INSERT OR REPLACE INTO conversations (id, title, mode, pinned, created_at, updated_at)
                VALUES (?, ?, 'assistant', 0, ?, ?)
            """, (session_id, title, now, now))
        else:
            if role == "user" and (existing_row["title"] in ("New Chat", "", None)):
                new_title = _derive_session_title(content)
                c.execute("UPDATE sessions SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (new_title, session_id))
                c.execute("UPDATE conversations SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (new_title, session_id))
            else:
                c.execute("UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (session_id,))
                c.execute("UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (session_id,))

        c.execute("""
            INSERT INTO messages (session_id, conversation_id, role, content, token_count, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (session_id, session_id, role, content, token_count, meta_json))
        msg_id = c.lastrowid

        c.execute("INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
                  (session_id, role, content))

        return int(msg_id or 0)


def get_last_session_message(session_id: str) -> dict[str, Any] | None:
    """Fetches only the latest message in O(log M) via index instead of scanning 200 rows."""
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""
            SELECT id, session_id, role, content, token_count, metadata, created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT 1
        """, (session_id,))
        row = c.fetchone()
        return dict(row) if row else None


def get_session_messages(session_id: str, limit: int = 200) -> list[dict[str, Any]]:
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""
            SELECT id, session_id, role, content, token_count, metadata, created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY id ASC
            LIMIT ?
        """, (session_id, limit))
        rows = [dict(r) for r in c.fetchall()]
        if not rows:
            c.execute("""
                SELECT id, session_id, role, content, token_count, metadata, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id ASC
                LIMIT ?
            """, (session_id, limit))
            rows = [dict(r) for r in c.fetchall()]
        return rows


get_conversation_messages = get_session_messages


def clear_session_messages(session_id: str) -> bool:
    with get_db() as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ? OR conversation_id = ?", (session_id, session_id))
        conn.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
        return True


clear_conversation_messages = clear_session_messages


def get_context_window(session_id: str, max_tokens: int = 4200) -> list[dict[str, str]]:
    """
    Sliding-window context manager:
    Packs historical messages under max_tokens budget while preserving initial user intent and recent turns.
    """
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""
            SELECT role, content, token_count FROM messages
            WHERE session_id = ?
            ORDER BY id ASC
        """, (session_id,))
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


save_chat_message = save_message


def get_recent_chat(session_id: str, limit: int = 8) -> list[dict[str, str]]:
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


def clear_chat_history(session_id: str | None = None) -> None:
    with get_db() as conn:
        if session_id:
            conn.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM messages WHERE session_id = ? OR conversation_id = ?", (session_id, session_id))
        else:
            conn.execute("DELETE FROM chat_history")
            conn.execute("DELETE FROM messages")


# --- Audit & Logging ---

def log_event(event_type: str, details: str, status: str = "info") -> None:
    with get_db() as conn:
        conn.execute(
            "INSERT INTO audit_log (event_type, details, status) VALUES (?, ?, ?)",
            (event_type, str(details), status)
        )


def get_recent_audit_logs(limit: int = 25) -> list[dict[str, Any]]:
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT id, event_type, details, status, created_at FROM audit_log ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(r) for r in c.fetchall()]


# --- Knowledge & Memory Retrieval ---

def get_counts() -> dict[str, int]:
    """O(1) aggregate count query for status endpoint (avoids loading entire tables into RAM)."""
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""
            SELECT
                (SELECT COUNT(*) FROM knowledge) AS knowledge_count,
                (SELECT COUNT(*) FROM gaps WHERE status IN ('pending', 'failed')) AS pending_gaps
        """)
        row = c.fetchone()
        if row:
            return {"knowledge_count": int(row["knowledge_count"]), "pending_gaps": int(row["pending_gaps"])}
        return {"knowledge_count": 0, "pending_gaps": 0}


def add_knowledge(topic: str, lesson: str, code_example: str = "", tags: str = "", source: str = "distilled") -> int:
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""
            INSERT INTO knowledge (topic, lesson, code_example, tags, source)
            VALUES (?, ?, ?, ?, ?)
        """, (topic.strip(), lesson.strip(), code_example.strip() if code_example else "", tags.strip(), source))
        kid = int(c.lastrowid or 0)
        c.execute(
            "INSERT INTO audit_log (event_type, details, status) VALUES (?, ?, ?)",
            ("knowledge_added", f"Assimilated: {topic}", "success")
        )
        return kid


def update_knowledge(knowledge_id: int, topic: str, lesson: str, code_example: str = "", tags: str = "") -> bool:
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""
            UPDATE knowledge
            SET topic = ?, lesson = ?, code_example = ?, tags = ?
            WHERE id = ?
        """, (topic.strip(), lesson.strip(), (code_example or "").strip(), (tags or "").strip(), knowledge_id))
        c.execute(
            "INSERT INTO audit_log (event_type, details, status) VALUES (?, ?, ?)",
            ("knowledge_updated", f"Updated knowledge ID: {knowledge_id} ({topic})", "info")
        )
        return True


def delete_knowledge(knowledge_id: int) -> bool:
    with get_db() as conn:
        c = conn.cursor()
        c.execute("DELETE FROM knowledge WHERE id = ?", (knowledge_id,))
        c.execute(
            "INSERT INTO audit_log (event_type, details, status) VALUES (?, ?, ?)",
            ("knowledge_deleted", f"Deleted knowledge ID: {knowledge_id}", "info")
        )
        return True


def search_knowledge(query: str, limit: int = 3) -> list[dict[str, Any]]:
    if not query or not query.strip():
        return []

    tokens = [t.lower() for t in _WORD_TOKEN_RE.findall(query)]
    if not tokens:
        return []

    with get_db() as conn:
        c = conn.cursor()
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
        except sqlite3.Error:
            pass

        clauses = []
        params: list[Any] = []
        for tok in tokens[:4]:
            clauses.append("(topic LIKE ? OR tags LIKE ?)")
            params.extend([f"%{tok}%", f"%{tok}%"])

        sql = f"SELECT id, topic, lesson, code_example, tags FROM knowledge WHERE {' OR '.join(clauses)} ORDER BY id DESC LIMIT ?"
        params.append(limit)
        try:
            c.execute(sql, params)
            return [dict(r) for r in c.fetchall()]
        except sqlite3.Error:
            return []


def get_all_knowledge(limit: int = 100) -> list[dict[str, Any]]:
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
            return int(existing["id"])

        c.execute("INSERT INTO gaps (question, context, status) VALUES (?, ?, 'pending')",
                  (question.strip(), context.strip()))
        gid = int(c.lastrowid or 0)
        c.execute(
            "INSERT INTO audit_log (event_type, details, status) VALUES (?, ?, ?)",
            ("gap_recorded", f"Recorded gap #{gid}: {question}", "warning")
        )
        return gid


def get_unresolved_gaps(limit: int = 50) -> list[dict[str, Any]]:
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""
            SELECT id, question, context, status, attempts, last_error, created_at
            FROM gaps WHERE status IN ('pending', 'failed')
            ORDER BY id ASC LIMIT ?
        """, (limit,))
        return [dict(r) for r in c.fetchall()]


def set_gap_status(gap_id: int, status: str, error_msg: str = "") -> None:
    with get_db() as conn:
        now = datetime.datetime.now().isoformat() if status == "resolved" else None
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
        return int(c.lastrowid or 0)


def get_todos() -> list[dict[str, Any]]:
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
        return int(c.lastrowid or 0)


def update_note(note_id: int, title: str, content: str, tags: str = "") -> bool:
    with get_db() as conn:
        conn.execute("""
            UPDATE notes
            SET title = ?, content = ?, tags = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (title.strip(), content.strip(), tags.strip(), note_id))
        return True


def get_notes() -> list[dict[str, Any]]:
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT id, title, content, tags, updated_at FROM notes ORDER BY id DESC")
        return [dict(r) for r in c.fetchall()]


def delete_note(note_id: int) -> bool:
    with get_db() as conn:
        conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        return True


# --- Hardware Diagnostics & Telemetry ---

def get_system_telemetry() -> dict[str, str]:
    stats = {
        "os": f"{platform.system()} {platform.release()}",
        "arch": platform.machine(),
        "temperature": "N/A",
        "cpu_usage": f"{os.cpu_count() or 1} Cores Active",
        "memory": "N/A",
        "disk": "N/A",
        "uptime": "N/A"
    }

    for zone in ["/sys/class/thermal/thermal_zone0/temp", "/sys/devices/virtual/thermal/thermal_zone0/temp"]:
        if os.path.exists(zone):
            try:
                with open(zone, encoding="utf-8") as f:
                    temp_c = int(f.read().strip()) / 1000.0
                    stats["temperature"] = f"{temp_c:.1f}°C"
                    break
            except (OSError, ValueError):
                pass

    if os.path.exists("/proc/loadavg"):
        try:
            with open("/proc/loadavg", encoding="utf-8") as f:
                parts = f.read().split()[:3]
                stats["cpu_usage"] = f"Load (1/5/15m): {', '.join(parts)}"
        except OSError:
            pass

    if os.path.exists("/proc/meminfo"):
        try:
            mem = {}
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        mem[parts[0].strip()] = int(parts[1].strip().split()[0])
            total_mb = mem.get("MemTotal", 0) // 1024
            avail_mb = mem.get("MemAvailable", 0) // 1024
            used_mb = total_mb - avail_mb
            pct = int((used_mb / max(total_mb, 1)) * 100)
            stats["memory"] = f"{used_mb}MB / {total_mb}MB ({pct}%)"
        except (OSError, ValueError):
            pass

    if os.path.exists("/proc/uptime"):
        try:
            with open("/proc/uptime", encoding="utf-8") as f:
                uptime_sec = float(f.read().split()[0])
                days = int(uptime_sec // 86400)
                hours = int((uptime_sec % 86400) // 3600)
                mins = int((uptime_sec % 3600) // 60)
                stats["uptime"] = f"{days}d {hours}h {mins}m"
        except (OSError, ValueError):
            pass

    try:
        import psutil
        if stats["cpu_usage"].endswith("Cores Active"):
            cpu_pct = psutil.cpu_percent(interval=None)
            stats["cpu_usage"] = f"{cpu_pct:.1f}% ({os.cpu_count() or 1} cores)"
        if stats["memory"] == "N/A":
            vm = psutil.virtual_memory()
            total_mb = vm.total // (1024 * 1024)
            used_mb = (vm.total - vm.available) // (1024 * 1024)
            stats["memory"] = f"{used_mb}MB / {total_mb}MB ({int(vm.percent)}%)"
        if stats["uptime"] == "N/A":
            uptime_sec = max(0.0, time.time() - psutil.boot_time())
            days = int(uptime_sec // 86400)
            hours = int((uptime_sec % 86400) // 3600)
            mins = int((uptime_sec % 3600) // 60)
            stats["uptime"] = f"{days}d {hours}h {mins}m"
        if stats["temperature"] == "N/A" and hasattr(psutil, "sensors_temperatures"):
            temps = psutil.sensors_temperatures() or {}
            for entries in temps.values():
                if entries and getattr(entries[0], "current", None):
                    stats["temperature"] = f"{entries[0].current:.1f}°C"
                    break
    except Exception:
        pass

    if platform.system() == "Windows":
        try:
            import ctypes
            if stats["memory"] == "N/A":
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]
                mem_status = MEMORYSTATUSEX()
                mem_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem_status)):
                    total_mb = int(mem_status.ullTotalPhys // (1024 * 1024))
                    avail_mb = int(mem_status.ullAvailPhys // (1024 * 1024))
                    used_mb = total_mb - avail_mb
                    stats["memory"] = f"{used_mb}MB / {total_mb}MB ({mem_status.dwMemoryLoad}%)"
            if stats["uptime"] == "N/A":
                tick_ms = ctypes.windll.kernel32.GetTickCount64()
                uptime_sec = tick_ms / 1000.0
                days = int(uptime_sec // 86400)
                hours = int((uptime_sec % 86400) // 3600)
                mins = int((uptime_sec % 3600) // 60)
                stats["uptime"] = f"{days}d {hours}h {mins}m"
        except Exception:
            pass

    if stats["temperature"] == "N/A":
        stats["temperature"] = "Nominal ( Passive )"

    try:
        root_dir = (os.path.splitdrive(os.path.abspath(__file__))[0] + "\\") if platform.system() == "Windows" else "/"
        total, used, _free = shutil.disk_usage(root_dir)
        total_gb = total // (2**30)
        used_gb = used // (2**30)
        pct = int((used / max(total, 1)) * 100)
        stats["disk"] = f"{used_gb}GB / {total_gb}GB ({pct}%)"
    except OSError:
        pass

    return stats


init_db()
