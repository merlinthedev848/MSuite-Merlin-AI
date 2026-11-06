import datetime
import hashlib
import json
import os
import platform
import re
import secrets
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


def close_connection() -> None:
    """Closes the current thread's cached SQLite connection and resets local state."""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
        _local.conn = None
        _local.db_path = None
        _local.tx_depth = 0


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
        db_dir = os.path.dirname(os.path.abspath(DB_PATH))
        os.makedirs(db_dir, exist_ok=True)
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
    Thread-safe transactional context manager with re-entrancy support.
    Outer block controls the single atomic transaction commit/rollback.
    Lock contention is governed by SQLite's busy_timeout PRAGMA (30s).
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
    try:
        yield conn
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
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

            # 0. User Accounts & Session Tokens (Per-User Saved Chats)
            c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                display_name TEXT DEFAULT '',
                is_admin INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);")

            c.execute("PRAGMA table_info(users);")
            u_cols = {col[1] for col in c.fetchall()}
            if "is_admin" not in u_cols:
                c.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER DEFAULT 0;")

            c.execute("""
            CREATE TABLE IF NOT EXISTS user_tokens (
                token TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP
            );
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_tokens_user ON user_tokens(user_id);")

            # 1. Sessions / Conversations
            c.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                mode TEXT DEFAULT 'assistant',
                pinned INTEGER DEFAULT 0,
                user_id TEXT DEFAULT NULL,
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
                user_id TEXT DEFAULT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            c.execute("PRAGMA table_info(sessions);")
            s_cols = {col[1] for col in c.fetchall()}
            if "user_id" not in s_cols:
                c.execute("ALTER TABLE sessions ADD COLUMN user_id TEXT DEFAULT NULL;")
            c.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id, updated_at DESC);")

            c.execute("PRAGMA table_info(conversations);")
            c_cols = {col[1] for col in c.fetchall()}
            if "user_id" not in c_cols:
                c.execute("ALTER TABLE conversations ADD COLUMN user_id TEXT DEFAULT NULL;")
            c.execute("CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id, updated_at DESC);")

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
            c.execute("CREATE INDEX IF NOT EXISTS idx_todos_status ON todos(completed ASC, id DESC);")

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
            c.execute("CREATE INDEX IF NOT EXISTS idx_notes_updated ON notes(updated_at DESC);")

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
            c.execute("CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_log(id DESC);")

            # 9. Calendar Events (Automated Telephony & Assistant Scheduling)
            c.execute("""
            CREATE TABLE IF NOT EXISTS calendar_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT DEFAULT '',
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                attendee_name TEXT DEFAULT '',
                attendee_phone TEXT DEFAULT '',
                attendee_email TEXT DEFAULT '',
                status TEXT DEFAULT 'confirmed',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_calendar_start ON calendar_events(start_time ASC);")
            c.execute("CREATE INDEX IF NOT EXISTS idx_calendar_status_start ON calendar_events(status, start_time ASC);")

            # 10. Dense Vector Knowledge Embeddings (RAG)
            c.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                knowledge_id INTEGER NOT NULL REFERENCES knowledge(id) ON DELETE CASCADE,
                chunk_index INTEGER DEFAULT 0,
                chunk_text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                dimension INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_embeddings_kid ON knowledge_embeddings(knowledge_id);")

        _initialized_db_path = DB_PATH
        if not os.getenv("PYTEST_CURRENT_TEST"):
            try:
                seed_foundational_datasets()
            except Exception:
                pass


# --- User Accounts & Session Authentication ---

def hash_password(password: str, salt: str) -> str:
    """Computes PBKDF2-HMAC-SHA256 hash with 100,000 iterations."""
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000).hex()


def create_user(username: str, password: str, display_name: str = "", is_admin: bool | None = None) -> dict[str, Any]:
    """Registers a new user account with secure hashed password."""
    clean_username = username.strip().lower()
    if not clean_username or len(clean_username) < 3:
        raise ValueError("Username must be at least 3 characters long.")
    if not re.match(r"^[a-zA-Z0-9_.-]+$", clean_username):
        raise ValueError("Username can only contain alphanumeric characters, dots, underscores, or dashes.")
    if not password or len(password) < 4:
        raise ValueError("Password must be at least 4 characters long.")

    uid = str(uuid.uuid4())[:12]
    salt = secrets.token_hex(16)
    pw_hash = hash_password(password, salt)
    clean_display = display_name.strip() or clean_username

    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT id FROM users WHERE username = ?", (clean_username,))
        if c.fetchone():
            raise ValueError(f"Username '{clean_username}' already exists.")

        # Determine admin status: explicit argument, standard admin aliases, or first registered user
        if is_admin is not None:
            admin_flag = 1 if is_admin else 0
        elif clean_username in ("admin", "merlin", "administrator", "root"):
            admin_flag = 1
        else:
            c.execute("SELECT COUNT(*) as total_users FROM users;")
            row_count = c.fetchone()["total_users"]
            admin_flag = 1 if row_count == 0 else 0

        now = datetime.datetime.now().isoformat()
        c.execute("""
            INSERT INTO users (id, username, password_hash, salt, display_name, is_admin, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (uid, clean_username, pw_hash, salt, clean_display, admin_flag, now))

    token = create_user_token(uid)
    return {
        "id": uid,
        "username": clean_username,
        "display_name": clean_display,
        "is_admin": bool(admin_flag),
        "token": token,
    }


def authenticate_user(username: str, password: str) -> dict[str, Any] | None:
    """Authenticates username and password. Returns user dict with new token if valid."""
    clean_username = username.strip().lower()
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT id, username, password_hash, salt, display_name, is_admin FROM users WHERE username = ?", (clean_username,))
        row = c.fetchone()
        if not row:
            return None

        computed_hash = hash_password(password, row["salt"])
        if not secrets.compare_digest(computed_hash, row["password_hash"]):
            return None

        uid = row["id"]
        is_adm = bool(row["is_admin"]) or row["username"].lower() in ("admin", "merlin", "administrator", "root")
        token = create_user_token(uid)
        return {
            "id": uid,
            "username": row["username"],
            "display_name": row["display_name"] or row["username"],
            "is_admin": is_adm,
            "token": token,
        }


def create_user_token(user_id: str, days_valid: int = 30) -> str:
    """Generates and stores a new cryptographically secure session token."""
    token = secrets.token_urlsafe(32)
    now_dt = datetime.datetime.now()
    now = now_dt.isoformat()
    expires_at = (now_dt + datetime.timedelta(days=max(1, int(days_valid)))).isoformat()
    with get_db() as conn:
        conn.execute("""
            INSERT INTO user_tokens (token, user_id, created_at, expires_at)
            VALUES (?, ?, ?, ?)
        """, (token, user_id, now, expires_at))
    return token


def get_user_by_token(token: str) -> dict[str, Any] | None:
    """Resolves an auth token to its active user record, verifying expiration."""
    if not token or not token.strip():
        return None
    now_iso = datetime.datetime.now().isoformat()
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""
            SELECT u.id, u.username, u.display_name, u.is_admin, u.created_at, t.expires_at
            FROM user_tokens t
            JOIN users u ON u.id = t.user_id
            WHERE t.token = ? AND (t.expires_at IS NULL OR t.expires_at > ?)
        """, (token.strip(), now_iso))
        row = c.fetchone()
        if not row:
            return None
        res = dict(row)
        res["is_admin"] = bool(res.get("is_admin")) or res["username"].lower() in ("admin", "merlin", "administrator", "root")
        return res


def delete_user_token(token: str) -> bool:
    """Invalidates an auth token upon logout."""
    if not token:
        return False
    with get_db() as conn:
        c = conn.cursor()
        c.execute("DELETE FROM user_tokens WHERE token = ?", (token.strip(),))
        return c.rowcount > 0


def associate_session_with_user(session_id: str, user_id: str) -> bool:
    """Assigns a session to a user account, persisting it if it was a transient guest session."""
    if not user_id:
        return False
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT id FROM sessions WHERE id = ?", (session_id,))
        row = c.fetchone()
        now = datetime.datetime.now().isoformat()
        if row:
            c.execute("UPDATE sessions SET user_id = ?, updated_at = ? WHERE id = ?", (user_id, now, session_id))
            c.execute("UPDATE conversations SET user_id = ?, updated_at = ? WHERE id = ?", (user_id, now, session_id))
        else:
            c.execute("""
                INSERT OR REPLACE INTO sessions (id, title, mode, pinned, user_id, created_at, updated_at)
                VALUES (?, 'Claimed Chat', 'assistant', 0, ?, ?, ?)
            """, (session_id, user_id, now, now))
            c.execute("""
                INSERT OR REPLACE INTO conversations (id, title, mode, pinned, user_id, created_at, updated_at)
                VALUES (?, 'Claimed Chat', 'assistant', 0, ?, ?, ?)
            """, (session_id, user_id, now, now))
        return True


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
    conv_id: str | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    sid = session_id or conv_id or str(uuid.uuid4())[:8]
    clean_title = (title or "New Chat").strip() or "New Chat"
    now = datetime.datetime.now().isoformat()
    if not user_id:
        # STRICT GUEST MODE: No conversation should be stored in guest mode!
        # Return a transient in-memory session object without persisting to database.
        return {
            "id": sid,
            "title": clean_title,
            "mode": mode,
            "pinned": 0,
            "user_id": None,
            "created_at": now,
            "updated_at": now,
            "guest": True,
        }

    with get_db() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO sessions (id, title, mode, pinned, user_id, created_at, updated_at)
            VALUES (?, ?, ?, 0, ?, ?, ?)
        """, (sid, clean_title, mode, user_id, now, now))
        conn.execute("""
            INSERT OR REPLACE INTO conversations (id, title, mode, pinned, user_id, created_at, updated_at)
            VALUES (?, ?, ?, 0, ?, ?, ?)
        """, (sid, clean_title, mode, user_id, now, now))
    return {"id": sid, "title": clean_title, "mode": mode, "pinned": 0, "user_id": user_id, "created_at": now, "updated_at": now}


def get_sessions(limit: int = 100, user_id: str | None = None, guest_only: bool = False) -> list[dict[str, Any]]:
    """Fetches sessions using indexed session_id lookup. Scoped per user when user_id is provided."""
    if not user_id:
        # In Guest mode, no saved conversations exist or should be returned
        return []

    with get_db() as conn:
        c = conn.cursor()
        sql_sess = """
            SELECT s.id, s.title, s.mode, s.pinned, s.user_id, s.created_at, s.updated_at,
                   (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) as message_count,
                   (SELECT content FROM messages m WHERE m.session_id = s.id ORDER BY m.id DESC LIMIT 1) as last_message
            FROM sessions s
            WHERE s.user_id = ?
            ORDER BY s.pinned DESC, s.updated_at DESC
            LIMIT ?
        """
        c.execute(sql_sess, (user_id, limit))
        return [dict(r) for r in c.fetchall()]


def get_session(session_id: str) -> dict[str, Any] | None:
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT id, title, mode, pinned, user_id, created_at, updated_at FROM sessions WHERE id = ?", (session_id,))
        row = c.fetchone()
        if not row:
            c.execute("SELECT id, title, mode, pinned, user_id, created_at, updated_at FROM conversations WHERE id = ?", (session_id,))
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

def save_message(
    session_id: str,
    role: str,
    content: str,
    metadata: dict | None = None,
    user_id: str | None = None,
) -> int:
    token_count = estimate_tokens(content)
    meta_json = json.dumps(metadata or {})

    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT id, title, user_id FROM sessions WHERE id = ?", (session_id,))
        existing_row = c.fetchone()
        if not existing_row:
            c.execute("SELECT id, title, user_id FROM conversations WHERE id = ?", (session_id,))
            existing_row = c.fetchone()

        resolved_user_id = user_id or (existing_row["user_id"] if existing_row else None)

        # STRICT GUEST MODE RULE:
        # No conversation should be stored in "guest mode" (resolved_user_id is None).
        # Only conversations can be saved in Signed in mode.
        if not resolved_user_id:
            return 0

        if not existing_row:
            title = _derive_session_title(content) if role == "user" else "New Chat"
            now = datetime.datetime.now().isoformat()
            c.execute("""
                INSERT OR REPLACE INTO sessions (id, title, mode, pinned, user_id, created_at, updated_at)
                VALUES (?, ?, 'assistant', 0, ?, ?, ?)
            """, (session_id, title, resolved_user_id, now, now))
            c.execute("""
                INSERT OR REPLACE INTO conversations (id, title, mode, pinned, user_id, created_at, updated_at)
                VALUES (?, ?, 'assistant', 0, ?, ?, ?)
            """, (session_id, title, resolved_user_id, now, now))
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

def get_counts(include_all: bool = False) -> dict[str, int]:
    """O(1) aggregate count query for status endpoint (avoids loading entire tables into RAM)."""
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""
            SELECT
                (SELECT COUNT(*) FROM knowledge) AS knowledge_count,
                (SELECT COUNT(*) FROM gaps WHERE status IN ('pending', 'failed')) AS pending_gaps,
                (SELECT COUNT(*) FROM calendar_events WHERE status = 'confirmed') AS calendar_count,
                (SELECT COUNT(*) FROM knowledge_embeddings) AS embeddings_count
        """)
        row = c.fetchone()
        if row:
            row_dict = dict(row)
            res = {
                "knowledge_count": int(row_dict.get("knowledge_count", 0)),
                "pending_gaps": int(row_dict.get("pending_gaps", 0)),
            }
            if include_all:
                res["calendar_count"] = int(row_dict.get("calendar_count", 0))
                res["embeddings_count"] = int(row_dict.get("embeddings_count", 0))
            return res
        res = {"knowledge_count": 0, "pending_gaps": 0}
        if include_all:
            res["calendar_count"] = 0
            res["embeddings_count"] = 0
        return res


def add_knowledge(topic: str, lesson: str, code_example: str = "", tags: str = "", source: str = "distilled") -> int:
    clean_topic = topic.strip()
    clean_lesson = lesson.strip()
    clean_code = (code_example or "").strip()
    clean_tags = (tags or "").strip()

    with get_db() as conn:
        c = conn.cursor()
        # Intelligent de-duplication: check if exact or case-insensitive topic already exists
        c.execute("SELECT id, lesson, code_example, tags FROM knowledge WHERE lower(topic) = lower(?) LIMIT 1", (clean_topic,))
        existing = c.fetchone()
        if existing:
            kid = int(existing["id"])
            # If the incoming entry provides missing code or richer content, enrich the existing record
            if (clean_code and not existing["code_example"]) or (len(clean_lesson) > len(existing["lesson"])):
                existing_tags = [t.strip() for t in (existing["tags"] or "").split(",") if t.strip()]
                new_tags = [t.strip() for t in clean_tags.split(",") if t.strip()]
                merged_tags = ", ".join(sorted(set(existing_tags + new_tags)))
                chosen_code = clean_code if clean_code else (existing["code_example"] or "")
                c.execute("""
                    UPDATE knowledge
                    SET lesson = ?, code_example = ?, tags = ?, source = ?
                    WHERE id = ?
                """, (clean_lesson, chosen_code, merged_tags, source, kid))
            try:
                index_knowledge_embeddings(kid, f"{clean_topic}\n{clean_lesson}\n{chosen_code}")
            except Exception:
                pass
            return kid

        c.execute("""
            INSERT INTO knowledge (topic, lesson, code_example, tags, source)
            VALUES (?, ?, ?, ?, ?)
        """, (clean_topic, clean_lesson, clean_code, clean_tags, source))
        kid = int(c.lastrowid or 0)
        c.execute(
            "INSERT INTO audit_log (event_type, details, status) VALUES (?, ?, ?)",
            ("knowledge_added", f"Assimilated: {clean_topic}", "success")
        )
        try:
            index_knowledge_embeddings(kid, f"{clean_topic}\n{clean_lesson}\n{clean_code}")
        except Exception:
            pass
        invalidate_dense_cache()
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
                ORDER BY bm25(knowledge_fts, 5.0, 2.0, 3.0, 1.5)
                LIMIT ?
            """, (fts_query, limit))
            results = [dict(r) for r in c.fetchall()]
            if results:
                return results
        except sqlite3.Error:
            # Fallback if bm25 weights or rank not supported
            try:
                c.execute("""
                    SELECT k.id, k.topic, k.lesson, k.code_example, k.tags
                    FROM knowledge_fts fts
                    JOIN knowledge k ON k.id = fts.rowid
                    WHERE knowledge_fts MATCH ?
                    ORDER BY rank
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
            clauses.append("(topic LIKE ? OR lesson LIKE ? OR tags LIKE ? OR code_example LIKE ?)")
            pat = f"%{tok}%"
            params.extend([pat, pat, pat, pat])

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


# --- Dense Vector RAG & Hybrid Search Engine ---

def index_knowledge_embeddings(knowledge_id: int, text: str | None = None) -> int:
    """
    Generates and stores dense vector embeddings for a knowledge entry.
    Chunks longer entries so fine-grained matching is achieved.
    """
    try:
        from embeddings import chunk_text, generate_embedding, pack_embedding
    except ImportError:
        return 0

    with get_db() as conn:
        c = conn.cursor()
        if text is None:
            c.execute("SELECT topic, lesson, code_example FROM knowledge WHERE id = ?", (knowledge_id,))
            row = c.fetchone()
            if not row:
                return 0
            text = f"{row['topic']}\n{row['lesson']}\n{row['code_example']}"

        chunks = chunk_text(text, chunk_size=400, chunk_overlap=60)
        c.execute("DELETE FROM knowledge_embeddings WHERE knowledge_id = ?", (knowledge_id,))
        indexed_count = 0
        for idx, chunk in enumerate(chunks):
            vec = generate_embedding(chunk)
            packed = pack_embedding(vec)
            c.execute("""
                INSERT INTO knowledge_embeddings (knowledge_id, chunk_index, chunk_text, embedding, dimension)
                VALUES (?, ?, ?, ?, ?)
            """, (knowledge_id, idx, chunk, packed, len(vec)))
            indexed_count += 1
        return indexed_count


def seed_foundational_datasets(force: bool = False) -> int:
    """
    Seeds foundational multi-domain dataset knowledge into SQLite + FTS5 + Dense Embeddings.
    Covers Coding, Personal Assistant & Productivity, Document Extraction, and Reasoning.
    """
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) as cnt FROM knowledge WHERE source LIKE 'curated_dataset%';")
        cnt = c.fetchone()["cnt"]
        if cnt > 0 and not force:
            return cnt

    seeds = [
        # 💻 Coding & Systems Engineering
        {
            "topic": "Python AsyncIO Non-Blocking Concurrency & Task Supervisors",
            "lesson": "High-throughput asynchronous servers in Python must never run blocking I/O on the main event loop. Offload CPU-heavy computation or blocking generators to a worker thread via asyncio.to_thread or run_in_executor, communicating back through threadsafe queues (loop.call_soon_threadsafe). Always catch task exceptions to prevent silent event loop crashes.",
            "code_example": "import asyncio\n\nasync def worker_queue_consumer(queue: asyncio.Queue):\n    while True:\n        item = await queue.get()\n        if item is None:\n            break\n        await process_item(item)\n        queue.task_done()",
            "tags": "python,asyncio,concurrency,architecture",
            "source": "curated_dataset:coding",
        },
        {
            "topic": "FastAPI Resilient Streaming (SSE & WebSockets)",
            "lesson": "FastAPI streaming endpoints require defensive lifecycle management. For SSE, yield formatted 'data: ...\\n\\n' frames followed by '[DONE]'. For bidirectional WebSockets, always wrap frame processing in try/finally blocks to ensure generation flags and server-side locks are released even if the client disconnects abruptly.",
            "code_example": "@app.get('/api/chat/stream')\nasync def stream_chat(prompt: str):\n    async def event_generator():\n        for chunk in engine.generate_stream(prompt):\n            yield f'data: {json.dumps({\"chunk\": chunk})}\\n\\n'\n        yield 'data: [DONE]\\n\\n'\n    return StreamingResponse(event_generator(), media_type='text/event-stream')",
            "tags": "fastapi,streaming,sse,websocket,networking",
            "source": "curated_dataset:coding",
        },
        {
            "topic": "SQLite FTS5 Full-Text Search and BM25 Ranking Optimization",
            "lesson": "SQLite FTS5 provides instantaneous full-text lexical search with zero external dependencies. Configure prefix matching ('token*') and order by bm25() with custom column weights. Combine with dense vector similarity via Reciprocal Rank Fusion (RRF: 1/(60 + rank_bm25) + 1/(60 + rank_vector)) for state-of-the-art hybrid RAG retrieval.",
            "code_example": "SELECT k.id, k.topic, k.lesson FROM knowledge_fts fts\nJOIN knowledge k ON k.id = fts.rowid\nWHERE knowledge_fts MATCH ?\nORDER BY bm25(knowledge_fts, 5.0, 2.0, 3.0, 1.5)\nLIMIT 10;",
            "tags": "sqlite,fts5,bm25,search,rag",
            "source": "curated_dataset:coding",
        },

        # 🤖 Personal Assistant & Productivity
        {
            "topic": "Deterministic Calendar Slot Conflict Resolution & Scheduling",
            "lesson": "An intelligent personal assistant must strictly avoid double-booking by enforcing half-open interval checking: slot A overlaps slot B if and only if (start_A < end_B) AND (end_A > start_B). All timestamps should be normalized to ISO 8601 strings and compared deterministically in UTC or local timezone with conflict logging.",
            "code_example": "def has_conflict(start_a, end_a, start_b, end_b):\n    return (start_a < end_b) and (end_a > start_b)",
            "tags": "scheduling,calendar,assistant,productivity",
            "source": "curated_dataset:assistant",
        },
        {
            "topic": "Executive Task Prioritization & Eisenhower Matrix Workflows",
            "lesson": "When triaging daily developer or executive tasks, categorize items into Urgent/Important (Do First), Important/Not Urgent (Schedule), Urgent/Not Important (Delegate), and Neither (Eliminate). When managing tasks in local assistants, track priority tags, due dates, and completion status with audit timestamps.",
            "code_example": "SELECT id, title, priority, due_date FROM todos\nWHERE completed = 0\nORDER BY CASE priority WHEN 'urgent' THEN 1 WHEN 'high' THEN 2 WHEN 'medium' THEN 3 ELSE 4 END, due_date ASC;",
            "tags": "productivity,task_management,eisenhower,workflows",
            "source": "curated_dataset:assistant",
        },

        # 📄 Document Analysis & Summarization
        {
            "topic": "Multi-Format Document Parsing and Structural Chunking",
            "lesson": "Ingesting arbitrary user documents (PDF, Markdown, codebases, text) requires segmenting content into semantically coherent chunks (300-500 tokens) with a 10-15% sliding window overlap. This preserves contextual continuity across chunk boundaries and prevents key entity splits during vector search.",
            "code_example": "def chunk_document(text, chunk_size=400, overlap=60):\n    words = text.split()\n    chunks, i = [], 0\n    while i < len(words):\n        chunks.append(' '.join(words[i:i + chunk_size]))\n        i += chunk_size - overlap\n    return chunks",
            "tags": "nlp,chunking,rag,documents,summarization",
            "source": "curated_dataset:documents",
        },
        {
            "topic": "Executive Summary & Key Action Item Distillation",
            "lesson": "When summarizing transcripts, technical papers, or meeting dialogues, follow the three-tier summary pattern: 1) Executive Bottom Line (1-2 sentences), 2) Key Strategic Decisions & Metrics (bullet points), 3) Concrete Next Steps with Assigned Owners and Deadlines.",
            "code_example": "# Summary Pattern:\n## Executive Summary\n- Primary objective & outcome\n## Strategic Findings\n- Point 1 (Impact)\n- Point 2 (Metrics)\n## Action Items\n- [ ] Owner: Task (Target Date)",
            "tags": "summarization,dialogue,executive_brief,nlp",
            "source": "curated_dataset:documents",
        },

        # 💬 Natural Conversation & Multi-Turn Reasoning
        {
            "topic": "Chain-of-Thought Decomposition for Complex Problem Solving",
            "lesson": "For mathematical, algorithmic, or architectural reasoning, guide the model through step-by-step problem decomposition. State assumptions explicitly, isolate variables, evaluate edge cases (e.g., null pointers, network timeouts, zero divisions), and state the verified conclusion at the end.",
            "code_example": "# Reasoning Pattern:\n1. Problem Understanding: Clarify inputs, outputs, constraints.\n2. Decomposition: Break down into discrete sub-problems.\n3. Execution & Verification: Step-by-step resolution.\n4. Final Synthesis: Direct, actionable answer.",
            "tags": "reasoning,chain_of_thought,logic,problem_solving",
            "source": "curated_dataset:reasoning",
        },
        {
            "topic": "Empathetic, Assertive & Grounded Conversational Alignment",
            "lesson": "A high-capability AI assistant balances warm professionalism with strict factual precision. Acknowledge user constraints, clarify ambiguities before making destructive assumptions, and maintain zero-hallucination discipline by stating confidence boundaries when information is missing.",
            "code_example": "# Response Guideline:\n- Directness: Put the answer first, not preamble.\n- Clarity: Use structured formatting (lists, bold headings).\n- Grounding: Cite specific files, lines, or parameters.",
            "tags": "conversation,alignment,chat,communication",
            "source": "curated_dataset:conversation",
        },
    ]

    added = 0
    for s in seeds:
        kid = add_knowledge(
            topic=s["topic"],
            lesson=s["lesson"],
            code_example=s["code_example"],
            tags=s["tags"],
            source=s["source"],
        )
        try:
            index_knowledge_embeddings(kid)
        except Exception:
            pass
        added += 1
    return added


_DENSE_CACHE: list[dict[str, Any]] = []
_DENSE_CACHE_EXPIRY: float = 0.0


def invalidate_dense_cache() -> None:
    global _DENSE_CACHE, _DENSE_CACHE_EXPIRY
    _DENSE_CACHE = []
    _DENSE_CACHE_EXPIRY = 0.0


def search_knowledge_dense(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """
    Executes dense vector search across indexed knowledge chunks using cosine similarity.
    Uses in-memory caching to achieve ultra-fast sub-millisecond retrieval speeds.
    """
    global _DENSE_CACHE, _DENSE_CACHE_EXPIRY
    try:
        from embeddings import cosine_similarity, generate_embedding
    except ImportError:
        return []

    now = time.time()
    rows = _DENSE_CACHE
    if not rows or now > _DENSE_CACHE_EXPIRY:
        with get_db() as conn:
            c = conn.cursor()
            c.execute("""
                SELECT ke.knowledge_id, ke.chunk_text, ke.embedding, k.topic, k.lesson, k.code_example, k.tags
                FROM knowledge_embeddings ke
                JOIN knowledge k ON k.id = ke.knowledge_id
            """)
            fetched = c.fetchall()
            _DENSE_CACHE = [dict(r) for r in fetched]
            _DENSE_CACHE_EXPIRY = now + 60.0
            rows = _DENSE_CACHE

    if not rows:
        return []

    query_vec = generate_embedding(query)
    scored: list[tuple[float, dict[str, Any]]] = []
    seen_ids: set[int] = set()
    for r in rows:
        kid = r["knowledge_id"]
        sim = cosine_similarity(query_vec, r["embedding"])
        if kid not in seen_ids or any(item["id"] == kid and sim > s for s, item in scored):
            seen_ids.add(kid)
            scored.append((sim, {
                "id": kid,
                "topic": r["topic"],
                "lesson": r["lesson"],
                "code_example": r["code_example"],
                "tags": r["tags"],
                "similarity": round(sim, 4),
            }))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:limit]]


def search_knowledge_hybrid(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """
    Hybrid Reciprocal Rank Fusion (RRF) search:
    Combines sparse lexical BM25 ranking (knowledge_fts) and dense semantic vector search (knowledge_embeddings).
    RRF score = 1/(60 + rank_bm25) + 1/(60 + rank_vector).
    """
    if not query or not query.strip():
        return []

    sparse_results = search_knowledge(query, limit=limit * 2)
    dense_results = search_knowledge_dense(query, limit=limit * 2)

    if not dense_results:
        return sparse_results[:limit]
    if not sparse_results:
        return dense_results[:limit]

    rrf_scores: dict[int, float] = {}
    items: dict[int, dict[str, Any]] = {}

    for rank, item in enumerate(sparse_results):
        kid = item["id"]
        rrf_scores[kid] = rrf_scores.get(kid, 0.0) + 1.0 / (60.0 + rank)
        items[kid] = item

    for rank, item in enumerate(dense_results):
        kid = item["id"]
        rrf_scores[kid] = rrf_scores.get(kid, 0.0) + 1.0 / (60.0 + rank)
        if kid not in items:
            items[kid] = item

    sorted_kids = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
    return [items[k] for k in sorted_kids[:limit]]


# --- Calendar Scheduling Management ---

def book_calendar_event(
    title: str,
    start_time: str,
    end_time: str,
    attendee_name: str,
    attendee_phone: str = "",
    attendee_email: str = "",
    description: str = "",
) -> dict[str, Any]:
    """
    Books an appointment on the calendar.
    Validates timestamps and enforces double-booking prevention against active confirmed events.
    """
    clean_title = (title or "Appointment").strip()
    clean_start = (start_time or "").strip()
    clean_end = (end_time or "").strip()
    clean_name = (attendee_name or "Guest").strip()

    if not clean_start or not clean_end:
        return {"status": "error", "error": "Both start_time and end_time are required."}

    with get_db() as conn:
        c = conn.cursor()
        c.execute("""
            SELECT id, title, start_time, end_time, attendee_name
            FROM calendar_events
            WHERE status = 'confirmed'
              AND NOT (end_time <= ? OR start_time >= ?)
        """, (clean_start, clean_end))
        conflicts = [dict(r) for r in c.fetchall()]
        if conflicts:
            return {
                "status": "conflict",
                "error": f"Requested slot conflicts with an existing appointment: '{conflicts[0]['title']}' ({conflicts[0]['start_time']} to {conflicts[0]['end_time']})",
                "conflicts": conflicts,
            }

        c.execute("""
            INSERT INTO calendar_events (title, description, start_time, end_time, attendee_name, attendee_phone, attendee_email, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'confirmed')
        """, (clean_title, description.strip(), clean_start, clean_end, clean_name, (attendee_phone or "").strip(), (attendee_email or "").strip()))
        event_id = int(c.lastrowid or 0)
        c.execute(
            "INSERT INTO audit_log (event_type, details, status) VALUES (?, ?, ?)",
            ("calendar_booked", f"Booked appointment ID {event_id}: '{clean_title}' for {clean_name} ({clean_start} - {clean_end})", "success")
        )

        return {
            "status": "confirmed",
            "event_id": event_id,
            "title": clean_title,
            "start_time": clean_start,
            "end_time": clean_end,
            "attendee_name": clean_name,
            "attendee_phone": attendee_phone,
            "attendee_email": attendee_email,
            "description": description,
        }


def check_calendar_availability(start_time: str, end_time: str) -> dict[str, Any]:
    """Checks whether the specified time window has any conflicting confirmed appointments."""
    clean_start = (start_time or "").strip()
    clean_end = (end_time or "").strip()
    if not clean_start or not clean_end:
        return {"available": False, "error": "Invalid start or end time specified."}

    with get_db() as conn:
        c = conn.cursor()
        c.execute("""
            SELECT id, title, start_time, end_time, attendee_name
            FROM calendar_events
            WHERE status = 'confirmed'
              AND NOT (end_time <= ? OR start_time >= ?)
        """, (clean_start, clean_end))
        conflicts = [dict(r) for r in c.fetchall()]
        return {
            "available": len(conflicts) == 0,
            "start_time": clean_start,
            "end_time": clean_end,
            "conflicts": conflicts,
        }


def get_calendar_events(
    start_time: str | None = None,
    end_time: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Retrieves calendar appointments optionally filtered by start / end date."""
    with get_db() as conn:
        c = conn.cursor()
        query = "SELECT * FROM calendar_events WHERE 1=1"
        params: list[Any] = []
        if start_time:
            query += " AND start_time >= ?"
            params.append(start_time.strip())
        if end_time:
            query += " AND start_time <= ?"
            params.append(end_time.strip())
        query += " ORDER BY start_time ASC LIMIT ?"
        params.append(limit)
        c.execute(query, params)
        return [dict(r) for r in c.fetchall()]


def cancel_calendar_event(event_id: int) -> bool:
    """Marks a calendar appointment as cancelled."""
    with get_db() as conn:
        c = conn.cursor()
        c.execute("UPDATE calendar_events SET status = 'cancelled' WHERE id = ?", (event_id,))
        if c.rowcount > 0:
            c.execute(
                "INSERT INTO audit_log (event_type, details, status) VALUES (?, ?, ?)",
                ("calendar_cancelled", f"Cancelled event ID {event_id}", "warning")
            )
            return True
        return False


def delete_calendar_event(event_id: int) -> bool:
    """Deletes an event record from the database completely."""
    with get_db() as conn:
        c = conn.cursor()
        c.execute("DELETE FROM calendar_events WHERE id = ?", (event_id,))
        return c.rowcount > 0


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
        c = conn.cursor()
        c.execute("DELETE FROM gaps WHERE id = ?", (gap_id,))
        return (c.rowcount or 0) > 0


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
        c = conn.cursor()
        c.execute("DELETE FROM todos WHERE id = ?", (todo_id,))
        return (c.rowcount or 0) > 0


def add_note(title: str, content: str, tags: str = "") -> int:
    with get_db() as conn:
        c = conn.cursor()
        c.execute("INSERT INTO notes (title, content, tags) VALUES (?, ?, ?)",
                  (title.strip(), content.strip(), tags.strip()))
        return int(c.lastrowid or 0)


def update_note(note_id: int, title: str, content: str, tags: str = "") -> bool:
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""
            UPDATE notes
            SET title = ?, content = ?, tags = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (title.strip(), content.strip(), tags.strip(), note_id))
        return (c.rowcount or 0) > 0


def get_notes() -> list[dict[str, Any]]:
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT id, title, content, tags, updated_at FROM notes ORDER BY id DESC")
        return [dict(r) for r in c.fetchall()]


def delete_note(note_id: int) -> bool:
    with get_db() as conn:
        c = conn.cursor()
        c.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        return (c.rowcount or 0) > 0


def search_notes(query: str, limit: int = 50) -> list[dict[str, Any]]:
    """Searches notes by title, content, or tags."""
    if not query or not query.strip():
        return get_notes()
    clean_q = f"%{query.strip()}%"
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""
            SELECT id, title, content, tags, updated_at
            FROM notes
            WHERE title LIKE ? OR content LIKE ? OR tags LIKE ?
            ORDER BY id DESC
            LIMIT ?
        """, (clean_q, clean_q, clean_q, limit))
        return [dict(r) for r in c.fetchall()]


def checkpoint_and_vacuum() -> dict[str, Any]:
    """Runs SQLite WAL checkpoint, token cleanup, and optimization to keep file sizes and lookup latencies optimal."""
    with get_db() as conn:
        conn.execute("DELETE FROM user_tokens WHERE expires_at IS NOT NULL AND expires_at < CURRENT_TIMESTAMP;")
    # PRAGMAs wal_checkpoint and optimize must execute outside an active write transaction
    conn = get_connection()
    try:
        conn.execute("PRAGMA wal_checkpoint(PASSIVE);")
        conn.execute("PRAGMA optimize;")
    except sqlite3.OperationalError:
        pass
    log_event("db_maintenance", "Completed SQLite WAL checkpoint, token cleanup, and query planner optimization", "info")
    return {"status": "ok", "message": "Database optimized and WAL checkpointed successfully."}


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

    # GPU Diagnostics
    stats["gpu"] = "CPU / Host RAM"
    if os.path.exists("/dev/dri"):
        render_nodes = [f for f in os.listdir("/dev/dri") if f.startswith("renderD")]
        if render_nodes:
            stats["gpu"] = f"Hardware DRI ({', '.join(render_nodes)})"
    if stats["gpu"].startswith("CPU") and shutil.which("nvidia-smi"):
        try:
            import subprocess
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                text=True, timeout=1.5
            ).strip()
            if out:
                stats["gpu"] = f"NVIDIA {out.splitlines()[0]}"
        except Exception:
            pass

    return stats


init_db()
