import os
import sqlite3
import datetime
import platform
import shutil
import time

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assistant_memory.db")

def get_db():
    """
    Creates a thread-safe connection to SQLite with WAL mode enabled
    and a 30-second busy timeout to prevent 'database is locked' errors under concurrency.
    """
    conn = sqlite3.connect(DB_PATH, timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    """Initializes tables, indices, and audit logging."""
    with get_db() as conn:
        c = conn.cursor()
        
        # Knowledge Base
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
        c.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_topic ON knowledge(topic);")
        c.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_tags ON knowledge(tags);")

        # Knowledge Gaps (State Machine: pending -> processing -> resolved / failed)
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

        # Schema migration check for existing databases
        c.execute("PRAGMA table_info(gaps);")
        gap_cols = [col[1] for col in c.fetchall()]
        if "status" not in gap_cols:
            c.execute("ALTER TABLE gaps ADD COLUMN status TEXT DEFAULT 'pending';")
        if "attempts" not in gap_cols:
            c.execute("ALTER TABLE gaps ADD COLUMN attempts INTEGER DEFAULT 0;")
        if "last_error" not in gap_cols:
            c.execute("ALTER TABLE gaps ADD COLUMN last_error TEXT DEFAULT '';")
        if "resolved_at" not in gap_cols:
            c.execute("ALTER TABLE gaps ADD COLUMN resolved_at TIMESTAMP;")

        c.execute("CREATE INDEX IF NOT EXISTS idx_gaps_status ON gaps(status);")

        # Personal Assistant: To-Dos
        c.execute("""
        CREATE TABLE IF NOT EXISTS todos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task TEXT NOT NULL,
            completed INTEGER DEFAULT 0,
            priority TEXT DEFAULT 'normal', -- 'low', 'normal', 'high'
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP
        );
        """)

        # Personal Assistant: Notes
        c.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            tags TEXT DEFAULT '',
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # Chat History & Context Windows
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

        # System & Cron Audit Log
        c.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            details TEXT NOT NULL,
            status TEXT DEFAULT 'info',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

# --- Audit & Logging ---

def log_event(event_type, details, status="info"):
    with get_db() as conn:
        conn.execute("INSERT INTO audit_log (event_type, details, status) VALUES (?, ?, ?)",
                     (event_type, str(details), status))

def get_recent_audit_logs(limit=20):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT id, event_type, details, status, created_at FROM audit_log ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(r) for r in c.fetchall()]

# --- Knowledge & Memory Retrieval ---

def add_knowledge(topic, lesson, code_example="", tags="", source="distilled"):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""
            INSERT INTO knowledge (topic, lesson, code_example, tags, source)
            VALUES (?, ?, ?, ?, ?)
        """, (topic.strip(), lesson.strip(), code_example.strip() if code_example else "", tags.strip(), source))
        log_event("knowledge_added", f"Assimilated topic: {topic}", "success")
        return c.lastrowid

def delete_knowledge(knowledge_id):
    with get_db() as conn:
        conn.execute("DELETE FROM knowledge WHERE id = ?", (knowledge_id,))
        log_event("knowledge_deleted", f"Deleted knowledge ID: {knowledge_id}", "info")
        return True

def search_knowledge(query, limit=3):
    """
    Ranks knowledge by token overlap across topic, lesson, tags, and code.
    Thread-safe and handles empty queries gracefully.
    """
    if not query or not query.strip():
        return []

    tokens = [t.lower() for t in query.replace(",", " ").replace(".", " ").split() if len(t) > 2]
    if not tokens:
        return []

    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT id, topic, lesson, code_example, tags FROM knowledge ORDER BY id DESC")
        rows = c.fetchall()

    scored = []
    for r in rows:
        haystack = f"{r['topic']} {r['lesson']} {r['tags']} {r['code_example'] or ''}".lower()
        score = sum(haystack.count(tok) * (3 if tok in r['topic'].lower() else 1) for tok in tokens)
        if score > 0:
            scored.append((score, dict(r)))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:limit]]

def get_all_knowledge(limit=100):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""
            SELECT id, topic, lesson, code_example, tags, source, created_at
            FROM knowledge ORDER BY id DESC LIMIT ?
        """, (limit,))
        return [dict(r) for r in c.fetchall()]

# --- Gaps State Machine ---

def add_gap(question, context=""):
    with get_db() as conn:
        c = conn.cursor()
        # Avoid exact duplicate pending gaps
        c.execute("SELECT id FROM gaps WHERE question = ? AND status = 'pending'", (question.strip(),))
        existing = c.fetchone()
        if existing:
            return existing["id"]

        c.execute("INSERT INTO gaps (question, context, status) VALUES (?, ?, 'pending')",
                  (question.strip(), context.strip()))
        gid = c.lastrowid
        log_event("gap_recorded", f"Recorded gap #{gid}: {question}", "warning")
        return gid

def get_unresolved_gaps(limit=50):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""
            SELECT id, question, context, status, attempts, last_error, created_at
            FROM gaps WHERE status IN ('pending', 'failed')
            ORDER BY id ASC LIMIT ?
        """, (limit,))
        return [dict(r) for r in c.fetchall()]

def set_gap_status(gap_id, status, error_msg=""):
    with get_db() as conn:
        now = datetime.datetime.now().isoformat() if status == 'resolved' else None
        conn.execute("""
            UPDATE gaps 
            SET status = ?, last_error = ?, attempts = attempts + 1, resolved_at = ?
            WHERE id = ?
        """, (status, error_msg, now, gap_id))

def delete_gap(gap_id):
    with get_db() as conn:
        conn.execute("DELETE FROM gaps WHERE id = ?", (gap_id,))
        return True

# --- Personal Assistant: Tasks & Notes ---

def add_todo(task, priority="normal"):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("INSERT INTO todos (task, priority) VALUES (?, ?)", (task.strip(), priority))
        return c.lastrowid

def get_todos():
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT id, task, completed, priority, created_at, completed_at FROM todos ORDER BY completed ASC, id DESC")
        return [dict(r) for r in c.fetchall()]

def toggle_todo(todo_id):
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

def delete_todo(todo_id):
    with get_db() as conn:
        conn.execute("DELETE FROM todos WHERE id = ?", (todo_id,))
        return True

def add_note(title, content, tags=""):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("INSERT INTO notes (title, content, tags) VALUES (?, ?, ?)",
                  (title.strip(), content.strip(), tags.strip()))
        return c.lastrowid

def update_note(note_id, title, content, tags=""):
    with get_db() as conn:
        conn.execute("""
            UPDATE notes 
            SET title = ?, content = ?, tags = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (title.strip(), content.strip(), tags.strip(), note_id))
        return True

def get_notes():
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT id, title, content, tags, updated_at FROM notes ORDER BY id DESC")
        return [dict(r) for r in c.fetchall()]

def delete_note(note_id):
    with get_db() as conn:
        conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        return True

# --- Chat History & State ---

def save_chat_message(session_id, role, content):
    with get_db() as conn:
        conn.execute("INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
                     (session_id, role, content))

def get_recent_chat(session_id, limit=8):
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

def clear_chat_history(session_id=None):
    with get_db() as conn:
        if session_id:
            conn.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
        else:
            conn.execute("DELETE FROM chat_history")

# --- Hardware Diagnostics & Telemetry ---

def get_system_telemetry():
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
        total, used, free = shutil.disk_usage("/")
        total_gb = total // (2**30)
        used_gb = used // (2**30)
        pct = int((used / max(total, 1)) * 100)
        stats["disk"] = f"{used_gb}GB / {total_gb}GB ({pct}%)"
    except Exception:
        pass

    return stats
