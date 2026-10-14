import os
import sqlite3
import datetime
import platform

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assistant_memory.db")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    # Learned knowledge base (populated via teacher sync or manual lessons)
    c.execute("""
    CREATE TABLE IF NOT EXISTS knowledge (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        topic TEXT NOT NULL,
        lesson TEXT NOT NULL,
        code_example TEXT,
        tags TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    # Knowledge gaps / questions to ask the teacher
    c.execute("""
    CREATE TABLE IF NOT EXISTS gaps (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT NOT NULL,
        context TEXT,
        resolved INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    # Personal Assistant: To-Do items
    c.execute("""
    CREATE TABLE IF NOT EXISTS todos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task TEXT NOT NULL,
        completed INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    # Personal Assistant: Notes
    c.execute("""
    CREATE TABLE IF NOT EXISTS notes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    # Conversation History
    c.execute("""
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()

# --- Knowledge & Self-Learning ---

def add_knowledge(topic, lesson, code_example="", tags=""):
    conn = get_db()
    c = conn.cursor()
    c.execute("INSERT INTO knowledge (topic, lesson, code_example, tags) VALUES (?, ?, ?, ?)",
              (topic, lesson, code_example, tags))
    conn.commit()
    row_id = c.lastrowid
    conn.close()
    return row_id

def search_knowledge(query, limit=3):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT topic, lesson, code_example FROM knowledge ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()

    if not rows or not query:
        return []

    words = set(query.lower().replace(",", " ").replace(".", " ").split())
    scored = []
    for r in rows:
        text = f"{r['topic']} {r['lesson']} {r['code_example'] or ''}".lower()
        score = sum(1 for w in words if len(w) > 2 and w in text)
        if score > 0:
            scored.append((score, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:limit]]

def get_all_knowledge():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT id, topic, lesson, code_example, tags, created_at FROM knowledge ORDER BY id DESC")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows

def add_gap(question, context=""):
    conn = get_db()
    c = conn.cursor()
    c.execute("INSERT INTO gaps (question, context, resolved) VALUES (?, ?, 0)", (question, context))
    conn.commit()
    row_id = c.lastrowid
    conn.close()
    return row_id

def get_unresolved_gaps():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT id, question, context, created_at FROM gaps WHERE resolved = 0 ORDER BY id ASC")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows

def resolve_gap(gap_id):
    conn = get_db()
    c = conn.cursor()
    c.execute("UPDATE gaps SET resolved = 1 WHERE id = ?", (gap_id,))
    conn.commit()
    conn.close()

# --- Personal Assistant (Todos & Notes) ---

def add_todo(task):
    conn = get_db()
    c = conn.cursor()
    c.execute("INSERT INTO todos (task, completed) VALUES (?, 0)", (task,))
    conn.commit()
    row_id = c.lastrowid
    conn.close()
    return row_id

def get_todos():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT id, task, completed, created_at FROM todos ORDER BY completed ASC, id DESC")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows

def toggle_todo(todo_id):
    conn = get_db()
    c = conn.cursor()
    c.execute("UPDATE todos SET completed = 1 - completed WHERE id = ?", (todo_id,))
    conn.commit()
    conn.close()

def delete_todo(todo_id):
    conn = get_db()
    c = conn.cursor()
    c.execute("DELETE FROM todos WHERE id = ?", (todo_id,))
    conn.commit()
    conn.close()

def add_note(title, content):
    conn = get_db()
    c = conn.cursor()
    c.execute("INSERT INTO notes (title, content) VALUES (?, ?)", (title, content))
    conn.commit()
    row_id = c.lastrowid
    conn.close()
    return row_id

def get_notes():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT id, title, content, updated_at FROM notes ORDER BY id DESC")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows

def delete_note(note_id):
    conn = get_db()
    c = conn.cursor()
    c.execute("DELETE FROM notes WHERE id = ?", (note_id,))
    conn.commit()
    conn.close()

# --- Chat History ---

def save_chat_message(session_id, role, content):
    conn = get_db()
    c = conn.cursor()
    c.execute("INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
              (session_id, role, content))
    conn.commit()
    conn.close()

def get_recent_chat(session_id, limit=8):
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        SELECT role, content FROM (
            SELECT id, role, content FROM chat_history 
            WHERE session_id = ? 
            ORDER BY id DESC LIMIT ?
        ) ORDER BY id ASC
    """, (session_id, limit))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows

# --- System Diagnostics (Pi 5 & Linux) ---

def get_system_telemetry():
    stats = {
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "temperature": "N/A",
        "cpu_usage": "N/A",
        "memory": "N/A",
        "disk": "N/A"
    }

    # Raspberry Pi / Linux CPU Temperature
    temp_path = "/sys/class/thermal/thermal_zone0/temp"
    if os.path.exists(temp_path):
        try:
            with open(temp_path, "r") as f:
                temp_c = int(f.read().strip()) / 1000.0
                stats["temperature"] = f"{temp_c:.1f}°C"
        except Exception:
            pass

    # Linux /proc/loadavg
    if os.path.exists("/proc/loadavg"):
        try:
            with open("/proc/loadavg", "r") as f:
                loads = f.read().split()[:3]
                stats["cpu_usage"] = f"Load: {', '.join(loads)}"
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
                        key = parts[0].strip()
                        val = parts[1].strip().split()[0]
                        mem[key] = int(val)
            total_mb = mem.get("MemTotal", 0) // 1024
            avail_mb = mem.get("MemAvailable", 0) // 1024
            used_mb = total_mb - avail_mb
            stats["memory"] = f"{used_mb}MB / {total_mb}MB ({int((used_mb/max(total_mb,1))*100)}%)"
        except Exception:
            pass

    # Disk usage
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        stats["disk"] = f"{used // (2**30)}GB / {total // (2**30)}GB ({int(used/max(total,1)*100)}%)"
    except Exception:
        pass

    return stats
