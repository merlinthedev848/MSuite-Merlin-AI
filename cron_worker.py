#!/usr/bin/env python3
import os
import socket
import sys
import time

from database import (
    get_db,
    get_system_telemetry,
    get_unresolved_gaps,
    init_db,
    log_event,
)
from sync_teacher import run_sync

LOCK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cron.lock")


class ProcessLock:
    """
    Cross-platform file lock mechanism to prevent overlapping cron worker runs.
    Uses fcntl on Unix/Linux/Pi and msvcrt on Windows.
    """
    def __init__(self, lockfile: str):
        self.lockfile = lockfile
        self.handle = None

    def acquire(self) -> bool:
        try:
            self.handle = open(self.lockfile, "w", encoding="utf-8")
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.handle.write(str(os.getpid()))
            self.handle.flush()
            return True
        except OSError:
            if self.handle is not None:
                try:
                    self.handle.close()
                except OSError:
                    pass
                self.handle = None
            return False

    def release(self) -> None:
        try:
            if self.handle:
                if os.name == "nt":
                    import msvcrt
                    try:
                        msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
                    except OSError:
                        pass
                else:
                    import fcntl
                    try:
                        fcntl.flock(self.handle, fcntl.LOCK_UN)
                    except OSError:
                        pass
                self.handle.close()
                self.handle = None
                if os.path.exists(self.lockfile):
                    os.remove(self.lockfile)
        except OSError:
            pass


def check_internet_connectivity(host: str = "8.8.8.8", port: int = 53, timeout: float = 3.0) -> bool:
    """Verifies outbound internet connectivity without mutating global socket.setdefaulttimeout or leaking sockets."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def purge_stale_history(retention_days: int = 14) -> None:
    """
    Cleans up old chat records to keep database footprint minimal.
    Cascades across messages, chat_history, and removes orphaned empty sessions.
    """
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""
            DELETE FROM messages
            WHERE created_at < datetime('now', '-' || ? || ' days')
        """, (retention_days,))
        deleted_msgs = c.rowcount

        c.execute("""
            DELETE FROM chat_history
            WHERE created_at < datetime('now', '-' || ? || ' days')
        """, (retention_days,))

        c.execute("""
            DELETE FROM sessions
            WHERE pinned = 0
              AND id NOT IN (SELECT DISTINCT session_id FROM messages WHERE session_id IS NOT NULL)
              AND id NOT IN (SELECT DISTINCT conversation_id FROM messages WHERE conversation_id IS NOT NULL)
        """)
        deleted_sessions = c.rowcount

        c.execute("""
            DELETE FROM conversations
            WHERE pinned = 0
              AND id NOT IN (SELECT DISTINCT session_id FROM messages WHERE session_id IS NOT NULL)
              AND id NOT IN (SELECT DISTINCT conversation_id FROM messages WHERE conversation_id IS NOT NULL)
        """)

        c.execute("PRAGMA optimize;")

        if deleted_msgs > 0 or deleted_sessions > 0:
            log_event(
                "cron_cleanup",
                f"Purged {deleted_msgs} stale messages and {deleted_sessions} empty unpinned sessions (older than {retention_days} days)",
                "info",
            )


def run_cron_cycle() -> None:
    start_time = time.time()
    init_db()

    lock = ProcessLock(LOCK_FILE)
    if not lock.acquire():
        print("[!] Cron overlapping detected: Another worker is already executing. Exiting cleanly.")
        sys.exit(0)

    try:
        log_event("cron_start", f"Cron worker started (PID: {os.getpid()})", "info")

        telemetry = get_system_telemetry()
        log_event("telemetry_heartbeat", f"Temp: {telemetry['temperature']} | Mem: {telemetry['memory']} | Disk: {telemetry['disk']}", "info")

        purge_stale_history(retention_days=14)

        is_online = check_internet_connectivity()
        api_key = (
            os.getenv("TEACHER_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("GROQ_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )

        if is_online:
            if api_key:
                unresolved = get_unresolved_gaps()
                if unresolved:
                    print(f"[*] Online and found {len(unresolved)} gaps. Running autonomous distillation...")
                    run_sync(api_key)
                else:
                    print("[*] Online, but no gaps pending resolution.")
            else:
                print("[*] Online, but TEACHER_API_KEY environment variable not configured.")
        else:
            print("[*] Device is completely offline. Skipping external teacher sync.")

        elapsed = time.time() - start_time
        log_event("cron_teardown", f"Cron cycle finished in {elapsed:.2f}s", "success")
        print(f"[OK] Cron cycle finished cleanly in {elapsed:.2f}s.")

    except Exception as e:
        log_event("cron_error", f"Unhandled exception in cron worker: {e}", "danger")
        print(f"[!] Cron execution failure: {e}", file=sys.stderr)
        sys.exit(1)

    finally:
        lock.release()


if __name__ == "__main__":
    run_cron_cycle()
