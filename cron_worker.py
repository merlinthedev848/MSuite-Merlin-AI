#!/usr/bin/env python3
import os
import sys
import time
import socket
import urllib.request
from memory import (
    init_db,
    get_system_telemetry,
    log_event,
    get_unresolved_gaps,
    get_db
)
from sync_teacher import run_sync

LOCK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cron.lock")

class ProcessLock:
    """
    Cross-platform file lock mechanism to prevent overlapping cron worker runs.
    Uses fcntl on Unix/Linux/Pi and msvcrt on Windows.
    """
    def __init__(self, lockfile):
        self.lockfile = lockfile
        self.handle = None

    def acquire(self):
        try:
            self.handle = open(self.lockfile, "w")
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.handle.write(str(os.getpid()))
            self.handle.flush()
            return True
        except (IOError, OSError):
            return False

    def release(self):
        try:
            if self.handle:
                if os.name == "nt":
                    import msvcrt
                    try:
                        msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
                    except Exception:
                        pass
                else:
                    import fcntl
                    try:
                        fcntl.flock(self.handle, fcntl.LOCK_UN)
                    except Exception:
                        pass
                self.handle.close()
                if os.path.exists(self.lockfile):
                    os.remove(self.lockfile)
        except Exception:
            pass

def check_internet_connectivity(host="8.8.8.8", port=53, timeout=3):
    """Verifies whether the host currently has outbound internet connectivity."""
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error:
        return False

def purge_stale_history(retention_days=30):
    """Cleans up old chat records to keep database footprint minimal."""
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""
            DELETE FROM chat_history 
            WHERE created_at < datetime('now', '-' || ? || ' days')
        """, (retention_days,))
        deleted = c.rowcount
        if deleted > 0:
            log_event("cron_cleanup", f"Purged {deleted} chat records older than {retention_days} days", "info")

def run_cron_cycle():
    start_time = time.time()
    init_db()

    # Step 1: Concurrency Lock Acquisition
    lock = ProcessLock(LOCK_FILE)
    if not lock.acquire():
        print("[!] Cron overlapping detected: Another worker is already executing. Exiting cleanly.")
        sys.exit(0)

    try:
        log_event("cron_start", f"Cron worker started (PID: {os.getpid()})", "info")

        # Step 2: System Health & Telemetry Audit
        telemetry = get_system_telemetry()
        log_event("telemetry_heartbeat", f"Temp: {telemetry['temperature']} | Mem: {telemetry['memory']} | Disk: {telemetry['disk']}", "info")

        # Step 3: Database Housekeeping
        purge_stale_history(retention_days=14)

        # Step 4: Connectivity & Autonomous Distillation Check
        is_online = check_internet_connectivity()
        api_key = os.getenv("TEACHER_API_KEY")

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
        # Step 5: Teardown & Lock Release
        lock.release()

if __name__ == "__main__":
    run_cron_cycle()
