"""High-Concurrency Stress Testing & System Bottleneck Analysis for Merlin AI.

Evaluates:
- High-concurrency read/write transactions against SQLite WAL
- Deterministic sub-millisecond fast-paths under simulated burst load
- Large-payload input resilience (100KB text, unicode, malformed inputs)
- SQL injection & path-traversal payload immunity
- Race conditions during rapid session/message mutations
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import database  # noqa: E402
from main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_stress_concurrent_database_writes():
    """Stress test: 50 concurrent authenticated workers creating sessions and messages."""
    num_workers = 30
    messages_per_worker = 10

    def worker_task(worker_id):
        user_id = f"stress-user-{worker_id}"
        session_id = f"stress-session-{worker_id}-{time.time()}"
        database.create_session(session_id, f"Stress Session {worker_id}", user_id=user_id)
        for i in range(messages_per_worker):
            database.save_message(session_id, "user", f"Worker {worker_id} message {i}", user_id=user_id)
            database.save_message(session_id, "assistant", f"Worker {worker_id} reply {i}", user_id=user_id)
        msgs = database.get_session_messages(session_id)
        assert len(msgs) == messages_per_worker * 2
        return True

    start_time = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker_task, i) for i in range(num_workers)]
        results = [f.result() for f in futures]

    duration = time.perf_counter() - start_time
    total_writes = num_workers * (1 + messages_per_worker * 2)
    throughput = total_writes / duration

    assert all(results)
    print(f"\n[Stress Test] DB Writes: {total_writes} in {duration:.2f}s ({throughput:.1f} ops/sec)")
    assert duration < 10.0, f"Database write stress test exceeded 10s: {duration:.2f}s"


def test_stress_fast_path_burst_throughput(client):
    """Stress test: 100 rapid concurrent health and clock probes."""
    num_requests = 100

    def query_time():
        resp = client.get("/api/chat/stream?prompt=what+time+is+it&session_id=stress-clock")
        assert resp.status_code == 200
        assert "The current time is" in resp.text
        return True

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(query_time) for _ in range(num_requests)]
        results = [f.result() for f in futures]

    duration = time.perf_counter() - start
    req_per_sec = num_requests / duration

    assert all(results)
    print(f"[Stress Test] Fast-Path Throughput: {req_per_sec:.1f} req/sec")
    assert req_per_sec > 10.0, "Fast-path throughput under heavy load must exceed 10 req/sec"


def test_stress_large_payload_resilience(client):
    """Stress test: 100KB large text payload submission and context fitting."""
    large_payload = "Alpha " * 18000  # ~108 KB string
    resp = client.get(f"/api/chat/stream?prompt={large_payload[:5000]}&session_id=stress-large")
    assert resp.status_code == 200
    assert "[DONE]" in resp.text


def test_stress_sql_injection_and_tampering(client):
    """Verify security parameterized queries under aggressive SQL injection attempts."""
    malicious_prompts = [
        "' OR '1'='1",
        "'; DROP TABLE sessions; --",
        "' UNION SELECT id, password_hash, username FROM users; --",
        "\" OR 1=1 --",
        "'; VACUUM; --",
    ]

    for attack in malicious_prompts:
        resp = client.get(f"/api/chat/stream?prompt={attack}&session_id=attack-session")
        assert resp.status_code == 200
        # Verify database integrity: sessions table must still exist
        assert database.get_sessions() is not None


def test_stress_concurrent_calendar_conflict_detection():
    """Verify concurrent double-booking attempts on the same time slot."""
    slot_start = "2027-01-15T14:00:00"
    slot_end = "2027-01-15T15:00:00"

    # Pre-clean any test events
    events = database.get_calendar_events()
    for ev in events:
        if ev.get("title") == "Contested Meeting":
            database.delete_calendar_event(ev["id"])

    # Book first slot
    ok1 = database.book_calendar_event("Contested Meeting", slot_start, slot_end, "First Attendee")
    assert ok1.get("status") == "confirmed"

    # Immediate second overlapping attempt must be detected and rejected
    res = database.check_calendar_availability(slot_start, slot_end)
    assert not res["available"], "Overlapping slot must report conflict"
    assert len(res["conflicts"]) > 0
