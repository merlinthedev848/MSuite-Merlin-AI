import os
import sys
import json
import re
import ast
import urllib.request
import urllib.error
import threading
from memory import (
    init_db,
    get_unresolved_gaps,
    set_gap_status,
    add_knowledge,
    log_event
)

DATASET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_dataset.jsonl")

# Concurrency & Sync State Tracker
_sync_lock = threading.Lock()
sync_state = {
    "is_syncing": False,
    "progress": 0,
    "total": 0,
    "current_task": "",
    "last_result": "idle"
}

DEFAULT_SYSTEM_PROMPT = """You are a senior systems architect and AI compiler specialist mentoring an offline assistant.
The assistant runs on a Raspberry Pi / Linux server.
Provide your answer STRICTLY as a single JSON object with these keys:
- "topic": (string) Specific descriptive title
- "lesson": (string) Concrete, high-yield explanation of the solution, architecture, and principles
- "code_example": (string) Complete, production-grade, well-commented code
- "tags": (string) Comma-separated tags (e.g. "python,networking,linux")
Output ONLY the JSON object. Do not add conversational text or unescaped characters.
"""

def extract_json_from_response(raw_text):
    """
    Robust JSON extractor that handles markdown code fences, conversational preamble,
    and trailing commentary.
    """
    cleaned = raw_text.strip()

    # Try direct parse
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Strip code block fences
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except Exception:
            pass

    # Greedy search for the outer-most balanced curly braces
    brace_match = re.search(r"(\{[\s\S]*\})", cleaned)
    if brace_match:
        try:
            return json.loads(brace_match.group(1).strip())
        except Exception:
            pass

    raise ValueError(f"Could not parse valid JSON from model response: {cleaned[:200]}...")

def validate_code_snippet(code, tags=""):
    """
    Validates Python syntax using the Abstract Syntax Tree (ast.parse)
    without executing the code.
    """
    if not code or not code.strip():
        return True, "No code provided"

    is_python = "python" in tags.lower() or "def " in code or "import " in code or "class " in code
    if is_python:
        try:
            ast.parse(code)
            return True, "AST syntax valid"
        except SyntaxError as se:
            return False, f"SyntaxError line {se.lineno}: {se.msg}"
        except Exception as e:
            return False, f"AST parse error: {e}"

    return True, "Non-Python or skipped"

def query_commercial_ai(prompt, api_key, endpoint="https://api.openai.com/v1/chat/completions", model="gpt-4o-mini"):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "Aether-Self-Learner/1.0"
    }

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }).encode("utf-8")

    req = urllib.request.Request(endpoint, data=payload, headers=headers)
    with urllib.request.urlopen(req, timeout=90) as resp:
        res_json = json.loads(resp.read().decode("utf-8"))
        raw_content = res_json["choices"][0]["message"]["content"]
        return extract_json_from_response(raw_content)

def run_sync(api_key, endpoint=None, model=None, topic_to_explore=None):
    """
    Thread-safe synchronization pipeline with gap state transitions.
    """
    global sync_state
    if not _sync_lock.acquire(blocking=False):
        log_event("sync_aborted", "Sync already in progress; skipped concurrent trigger", "warning")
        return {"status": "error", "message": "Synchronization is already running."}

    init_db()
    endpoint = endpoint or os.getenv("TEACHER_ENDPOINT", "https://api.openai.com/v1/chat/completions")
    model = model or os.getenv("TEACHER_MODEL", "gpt-4o-mini")

    try:
        sync_state["is_syncing"] = True
        sync_state["last_result"] = "running"
        log_event("sync_started", f"Starting teacher sync with model: {model}", "info")

        # 1. Custom Topic Discovery
        if topic_to_explore and topic_to_explore.strip():
            sync_state["current_task"] = f"Exploring topic: {topic_to_explore}"
            prompt = f"Teach me how to master: '{topic_to_explore}'. Provide a deep, production-ready lesson and complete working code."
            try:
                result = query_commercial_ai(prompt, api_key, endpoint, model)
                valid, msg = validate_code_snippet(result.get("code_example", ""), result.get("tags", ""))
                if valid:
                    add_knowledge(result["topic"], result["lesson"], result.get("code_example", ""), result.get("tags", ""), source=f"teacher:{model}")
                    with open(DATASET_FILE, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"prompt": topic_to_explore, "response": result.get("code_example", "")}) + "\n")
                    log_event("sync_topic_success", f"Learned topic: {result['topic']}", "success")
                else:
                    log_event("sync_code_invalid", f"Rejected topic '{topic_to_explore}': {msg}", "danger")
            except Exception as e:
                log_event("sync_topic_failed", f"Failed exploring '{topic_to_explore}': {e}", "danger")

        # 2. Process Unresolved Gaps
        gaps = get_unresolved_gaps()
        sync_state["total"] = len(gaps)
        sync_state["progress"] = 0

        for idx, gap in enumerate(gaps):
            sync_state["progress"] = idx + 1
            sync_state["current_task"] = f"Resolving gap #{gap['id']}: {gap['question'][:30]}..."
            set_gap_status(gap["id"], "processing")

            prompt = (
                f"The local offline AI struggled with the following task/question:\n"
                f"Question: {gap['question']}\n"
                f"Context: {gap.get('context', '')}\n"
                f"Explain how to solve this properly, provide robust code, and key principles to remember."
            )

            try:
                result = query_commercial_ai(prompt, api_key, endpoint, model)
                code = result.get("code_example", "")
                tags = result.get("tags", "")
                valid, msg = validate_code_snippet(code, tags)

                if valid:
                    add_knowledge(result["topic"], result["lesson"], code, tags, source=f"teacher_gap:{gap['id']}")
                    set_gap_status(gap["id"], "resolved")
                    with open(DATASET_FILE, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"prompt": gap["question"], "response": code}) + "\n")
                    log_event("gap_resolved", f"Gap #{gap['id']} resolved as '{result['topic']}'", "success")
                else:
                    set_gap_status(gap["id"], "failed", error_msg=msg)
                    log_event("gap_failed_validation", f"Gap #{gap['id']} code validation failed: {msg}", "warning")

            except Exception as e:
                set_gap_status(gap["id"], "failed", error_msg=str(e))
                log_event("gap_error", f"Error querying teacher for gap #{gap['id']}: {e}", "danger")

        sync_state["last_result"] = "completed"
        log_event("sync_finished", "Teacher synchronization completed successfully", "success")
        return {"status": "success", "message": "Teacher synchronization completed."}

    finally:
        sync_state["is_syncing"] = False
        sync_state["current_task"] = ""
        _sync_lock.release()

if __name__ == "__main__":
    key = os.getenv("TEACHER_API_KEY")
    if not key and len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        key = sys.argv[1]
    if not key:
        key = input("Enter Commercial AI API Key (OpenAI / OpenRouter / Gemini): ").strip()
    
    if not key:
        print("[!] Error: API key required to connect with commercial AI.")
        sys.exit(1)

    custom_topic = None
    if "--topic" in sys.argv:
        idx = sys.argv.index("--topic")
        if idx + 1 < len(sys.argv):
            custom_topic = sys.argv[idx + 1]

    res = run_sync(key, topic_to_explore=custom_topic)
    print(res)
