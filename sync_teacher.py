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
    "last_result": "idle",
    "last_error": None
}

DEFAULT_SYSTEM_PROMPT = """You are a senior systems architect and AI compiler specialist mentoring Merlin AI, an offline assistant.
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

def clean_code_snippet(code):
    """
    Strips markdown code fences (e.g. ```python ... ```) that models sometimes return inside JSON strings.
    """
    if not code:
        return ""
    cleaned = code.strip()
    match = re.search(r"```(?:python|py)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return cleaned

def validate_code_snippet(code, tags=""):
    """
    Validates Python syntax using the Abstract Syntax Tree (ast.parse)
    without executing the code.
    """
    if not code or not code.strip():
        return True, "No code provided"

    cleaned_code = clean_code_snippet(code)
    is_python = "python" in tags.lower() or "def " in cleaned_code or "import " in cleaned_code or "class " in cleaned_code
    if is_python:
        try:
            ast.parse(cleaned_code)
            return True, "AST syntax valid"
        except SyntaxError as se:
            return False, f"SyntaxError line {se.lineno}: {se.msg}"
        except Exception as e:
            return False, f"AST parse error: {e}"

    return True, "Non-Python or skipped"

def resolve_teacher_provider(api_key, endpoint=None, model=None):
    """
    Auto-detects API provider and default models based on key prefixes or environment variables:
    - AIza... or gemini model -> Google Gemini (FREE Tier: gemini-2.0-flash, gemini-1.5-flash)
    - gsk_... -> Groq (Free high-speed tier: llama-3.3-70b-versatile)
    - sk-or-... -> OpenRouter (Free models: meta-llama/llama-3.3-70b-instruct:free)
    - sk-... -> OpenAI (gpt-4o-mini)
    """
    key = (api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("TEACHER_API_KEY") or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    ep = endpoint.strip() if endpoint else ""
    mdl = model.strip() if model else ""

    if ep and mdl:
        return ep, mdl

    if key.startswith("AIza") or (mdl and mdl.startswith("gemini")) or (ep and "googleapis.com" in ep):
        # Google Gemini OpenAI-compatible or native endpoint
        default_gemini_ep = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        return ep or default_gemini_ep, mdl or "gemini-2.0-flash"
    elif key.startswith("gsk_"):
        # Groq (Free fast tier)
        return ep or "https://api.groq.com/openai/v1/chat/completions", mdl or "llama-3.3-70b-versatile"
    elif key.startswith("sk-or-"):
        # OpenRouter (Free models available)
        return ep or "https://openrouter.ai/api/v1/chat/completions", mdl or "google/gemini-2.0-flash-exp:free"
    else:
        # Default OpenAI or custom
        return ep or os.getenv("TEACHER_ENDPOINT", "https://api.openai.com/v1/chat/completions"), mdl or os.getenv("TEACHER_MODEL", "gpt-4o-mini")

def discover_gemini_models(api_key_clean):
    """
    Dynamically queries Google's ListModels API to discover accessible models.
    Catches 400/403 directly to give the user immediate feedback if the key is invalid or Generative Language API is disabled.
    """
    discovered = []
    working_version = "v1beta"
    for ver in ["v1beta", "v1"]:
        url = f"https://generativelanguage.googleapis.com/{ver}/models?key={api_key_clean}"
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Merlin-AI-Teacher/1.0"}
            )
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                for m in data.get("models", []):
                    methods = m.get("supportedGenerationMethods", [])
                    if "generateContent" in methods:
                        m_name = m.get("name", "").replace("models/", "")
                        if m_name and m_name not in discovered:
                            discovered.append(m_name)
                if discovered:
                    working_version = ver
                    break
        except urllib.error.HTTPError as he:
            err_msg = ""
            try:
                err_data = json.loads(he.read().decode("utf-8", errors="replace"))
                if "error" in err_data:
                    err_msg = err_data["error"].get("message") or str(err_data["error"])
            except Exception:
                pass
            if he.code in (400, 401, 403):
                raise RuntimeError(f"Google Gemini Authentication Error ({he.code}): {err_msg or he.reason}")
        except Exception:
            pass
    return discovered, working_version

def fetch_live_models_for_provider(provider, api_key):
    """
    Queries the provider's live models API and returns a list of verified model IDs.
    Eliminates hardcoded model lists by pulling the exact models authorized on that API key.
    """
    api_key_clean = (api_key or "").strip().strip("'\"")
    models = []

    if provider == "gemini":
        discovered, _ = discover_gemini_models(api_key_clean)
        return discovered or ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]

    elif provider == "groq":
        url = "https://api.groq.com/openai/v1/models"
        headers = {"Authorization": f"Bearer {api_key_clean}", "User-Agent": "Merlin-AI/1.0"}
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                for m in data.get("data", []):
                    m_id = m.get("id")
                    if m_id and not m.get("deleted", False):
                        models.append(m_id)
        except Exception:
            pass
        return models or ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "deepseek-r1-distill-llama-70b"]

    elif provider == "openrouter":
        url = "https://openrouter.ai/api/v1/models"
        headers = {"User-Agent": "Merlin-AI/1.0"}
        if api_key_clean:
            headers["Authorization"] = f"Bearer {api_key_clean}"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                for m in data.get("data", []):
                    m_id = m.get("id", "")
                    if ":free" in m_id:
                        models.append(m_id)
        except Exception:
            pass
        return models or ["google/gemini-2.0-flash-exp:free", "deepseek/deepseek-r1:free", "qwen/qwen-2.5-coder-32b-instruct:free"]

    elif provider == "openai":
        url = "https://api.openai.com/v1/models"
        headers = {"Authorization": f"Bearer {api_key_clean}", "User-Agent": "Merlin-AI/1.0"}
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                for m in data.get("data", []):
                    m_id = m.get("id", "")
                    if "gpt" in m_id and "instruct" not in m_id and "realtime" not in m_id:
                        models.append(m_id)
        except Exception:
            pass
        return models or ["gpt-4o-mini", "gpt-4o"]

    return []

def query_commercial_ai(prompt, api_key, endpoint=None, model="gemini-1.5-flash"):
    api_key_clean = (api_key or os.getenv("TEACHER_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip().strip("'\"")
    if not api_key_clean:
        raise ValueError("Teacher API key is required. Enter your key in the Sync dialog or set TEACHER_API_KEY.")

    if " " in api_key_clean or api_key_clean.startswith(("cd ", "sudo ", "git ", "./", "rm ")):
        raise ValueError("Invalid API key: It looks like a terminal command was pasted into the API key field. Please enter a valid API key.")

    model_clean = (model or "gemini-1.5-flash").strip()
    is_gemini = (
        api_key_clean.startswith("AIza")
        or "gemini" in model_clean.lower()
        or (endpoint and "googleapis.com" in endpoint)
    )

    # Strategy 1: For Google Gemini, query official REST API with dynamic model discovery and automatic fallback
    if is_gemini:
        clean_model = model_clean.replace("models/", "")
        discovered_models, api_ver = discover_gemini_models(api_key_clean)

        candidate_models = []
        if clean_model in discovered_models:
            candidate_models.append(clean_model)
        else:
            for m in discovered_models:
                if clean_model in m:
                    candidate_models.append(m)

        for m in sorted(discovered_models, key=lambda x: (0 if "flash" in x else 1, 0 if "1.5" in x or "2.0" in x else 1)):
            if m not in candidate_models:
                candidate_models.append(m)

        if not candidate_models:
            candidate_models = [clean_model]
            for fb in ["gemini-1.5-flash", "gemini-1.5-flash-latest", "gemini-2.0-flash", "gemini-1.5-pro", "gemini-pro"]:
                if fb not in candidate_models:
                    candidate_models.append(fb)

        last_error = None
        for candidate in candidate_models:
            for ver in [api_ver, "v1beta", "v1"]:
                native_url = f"https://generativelanguage.googleapis.com/{ver}/models/{candidate}:generateContent?key={api_key_clean}"
                native_payload = {
                    "contents": [
                        {"parts": [{"text": prompt}]}
                    ],
                    "generationConfig": {
                        "temperature": 0.2
                    }
                }
                if ver == "v1beta":
                    native_payload["system_instruction"] = {
                        "parts": [{"text": DEFAULT_SYSTEM_PROMPT}]
                    }
                    native_payload["generationConfig"]["responseMimeType"] = "application/json"
                else:
                    native_payload["contents"][0]["parts"][0]["text"] = f"{DEFAULT_SYSTEM_PROMPT}\n\nTask: {prompt}"

                try:
                    req = urllib.request.Request(
                        native_url,
                        data=json.dumps(native_payload).encode("utf-8"),
                        headers={"Content-Type": "application/json", "User-Agent": "Merlin-AI-Teacher/1.0"}
                    )
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        res_json = json.loads(resp.read().decode("utf-8"))
                        candidate_obj = res_json.get("candidates", [{}])[0]
                        content_obj = candidate_obj.get("content", {})
                        parts = content_obj.get("parts", [])
                        raw_content = parts[0].get("text", "") if parts else ""
                        result = extract_json_from_response(raw_content)
                        if "code_example" in result:
                            result["code_example"] = clean_code_snippet(result["code_example"])
                        return result
                except urllib.error.HTTPError as he:
                    err_msg = ""
                    try:
                        err_data = json.loads(he.read().decode("utf-8", errors="replace"))
                        if "error" in err_data:
                            err_msg = err_data["error"].get("message") or str(err_data["error"])
                    except Exception:
                        pass
                    if he.code == 404:
                        last_error = f"{candidate} ({ver}): {err_msg or 'Not found'}"
                        continue
                    if he.code in (400, 401, 403):
                        raise RuntimeError(f"Google Gemini Error ({candidate}): {err_msg or f'HTTP {he.code}: {he.reason}'}")
                    last_error = f"{candidate} ({ver}): HTTP {he.code} {err_msg}"
                except Exception as e:
                    last_error = str(e)
                    continue

        # Strategy 1B: Fallback to Google Gemini OpenAI-compatible completions endpoint
        try:
            openai_gemini_url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
            oa_headers = {
                "Content-Type": "application/json",
                "User-Agent": "Merlin-AI-Teacher/1.0",
                "Authorization": f"Bearer {api_key_clean}"
            }
            oa_payload = {
                "model": clean_model,
                "messages": [
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2
            }
            req_oa = urllib.request.Request(openai_gemini_url, data=json.dumps(oa_payload).encode("utf-8"), headers=oa_headers)
            with urllib.request.urlopen(req_oa, timeout=60) as resp_oa:
                res_json = json.loads(resp_oa.read().decode("utf-8"))
                if "choices" in res_json and res_json["choices"]:
                    raw_content = res_json["choices"][0]["message"]["content"]
                    result = extract_json_from_response(raw_content)
                    if "code_example" in result:
                        result["code_example"] = clean_code_snippet(result["code_example"])
                    return result
        except Exception:
            pass

        raise RuntimeError(f"Google Gemini Model Error: Could not generate content with models {candidate_models[:3]}. ({last_error})")

    # Strategy 2: Standard OpenAI-compatible completion endpoint (Groq, OpenRouter, OpenAI)
    target_url = endpoint or "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Merlin-AI-Teacher/1.0",
        "Authorization": f"Bearer {api_key_clean}"
    }

    if "openrouter.ai" in target_url:
        headers["HTTP-Referer"] = "http://localhost:5000"
        headers["X-Title"] = "Merlin AI Offline Assistant"

    payload_data = {
        "model": model_clean,
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    req = urllib.request.Request(target_url, data=json.dumps(payload_data).encode("utf-8"), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            res_json = json.loads(resp.read().decode("utf-8"))
            if "choices" in res_json and res_json["choices"]:
                raw_content = res_json["choices"][0]["message"]["content"]
            else:
                raw_content = str(res_json)
            result = extract_json_from_response(raw_content)
            if "code_example" in result:
                result["code_example"] = clean_code_snippet(result["code_example"])
            return result
    except urllib.error.HTTPError as he:
        err_msg = ""
        try:
            err_data = json.loads(he.read().decode("utf-8", errors="replace"))
            if "error" in err_data:
                err_msg = err_data["error"].get("message") or str(err_data["error"])
        except Exception:
            pass
        if "unavailable for free" in err_msg.lower() or ("not found" in err_msg.lower() and ":free" in model_clean):
            for fb_model in ["google/gemini-2.0-flash-exp:free", "deepseek/deepseek-r1:free", "qwen/qwen-2.5-coder-32b-instruct:free"]:
                if fb_model != model_clean:
                    payload_data["model"] = fb_model
                    try:
                        req_fb = urllib.request.Request(target_url, data=json.dumps(payload_data).encode("utf-8"), headers=headers)
                        with urllib.request.urlopen(req_fb, timeout=60) as resp_fb:
                            res_json = json.loads(resp_fb.read().decode("utf-8"))
                            raw_content = res_json["choices"][0]["message"]["content"] if "choices" in res_json and res_json["choices"] else str(res_json)
                            result = extract_json_from_response(raw_content)
                            if "code_example" in result:
                                result["code_example"] = clean_code_snippet(result["code_example"])
                            return result
                    except Exception:
                        continue
        if not err_msg:
            err_msg = f"HTTP {he.code}: {he.reason}"
        raise RuntimeError(f"Teacher API Error ({model_clean}): {err_msg}")
    except Exception as e:
        raise RuntimeError(f"Teacher Connection Error ({model_clean}): {e}")

def run_sync(api_key, endpoint=None, model=None, topic_to_explore=None):
    """
    Thread-safe synchronization pipeline with gap state transitions.
    """
    global sync_state
    if not _sync_lock.acquire(blocking=False):
        log_event("sync_aborted", "Sync already in progress; skipped concurrent trigger", "warning")
        return {"status": "error", "message": "Synchronization is already running."}

    init_db()
    endpoint, model = resolve_teacher_provider(api_key, endpoint, model)

    try:
        sync_state["is_syncing"] = True
        sync_state["last_result"] = "running"
        sync_state["last_error"] = None
        log_event("sync_started", f"Starting teacher sync with model: {model} at {endpoint}", "info")

        had_error = False

        # 1. Custom Topic Discovery
        if topic_to_explore and topic_to_explore.strip():
            sync_state["current_task"] = f"Exploring topic: {topic_to_explore}"
            prompt = f"Teach me how to master: '{topic_to_explore}'. Provide a deep, production-ready lesson and complete working code."
            try:
                result = query_commercial_ai(prompt, api_key, endpoint, model)
                code = clean_code_snippet(result.get("code_example", ""))
                valid, msg = validate_code_snippet(code, result.get("tags", ""))
                if valid:
                    add_knowledge(result["topic"], result["lesson"], code, result.get("tags", ""), source=f"teacher:{model}")
                    with open(DATASET_FILE, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"prompt": topic_to_explore, "response": code}) + "\n")
                    log_event("sync_topic_success", f"Learned topic: {result['topic']}", "success")
                else:
                    had_error = True
                    sync_state["last_error"] = f"Validation failed: {msg}"
                    log_event("sync_code_invalid", f"Rejected topic '{topic_to_explore}': {msg}", "danger")
            except Exception as e:
                had_error = True
                sync_state["last_error"] = str(e)
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
                had_error = True
                sync_state["last_error"] = str(e)
                set_gap_status(gap["id"], "failed", error_msg=str(e))
                log_event("gap_error", f"Error querying teacher for gap #{gap['id']}: {e}", "danger")

        if had_error:
            sync_state["last_result"] = "error"
            return {"status": "error", "message": sync_state.get("last_error", "Sync encountered an error.")}
        else:
            sync_state["last_result"] = "completed"
            log_event("sync_finished", "Teacher synchronization completed successfully", "success")
            return {"status": "success", "message": "Teacher synchronization completed."}

    finally:
        sync_state["is_syncing"] = False
        sync_state["current_task"] = ""
        _sync_lock.release()

def query_gemini_multimodal(prompt, base64_data, mime_type, api_key):
    """
    Direct multimodal query to Gemini (v1beta) supporting images and audio.
    """
    api_key_clean = (api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip().strip("'\"")
    if not api_key_clean:
        raise ValueError("Google Gemini API key required for multimodal analysis.")

    models = ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]
    last_err = None

    for m in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={api_key_clean}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": base64_data
                            }
                        },
                        {
                            "text": prompt or "Analyze this file in detail and provide insights, summaries, and code or transcripts where appropriate."
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2
            }
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "User-Agent": "Merlin-AI/1.0"}
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                res_json = json.loads(resp.read().decode("utf-8"))
                candidates = res_json.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", "")
        except urllib.error.HTTPError as he:
            err_msg = ""
            try:
                err_data = json.loads(he.read().decode("utf-8", errors="replace"))
                if "error" in err_data:
                    err_msg = err_data["error"].get("message")
            except Exception:
                pass
            last_err = err_msg or f"HTTP {he.code}: {he.reason}"
        except Exception as e:
            last_err = str(e)

    raise RuntimeError(f"Gemini multimodal analysis failed: {last_err}")

if __name__ == "__main__":
    key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("TEACHER_API_KEY")
        or os.getenv("GROQ_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not key and len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        key = sys.argv[1]
    if not key:
        key = input("Enter Teacher API Key (OpenRouter / Groq / OpenAI): ").strip()
    
    if not key:
        print("[!] Error: API key required to connect with teacher AI.")
        sys.exit(1)

    custom_topic = None
    if "--topic" in sys.argv:
        idx = sys.argv.index("--topic")
        if idx + 1 < len(sys.argv):
            custom_topic = sys.argv[idx + 1]

    res = run_sync(key, topic_to_explore=custom_topic)
    print(res)
