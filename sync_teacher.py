import os
import sys
import json
import urllib.request
import urllib.error
from memory import init_db, get_unresolved_gaps, resolve_gap, add_knowledge

DATASET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_dataset.jsonl")

DEFAULT_SYSTEM_PROMPT = """You are a master software engineering mentor and AI architect.
A junior offline AI running on a Raspberry Pi / Linux server is consulting you to improve its capabilities.
You must return ONLY a valid JSON object with the following fields:
- "topic": (string) Short descriptive title of the topic or concept
- "lesson": (string) Clear, direct, high-yield explanation of the solution, theory, or architectural pattern
- "code_example": (string) Clean, production-grade, working code sample with comments
- "tags": (string) Comma-separated tags (e.g. "python,networking,linux")
Do not include markdown code fences (like ```json), just the raw JSON object.
"""

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
        "temperature": 0.2,
        "response_format": {"type": "json_object"} if "openai" in endpoint or "gpt" in model else None
    }).encode("utf-8")

    req = urllib.request.Request(endpoint, data=payload, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            raw_text = data["choices"][0]["message"]["content"].strip()
            # Clean possible markdown wrapping if returned
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            return json.loads(raw_text.strip())
    except Exception as e:
        print(f"[!] Error contacting commercial AI: {e}")
        return None

def test_code_snippet(code, tags=""):
    """Validates python code syntax before allowing it to be recorded into memory."""
    if "python" in tags.lower() or "def " in code or "import " in code:
        try:
            compile(code, "<distilled_test>", "exec")
            return True, "Syntax valid"
        except SyntaxError as se:
            return False, f"SyntaxError: {se}"
    return True, "Non-Python or skipped"

def run_sync(api_key, endpoint=None, model=None, topic_to_explore=None):
    init_db()
    endpoint = endpoint or os.getenv("TEACHER_ENDPOINT", "https://api.openai.com/v1/chat/completions")
    model = model or os.getenv("TEACHER_MODEL", "gpt-4o-mini")

    gaps = get_unresolved_gaps()
    
    if not gaps and not topic_to_explore:
        print("[*] No outstanding gaps recorded in memory.")
        choice = input("Would you like the assistant to autonomously learn a new advanced coding topic? (y/N): ").strip().lower()
        if choice == 'y':
            topic_to_explore = input("Enter a topic or domain to master (e.g., 'Async socket server in Python', 'Optimized sqlite indexing'): ").strip()
        else:
            print("[*] Nothing to do. Local AI is synchronized.")
            return

    # Process explicit topic exploration if requested
    if topic_to_explore:
        print(f"\n[*] Interviewing teacher on topic: '{topic_to_explore}'...")
        prompt = f"Teach me how to master: '{topic_to_explore}'. Provide a deep, production-ready lesson and complete working code."
        result = query_commercial_ai(prompt, api_key, endpoint, model)
        if result and "topic" in result and "lesson" in result:
            code = result.get("code_example", "")
            tags = result.get("tags", "")
            valid, msg = test_code_snippet(code, tags)
            if valid:
                add_knowledge(result["topic"], result["lesson"], code, tags)
                with open(DATASET_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"prompt": topic_to_explore, "response": code}) + "\n")
                print(f"[✓] Successfully assimilated skill: '{result['topic']}' into memory!")
            else:
                print(f"[!] Code failed verification: {msg}")

    # Process all unresolved gaps accumulated during offline sessions
    for gap in gaps:
        print(f"\n[*] Consulting teacher regarding recorded gap #{gap['id']}: '{gap['question']}'")
        prompt = (
            f"The local offline AI struggled with the following task/question:\n"
            f"Question: {gap['question']}\n"
            f"Context: {gap['context']}\n"
            f"Explain how to solve this properly, provide robust code, and key principles to remember."
        )
        result = query_commercial_ai(prompt, api_key, endpoint, model)
        if result and "topic" in result and "lesson" in result:
            code = result.get("code_example", "")
            tags = result.get("tags", "")
            valid, msg = test_code_snippet(code, tags)
            if valid:
                add_knowledge(result["topic"], result["lesson"], code, tags)
                resolve_gap(gap["id"])
                with open(DATASET_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"prompt": gap["question"], "response": code}) + "\n")
                print(f"[✓] Gap #{gap['id']} resolved and assimilated as '{result['topic']}'!")
            else:
                print(f"[!] Teacher's code for gap #{gap['id']} failed syntax check ({msg}). Retaining gap.")

    print("\n[✓] Teacher sync session completed. All learned skills are permanently indexed in SQLite.")

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

    run_sync(key, topic_to_explore=custom_topic)
