import os
import sys
import json
import re
import ast
import threading
from memory import init_db, add_knowledge, log_event

DEFAULT_DATASET = "openbmb/UltraData-Stack-v3-L2"
DATASET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_dataset.jsonl")
LOCAL_DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "datasets")

# Concurrency & Progress Tracker
_import_lock = threading.Lock()
import_state = {
    "is_importing": False,
    "progress": 0,
    "total": 0,
    "current_dataset": DEFAULT_DATASET,
    "last_result": "idle"
}

POPULAR_DATASETS = [
    # 💻 CODING & SOFTWARE ENGINEERING
    {
        "id": "openbmb/UltraData-Stack-v3-L2",
        "repo": "openbmb/UltraData-Stack-v3-L2",
        "subset": None,
        "name": "openbmb/UltraData-Stack-v3-L2 (Default: Filtered High-Yield Code & Data Stack)",
        "category": "💻 Coding & Tech",
        "default_limit": 50,
        "is_default": True
    },
    {
        "id": "openbmb/UltraData-SFT-Agent-2609:Code-Agent",
        "repo": "openbmb/UltraData-SFT-Agent-2609",
        "subset": "Code-Agent",
        "name": "openbmb/UltraData-SFT-Agent-2609 [Code-Agent] (Agentic Coding & Tool Use)",
        "category": "💻 Coding & Tech",
        "default_limit": 50,
        "is_default": False
    },
    {
        "id": "sahil2801/CodeAlpaca-20k",
        "repo": "sahil2801/CodeAlpaca-20k",
        "subset": None,
        "name": "sahil2801/CodeAlpaca-20k (20k Coding Tasks & Solutions)",
        "category": "💻 Coding & Tech",
        "default_limit": 50,
        "is_default": False
    },
    {
        "id": "iamtarun/python_code_instructions_18k_alpaca",
        "repo": "iamtarun/python_code_instructions_18k_alpaca",
        "subset": None,
        "name": "iamtarun/python_code_instructions_18k_alpaca (18k Python Problems & Functions)",
        "category": "💻 Coding & Tech",
        "default_limit": 50,
        "is_default": False
    },

    # 🤖 PERSONAL ASSISTANT (PA) & PRODUCTIVITY
    {
        "id": "databricks/databricks-dolly-15k",
        "repo": "databricks/databricks-dolly-15k",
        "subset": None,
        "name": "databricks/databricks-dolly-15k (Human PA: Scheduling, Brainstorming, QA & Tasks)",
        "category": "🤖 Personal Assistant (PA)",
        "default_limit": 50,
        "is_default": False
    },
    {
        "id": "teknium/OpenHermes-2.5",
        "repo": "teknium/OpenHermes-2.5",
        "subset": None,
        "name": "teknium/OpenHermes-2.5 (Gold-Standard General Assistant & Reasoning)",
        "category": "🤖 Personal Assistant (PA)",
        "default_limit": 50,
        "is_default": False
    },
    {
        "id": "openbmb/UltraData-SFT-Agent-2609:General-Agent",
        "repo": "openbmb/UltraData-SFT-Agent-2609",
        "subset": "General-Agent",
        "name": "openbmb/UltraData-SFT-Agent-2609 [General-Agent] (Multi-Step Task Planning)",
        "category": "🤖 Personal Assistant (PA)",
        "default_limit": 50,
        "is_default": False
    },

    # 📄 DOCUMENTS, SUMMARIZATION & KNOWLEDGE
    {
        "id": "knkarthick/dialogsum",
        "repo": "knkarthick/dialogsum",
        "subset": None,
        "name": "knkarthick/dialogsum (Document, Meeting & Dialogue Summarization)",
        "category": "📄 Documents & Summaries",
        "default_limit": 50,
        "is_default": False
    },
    {
        "id": "samsum",
        "repo": "samsum",
        "subset": None,
        "name": "samsum (Notes, Messaging & Key Takeaway Summarization)",
        "category": "📄 Documents & Summaries",
        "default_limit": 50,
        "is_default": False
    },
    {
        "id": "openbmb/UltraData-SFT-Agent-2609:Search-Agent",
        "repo": "openbmb/UltraData-SFT-Agent-2609",
        "subset": "Search-Agent",
        "name": "openbmb/UltraData-SFT-Agent-2609 [Search-Agent] (Document Search & Fact Retrieval)",
        "category": "📄 Documents & Summaries",
        "default_limit": 50,
        "is_default": False
    },

    # 💬 NATURAL CONVERSATION & ALIGNMENT
    {
        "id": "HuggingFaceH4/ultrachat_200k",
        "repo": "HuggingFaceH4/ultrachat_200k",
        "subset": None,
        "name": "HuggingFaceH4/ultrachat_200k (High-Quality Multi-Turn Conversation)",
        "category": "💬 Conversation & Chat",
        "default_limit": 50,
        "is_default": False
    },
    {
        "id": "Anthropic/hh-rlhf",
        "repo": "Anthropic/hh-rlhf",
        "subset": None,
        "name": "Anthropic/hh-rlhf (Helpful & Harmless Dialogue Alignment)",
        "category": "💬 Conversation & Chat",
        "default_limit": 50,
        "is_default": False
    }
]

def ensure_datasets_installed():
    """
    Ensures the Hugging Face 'datasets' library is available.
    Automatically installs via pip if missing.
    """
    try:
        import datasets
        return True
    except ImportError:
        print("[*] Hugging Face 'datasets' library not found. Installing automatically via pip...")
        import subprocess
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "datasets>=3.0.0", "huggingface-hub>=0.28.0"
            ])
            import datasets
            print("[✓] Successfully installed Hugging Face datasets.")
            return True
        except Exception as e:
            print(f"[!] Failed to auto-install datasets: {e}")
            return False

def get_local_repo_path(repo_id=DEFAULT_DATASET, subset=None):
    """Returns local disk storage path for a given dataset repository and optional subset."""
    if ":" in repo_id and not subset:
        repo_id, subset = repo_id.split(":", 1)
    clean_name = repo_id.replace("/", "_")
    if subset:
        clean_name += f"_{subset.replace('/', '_')}"
    return os.path.join(LOCAL_DATASETS_DIR, clean_name)

def load_local_dataset(repo_id=DEFAULT_DATASET, subset=None, split="train", streaming=False, token=None):
    """
    Loads dataset locally from disk if downloaded, otherwise streams from Hugging Face
    and caches locally in data/datasets. Supports subsets and gated HF tokens.
    """
    ensure_datasets_installed()
    import datasets

    if ":" in repo_id and not subset:
        repo_id, subset = repo_id.split(":", 1)

    os.makedirs(LOCAL_DATASETS_DIR, exist_ok=True)
    local_path = get_local_repo_path(repo_id, subset)

    # 1. Check if saved to disk locally
    if os.path.exists(local_path) and os.listdir(local_path):
        try:
            desc = f"'{repo_id}' [{subset}]" if subset else f"'{repo_id}'"
            print(f"[*] Loading {desc} from local disk: {local_path}")
            return datasets.load_from_disk(local_path)
        except Exception as e:
            print(f"[!] Note: Could not load from disk ({e}), falling back to HF cache...")

    # 2. Load with local cache_dir
    hf_token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    kwargs = {
        "split": split,
        "cache_dir": LOCAL_DATASETS_DIR,
        "streaming": streaming
    }
    if hf_token:
        kwargs["token"] = hf_token

    desc = f"'{repo_id}' [{subset}]" if subset else f"'{repo_id}'"
    print(f"[*] Loading {desc} (cache_dir={LOCAL_DATASETS_DIR}, streaming={streaming})...")

    if subset:
        return datasets.load_dataset(repo_id, subset, **kwargs)
    else:
        return datasets.load_dataset(repo_id, **kwargs)

def download_and_save_locally(repo_id=DEFAULT_DATASET, subset=None, limit=200, split="train", token=None):
    """
    Downloads dataset samples from Hugging Face and saves them permanently
    to local disk (data/datasets/<repo_id>) for 100% offline access.
    """
    ensure_datasets_installed()
    import datasets

    if ":" in repo_id and not subset:
        repo_id, subset = repo_id.split(":", 1)

    os.makedirs(LOCAL_DATASETS_DIR, exist_ok=True)
    local_path = get_local_repo_path(repo_id, subset)

    desc = f"'{repo_id}' [{subset}]" if subset else f"'{repo_id}'"
    print(f"[*] Downloading {limit} samples from {desc} for local offline storage...")

    hf_token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    kwargs = {
        "split": split,
        "cache_dir": LOCAL_DATASETS_DIR,
        "streaming": True
    }
    if hf_token:
        kwargs["token"] = hf_token

    if subset:
        stream_ds = datasets.load_dataset(repo_id, subset, **kwargs)
    else:
        stream_ds = datasets.load_dataset(repo_id, **kwargs)

    samples = []
    for i, item in enumerate(stream_ds):
        if i >= limit:
            break
        samples.append(item)

    local_ds = datasets.Dataset.from_list(samples)
    local_ds.save_to_disk(local_path)
    print(f"[✓] Saved {len(samples)} samples locally to: {local_path}")
    return local_path

def extract_code_and_lesson(text):
    """Extracts markdown code fences or separates text explanations from code."""
    if not text:
        return "", ""
    code_match = re.search(r"```(?:[a-zA-Z0-9_\-]+)?\s*([\s\S]*?)\s*```", text)
    if code_match:
        code = code_match.group(1).strip()
        lesson = re.sub(r"```(?:[a-zA-Z0-9_\-]+)?\s*[\s\S]*?\s*```", "", text).strip()
        return lesson or "Practical implementation.", code
    return text.strip(), ""

def validate_code(code):
    """Validates Python syntax if code looks like Python."""
    if not code:
        return True
    if any(k in code for k in ["def ", "import ", "class ", "print("]):
        try:
            ast.parse(code)
            return True
        except Exception:
            return False
    return True

def parse_record(item):
    """
    Normalizes diverse Hugging Face dataset schemas into a standard (topic, lesson, code, tags).
    Includes direct support for:
    - openbmb/UltraData-Stack-v3-L2
    - openbmb/UltraData-SFT-Agent-2609 (Code-Agent, General-Agent, Search-Agent)
    - Anthropic/hh-rlhf
    - Standard Alpaca, ShareGPT, and QA schemas.
    """
    topic = ""
    lesson = ""
    code = ""
    tags = "huggingface,dataset"

    # Schema 1: Agent SFT multi-turn dialogues (openbmb/UltraData-SFT-Agent-2609)
    if "conversations" in item and isinstance(item["conversations"], list):
        user_turn = next((c.get("value") or c.get("content") for c in item["conversations"] if c.get("from") in ["user", "human"] or c.get("role") in ["user", "human"]), "")
        asst_turns = [c.get("value") or c.get("content") for c in item["conversations"] if c.get("from") in ["assistant", "gpt", "agent"] or c.get("role") in ["assistant", "gpt", "agent"]]
        topic = str(user_turn) if user_turn else "Agent Task"
        full_asst = "\n\n".join(str(t) for t in asst_turns if t)
        lesson, code = extract_code_and_lesson(full_asst)
        tags = "agent,sft,tool_use,action"

    elif "messages" in item and isinstance(item["messages"], list):
        user_msg = next((m.get("content", "") for m in item["messages"] if m.get("role") in ["user", "human"]), "")
        asst_msgs = [m.get("content", "") for m in item["messages"] if m.get("role") in ["assistant", "agent", "tool"]]
        topic = str(user_msg) if user_msg else "Agent Interaction"
        full_asst = "\n\n".join(str(m) for m in asst_msgs if m)
        lesson, code = extract_code_and_lesson(full_asst)
        tags = "agent,sft,tool_use,action"

    # Schema 2: Anthropic HH-RLHF (chosen / rejected dialogue turns)
    elif "chosen" in item and item["chosen"]:
        chosen_dialogue = str(item["chosen"]).strip()
        if "\n\nAssistant:" in chosen_dialogue:
            parts = chosen_dialogue.rsplit("\n\nAssistant:", 1)
            raw_assistant = parts[1].strip()
            human_parts = parts[0].rsplit("\n\nHuman:", 1)
            raw_human = human_parts[1].strip() if len(human_parts) > 1 else human_parts[0].strip()
            if raw_human.lower().startswith("human:"):
                raw_human = raw_human[6:].strip()
            topic = raw_human
            lesson, code = extract_code_and_lesson(raw_assistant)
            tags = "anthropic,rlhf,alignment,dialogue"
        else:
            topic = chosen_dialogue[:150]
            lesson, code = extract_code_and_lesson(chosen_dialogue)
            tags = "anthropic,rlhf"

    # Schema 3: openbmb/UltraData-Stack-v3-L2 and pretraining corpuses (text, content, markdown)
    elif "text" in item and item["text"]:
        raw_text = str(item["text"]).strip()
        lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
        if lines:
            topic = lines[0][:150].lstrip("#").strip()
            lesson, code = extract_code_and_lesson(raw_text)
            if not lesson:
                lesson = raw_text[:500]

    elif "content" in item and item["content"]:
        raw_text = str(item["content"]).strip()
        lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
        if lines:
            topic = lines[0][:150].lstrip("#").strip()
            lesson, code = extract_code_and_lesson(raw_text)
            if not lesson:
                lesson = raw_text[:500]

    # Schema 4: Document & Dialogue Summarization (knkarthick/dialogsum, samsum)
    elif "dialogue" in item and "summary" in item:
        dialogue = str(item["dialogue"]).strip()
        summary = str(item["summary"]).strip()
        topic_preview = str(item.get("topic") or dialogue)[:120].strip()
        topic = f"Summarize: {topic_preview}"
        lesson = f"Document / Dialogue Content:\n{dialogue}\n\nSummary / Key Points:\n{summary}"
        code = ""
        tags = "document,summarization,notes"

    # Schema 5: Human PA & Instruction with context (databricks/databricks-dolly-15k, alpaca, openhermes)
    elif "instruction" in item and ("output" in item or "response" in item):
        topic = str(item["instruction"]).strip()
        ctx = item.get("context") or item.get("input")
        raw_output = str(item.get("output") or item.get("response") or "")
        if ctx and str(ctx).strip():
            combined = f"Context:\n{str(ctx).strip()}\n\n{raw_output}"
        else:
            combined = raw_output
        lesson, code = extract_code_and_lesson(combined)
        cat = item.get("category", "")
        tags = f"pa,instruction,{cat}" if cat else "pa,instruction"

    # Schema 6: QA / Question-Answer
    elif "question" in item and "answer" in item:
        topic = str(item["question"])
        lesson, code = extract_code_and_lesson(str(item["answer"]))

    # Schema 6: Prompt / Completion
    elif "prompt" in item and ("completion" in item or "response" in item):
        topic = str(item["prompt"])
        raw_res = str(item.get("completion") or item.get("response") or "")
        lesson, code = extract_code_and_lesson(raw_res)

    # Fallback: scan first two string values
    else:
        str_vals = [str(v) for v in item.values() if isinstance(v, str) and len(str(v).strip()) > 10]
        if len(str_vals) >= 2:
            topic = str_vals[0][:150]
            lesson, code = extract_code_and_lesson(str_vals[1])

    if not topic:
        return None

    # Topic length sanitation
    topic = topic.strip()
    if len(topic) > 180:
        topic = topic[:177] + "..."

    return {
        "topic": topic,
        "lesson": lesson or topic,
        "code_example": code,
        "tags": tags
    }

def import_hf_dataset(repo_id=DEFAULT_DATASET, subset=None, limit=50, split="train", save_local=False, token=None):
    """
    Streams and assimilates a dataset from Hugging Face into Merlin's offline memory.
    If save_local is True, also saves the samples locally to data/datasets.
    """
    global import_state

    if not _import_lock.acquire(blocking=False):
        return {"status": "error", "message": "Another dataset import is already in progress."}

    if not ensure_datasets_installed():
        _import_lock.release()
        return {"status": "error", "message": "Failed to install Hugging Face 'datasets' library."}

    init_db()
    if ":" in repo_id and not subset:
        repo_id, subset = repo_id.split(":", 1)

    repo_id = (repo_id or DEFAULT_DATASET).strip()
    limit = max(1, min(int(limit), 2000))
    display_name = f"{repo_id}:{subset}" if subset else repo_id

    try:
        import_state["is_importing"] = True
        import_state["progress"] = 0
        import_state["total"] = limit
        import_state["current_dataset"] = display_name
        import_state["last_result"] = "running"
        log_event("dataset_import_started", f"Importing {limit} records from {display_name}", "info")

        # Load local or remote dataset
        ds = load_local_dataset(repo_id=repo_id, subset=subset, split=split, streaming=True, token=token)
        count = 0
        raw_samples = []

        for item in ds:
            if count >= limit:
                break

            parsed = parse_record(item)
            if not parsed or not parsed["topic"]:
                continue

            # Validate code snippet if present
            if parsed["code_example"] and not validate_code(parsed["code_example"]):
                continue

            # Assimilate into SQLite knowledge base
            source_tag = f"hf:{display_name}"
            add_knowledge(
                topic=parsed["topic"],
                lesson=parsed["lesson"],
                code_example=parsed["code_example"],
                tags=parsed["tags"],
                source=source_tag
            )

            # Append to training dataset JSONL
            training_sample = {
                "prompt": parsed["topic"],
                "response": parsed["code_example"] if parsed["code_example"] else parsed["lesson"]
            }
            with open(DATASET_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(training_sample) + "\n")

            if save_local:
                raw_samples.append(item)

            count += 1
            import_state["progress"] = count

        if save_local and raw_samples:
            import datasets
            local_path = get_local_repo_path(repo_id, subset)
            os.makedirs(LOCAL_DATASETS_DIR, exist_ok=True)
            datasets.Dataset.from_list(raw_samples).save_to_disk(local_path)
            log_event("dataset_saved_locally", f"Saved {len(raw_samples)} samples to {local_path}", "success")

        import_state["last_result"] = "completed"
        log_event("dataset_import_finished", f"Successfully imported {count} items from {display_name}", "success")
        return {
            "status": "success",
            "imported_count": count,
            "dataset": display_name,
            "message": f"Successfully imported and assimilated {count} samples from {display_name}."
        }

    except Exception as e:
        import_state["last_result"] = f"failed: {e}"
        log_event("dataset_import_error", f"Error importing {display_name}: {e}", "danger")
        return {"status": "error", "message": f"Dataset import failed: {e}"}

    finally:
        import_state["is_importing"] = False
        _import_lock.release()

if __name__ == "__main__":
    repo = DEFAULT_DATASET
    subset = None
    lim = 50
    save_to_local_disk = False
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

    args = sys.argv[1:]
    if "--download-local" in args:
        save_to_local_disk = True
        args.remove("--download-local")

    if "--token" in args:
        try:
            t_idx = args.index("--token")
            token = args[t_idx + 1]
            args.pop(t_idx + 1)
            args.pop(t_idx)
        except Exception:
            pass

    if "--limit" in args:
        try:
            idx = args.index("--limit")
            lim = int(args[idx + 1])
            args.pop(idx + 1)
            args.pop(idx)
        except Exception:
            pass

    if "--subset" in args:
        try:
            s_idx = args.index("--subset")
            subset = args[s_idx + 1]
            args.pop(s_idx + 1)
            args.pop(s_idx)
        except Exception:
            pass

    if args and not args[0].startswith("--"):
        repo = args[0]
        if ":" in repo and not subset:
            repo, subset = repo.split(":", 1)

    print("==========================================================")
    print(f"  Merlin AI — Hugging Face Dataset Assimilator")
    print(f"  Target Dataset: {repo}")
    if subset:
        print(f"  Subset / Config:{subset}")
    print(f"  Sample Limit:   {lim}")
    print(f"  Save Local:     {save_to_local_disk}")
    if token:
        print(f"  Auth Token:     [Configured]")
    print("==========================================================")
    res = import_hf_dataset(repo, subset=subset, limit=lim, save_local=save_to_local_disk, token=token)
    print(f"[✓] {res.get('message', res)}")
