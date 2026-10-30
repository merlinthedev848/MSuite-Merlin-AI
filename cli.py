import os
import sys

from database import (
    add_gap,
    add_todo,
    get_all_knowledge,
    get_system_telemetry,
    get_todos,
    init_db,
)
from engine import InferenceEngine


def main():
    init_db()
    print("==========================================================")
    print("  MERLIN AI — OFFLINE ASSISTANT & CODER (CLI MODE)")
    print("  Commands: :teach <topic>, :dataset <repo> [N], :tasks, :addtask, :skills, :gap, :sys, :exit")
    print("==========================================================")

    engine = InferenceEngine()
    session_id = "cli_session"

    while True:
        try:
            user_input = input("\n[MERLIN AI] > ").strip()
            if not user_input:
                continue

            if user_input.lower() in [":exit", "exit", "quit", ":q"]:
                print("Goodbye!")
                break

            elif user_input.lower() == ":tasks":
                todos = get_todos()
                print("\n--- To-Do List ---")
                if not todos:
                    print("No tasks recorded.")
                for t in todos:
                    status = "[✓]" if t["completed"] else "[ ]"
                    print(f"{status} #{t['id']}: {t['task']}")
                continue

            elif user_input.startswith(":addtask "):
                task = user_input[9:].strip()
                tid = add_todo(task)
                print(f"[✓] Task #{tid} added.")
                continue

            elif user_input.lower() == ":skills":
                skills = get_all_knowledge()
                print(f"\n--- Learned Knowledge Base ({len(skills)} entries) ---")
                for s in skills[:5]:
                    print(f"• {s['topic']}: {s['lesson'][:100]}...")
                continue

            elif user_input.startswith(":gap "):
                gap = user_input[5:].strip()
                gid = add_gap(gap)
                print(f"[✓] Knowledge gap #{gid} recorded for future teacher sync.")
                continue

            elif user_input.lower() == ":sys":
                stats = get_system_telemetry()
                print("\n--- System Diagnostics ---")
                for k, v in stats.items():
                    print(f"{k.capitalize()}: {v}")
                continue

            elif user_input.startswith(":teach"):
                topic = user_input[6:].strip()
                key = (
                    os.getenv("GEMINI_API_KEY")
                    or os.getenv("GOOGLE_API_KEY")
                    or os.getenv("TEACHER_API_KEY")
                    or os.getenv("GROQ_API_KEY")
                    or os.getenv("OPENAI_API_KEY")
                )
                if not key:
                    key = input("Enter Teacher API Key: ").strip()
                if not key:
                    print("Teacher sync cancelled: API key required.")
                    continue
                print("[*] Distilling knowledge via Teacher AI...")
                from sync_teacher import run_sync
                res = run_sync(key, topic_to_explore=topic or None)
                print(f"[✓] Result: {res.get('message', 'Completed')}")
                continue

            elif user_input.startswith(":dataset"):
                parts = user_input[8:].strip().split()
                repo_id = parts[0] if parts else "openbmb/UltraData-Stack-v3-L2"
                limit = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 50
                print(f"[*] Assimilating and caching locally {limit} samples from: {repo_id}...")
                from import_dataset import import_hf_dataset
                res = import_hf_dataset(repo_id, limit=limit, save_local=True)
                print(f"[✓] Result: {res.get('message', 'Done')}")
                continue



            # Standard chat/coding generation (Unified Dual-Power Coder & Assistant)
            print("\n--- Response ---")
            for chunk in engine.generate_stream(user_input, session_id=session_id):
                sys.stdout.write(chunk)
                sys.stdout.flush()
            print()

        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
