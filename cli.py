import os
import sys
from memory import init_db, add_todo, get_todos, get_all_knowledge, add_gap, get_system_telemetry
from engine import InferenceEngine

def main():
    init_db()
    print("==========================================================")
    print("  AETHER OFFLINE AI ASSISTANT & CODER (CLI MODE)")
    print("  Commands: :tasks, :addtask <task>, :skills, :gap <question>, :sys, :exit")
    print("==========================================================")

    engine = InferenceEngine()
    session_id = "cli_session"
    mode = "coder"

    while True:
        try:
            user_input = input(f"\n[{mode.upper()}] > ").strip()
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

            elif user_input.lower() == ":toggle":
                mode = "assistant" if mode == "coder" else "coder"
                print(f"Switched mode to: {mode}")
                continue

            # Standard chat/coding generation
            print("\n--- Response ---")
            for chunk in engine.generate_stream(user_input, session_id=session_id, mode=mode):
                sys.stdout.write(chunk)
                sys.stdout.flush()
            print()

        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
