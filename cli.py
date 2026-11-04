import sys

from database import (
    add_gap,
    add_note,
    add_todo,
    checkpoint_and_vacuum,
    clear_session_messages,
    delete_note,
    delete_todo,
    get_all_knowledge,
    get_notes,
    get_system_telemetry,
    get_todos,
    init_db,
    search_notes,
    toggle_todo,
)
from engine import InferenceEngine
from sync_teacher import load_saved_api_key, run_sync

HELP_TEXT = """
Commands:
  :help, :?               Show this help message
  :tasks                  List all to-do items
  :addtask <text>         Add a new task
  :done <id>              Toggle task completion
  :deltask <id>           Delete a task by ID
  :notes                  View recent technical notes
  :addnote <title> | <txt>Add a new note
  :delnote <id>           Delete a note by ID
  :searchnote <query>     Search notes by keyword
  :skills                 List learned knowledge base entries
  :gap <question>         Record an offline knowledge gap
  :models                 List available inference models
  :model <name>           Switch active model
  :sys                    Display hardware & system telemetry
  :vacuum                 Optimize SQLite database & WAL checkpoint
  :clear                  Clear current CLI session history
  :teach <topic>          Distill a topic using Teacher AI
  :dataset <repo> [N]     Assimilate Hugging Face dataset
  :exit, :q, quit         Exit Merlin CLI
"""


def main():
    init_db()
    print("==========================================================")
    print("  MERLIN AI — OFFLINE ASSISTANT & CODER (CLI MODE)")
    print("  Type :help for commands, or type any prompt to begin.")
    print("==========================================================")

    engine = InferenceEngine()
    session_id = "cli_session"

    while True:
        try:
            try:
                user_input = input("\n[MERLIN AI] > ").strip()
            except EOFError:
                print("\nGoodbye!")
                break
            if not user_input:
                continue

            if user_input.lower() in [":exit", "exit", "quit", ":q"]:
                print("Goodbye!")
                break

            elif user_input.lower() in [":help", ":?", "help"]:
                print(HELP_TEXT)
                continue

            elif user_input.lower() == ":tasks":
                todos = get_todos()
                print("\n--- To-Do List ---")
                if not todos:
                    print("No tasks recorded.")
                for t in todos:
                    status = "[✓]" if t["completed"] else "[ ]"
                    print(f"{status} #{t['id']} [{t.get('priority', 'normal')}]: {t['task']}")
                continue

            elif user_input.startswith(":addtask "):
                task = user_input[9:].strip()
                tid = add_todo(task)
                print(f"[✓] Task #{tid} added.")
                continue

            elif user_input.startswith(":done "):
                raw_id = user_input[6:].strip()
                if raw_id.isdigit():
                    toggled = toggle_todo(int(raw_id))
                    print(f"[{'✓' if toggled else '!'}] Task #{raw_id} {'updated' if toggled else 'not found'}.")
                else:
                    print("[!] Usage: :done <numeric task id>")
                continue

            elif user_input.startswith(":deltask "):
                raw_id = user_input[9:].strip()
                if raw_id.isdigit():
                    deleted = delete_todo(int(raw_id))
                    print(f"[{'✓' if deleted else '!'}] Task #{raw_id} {'deleted' if deleted else 'not found'}.")
                else:
                    print("[!] Usage: :deltask <numeric task id>")
                continue

            elif user_input.lower() == ":notes":
                notes = get_notes()
                print(f"\n--- Technical Notes ({len(notes)} entries) ---")
                if not notes:
                    print("No notes recorded yet.")
                for n in notes[:10]:
                    print(f"• #{n['id']} {n['title']} ({n.get('tags', '')})\n  {n['content'][:120]}...\n")
                continue

            elif user_input.startswith(":addnote "):
                payload = user_input[9:].strip()
                if "|" in payload:
                    title, content = payload.split("|", 1)
                else:
                    title, content = "CLI Note", payload
                nid = add_note(title.strip(), content.strip())
                print(f"[✓] Note #{nid} saved.")
                continue

            elif user_input.startswith(":delnote "):
                raw_id = user_input[9:].strip()
                if raw_id.isdigit():
                    deleted = delete_note(int(raw_id))
                    print(f"[{'✓' if deleted else '!'}] Note #{raw_id} {'deleted' if deleted else 'not found'}.")
                else:
                    print("[!] Usage: :delnote <numeric note id>")
                continue

            elif user_input.startswith(":searchnote "):
                q = user_input[12:].strip()
                matches = search_notes(q)
                print(f"\n--- Notes matching '{q}' ({len(matches)} results) ---")
                for m in matches:
                    print(f"• #{m['id']} {m['title']}: {m['content'][:100]}...")
                continue

            elif user_input.lower() == ":skills":
                skills = get_all_knowledge()
                print(f"\n--- Learned Knowledge Base ({len(skills)} entries) ---")
                if not skills:
                    print("No skills assimilated yet.")
                for s in skills[:8]:
                    print(f"• #{s['id']} {s['topic']} [{s.get('source', '')}]\n  {s['lesson'][:120]}...")
                continue

            elif user_input.startswith(":gap "):
                gap = user_input[5:].strip()
                gid = add_gap(gap)
                print(f"[✓] Knowledge gap #{gid} recorded for future teacher sync.")
                continue

            elif user_input.lower() == ":models":
                models = engine.get_available_models()
                print(f"\n--- Model Registry (Backend: {engine.backend}) ---")
                print(f"Active: {engine.active_model_name}")
                for m in models:
                    active_marker = " [ACTIVE]" if m.get("is_active") else ""
                    print(f"• {m['name']} ({m['backend']}){active_marker}")
                continue

            elif user_input.startswith(":model "):
                target_model = user_input[7:].strip()
                success, msg = engine.set_active_model(target_model)
                prefix = "[✓]" if success else "[!]"
                print(f"{prefix} {msg}")
                continue

            elif user_input.lower() == ":vacuum":
                res = checkpoint_and_vacuum()
                print(f"[✓] {res.get('message', 'Maintenance completed.')}")
                continue

            elif user_input.lower() == ":clear":
                clear_session_messages(session_id)
                print("[✓] Session messages cleared.")
                continue

            elif user_input.lower() == ":sys":
                stats = get_system_telemetry()
                print("\n--- System Diagnostics ---")
                for k, v in stats.items():
                    print(f"{k.capitalize()}: {v}")
                continue

            elif user_input.startswith(":teach"):
                topic = user_input[6:].strip()
                key = load_saved_api_key()
                if not key:
                    key = input("Enter Teacher API Key: ").strip()
                if not key:
                    print("Teacher sync cancelled: API key required.")
                    continue
                print("[*] Distilling knowledge via Teacher AI...")
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

            # Standard chat/coding generation (with stream cancel on KeyboardInterrupt)
            print("\n--- Response ---")
            try:
                for chunk in engine.generate_stream(user_input, session_id=session_id):
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                print()
            except KeyboardInterrupt:
                engine.stop_generation(session_id)
                print("\n[Generation cancelled by user]")

        except KeyboardInterrupt:
            print("\nType :exit or press Ctrl+D to quit.")
            continue


if __name__ == "__main__":
    main()
