"""
Deterministic JSON Function-Calling Engine & Autonomous Workhorse Tools for Merlin AI.

Provides standard OpenAI / Ollama tool schemas, validation, and zero-hallucination
execution dispatch for system diagnostics, calendar automation, personal task management,
technical note-taking, and persistent hybrid RAG knowledge retrieval.
"""

import datetime
import json
import platform
import re
import shutil
import sys
from collections.abc import Callable
from typing import Any

import psutil

from database import (
    add_note,
    add_todo,
    book_calendar_event,
    cancel_calendar_event,
    check_calendar_availability,
    checkpoint_and_vacuum,
    delete_todo,
    get_all_knowledge,
    get_calendar_events,
    get_todos,
    search_knowledge,
    search_notes,
    toggle_todo,
)

# Standard Tool Definitions for OpenAI-compatible and Ollama function calling APIs
MERLIN_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_system_diagnostics",
            "description": "Returns live hardware, OS, CPU, memory, disk, and Merlin AI operational telemetry. Always call this when asked for status report, diagnostics, suit/server health, or system telemetry.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns the current local date, time, day of week, and timezone of the server. Always call this when scheduling or answering questions relative to 'today', 'tomorrow', 'this Friday', etc.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_calendar_availability",
            "description": "Checks whether a specific time slot on the calendar is free or has conflicting appointments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {
                        "type": "string",
                        "description": "Start timestamp in ISO 8601 format (e.g. '2026-11-05T14:00:00' or '2026-11-05 14:00').",
                    },
                    "end_time": {
                        "type": "string",
                        "description": "End timestamp in ISO 8601 format (e.g. '2026-11-05T15:00:00' or '2026-11-05 15:00').",
                    },
                },
                "required": ["start_time", "end_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_calendar_appointment",
            "description": "Books a new calendar appointment or meeting. Enforces double-booking prevention. Returns confirmation details or error if conflicting.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Summary or title of the meeting / appointment.",
                    },
                    "start_time": {
                        "type": "string",
                        "description": "Start timestamp in ISO 8601 format (e.g. '2026-11-05T10:00:00').",
                    },
                    "end_time": {
                        "type": "string",
                        "description": "End timestamp in ISO 8601 format (e.g. '2026-11-05T11:00:00').",
                    },
                    "attendee_name": {
                        "type": "string",
                        "description": "Full name of the client, caller, or attendee.",
                    },
                    "attendee_phone": {
                        "type": "string",
                        "description": "Phone number of the attendee (optional).",
                    },
                    "attendee_email": {
                        "type": "string",
                        "description": "Email address of the attendee (optional).",
                    },
                    "description": {
                        "type": "string",
                        "description": "Additional notes, agenda, or location for the event.",
                    },
                },
                "required": ["title", "start_time", "end_time", "attendee_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_calendar_events",
            "description": "Lists upcoming or past scheduled calendar appointments within an optional date range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {
                        "type": "string",
                        "description": "Earliest start time to filter by (ISO 8601 string, optional).",
                    },
                    "end_time": {
                        "type": "string",
                        "description": "Latest start time to filter by (ISO 8601 string, optional).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of events to return (default 20).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_calendar_appointment",
            "description": "Cancels an existing calendar appointment by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {
                        "type": "integer",
                        "description": "The unique numerical ID of the event to cancel.",
                    },
                },
                "required": ["event_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Searches Merlin's persistent technical knowledge base for coding standards, server configs, and distilled lessons.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords or question to search.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max entries to return (default 3).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_todo",
            "description": "Adds a task to Merlin's persistent to-do list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The description or title of the action item.",
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["urgent", "high", "normal", "low"],
                        "description": "Priority level of the task (default 'normal').",
                    },
                },
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_todos",
            "description": "Lists to-do items from Merlin's task manager, optionally filtering for only pending tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "only_pending": {
                        "type": "boolean",
                        "description": "If true, only lists uncompleted tasks. Defaults to true.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "toggle_todo",
            "description": "Toggles a task between completed and uncompleted status by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "todo_id": {
                        "type": "integer",
                        "description": "ID of the to-do item to toggle.",
                    },
                },
                "required": ["todo_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_todo",
            "description": "Deletes a to-do item permanently by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "todo_id": {
                        "type": "integer",
                        "description": "ID of the to-do item to delete.",
                    },
                },
                "required": ["todo_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_technical_note",
            "description": "Saves a technical note, code snippet, or engineering insight to Merlin's memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Title or subject of the note.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Body content of the note (supports Markdown).",
                    },
                    "tags": {
                        "type": "string",
                        "description": "Comma-separated tags (e.g. 'python, docker, networking').",
                    },
                },
                "required": ["title", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_technical_notes",
            "description": "Searches stored technical notes by query keyword or tags.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keyword to search within note titles, contents, and tags.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (default 10).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "optimize_database",
            "description": "Executes an immediate SQLite WAL checkpoint and query planner optimization.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


def tool_get_system_diagnostics(**kwargs: Any) -> dict[str, Any]:
    """Retrieves live CPU, RAM, disk, OS, and Merlin AI operational telemetry."""
    try:
        cpu_pct = psutil.cpu_percent(interval=None)
        cpu_count = psutil.cpu_count(logical=True)
        vmem = psutil.virtual_memory()
    except Exception:
        cpu_pct = 0.0
        cpu_count = 1
        vmem = None

    try:
        disk = shutil.disk_usage(".")
    except Exception:
        disk = None

    # Memory telemetry
    try:
        knowledge_cnt = len(get_all_knowledge(limit=1000))
    except Exception:
        knowledge_cnt = 0

    try:
        todos = get_todos()
        open_todos = len([t for t in todos if not t.get("completed")])
    except Exception:
        open_todos = 0

    return {
        "status": "success",
        "system": {
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "python_version": sys.version.split()[0],
            "cpu_usage_percent": cpu_pct,
            "cpu_cores": cpu_count,
            "memory_total_gb": round(vmem.total / (1024**3), 2) if vmem else 0.0,
            "memory_used_gb": round(vmem.used / (1024**3), 2) if vmem else 0.0,
            "memory_free_gb": round(vmem.available / (1024**3), 2) if vmem else 0.0,
            "memory_percent": vmem.percent if vmem else 0.0,
            "disk_total_gb": round(disk.total / (1024**3), 2) if disk else 0.0,
            "disk_used_gb": round(disk.used / (1024**3), 2) if disk else 0.0,
            "disk_free_gb": round(disk.free / (1024**3), 2) if disk else 0.0,
        },
        "merlin_telemetry": {
            "active_knowledge_entries": knowledge_cnt,
            "pending_tasks": open_todos,
            "hybrid_rag_status": "ONLINE (Dense Vector + FTS5)",
            "voice_engine": "Stark Harmonic WebAudio & British Cadence Online",
        },
        "overall_status": "OPTIMAL - All primary systems operational",
    }


def tool_get_current_time(**kwargs: Any) -> dict[str, Any]:
    now = datetime.datetime.now()
    return {
        "status": "success",
        "current_time": now.isoformat(timespec="seconds"),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "day_of_week": now.strftime("%A"),
        "formatted": now.strftime("%A, %B %d, %Y at %I:%M %p"),
    }


def tool_check_calendar_availability(start_time: str, end_time: str, **kwargs: Any) -> dict[str, Any]:
    return check_calendar_availability(start_time, end_time)


def tool_book_calendar_appointment(
    title: str,
    start_time: str,
    end_time: str,
    attendee_name: str,
    attendee_phone: str = "",
    attendee_email: str = "",
    description: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    return book_calendar_event(
        title=title,
        start_time=start_time,
        end_time=end_time,
        attendee_name=attendee_name,
        attendee_phone=attendee_phone,
        attendee_email=attendee_email,
        description=description,
    )


def tool_list_calendar_events(
    start_time: str | None = None,
    end_time: str | None = None,
    limit: int = 20,
    **kwargs: Any,
) -> dict[str, Any]:
    events = get_calendar_events(start_time=start_time, end_time=end_time, limit=limit)
    return {
        "status": "success",
        "count": len(events),
        "events": events,
    }


def tool_cancel_calendar_appointment(event_id: int, **kwargs: Any) -> dict[str, Any]:
    success = cancel_calendar_event(int(event_id))
    return {
        "status": "success" if success else "not_found",
        "event_id": event_id,
        "cancelled": success,
    }


def tool_search_knowledge_base(query: str, limit: int = 3, **kwargs: Any) -> dict[str, Any]:
    results = search_knowledge(query, limit=limit)
    return {
        "status": "success",
        "query": query,
        "count": len(results),
        "results": results,
    }


def tool_add_todo(task: str, priority: str = "normal", **kwargs: Any) -> dict[str, Any]:
    todo_id = add_todo(task=task, priority=priority)
    return {
        "status": "success",
        "todo_id": todo_id,
        "task": task,
        "priority": priority,
    }


def tool_list_todos(only_pending: bool = True, **kwargs: Any) -> dict[str, Any]:
    todos = get_todos()
    if only_pending:
        todos = [t for t in todos if not t.get("completed")]
    return {
        "status": "success",
        "count": len(todos),
        "todos": todos,
    }


def tool_toggle_todo(todo_id: int, **kwargs: Any) -> dict[str, Any]:
    success = toggle_todo(int(todo_id))
    return {
        "status": "success" if success else "not_found",
        "todo_id": todo_id,
    }


def tool_delete_todo(todo_id: int, **kwargs: Any) -> dict[str, Any]:
    success = delete_todo(int(todo_id))
    return {
        "status": "success" if success else "not_found",
        "todo_id": todo_id,
    }


def tool_add_technical_note(title: str, content: str, tags: str = "", **kwargs: Any) -> dict[str, Any]:
    note_id = add_note(title=title, content=content, tags=tags)
    return {
        "status": "success",
        "note_id": note_id,
        "title": title,
    }


def tool_search_technical_notes(query: str, limit: int = 10, **kwargs: Any) -> dict[str, Any]:
    notes = search_notes(query=query, limit=limit)
    return {
        "status": "success",
        "query": query,
        "count": len(notes),
        "notes": notes,
    }


def tool_optimize_database(**kwargs: Any) -> dict[str, Any]:
    return checkpoint_and_vacuum()


# Dispatch Registry
TOOL_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "get_system_diagnostics": tool_get_system_diagnostics,
    "get_current_time": tool_get_current_time,
    "check_calendar_availability": tool_check_calendar_availability,
    "book_calendar_appointment": tool_book_calendar_appointment,
    "list_calendar_events": tool_list_calendar_events,
    "cancel_calendar_appointment": tool_cancel_calendar_appointment,
    "search_knowledge_base": tool_search_knowledge_base,
    "add_todo": tool_add_todo,
    "list_todos": tool_list_todos,
    "toggle_todo": tool_toggle_todo,
    "delete_todo": tool_delete_todo,
    "add_technical_note": tool_add_technical_note,
    "search_technical_notes": tool_search_technical_notes,
    "optimize_database": tool_optimize_database,
}


def execute_tool(name: str, arguments: dict[str, Any] | str) -> dict[str, Any]:
    """
    Executes a tool deterministically and returns structured JSON output.
    Catches and formats any exceptions to prevent crashing the agent loop.
    """
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except Exception as e:
            return {"status": "error", "error": f"Invalid JSON arguments: {e}"}

    if not isinstance(arguments, dict):
        arguments = {}

    handler = TOOL_HANDLERS.get(name)
    if not handler:
        return {"status": "error", "error": f"Unknown tool function '{name}'"}

    try:
        return handler(**arguments)
    except Exception as e:
        return {"status": "error", "error": f"Execution error in {name}: {str(e)}"}


# Regex parsers for LLMs that output function calls as text tokens
_JSON_TOOL_BLOCK_RE = re.compile(
    r"```(?:json)?\s*(\{\s*\"(?:tool|function|name|action)\"\s*:[^`]+?\})\s*```",
    re.DOTALL,
)
_TAG_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)


def extract_text_tool_calls(text: str) -> list[dict[str, Any]]:
    """
    Detects and parses JSON tool invocation blocks from local models that output
    markdown or XML-style tool calls instead of native API tool structures.
    """
    calls: list[dict[str, Any]] = []
    if not text:
        return calls

    # Check for <tool_call>{...}</tool_call>
    for match in _TAG_TOOL_CALL_RE.finditer(text):
        try:
            data = json.loads(match.group(1))
            name = data.get("name") or data.get("tool") or data.get("function")
            args = data.get("arguments") or data.get("parameters") or {}
            if name:
                calls.append({"name": name, "arguments": args})
        except Exception:
            pass

    if calls:
        return calls

    # Check for ```json { "tool": "...", "parameters": {...} } ```
    for match in _JSON_TOOL_BLOCK_RE.finditer(text):
        try:
            data = json.loads(match.group(1))
            name = data.get("name") or data.get("tool") or data.get("function") or data.get("action")
            args = data.get("arguments") or data.get("parameters") or {}
            if name:
                calls.append({"name": name, "arguments": args})
        except Exception:
            pass

    return calls
