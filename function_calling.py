"""
Deterministic JSON Function-Calling Engine & Calendar Automation for Merlin AI.

Provides standard OpenAI / Ollama tool schemas, validation, and zero-hallucination
execution dispatch for calendar bookings, availability checks, and knowledge retrieval.
"""

import datetime
import json
import re
from collections.abc import Callable
from typing import Any

from database import (
    book_calendar_event,
    cancel_calendar_event,
    check_calendar_availability,
    get_calendar_events,
    search_knowledge,
)

# Standard Tool Definitions for OpenAI-compatible and Ollama function calling APIs
MERLIN_TOOLS: list[dict[str, Any]] = [
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
]


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


# Dispatch Registry
TOOL_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "get_current_time": tool_get_current_time,
    "check_calendar_availability": tool_check_calendar_availability,
    "book_calendar_appointment": tool_book_calendar_appointment,
    "list_calendar_events": tool_list_calendar_events,
    "cancel_calendar_appointment": tool_cancel_calendar_appointment,
    "search_knowledge_base": tool_search_knowledge_base,
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
