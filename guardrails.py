"""
Business action "tools" the assistant can invoke.

Each tool operates on mock JSON data under /data so the project is fully
self-contained and runnable without any external database or credentials.
The functions here are plain Python - they are wrapped as Anthropic tool
definitions in llm_service.py and dispatched by name.
"""
import json
import os
import uuid
from datetime import datetime, timezone
from difflib import get_close_matches
from threading import Lock

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMPLOYEES_PATH = os.path.join(BASE_DIR, "data", "employees.json")
TICKETS_PATH = os.path.join(BASE_DIR, "data", "tickets.json")

_tickets_lock = Lock()


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Tool 1: Create a support ticket
# ---------------------------------------------------------------------------
def create_ticket(subject: str, description: str = "", priority: str = "medium",
                   requested_by: str = "unknown") -> dict:
    priority = priority.lower() if priority else "medium"
    if priority not in {"low", "medium", "high", "urgent"}:
        priority = "medium"

    with _tickets_lock:
        tickets = _load_json(TICKETS_PATH)
        ticket = {
            "ticket_id": f"TCK-{uuid.uuid4().hex[:8].upper()}",
            "subject": subject,
            "description": description,
            "priority": priority,
            "status": "open",
            "requested_by": requested_by,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        tickets.append(ticket)
        _save_json(TICKETS_PATH, tickets)
    return ticket


# ---------------------------------------------------------------------------
# Tool 2: Look up employee information
# ---------------------------------------------------------------------------
def get_employee_info(name: str = "", employee_id: str = "") -> dict:
    employees = _load_json(EMPLOYEES_PATH)

    if employee_id:
        for emp in employees:
            if emp["id"].lower() == employee_id.lower():
                return {"found": True, "employee": emp}
        return {"found": False, "message": f"No employee found with id '{employee_id}'."}

    if not name:
        return {"found": False, "message": "No name or employee_id provided."}

    name_lower = name.lower().strip()
    # exact / substring match first
    for emp in employees:
        if name_lower == emp["name"].lower() or name_lower in emp["name"].lower():
            return {"found": True, "employee": emp}

    # fuzzy fallback for typos
    all_names = [e["name"] for e in employees]
    close = get_close_matches(name, all_names, n=1, cutoff=0.6)
    if close:
        emp = next(e for e in employees if e["name"] == close[0])
        return {"found": True, "employee": emp, "note": f"Closest match to '{name}'."}

    return {"found": False, "message": f"No employee found matching '{name}'."}


# ---------------------------------------------------------------------------
# Tool 3: Generate a simple report
# ---------------------------------------------------------------------------
def generate_report(department: str = "") -> dict:
    employees = _load_json(EMPLOYEES_PATH)
    with _tickets_lock:
        tickets = _load_json(TICKETS_PATH)

    if department:
        dept_lower = department.lower()
        employees = [e for e in employees if e["department"].lower() == dept_lower]

    dept_counts = {}
    for e in _load_json(EMPLOYEES_PATH):
        dept_counts[e["department"]] = dept_counts.get(e["department"], 0) + 1

    priority_counts = {}
    open_tickets = 0
    for t in tickets:
        priority_counts[t["priority"]] = priority_counts.get(t["priority"], 0) + 1
        if t["status"] == "open":
            open_tickets += 1

    return {
        "scope": department or "company-wide",
        "employee_count": len(employees),
        "headcount_by_department": dept_counts,
        "total_tickets": len(tickets),
        "open_tickets": open_tickets,
        "tickets_by_priority": priority_counts,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Dispatcher used by the LLM tool-calling layer
# ---------------------------------------------------------------------------
TOOL_FUNCTIONS = {
    "create_ticket": create_ticket,
    "get_employee_info": get_employee_info,
    "generate_report": generate_report,
}


def execute_tool(tool_name: str, tool_input: dict) -> dict:
    fn = TOOL_FUNCTIONS.get(tool_name)
    if not fn:
        return {"error": f"Unknown tool '{tool_name}'."}
    try:
        return fn(**tool_input)
    except TypeError as e:
        return {"error": f"Invalid arguments for '{tool_name}': {e}"}
