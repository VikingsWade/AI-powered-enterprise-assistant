"""
Core AI workflow.

ENGINEERING IMPROVEMENT (chosen): API / Tool Calling
------------------------------------------------------
Naive v0 of this project used keyword matching ("if 'ticket' in question")
to decide which business action to run, then string-formatted a canned
response. That's brittle: it breaks on phrasing it hasn't seen, can't
extract structured fields (subject/priority/employee name) reliably, and
can't combine "answer + action" in one coherent reply.

The improvement: the question is sent to Claude with a set of function
("tool") definitions (create_ticket, get_employee_info, generate_report).
The model decides - based on meaning, not keywords - whether a tool is
needed, and returns structured, typed arguments for it. We execute the
tool locally against mock data, feed the result back to the model, and let
it compose the final natural-language answer. This makes the system:
  - robust to free-form phrasing
  - able to ask clarifying questions when a required field is missing
    (Claude can just respond with text instead of calling a tool)
  - easy to extend (add a new business action = add one tool schema)

FALLBACK / ERROR HANDLING
------------------------------------------------------
If the Anthropic API is unreachable, unauthenticated, or errors out, we do
NOT want the whole endpoint to 500. Instead we fall back to a small
rule-based intent classifier that covers the same three actions using
simple keyword heuristics. This keeps the /ask endpoint functional (with
reduced NLU quality) even with zero external dependencies - which is also
what lets the two required test inputs run in this sandbox with no API key
configured.
"""
import os
import re
import traceback

from app.tools import execute_tool, get_employee_info
from app.memory import memory

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are an internal enterprise assistant for Acme Corp.
You can answer general questions and, when appropriate, take one of these
business actions using the tools available to you:
  - create_ticket: file an IT/HR/ops support ticket
  - get_employee_info: look up an employee's contact/department info
  - generate_report: produce a headcount / ticket summary report

Rules:
- Only call a tool when the user's request actually requires that action.
- If a required detail is missing (e.g. no ticket subject, or an employee
  name you don't recognize), ask a concise clarifying question instead of
  guessing.
- After a tool result comes back, summarize it clearly and helpfully in
  plain language - don't just dump raw JSON.
- Keep answers concise and professional.
"""

TOOLS = [
    {
        "name": "create_ticket",
        "description": "File a support/IT/HR ticket on behalf of the user. Use this when "
        "the user reports a problem, outage, request, or anything needing follow-up action.",
        "input_schema": {
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "Short summary of the issue."},
                "description": {"type": "string", "description": "Fuller description of the issue."},
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "urgent"],
                    "description": "Urgency of the ticket.",
                },
                "requested_by": {"type": "string", "description": "Name of the requester, if known."},
            },
            "required": ["subject"],
        },
    },
    {
        "name": "get_employee_info",
        "description": "Look up an employee's department, title, email, phone, manager, "
        "or location by name or employee id.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Employee's full or partial name."},
                "employee_id": {"type": "string", "description": "Employee id, e.g. E101."},
            },
        },
    },
    {
        "name": "generate_report",
        "description": "Generate a headcount and ticket summary report, optionally scoped "
        "to one department.",
        "input_schema": {
            "type": "object",
            "properties": {
                "department": {
                    "type": "string",
                    "description": "Department to scope the report to, e.g. 'Engineering'. "
                    "Leave blank for a company-wide report.",
                }
            },
        },
    },
]


class AssistantEngine:
    def __init__(self):
        self.client = None
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            try:
                import anthropic

                self.client = anthropic.Anthropic(api_key=api_key)
            except Exception:
                self.client = None

    # ------------------------------------------------------------------
    def ask(self, question: str, session_id: str = "default") -> dict:
        history = memory.get_history(session_id)

        if self.client:
            try:
                return self._ask_llm(question, history, session_id)
            except Exception as e:
                # Any API failure (auth, network, rate limit, malformed
                # response) falls back gracefully instead of a 500.
                print("LLM path failed, falling back:", e)
                traceback.print_exc()
                return self._ask_fallback(question, session_id, error=str(e))
        else:
            return self._ask_fallback(question, session_id)

    # ------------------------------------------------------------------
    def _ask_llm(self, question: str, history: list, session_id: str) -> dict:
        messages = history + [{"role": "user", "content": question}]

        response = self.client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        action_taken = None
        action_result = None

        # Loop in case the model wants to call a tool, see the result, and
        # then respond (Anthropic tool-use is a single extra round trip in
        # the common case, but we loop defensively).
        while response.stop_reason == "tool_use":
            tool_use_block = next(b for b in response.content if b.type == "tool_use")
            action_taken = tool_use_block.name
            action_result = execute_tool(tool_use_block.name, tool_use_block.input)

            messages.append({"role": "assistant", "content": response.content})
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_block.id,
                            "content": str(action_result),
                        }
                    ],
                }
            )
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

        final_text = "".join(b.text for b in response.content if b.type == "text").strip()

        memory.add_turn(session_id, "user", question)
        memory.add_turn(session_id, "assistant", final_text)

        return {
            "answer": final_text or "I've processed your request.",
            "action_taken": action_taken,
            "action_result": action_result,
            "mode": "llm",
        }

    # ------------------------------------------------------------------
    def _ask_fallback(self, question: str, session_id: str, error: str = None) -> dict:
        """Rule-based fallback used when no LLM is reachable. Covers the
        same three business actions with simple heuristics."""
        q = question.strip()
        q_lower = q.lower()
        action_taken = None
        action_result = None

        ticket_keywords = ["ticket", "broken", "broke", "not working", "issue", "problem", "crash",
                            "crashed", "outage", "down", "error", "help me fix", "vpn", "access"]
        employee_keywords = ["employee", "who is", "contact", "phone number", "email of",
                              "manager of", "department of", "reach"]
        report_keywords = ["report", "summary", "headcount", "how many employees", "stats", "statistics"]

        if any(k in q_lower for k in report_keywords):
            dept_match = re.search(r"(?:for|in)\s+([A-Za-z ]+?)(?:\s+department)?[\?\.]?$", q, re.IGNORECASE)
            department = dept_match.group(1).strip() if dept_match else ""
            action_taken = "generate_report"
            action_result = execute_tool("generate_report", {"department": department})
            answer = (
                f"Here's the {action_result['scope']} report: "
                f"{action_result['employee_count']} employees, "
                f"{action_result['open_tickets']} open ticket(s) out of {action_result['total_tickets']} total."
            )

        elif any(k in q_lower for k in employee_keywords):
            name = self._extract_name(q)
            action_taken = "get_employee_info"
            action_result = get_employee_info(name=name)
            if action_result.get("found"):
                emp = action_result["employee"]
                answer = (
                    f"{emp['name']} is a {emp['title']} in {emp['department']}. "
                    f"Email: {emp['email']}, Phone: {emp['phone']}, Location: {emp['location']}."
                )
            else:
                answer = (
                    f"I couldn't confidently identify which employee you mean from "
                    f"\"{q}\". Could you give me their full name or employee ID?"
                )

        elif any(k in q_lower for k in ticket_keywords):
            # Require a minimally usable subject; if the request is too
            # vague, ask a clarifying question instead of filing junk.
            if len(q.split()) < 4:
                answer = (
                    "I can file a support ticket for you, but I need a bit more detail - "
                    "what exactly is broken or what do you need help with?"
                )
            else:
                priority = "urgent" if any(w in q_lower for w in ["asap", "urgent", "immediately", "now"]) else "medium"
                action_taken = "create_ticket"
                action_result = execute_tool(
                    "create_ticket",
                    {"subject": q[:80], "description": q, "priority": priority, "requested_by": session_id},
                )
                answer = (
                    f"I've filed ticket {action_result['ticket_id']} "
                    f"(priority: {action_result['priority']}) for: \"{action_result['subject']}\". "
                    f"Someone from support will follow up."
                )
        else:
            answer = (
                "I'm running in fallback mode (no LLM connected) and I can help with: "
                "filing a support ticket, looking up employee info, or generating a report. "
                "Could you rephrase your request around one of those?"
            )

        memory.add_turn(session_id, "user", question)
        memory.add_turn(session_id, "assistant", answer)

        result = {
            "answer": answer,
            "action_taken": action_taken,
            "action_result": action_result,
            "mode": "fallback",
        }
        if error:
            result["fallback_reason"] = error
        return result

    @staticmethod
    def _extract_name(question: str) -> str:
        # crude but effective for demo purposes: capitalized word pairs
        matches = re.findall(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)\b", question)
        if matches:
            return matches[0]
        # fallback: text after "of" e.g. "email of john smith"
        m = re.search(r"of\s+([A-Za-z ]+)", question, re.IGNORECASE)
        return m.group(1).strip() if m else question


engine = AssistantEngine()
