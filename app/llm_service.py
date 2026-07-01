"""
Core AI workflow.

ENGINEERING IMPROVEMENT (chosen): API / Tool Calling
------------------------------------------------------
Naive v0 of this project used keyword matching ("if 'ticket' in question")
to decide which business action to run, then string-formatted a canned
response. That's brittle: it breaks on phrasing it hasn't seen, can't
extract structured fields (subject/priority/employee name) reliably, and
can't combine "answer + action" in one coherent reply.

The improvement: the question is sent to an LLM with a set of function
("tool") definitions (create_ticket, get_employee_info, generate_report).
The model decides - based on meaning, not keywords - whether a tool is
needed, and returns structured, typed arguments for it. We execute the
tool locally against mock data, feed the result back to the model, and let
it compose the final natural-language answer. This makes the system:
  - robust to free-form phrasing
  - able to ask clarifying questions when a required field is missing
    (the model can just respond with text instead of calling a tool)
  - easy to extend (add a new business action = add one tool schema)

PROVIDER STRATEGY (three tiers)
------------------------------------------------------
1. Gemini (google-genai SDK) - PRIMARY. Google's Gemini API has a genuinely
   free tier (no credit card required, generous daily request quota on
   Flash models), which makes it the best default for a project graders
   or reviewers should be able to run without paying for anything. Tool
   calling is done manually (automatic_function_calling disabled) so we
   can dispatch through the same `execute_tool` used by every other path.
2. Anthropic (Claude) - SECONDARY. Used automatically if ANTHROPIC_API_KEY
   is set and either Gemini isn't configured or the Gemini call itself
   fails. Uses the exact same tool schema (translated to Claude's
   `input_schema` shape) so behavior is consistent across providers.
3. Rule-based fallback - TERTIARY. If neither LLM is reachable, unreachable
   mid-request, unauthenticated, or errors out, we do NOT want the whole
   endpoint to 500. A small keyword/regex classifier covers the same three
   actions with reduced NLU quality. This keeps /ask functional with zero
   external dependencies.

Both tool schemas are generated from one shared TOOL_SPECS list so the
three actions only need to be defined once.
"""
import os
import re
import traceback

from app.tools import execute_tool, get_employee_info, list_employees
from app.memory import memory

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

SYSTEM_PROMPT = """You are an internal enterprise assistant.
You can answer general questions and, when appropriate, take one of these
business actions using the tools available to you:
  - create_ticket: file an IT/HR/ops support ticket
  - get_employee_info: look up ONE specific employee's contact/department/manager info
  - list_employees: list ALL employees, optionally filtered by department
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

# ---------------------------------------------------------------------------
# Single source of truth for the three business-action tool schemas. Each
# provider's SDK wants a slightly different shape, so we generate both from
# this one JSON-Schema-ish spec instead of maintaining two copies by hand.
# ---------------------------------------------------------------------------
TOOL_SPECS = [
    {
        "name": "create_ticket",
        "description": "File a support/IT/HR ticket on behalf of the user. Use this when "
        "the user reports a problem, outage, request, or anything needing follow-up action.",
        "parameters": {
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
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Employee's full or partial name."},
                "employee_id": {"type": "string", "description": "Employee id, e.g. E101."},
            },
        },
    },
    {
        "name": "list_employees",
        "description": "List all employees, optionally filtered by department. Use this "
        "when the user asks to see/list all employees, or all employees in a department "
        "(e.g. 'list all employees', 'who works in Sales'). Do NOT use this for a single "
        "named person - use get_employee_info for that.",
        "parameters": {
            "type": "object",
            "properties": {
                "department": {
                    "type": "string",
                    "description": "Department to filter by, e.g. 'Engineering'. Leave blank to list everyone.",
                }
            },
        },
    },
    {
        "name": "generate_report",
        "description": "Generate a headcount and ticket summary report, optionally scoped "
        "to one department.",
        "parameters": {
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

# Anthropic wants {"name", "description", "input_schema"}.
ANTHROPIC_TOOLS = [
    {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
    for t in TOOL_SPECS
]

# Gemini wants {"name", "description", "parameters"} wrapped in a Tool /
# function_declarations list. Wrapped into a types.Tool lazily in
# AssistantEngine.__init__ once we know google-genai is importable.
GEMINI_FUNCTION_DECLARATIONS = [
    {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}
    for t in TOOL_SPECS
]


class AssistantEngine:
    def __init__(self):
        self.gemini_client = None
        self.gemini_tools = None
        self.anthropic_client = None
        self._last_employee_by_session = {}  # crude pronoun-resolution cache for fallback mode

        # --- Primary: Gemini (free tier) -----------------------------------
        gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if gemini_key:
            try:
                from google import genai
                from google.genai import types

                self.gemini_client = genai.Client(api_key=gemini_key)
                self.gemini_tools = [types.Tool(function_declarations=GEMINI_FUNCTION_DECLARATIONS)]
            except Exception as e:
                print(f"[startup] Gemini client init failed: {e}")
                traceback.print_exc()
                self.gemini_client = None
                self.gemini_tools = None

        # --- Secondary: Anthropic (optional) --------------------------------
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_key:
            try:
                import anthropic

                self.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
            except Exception as e:
                print(f"[startup] Anthropic client init failed: {e}")
                traceback.print_exc()
                self.anthropic_client = None

    @property
    def active_provider(self):
        """Which provider will be tried first, for /health reporting."""
        if self.gemini_client:
            return "gemini"
        if self.anthropic_client:
            return "anthropic"
        return None

    # ------------------------------------------------------------------
    def ask(self, question: str, session_id: str = "default") -> dict:
        history = memory.get_history(session_id)

        if self.gemini_client:
            try:
                return self._ask_gemini(question, history, session_id)
            except Exception as e:
                print("Gemini path failed, falling back:", e)
                traceback.print_exc()
                if self.anthropic_client:
                    try:
                        return self._ask_anthropic(question, history, session_id)
                    except Exception as e2:
                        print("Anthropic path also failed, falling back:", e2)
                        traceback.print_exc()
                        return self._ask_fallback(
                            question, session_id, error=f"gemini: {e} | anthropic: {e2}"
                        )
                return self._ask_fallback(question, session_id, error=str(e))

        if self.anthropic_client:
            try:
                return self._ask_anthropic(question, history, session_id)
            except Exception as e:
                print("Anthropic path failed, falling back:", e)
                traceback.print_exc()
                return self._ask_fallback(question, session_id, error=str(e))

        return self._ask_fallback(question, session_id)

    # ------------------------------------------------------------------
    # PRIMARY: Gemini
    # ------------------------------------------------------------------
    def _ask_gemini(self, question: str, history: list, session_id: str) -> dict:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=self.gemini_tools,
            # We dispatch tool calls ourselves through execute_tool() so
            # behavior (and mock-data side effects) is identical across
            # every provider - so automatic function calling is disabled.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        contents = self._history_to_gemini_contents(history)
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=question)]))

        response = self.gemini_client.models.generate_content(
            model=GEMINI_MODEL, contents=contents, config=config,
        )

        action_taken = None
        action_result = None

        # Loop in case the model wants to call a tool, see the result, and
        # then respond (usually a single extra round trip, looped
        # defensively in case the model chains tool calls).
        hops = 0
        while response.function_calls and hops < 5:
            hops += 1
            function_call_content = response.candidates[0].content
            contents.append(function_call_content)

            response_parts = []
            for fc in response.function_calls:
                action_taken = fc.name
                action_result = execute_tool(fc.name, dict(fc.args or {}))
                response_parts.append(
                    types.Part.from_function_response(
                        name=fc.name, response={"result": action_result}
                    )
                )
            contents.append(types.Content(role="tool", parts=response_parts))

            response = self.gemini_client.models.generate_content(
                model=GEMINI_MODEL, contents=contents, config=config,
            )

        final_text = (response.text or "").strip()

        memory.add_turn(session_id, "user", question)
        memory.add_turn(session_id, "assistant", final_text)

        return {
            "answer": final_text or "I've processed your request.",
            "action_taken": action_taken,
            "action_result": action_result,
            "mode": "llm",
            "provider": "gemini",
        }

    @staticmethod
    def _history_to_gemini_contents(history: list):
        """Convert our provider-agnostic {"role": "user"/"assistant", "content": str}
        memory turns into Gemini's Content objects (role must be user/model)."""
        from google.genai import types

        contents = []
        for turn in history:
            role = "model" if turn["role"] == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=turn["content"])]))
        return contents

    # ------------------------------------------------------------------
    # SECONDARY: Anthropic
    # ------------------------------------------------------------------
    def _ask_anthropic(self, question: str, history: list, session_id: str) -> dict:
        messages = history + [{"role": "user", "content": question}]

        response = self.anthropic_client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=ANTHROPIC_TOOLS,
            messages=messages,
        )

        action_taken = None
        action_result = None

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
            response = self.anthropic_client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=ANTHROPIC_TOOLS,
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
            "provider": "anthropic",
        }

    # ------------------------------------------------------------------
    # TERTIARY: rule-based fallback (no external API required)
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
        list_keywords = ["list all", "list employees", "all employees", "every employee",
                          "who works in", "show me all", "how many employees", "employee list"]
        employee_keywords = ["employee", "who is", "contact", "phone", "email",
                              "manager", "department of", "reach", "tell me about",
                              "info about", "information about", "details about",
                              "his email", "her email", "his phone", "her phone",
                              "title of", "location of", "who works"]
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

        elif any(k in q_lower for k in list_keywords):
            dept_match = re.search(r"(?:in|for)\s+([A-Za-z ]+?)(?:\s+department)?[\?\.]?$", q, re.IGNORECASE)
            department = dept_match.group(1).strip() if dept_match else ""
            action_taken = "list_employees"
            action_result = list_employees(department=department)
            names = ", ".join(f"{e['name']} ({e['title']})" for e in action_result["employees"][:15])
            scope = f" in {department}" if department else ""
            answer = f"There are {action_result['count']} employee(s){scope}: {names}."

        elif any(k in q_lower for k in employee_keywords) or self._has_unresolved_pronoun(q_lower):
            name = self._extract_name(q)
            if not name or name.strip().lower() == q.strip().lower():
                # No explicit name found (e.g. "What department does he work in?") -
                # fall back to whichever employee was last discussed in this session.
                name = self._last_employee_by_session.get(session_id, "")

            action_taken = "get_employee_info"
            action_result = get_employee_info(name=name) if name else {"found": False}

            if action_result.get("found"):
                emp = action_result["employee"]
                self._last_employee_by_session[session_id] = emp["name"]
                asking_manager_only = "manager" in q_lower and not any(
                    w in q_lower for w in ["email", "phone", "contact", "location", "title"]
                )
                if asking_manager_only:
                    answer = f"{emp['name']}'s manager is {emp.get('manager', 'not on file')}."
                else:
                    answer = (
                        f"{emp['name']} is a {emp['title']} in {emp['department']}, "
                        f"reporting to {emp.get('manager', 'N/A')}. "
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
            "provider": None,
        }
        if error:
            result["fallback_reason"] = error
        return result

    @staticmethod
    def _has_unresolved_pronoun(q_lower: str) -> bool:
        pronouns = [" he ", " him ", " his ", " she ", " her ", " they ", " them ", " their "]
        padded = f" {q_lower} "
        return any(p in padded for p in pronouns) and any(
            w in q_lower for w in ["department", "email", "phone", "manager", "office",
                                    "location", "title", "role", "work in", "reach"]
        )

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