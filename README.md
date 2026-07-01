# Enterprise Assistant

An AI-powered internal assistant exposed as a single FastAPI endpoint,
`POST /ask`. It answers general questions and performs three business
actions — **create a support ticket**, **look up employee info**, and
**generate a report** — using Claude's tool-calling to decide *when* and
*how* to invoke them, with a rule-based fallback so the API never hard-fails
if the LLM is unreachable.

## 1. Architecture

```
Client
  │  POST /ask {"question": "...", "session_id": "..."}
  ▼
FastAPI (app/main.py)
  │  1. guardrails.validate_question()  -> reject empty/too-long/injection attempts
  ▼
AssistantEngine.ask()  (app/llm_service.py)
  │  2. load last N turns from ConversationMemory (app/memory.py)
  │
  ├── If ANTHROPIC_API_KEY set ──────────────────────────────────┐
  │      Claude (claude-sonnet-4-6) + 3 tool schemas              │
  │      (create_ticket / get_employee_info / generate_report)    │
  │      Model decides: answer directly, ask a clarifying         │
  │      question, or call a tool with structured arguments.      │
  │      -> execute_tool() runs it against mock JSON data          │
  │      -> tool result sent back to Claude to compose final reply│
  │                                                                 │
  └── Else, or if the API call throws ─────────────────────────────┘
         rule-based fallback: keyword intent classification +
         regex field extraction, same 3 actions, degraded NLU
  │
  ▼
Response: { answer, action_taken, action_result, mode, session_id }
```

**Business actions / mock data** (`app/tools.py`, `data/*.json`):
- `create_ticket(subject, description, priority, requested_by)` — appends to `data/tickets.json`, returns a generated ticket ID.
- `get_employee_info(name | employee_id)` — looks up `data/employees.json` (exact, substring, and fuzzy match).
- `generate_report(department?)` — aggregates headcount and ticket stats from the same mock data.

## 2. The mandatory "real engineering improvement": API / Tool Calling

**Naive v0** of this project decided which action to take with something
like `if "ticket" in question.lower(): create_ticket(...)`. That approach:
- breaks the moment phrasing changes ("my VPN is down" vs "file a ticket"),
- can't reliably extract structured fields (subject, priority, employee name),
- can't blend "answer + action" into one coherent, context-aware reply,
- has no way to ask a clarifying question when information is missing.

**What I changed:** the question (plus recent conversation history) is sent
to Claude along with three JSON-schema tool definitions. Claude decides,
based on meaning rather than keywords, whether a tool is needed and returns
typed, validated arguments for it. The result:

- Robust to free-form phrasing — "my VPN access broke this morning" is
  correctly understood as a ticket-worthy incident even with no keyword
  "ticket" in the sentence.
- The model can **ask a clarifying question instead of guessing** when a
  required field (like a ticket subject, or an unrecognized employee name)
  is missing — this doubles as request-validation at the semantic level.
- Adding a new business action is just adding one more tool schema + one
  more Python function — no branching logic to maintain.
- The tool result is fed back to Claude, which composes a natural final
  answer instead of returning raw JSON to the user.

I additionally layered **error handling / fallback logic** on top (see
`AssistantEngine.ask`): if the Anthropic client isn't configured, or the API
call throws for any reason (auth, network, rate limit), the endpoint falls
back to a small rule-based classifier that covers the same three actions.
This means `/ask` never 500s due to an upstream LLM outage, and it's also
what lets this whole project run and be graded with zero external API
dependencies (see Test Inputs below — both were run with no API key
configured, purely through the fallback path).

## 3. Guardrails (bonus, `app/guardrails.py`)

Lightweight, zero-latency, rule-based request validation runs before
anything touches the LLM: rejects empty/whitespace-only input, caps length
at 1500 chars, and blocks a small set of prompt-injection phrases
("ignore previous instructions", "reveal your system prompt", etc.) with a
400. This is on top of the tool-calling improvement above, not a
replacement for it — kept intentionally simple since the assignment asks
for one core improvement.

## 4. Test Inputs

Both were captured live against the running fallback path (no API key
configured in this environment) — see `test_client.py` to reproduce, or
run the curl commands below.

**Normal business query:**
```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "What is the email and phone number for John Smith?"}'
```
```json
{
  "answer": "John Smith is a Senior Backend Engineer in Engineering. Email: john.smith@acmecorp.com, Phone: +1-415-555-0192, Location: Remote - US.",
  "action_taken": "get_employee_info",
  "action_result": { "found": true, "employee": { "...": "..." } },
  "mode": "fallback"
}
```

**Challenging query** (no explicit "create a ticket" ask, implicit urgency,
requires the assistant to infer the correct action):
```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "My VPN access broke this morning and I cannot connect, please help ASAP"}'
```
```json
{
  "answer": "I've filed ticket TCK-0A63D9CF (priority: urgent) for: \"My VPN access broke this morning and I cannot connect, please help ASAP\". Someone from support will follow up.",
  "action_taken": "create_ticket",
  "action_result": { "ticket_id": "TCK-0A63D9CF", "priority": "urgent", "...": "..." },
  "mode": "fallback"
}
```

Two more edge cases are included in `test_client.py`: an empty-input
guardrail rejection (422) and a too-vague ticket request ("it broke") that
correctly triggers a clarifying question instead of filing a low-quality
ticket.

## 5. Running locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# optional — without this, the API runs fully functional in fallback mode
cp .env.example .env   # then add your ANTHROPIC_API_KEY

uvicorn app.main:app --reload
# -> http://localhost:8000/docs for interactive Swagger UI
```

Run the test script:
```bash
python test_client.py
```

## 6. Free deployment

**Option A — Render.com (recommended, free tier, `render.yaml` included):**
1. Push this folder to a new GitHub repo.
2. On [render.com](https://render.com) → New → Blueprint → connect the repo
   (it will read `render.yaml` automatically).
3. Add `ANTHROPIC_API_KEY` as an environment variable in the Render
   dashboard (optional — works without it).
4. Deploy. Render gives you a free `https://<name>.onrender.com` URL.

**Option B — Railway.app:** New Project → Deploy from GitHub repo → it
auto-detects the `Dockerfile` → add `ANTHROPIC_API_KEY` env var → deploy.

**Option C — Fly.io:** `fly launch` (detects the Dockerfile), then
`fly secrets set ANTHROPIC_API_KEY=...` and `fly deploy`.

All three have free tiers sufficient for a demo API.

## 7. Tradeoffs (for video discussion)

- **Speed vs. functionality:** the rule-based fallback answers instantly
  with zero external dependency, but has materially worse language
  understanding than the LLM tool-calling path — it can't handle phrasing
  outside its keyword lists. I kept both because the assignment values a
  working end-to-end demo over an assistant that's fully offline if the
  API key is missing/rate-limited.
- **Simplicity vs. scalability:** conversation memory is an in-process
  Python dict. It's trivial to reason about and needs zero infra, but it's
  lost on restart and won't work across multiple instances behind a load
  balancer — a production version would move this to Redis keyed by
  session/user.
- **Accuracy vs. development time:** I capped the guardrail layer at
  rule-based checks (length, empty input, a short injection-phrase list)
  rather than a second LLM call to classify malicious intent, trading some
  detection accuracy for near-zero added latency and complexity.

## 8. Project structure

```
enterprise-assistant/
├── app/
│   ├── main.py          # FastAPI app, /ask endpoint
│   ├── llm_service.py   # Claude tool-calling + fallback (core AI workflow)
│   ├── tools.py          # business actions against mock data
│   ├── memory.py          # conversation memory
│   └── guardrails.py     # request validation
├── data/
│   ├── employees.json    # mock employee directory
│   └── tickets.json      # mock ticket store (grows at runtime)
├── test_client.py        # scripted demo of the two required test inputs
├── requirements.txt
├── Dockerfile
├── render.yaml           # one-click free deploy on Render
└── .env.example
```
