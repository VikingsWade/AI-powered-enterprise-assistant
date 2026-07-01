# Enterprise Assistant

An AI-powered internal assistant exposed as a single FastAPI endpoint,
`POST /ask`. It answers general questions and performs three business
actions — **create a support ticket**, **look up employee info**, and
**generate a report** — using an LLM's tool-calling to decide *when* and
*how* to invoke them, with a rule-based fallback so the API never hard-fails
if no LLM is reachable.

The primary LLM provider is **Gemini** (`google-genai` SDK), chosen because
Google's free tier requires no credit card and gives ~1,500 requests/day on
Flash — so anyone can clone this repo and run the full LLM tool-calling path
for $0. Claude (Anthropic) is wired in as an optional secondary provider.

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
  ├── 1. PRIMARY: If GEMINI_API_KEY set ───────────────────────────────┐
  │      Gemini (gemini-2.5-flash, google-genai SDK) + 3 tool schemas  │
  │      (create_ticket / get_employee_info / generate_report)         │
  │      Model decides: answer directly, ask a clarifying              │
  │      question, or call a tool with structured arguments.           │
  │      -> execute_tool() runs it against mock JSON data              │
  │      -> tool result sent back to Gemini to compose final reply     │
  │                                                                    │
  ├── 2. SECONDARY: If Gemini absent/fails, and ANTHROPIC_API_KEY set─ ┤
  │      Same flow via Claude (claude-sonnet-4-6), same tool schemas   │
  │                                                                    │
  └── 3. TERTIARY: If neither LLM reachable ───────────────────────────┘
         rule-based fallback: keyword intent classification +
         regex field extraction, same 3 actions, degraded NLU
  │
  ▼
Response: { answer, action_taken, action_result, mode, provider, session_id }
```

**Business actions / mock data** (`app/tools.py`, `data/*.json`):
- `create_ticket(subject, description, priority, requested_by)` — appends to `data/tickets.json`, returns a generated ticket ID.
- `get_employee_info(name | employee_id)` — looks up `data/employees.json` (exact, substring, and fuzzy match), returns full profile including manager.
- `list_employees(department?)` — lists all employees, optionally scoped to one department.
- `generate_report(department?)` — aggregates headcount and ticket stats from the same mock data.

## 2. The mandatory "real engineering improvement": API / Tool Calling

**Naive v0** of this project decided which action to take with something
like `if "ticket" in question.lower(): create_ticket(...)`. That approach:
- breaks the moment phrasing changes ("my VPN is down" vs "file a ticket"),
- can't reliably extract structured fields (subject, priority, employee name),
- can't blend "answer + action" into one coherent, context-aware reply,
- has no way to ask a clarifying question when information is missing.

**What I changed:** the question (plus recent conversation history) is sent
to an LLM along with three JSON-schema tool definitions. The model decides,
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
- The tool result is fed back to the model, which composes a natural final
  answer instead of returning raw JSON to the user.

**Multi-provider, not single-vendor:** the three tool schemas are defined
once (`TOOL_SPECS` in `app/llm_service.py`) and translated into whichever
shape each provider's SDK wants — Gemini's `function_declarations` and
Claude's `input_schema`. This means the tool-calling logic isn't locked to
one vendor's API, and it's what makes the three-tier fallback in the
architecture diagram above possible: Gemini first (free), Claude second
(if configured), rule-based logic last (always available).

I additionally layered **error handling / fallback logic** on top (see
`AssistantEngine.ask`): if neither LLM client is configured, or an API call
throws for any reason (auth, network, rate limit), the endpoint falls back a
tier — down to a small rule-based classifier that covers the same three
actions if both LLMs are unavailable. This means `/ask` never 500s due to an
upstream LLM outage, and it's also what lets this whole project run and be
graded with zero external API dependencies (see Test Inputs below — both
were run with no API key configured, purely through the fallback path).

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
cp .env.example .env
# then add your GEMINI_API_KEY (free, no credit card: https://aistudio.google.com/apikey)
# ANTHROPIC_API_KEY is optional and only used as a secondary provider

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
3. Add `GEMINI_API_KEY` (and optionally `ANTHROPIC_API_KEY`) as environment
   variables in the Render dashboard — both are optional, the API works
   without them via the rule-based fallback.
4. Deploy. Render gives you a free `https://<name>.onrender.com` URL.

**Option B — Railway.app:** New Project → Deploy from GitHub repo → it
auto-detects the `Dockerfile` → add `GEMINI_API_KEY` env var → deploy.

**Option C — Fly.io:** `fly launch` (detects the Dockerfile), then
`fly secrets set GEMINI_API_KEY=...` and `fly deploy`.

All three have free tiers sufficient for a demo API.

## 7. Tradeoffs (for video discussion)

- **Speed vs. functionality:** the rule-based fallback answers instantly
  with zero external dependency, but has materially worse language
  understanding than the LLM tool-calling path — it can't handle phrasing
  outside its keyword lists. I kept both because the assignment values a
  working end-to-end demo over an assistant that's fully offline if the
  API key is missing/rate-limited.
- **Free tier vs. best-in-class model:** Gemini Flash is the primary
  provider specifically because it's free with no credit card, which
  matters for a project meant to be easy to clone and grade — but it's not
  necessarily the strongest model available. Claude is wired in as a
  same-schema secondary provider for anyone who wants to trade the free
  tier for a different model's tool-calling quality.
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
│   ├── llm_service.py   # Gemini (primary) + Claude (secondary) tool-calling
│   │                     # + rule-based fallback (core AI workflow)
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