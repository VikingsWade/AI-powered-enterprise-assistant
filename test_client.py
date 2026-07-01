"""
Demonstrates the two required test inputs against a running instance of the API.

Usage:
    python -m uvicorn app.main:app --reload   # in one terminal
    python test_client.py                      # in another terminal
"""
import json
import httpx

BASE_URL = "http://localhost:8000"


def call(question, session_id="test"):
    r = httpx.post(f"{BASE_URL}/ask", json={"question": question, "session_id": session_id}, timeout=30)
    print(f"HTTP {r.status_code}")
    print(json.dumps(r.json(), indent=2))
    print("-" * 70)


if __name__ == "__main__":
    print("=== TEST 1: Normal business query (employee lookup) ===")
    call("What is the email and phone number for John Smith?", session_id="normal-query")

    print("\n=== TEST 2: Challenging query (implicit action, urgency, no explicit 'ticket' ask) ===")
    call(
        "My VPN access broke this morning and I cannot connect, please help ASAP",
        session_id="challenging-query",
    )

    print("\n=== BONUS: Guardrail - invalid/empty input ===")
    call("", session_id="invalid-input")

    print("\n=== BONUS: Ambiguous/incomplete request -> clarifying question, no bad action taken ===")
    call("it broke", session_id="ambiguous-input")
