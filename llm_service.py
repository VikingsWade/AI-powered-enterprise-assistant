"""
Simple in-process conversation memory, keyed by session_id.

For a take-home project this is intentionally in-memory (a Python dict).
In production this would move to Redis / a database keyed by user or
session so it survives restarts and works across multiple API instances -
see README "Tradeoffs" section for the reasoning.
"""
from collections import defaultdict
from threading import Lock

MAX_TURNS_KEPT = 6  # last N user/assistant exchanges kept per session


class ConversationMemory:
    def __init__(self):
        self._store = defaultdict(list)
        self._lock = Lock()

    def add_turn(self, session_id: str, role: str, content: str):
        with self._lock:
            self._store[session_id].append({"role": role, "content": content})
            # Keep only the most recent turns to bound token usage
            if len(self._store[session_id]) > MAX_TURNS_KEPT * 2:
                self._store[session_id] = self._store[session_id][-MAX_TURNS_KEPT * 2 :]

    def get_history(self, session_id: str):
        with self._lock:
            return list(self._store[session_id])

    def clear(self, session_id: str):
        with self._lock:
            self._store.pop(session_id, None)


memory = ConversationMemory()
