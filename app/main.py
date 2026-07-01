from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.guardrails import validate_question
from app.llm_service import engine

app = FastAPI(
    title="Acme Corp Enterprise Assistant",
    description="AI-powered assistant that answers questions and performs business actions "
    "(ticket creation, employee lookup, reporting) via tool-calling with rule-based fallback.",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str = Field(..., description="The user's natural-language question or request.")
    session_id: Optional[str] = Field(
        "default", description="Optional session id to maintain conversation memory across calls."
    )


class AskResponse(BaseModel):
    answer: str
    action_taken: Optional[str] = None
    action_result: Optional[dict] = None
    mode: str
    session_id: str


@app.get("/")
def root():
    return {
        "service": "Acme Corp Enterprise Assistant",
        "status": "ok",
        "docs": "/docs",
        "endpoint": "POST /ask",
    }


@app.get("/health")
def health():
    return {"status": "healthy", "llm_connected": engine.client is not None}


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    try:
        cleaned_question = validate_question(payload.question)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid request: {e}")

    session_id = payload.session_id or "default"
    result = engine.ask(cleaned_question, session_id=session_id)

    return AskResponse(
        answer=result["answer"],
        action_taken=result.get("action_taken"),
        action_result=result.get("action_result"),
        mode=result.get("mode", "unknown"),
        session_id=session_id,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
