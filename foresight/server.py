"""FastAPI app: serves the chat page and streams agent events over SSE."""

from __future__ import annotations

import asyncio
import json
import os
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()

from . import sources  # noqa: E402
from .agent import Session, run_turn  # noqa: E402  (reads env at import)
from .forecaster import TimesFM  # noqa: E402
from .tools import ToolContext  # noqa: E402

WEB = Path(__file__).parent / "web"
MAX_SESSIONS = 200


class State:
    http = None
    model = TimesFM()
    model_ready = False
    llm: anthropic.AsyncAnthropic | None = None
    sessions: OrderedDict[str, Session] = OrderedDict()


state = State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.http = sources.new_client()
    state.llm = anthropic.AsyncAnthropic() if os.environ.get("ANTHROPIC_API_KEY") else None

    async def warm():
        # Loading the checkpoint takes a while; do it before the first question.
        await asyncio.to_thread(state.model.load)
        state.model_ready = True

    warming = asyncio.create_task(warm())
    yield
    warming.cancel()
    await state.http.aclose()


app = FastAPI(title="Foresight", lifespan=lifespan)


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=8, max_length=64)
    message: str = Field(min_length=1, max_length=2000)


def _session(session_id: str) -> Session:
    if session_id in state.sessions:
        state.sessions.move_to_end(session_id)
    else:
        state.sessions[session_id] = Session(ToolContext(http=state.http, model=state.model))
        while len(state.sessions) > MAX_SESSIONS:
            state.sessions.popitem(last=False)
    return state.sessions[session_id]


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


@app.post("/api/chat")
async def chat(req: ChatRequest):
    async def events():
        if state.llm is None:
            yield _sse(
                {
                    "type": "error",
                    "message": "No ANTHROPIC_API_KEY set. Add it to .env and restart the server.",
                }
            )
            return
        async for event in run_turn(_session(req.session_id), req.message, state.llm):
            yield _sse(event)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/health")
async def health():
    return {"model_ready": state.model_ready, "llm_configured": state.llm is not None}


@app.get("/")
async def index():
    return FileResponse(WEB / "index.html")


app.mount("/static", StaticFiles(directory=WEB), name="static")
