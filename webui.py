"""Browser UI for compare.py.

Same job as compare.py -- run one prompt through both SDKs and put the two
ledgers next to each other -- except the output is a page instead of a terminal
dump, so the itineraries are readable and the diff is visible at a glance.

    python webui.py            then open http://127.0.0.1:8000

Reads shared/settings.py like everything else, so TRIP_PROVIDER / TRIP_TOOLS /
TRIP_FAST_FAIL all apply here unchanged.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from adk_version.run import run_pipeline as adk_pipeline
from openai_version.run import run_pipeline as openai_pipeline
from shared import settings, trace

WEB = Path(__file__).resolve().parent / "web"

app = FastAPI(title="trip planner - SDK compare")

# trace.CALLS is ONE global list shared by both SDKs (that is the whole point of
# it), so two overlapping requests would interleave into each other's ledger.
# Serialise runs rather than pretend otherwise.
_RUN_LOCK = asyncio.Lock()


class RunRequest(BaseModel):
    prompt: str


async def _one(pipeline, prompt: str) -> dict:
    """Run a single pipeline in isolation and capture its ledger.

    reset-then-snapshot brackets each run for the same reason compare.py does
    it: skip either half and the second SDK's calls append onto the first.
    """
    trace.reset()
    started = time.perf_counter()
    try:
        text, ok = await pipeline(prompt), True
    except Exception as exc:                     # one side failing must not
        text, ok = trace.explain(exc), False     # hide the other. explain() digs
        #        the real cause out of ADK's TaskGroup ExceptionGroup
    return {
        "ok": ok,
        "text": text,
        "elapsed_ms": round((time.perf_counter() - started) * 1000),
        "calls": [asdict(c) for c in trace.snapshot()],
    }


@app.get("/api/config")
async def config() -> dict:
    return {"describe": settings.describe(), "tools": settings.TOOLS,
            "cooldown_s": settings.COOLDOWN_SECONDS}


@app.post("/api/run")
async def run(req: RunRequest) -> dict:
    async with _RUN_LOCK:
        adk = await _one(adk_pipeline, req.prompt)
        # Two full pipelines inside one minute exceed Groq's 8000 tokens/min.
        # See settings.COOLDOWN_SECONDS -- this is why a run takes ~90s.
        if settings.COOLDOWN_SECONDS:
            await asyncio.sleep(settings.COOLDOWN_SECONDS)
        openai = await _one(openai_pipeline, req.prompt)

    return {
        "prompt": req.prompt,
        "config": settings.describe(),
        "tools": settings.TOOLS,
        "cooldown_s": settings.COOLDOWN_SECONDS,
        "adk": adk,
        "openai": openai,
        "counts": {
            "adk": trace.call_counts([trace.Call(**c) for c in adk["calls"]]),
            "openai": trace.call_counts([trace.Call(**c) for c in openai["calls"]]),
        },
    }


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


if __name__ == "__main__":
    import uvicorn

    trace.configure_logging()
    print(f"config: {settings.describe()}")
    print("open http://127.0.0.1:8000")
    # reload=False on purpose: a reloader would re-import the agent modules and,
    # under TRIP_TOOLS=mcp, spawn a second set of server subprocesses.
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
