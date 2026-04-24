"""FastAPI server exposing the standard OpenEnv HTTP interface for the
AI Agent Oversight Tribunal.

Routes
------
POST /reset          → Observation JSON
POST /step           → {observation, reward, done, info}
GET  /state          → State JSON
GET  /health         → {ok, version, round_index}
GET  /info           → env metadata
GET  /dashboard      → serves built React app
GET  /stream/rounds  → Server-Sent Events for live UI
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from tribunal.env import TribunalEnvironment
from tribunal.schemas import (
    Action,
    JudgeVerdict,
    Observation,
    State,
    verdict_from_json,
    ParseError,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tribunal.server")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VERSION = "1.0.0"

# Dashboard dist directory — try multiple locations for dev vs Docker
_DASHBOARD_CANDIDATES = [
    Path(__file__).resolve().parent.parent.parent / "dashboard" / "dist",  # dev: src/tribunal -> root
    Path("/app/dashboard/dist"),  # Docker
]
_DASHBOARD_DIR = next((p for p in _DASHBOARD_CANDIDATES if p.is_dir()), _DASHBOARD_CANDIDATES[0])

# ---------------------------------------------------------------------------
# Global environment + lock
# ---------------------------------------------------------------------------

_env: Optional[TribunalEnvironment] = None
_lock = asyncio.Lock()
_sse_subscribers: list[asyncio.Queue] = []


def _get_env() -> TribunalEnvironment:
    global _env
    if _env is None:
        import os
        _env = TribunalEnvironment(
            seed=int(os.environ.get("TRIBUNAL_SEED", "42")),
            episodes_per_reset=int(os.environ.get("TRIBUNAL_EPISODES", "5")),
            failure_rate=float(os.environ.get("TRIBUNAL_FAILURE_RATE", "0.6")),
        )
    return _env


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    import os
    port = os.environ.get("TRIBUNAL_PORT", "7860")
    host = os.environ.get("TRIBUNAL_HOST", "0.0.0.0")
    logger.info("Tribunal server starting  (version %s, %s:%s)", VERSION, host, port)
    _get_env()
    yield
    env = _get_env()
    env.close()
    logger.info("Tribunal server stopped")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Agent Oversight Tribunal",
    version=VERSION,
    description="OpenEnv environment for Multi-Agent Interactions hackathon",
    lifespan=lifespan,
)

# CORS — permissive for hackathon
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Structured request logger middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def request_logger(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s  → %d  (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Routes: OpenEnv interface
# ---------------------------------------------------------------------------

@app.post("/reset")
async def reset_env(request: Request, seed: Optional[int] = None) -> Dict[str, Any]:
    """Reset the environment and return the first Observation."""
    # Accept seed from query param or JSON body
    if seed is None:
        try:
            body = await request.json()
            seed = body.get("seed")
        except Exception:
            pass
    async with _lock:
        env = _get_env()
        obs = env.reset(seed=seed)
        return obs.model_dump()


@app.post("/step")
async def step_env(request: Request) -> Dict[str, Any]:
    """Process a Judge verdict and return the next round.

    Accepts either:
      - Raw JSON of JudgeVerdict fields
      - {"verdict": {...}} (Action wrapper)
    """
    async with _lock:
        env = _get_env()
        body = await request.json()

        parse_failed = False
        try:
            # Try to parse as Action first
            if "verdict" in body and isinstance(body["verdict"], dict):
                verdict_data = body["verdict"]
            else:
                verdict_data = body

            # Coerce int keys
            if "failure_types" in verdict_data and isinstance(verdict_data["failure_types"], dict):
                verdict_data["failure_types"] = {
                    int(k): v for k, v in verdict_data["failure_types"].items()
                }
            if "per_worker_confidence" in verdict_data and isinstance(
                verdict_data["per_worker_confidence"], dict
            ):
                verdict_data["per_worker_confidence"] = {
                    int(k): v for k, v in verdict_data["per_worker_confidence"].items()
                }

            verdict = JudgeVerdict.model_validate(verdict_data)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Invalid verdict: {e}")

        action = Action(verdict=verdict)

        try:
            obs, reward, done, info = env.step(action)
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Push to SSE subscribers
        if env.round_events:
            latest = env.round_events[-1]
            for q in _sse_subscribers:
                try:
                    q.put_nowait(latest)
                except asyncio.QueueFull:
                    pass

        return {
            "observation": obs.model_dump(),
            "reward": reward,
            "done": done,
            "info": info,
        }


@app.get("/state")
async def get_state() -> Dict[str, Any]:
    """Return the current environment state."""
    env = _get_env()
    return env.state().model_dump()


@app.get("/health")
async def health() -> Dict[str, Any]:
    """Health check endpoint."""
    env = _get_env()
    st = env.state()
    return {
        "ok": True,
        "version": VERSION,
        "round_index": st.round_index,
    }


@app.get("/info")
async def env_info() -> Dict[str, Any]:
    """Environment metadata for judges."""
    return {
        "name": "AI Agent Oversight Tribunal",
        "version": VERSION,
        "theme": "multi-agent-interactions",
        "description": (
            "Four specialised AI workers complete tasks. A hidden failure "
            "injector corrupts some outputs. The Judge agent must identify "
            "which workers misbehaved, classify the failure type, and "
            "explain its reasoning."
        ),
        "worker_roles": ["SUMMARISER", "TICKET_RESOLVER", "NEGOTIATOR", "RESEARCHER"],
        "failure_types": ["HALLUCINATION", "COLLUSION", "MANIPULATION", "SILENCE"],
        "reward_components": [
            "identification",
            "type_classification",
            "explanation_quality",
            "calibration",
            "false_positive_penalty",
            "anti_hack_penalty",
        ],
        "links": {
            "repo": "https://github.com/shreeyanshi123/tribunal-env",
            "openenv_spec": "https://openenv.dev/spec",
        },
    }


# ---------------------------------------------------------------------------
# SSE streaming
# ---------------------------------------------------------------------------

@app.get("/stream/rounds")
async def stream_rounds():
    """Server-Sent Events endpoint pushing completed round data."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=50)
    _sse_subscribers.append(queue)

    async def event_generator():
        try:
            while True:
                data = await queue.get()
                yield f"data: {json.dumps(data, default=str)}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            _sse_subscribers.remove(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Dashboard (static files)
# ---------------------------------------------------------------------------

if _DASHBOARD_DIR.is_dir():
    app.mount("/dashboard", StaticFiles(directory=str(_DASHBOARD_DIR), html=True), name="dashboard")
else:
    @app.get("/dashboard")
    async def dashboard_placeholder():
        return HTMLResponse(
            "<html><body><h1>Dashboard</h1>"
            "<p>No built dashboard found. Run <code>cd dashboard && npm run build</code></p>"
            "</body></html>"
        )
