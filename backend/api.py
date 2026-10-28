"""Private, serialized CPU inference. No prompts are written to disk or logs."""

import os, time, secrets, threading
from collections import OrderedDict
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, Field
from .engine import Engine

app = FastAPI(title="Live Prompt X-ray")
engine = None
lock = threading.RLock()
sessions = {}
calls = 0


class Prompt(BaseModel):
    prompt: str = Field(min_length=1, max_length=4096)


class Intervention(BaseModel):
    run_id: str
    layer: int = Field(ge=0, lt=12)
    head: int = Field(ge=0, lt=12)


def authorize(authorization: str = Header(""), x_session_id: str = Header("")):
    expected = os.environ.get("XRAY_ACCESS_TOKEN", "")
    if not expected or not secrets.compare_digest(authorization, "Bearer " + expected):
        raise HTTPException(401, "Connect with your private beta access token.")
    if not 16 <= len(x_session_id) <= 100:
        raise HTTPException(400, "A valid session ID is required.")
    return x_session_id


def prune():
    now = time.monotonic()
    for session in list(sessions):
        for id, (created, _) in list(sessions[session].items()):
            if now - created > 600:
                del sessions[session][id]
        if not sessions[session]:
            del sessions[session]


def getrun(session, id):
    prune()
    if id not in sessions.get(session, {}):
        raise HTTPException(
            410, "Run expired or belongs to another session. Analyze the prompt again."
        )
    return sessions[session][id][1]


def summarize(run):
    return {k: v for k, v in run.items() if k != "heads"} | {"heads": []}


def execute(session, prompt, ablation=None):
    global engine, calls
    with lock:
        prune()
        if os.environ.get("XRAY_MODAL_PILOT"):
            import modal

            counter = modal.Dict.from_name(
                "live-prompt-xray-pilot-v1", create_if_missing=True
            )
            calls = counter.get("calls", 0)
        if calls >= 50:
            raise HTTPException(
                429,
                "The 50-run pilot is complete. Review usage before reopening live inference.",
            )
        if session not in sessions and len(sessions) >= 4:
            raise HTTPException(
                503,
                "The private beta is full. Try again after an inactive session expires.",
            )
        calls += 1
        if os.environ.get("XRAY_MODAL_PILOT"):
            counter["calls"] = calls
        if engine is None:
            engine = Engine()
        try:
            result = engine.run(prompt, ablation, compact=True)
        except AssertionError as e:
            raise HTTPException(422, str(e) or "Input failed model validation.")
        cache = sessions.setdefault(session, OrderedDict())
        cache[result["id"]] = (time.monotonic(), result)
        while len(cache) > 8:
            cache.popitem(last=False)
        return summarize(result)


@app.post("/api/analyze")
def analyze(body: Prompt, session=Depends(authorize)):
    return execute(session, body.prompt)


@app.get("/api/runs/{run_id}/head/{layer}/{head}")
def head(run_id: str, layer: int, head: int, session=Depends(authorize)):
    if not 0 <= layer < 12 or not 0 <= head < 12:
        raise HTTPException(422, "Head index out of range.")
    with lock:
        data = getrun(session, run_id)["heads"][layer][head]
        return {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in data.items()}


@app.post("/api/intervene")
def intervene(body: Intervention, session=Depends(authorize)):
    with lock:
        run = getrun(session, body.run_id)
        if "intervention" in run:
            raise HTTPException(
                409, "Restore the clean run before another intervention."
            )
        result = execute(session, run["prompt"], (body.layer, body.head))
        result["baseline_id"] = body.run_id
        return result


@app.get("/api/health")
def health():
    return {"status": "ready", "model": "GPT-2", "mode": "private CPU beta"}


@app.middleware("http")
async def private_response(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response
