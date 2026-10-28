import json, time, os
from pathlib import Path
import numpy as np, torch
from fastapi.testclient import TestClient
from . import api
from .engine import Engine, PRESETS

os.environ["XRAY_ACCESS_TOKEN"] = "local-test-only"
api.engine = Engine()
e = api.engine
client = TestClient(api.app)
headers = {
    "Authorization": "Bearer local-test-only",
    "X-Session-ID": "session-aaaaaaaaaaaa",
}
assert client.post("/api/analyze", json={"prompt": "hi"}).status_code == 401
result = client.post(
    "/api/analyze", headers=headers, json={"prompt": "The quick brown fox"}
)
assert result.status_code == 200, result.text
run = result.json()
id = run["id"]
assert client.get(f"/api/runs/{id}/head/0/0", headers=headers).status_code == 200
assert (
    client.get(
        f"/api/runs/{id}/head/0/0",
        headers=headers | {"X-Session-ID": "session-bbbbbbbbbbbb"},
    ).status_code
    == 410
)
assert client.get(f"/api/runs/{id}/head/12/0", headers=headers).status_code == 422
ablated = client.post(
    "/api/intervene", headers=headers, json={"run_id": id, "layer": 0, "head": 0}
)
assert ablated.status_code == 200
assert ablated.json()["probabilities"] != run["probabilities"]
restore = client.post(
    "/api/analyze", headers=headers, json={"prompt": run["prompt"]}
).json()
assert restore["probabilities"] == run["probabilities"]
assert (
    client.post(
        "/api/analyze", headers=headers, json={"prompt": " a" * 256}
    ).status_code
    == 422
)
for _ in range(8):
    client.post("/api/analyze", headers=headers, json={"prompt": "Once upon a"})
assert client.get(f"/api/runs/{id}/head/0/0", headers=headers).status_code == 410
latest = next(iter(api.sessions[headers["X-Session-ID"]]))
created, r = api.sessions[headers["X-Session-ID"]][latest]
api.sessions[headers["X-Session-ID"]][latest] = (created - 601, r)
assert client.get(f"/api/runs/{latest}/head/0/0", headers=headers).status_code == 410
_, tokens = e.ids("🙂 café  hello\n")
assert tokens[-1]["offset"][1] == len("🙂 café  hello\n".encode("utf-16-le")) // 2
latencies = {}
for length in [128, 256]:
    prompt = " hello" * (length - 1)
    timings = []
    for _ in range(30):
        start = time.perf_counter()
        r = e.run(prompt, compact=True)
        timings.append(time.perf_counter() - start)
    latencies[length] = {
        "runs": 30,
        "median_seconds": float(np.median(timings)),
        "p95_seconds": float(np.quantile(timings, 0.95)),
    }
    print(length, latencies[length], flush=True)
Path(__file__).resolve().parents[1].joinpath("docs/api-validation.json").write_text(
    json.dumps(
        {
            "checks": [
                "auth",
                "session isolation",
                "head range",
                "actual ablation",
                "exact restore",
                "token limit",
                "8-run eviction",
                "10-minute expiry",
                "UTF16 offsets",
            ],
            "latency_cpu_local": latencies,
        },
        indent=2,
    )
)
