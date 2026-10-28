import json, time, urllib.request, urllib.error
from pathlib import Path

root = Path(__file__).resolve().parents[1]
token = dict(
    line.split("=", 1)
    for line in (root / ".env.local").read_text().splitlines()
    if "=" in line
)["XRAY_ACCESS_TOKEN"]
base = "https://tonywangs--live-prompt-xray-web.us-east.modal.run"


def request(path, data=None, auth=True):
    headers = {
        "Content-Type": "application/json",
        "X-Session-ID": "release-validation-session",
    }
    if auth:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(
        base + "/api/" + path,
        data=json.dumps(data).encode() if data else None,
        headers=headers,
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


try:
    request("analyze", {"prompt": "Once upon a"}, False)
    raise AssertionError("Anonymous inference allowed")
except urllib.error.HTTPError as e:
    assert e.code == 401
start = time.perf_counter()
run = request("analyze", {"prompt": "Once upon a"})
print(
    "Live result:",
    run["predictions"][0],
    "wall seconds:",
    round(time.perf_counter() - start, 2),
    flush=True,
)
head = request(f"runs/{run['id']}/head/2/6")
assert len(head["attention"]) == 4
ablated = request("intervene", {"run_id": run["id"], "layer": 2, "head": 6})
assert ablated["probabilities"] != run["probabilities"]
print("Live intervention verified", flush=True)
(root / "docs/remote-validation.json").write_text(
    json.dumps(
        {
            "anonymous_inference": "401",
            "head_endpoint": "passed",
            "ablation_changes_output": True,
            "provenance": run["provenance"],
        },
        indent=2,
    )
)
