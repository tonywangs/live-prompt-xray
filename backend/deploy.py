"""Create the private beta secret without printing it, then deploy."""

import os, secrets, subprocess
from pathlib import Path
import modal

root = Path(__file__).resolve().parents[1]
envfile = root / ".env.local"
if not envfile.exists():
    envfile.write_text("XRAY_ACCESS_TOKEN=" + secrets.token_urlsafe(32) + "\n")
    envfile.chmod(0o600)
values = dict(
    line.split("=", 1) for line in envfile.read_text().splitlines() if "=" in line
)
modal.Secret.objects.create(
    "live-prompt-xray-access",
    {"XRAY_ACCESS_TOKEN": values["XRAY_ACCESS_TOKEN"]},
    allow_existing=True,
)
subprocess.run(
    [os.sys.executable, "-m", "modal", "deploy", "backend/modal_app.py"],
    cwd=root,
    check=True,
)
