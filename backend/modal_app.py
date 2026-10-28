"""CPU is sufficient for GPT-2. One bounded worker; no GPU allocation."""
from pathlib import Path
import modal
root=Path(__file__).resolve().parents[1]
image=(modal.Image.debian_slim(python_version='3.11').pip_install_from_requirements(str(root/'backend/requirements.txt')).env({'HF_HOME':'/cache','TOKENIZERS_PARALLELISM':'false','XRAY_MODAL_PILOT':'1'}).add_local_dir(str(root/'backend'),remote_path='/root/backend').add_local_file(str(root/'public/data/projection.json'),remote_path='/root/public/data/projection.json'))
app=modal.App('live-prompt-xray');volume=modal.Volume.from_name('live-prompt-xray-models',create_if_missing=True)
@app.function(image=image,cpu=4,memory=2048,max_containers=1,timeout=120,scaledown_window=60,volumes={'/cache':volume},secrets=[modal.Secret.from_name('live-prompt-xray-access')])
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def web():
 from backend.api import app
 return app
