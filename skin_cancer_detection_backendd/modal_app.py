"""Deploy the skin triage API on Modal.

    pip install modal
    modal setup                 # log in once (opens the browser)
    modal deploy modal_app.py   # prints the public URL

The folder must contain model/ (deploy_config.json, inference.py and the three
model_*.pth files), next to app.py and explain.py.
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")  # CPU-only torch
    .env({"MODEL_DIR": "/root/model", "TORCH_THREADS": "2"})
    .add_local_dir("model", remote_path="/root/model")
    .add_local_python_source("app", "explain")
)

modal_app = modal.App("skin-triage", image=image)


@modal_app.function(
    cpu=1.0,               # one physical core (two vCPUs)
    memory=2048,           # MiB; the three models need about 1.1 GB
    scaledown_window=300,  # stay warm for 5 minutes after the last request
    max_containers=1,      # never run more than one copy, to protect the free credits
    timeout=120,
)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def web():
    import app as api  # the FastAPI app; its lifespan loads the models at startup

    return api.app
