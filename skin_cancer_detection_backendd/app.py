import importlib.util
import io
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Literal

import torch
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

import explain

MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "model"))
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "10"))
API_KEY = os.getenv("API_KEY", "")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
TORCH_THREADS = int(os.getenv("TORCH_THREADS", str(os.cpu_count() or 1)))

Image.MAX_IMAGE_PIXELS = 50_000_000

DISCLAIMER = "Research prototype for dermoscopic images. It does not provide a medical diagnosis."
MESSAGES = {
    "benign": "Rated likely benign by this research system. This is not a diagnosis: see a doctor if the "
              "lesion changes, bleeds or itches.",
    "malignant": "Features suggestive of a malignant lesion. Please see a dermatologist promptly.",
    "refer": "The system is not confident about this lesion. Please have it examined by a dermatologist.",
}

state = {}
lock = threading.Lock()


def load_triage(folder):
    spec = importlib.util.spec_from_file_location("inference", os.path.join(folder, "inference.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state["inference"] = module
    return module.Triage(folder, device="cuda" if torch.cuda.is_available() else "cpu")


@asynccontextmanager
async def lifespan(app):
    torch.set_num_threads(TORCH_THREADS)
    triage = load_triage(MODEL_DIR)
    triage.predict(Image.new("RGB", (224, 224), (180, 140, 120)))
    state["triage"] = triage
    yield
    state.clear()


app = FastAPI(title="Skin lesion triage API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"])


class Prediction(BaseModel):
    decision: Literal["benign", "malignant", "refer"]
    message: str
    malignancy_prob: float
    defer_degree: float
    threshold: float
    defer_cutoff: float
    model_probs: Dict[str, float]
    model_disagreement: float
    inference_ms: float
    disclaimer: str


class Level(BaseModel):
    value: float
    level: str


class Disagreement(Level):
    std: float


class Rule(BaseModel):
    rule: str
    strength: float
    defer_output: float


class Explanation(BaseModel):
    summary: str
    confidence: Level
    disagreement: Disagreement
    models_above_threshold: List[str]
    rules: List[Rule]


class ExplainedPrediction(Prediction):
    explanation: Explanation
    heatmaps_png_base64: Dict[str, str]
    heatmap_note: str


def check_key(key):
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def read_image(upload):
    limit = int(MAX_UPLOAD_MB * 1024 * 1024)
    raw = upload.read(limit + 1)
    if len(raw) > limit:
        raise HTTPException(status_code=413, detail=f"Image is larger than {MAX_UPLOAD_MB:g} MB.")
    try:
        Image.open(io.BytesIO(raw)).verify()
        img = Image.open(io.BytesIO(raw))
        img.load()
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, SyntaxError):
        raise HTTPException(status_code=400, detail="The file is not a readable image (JPEG, PNG or WebP).")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    return {"status": "ok" if "triage" in state else "loading"}


@app.get("/info")
def info():
    cfg = state["triage"].cfg
    return {
        "models": cfg["model_order"],
        "fusion": cfg["fusion"],
        "threshold": cfg["threshold"],
        "defer_cutoff": cfg["defer_tau"],
        "confidence_input": cfg["fuzzy_conf"],
        "guarantee": f"target: error of automatically reported cases at most {cfg['alpha']:.0%} "
                     f"with {1 - cfg['delta']:.0%} confidence (approximate; on data like HAM10000)",
        "test_time_augmentation": cfg["tta"],
        "disclaimer": DISCLAIMER,
    }


@app.post("/predict", response_model=Prediction)
def predict(file: UploadFile = File(...), x_api_key: str = Header(default="")):
    check_key(x_api_key)
    img = read_image(file.file)
    triage = state["triage"]
    with lock:
        t0 = time.perf_counter()
        r = triage.predict(img)[0]
        ms = (time.perf_counter() - t0) * 1000
    return Prediction(**r, message=MESSAGES[r["decision"]], threshold=triage.cfg["threshold"],
                      defer_cutoff=triage.cfg["defer_tau"], inference_ms=round(ms, 1), disclaimer=DISCLAIMER)


@app.post("/explain", response_model=ExplainedPrediction)
def explain_case(file: UploadFile = File(...), x_api_key: str = Header(default="")):
    check_key(x_api_key)
    img = read_image(file.file)
    triage = state["triage"]
    with lock:
        t0 = time.perf_counter()
        r = triage.predict(img)[0]
        maps = explain.heatmaps(triage, img)
        ms = (time.perf_counter() - t0) * 1000
    return ExplainedPrediction(
        **r, message=MESSAGES[r["decision"]], threshold=triage.cfg["threshold"],
        defer_cutoff=triage.cfg["defer_tau"], inference_ms=round(ms, 1), disclaimer=DISCLAIMER,
        explanation=explain.fuzzy_explanation(state["inference"], triage.cfg, r),
        heatmaps_png_base64=maps,
        heatmap_note="Grad-CAM for the malignant class: warmer colours mark the image regions that most "
                     "increased each model's malignancy score. It shows where the models looked, not a "
                     "clinical reason, and can highlight artefacts such as hairs or ruler marks.")
