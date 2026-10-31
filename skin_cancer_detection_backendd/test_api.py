import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app as api


def image_bytes(fmt="JPEG", size=(450, 600), seed=0):
    rng = np.random.RandomState(seed)
    img = Image.fromarray((rng.rand(size[1], size[0], 3) * 255).astype("uint8"))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    with TestClient(api.app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_root_redirects_to_docs(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "/docs"


def test_info(client):
    info = client.get("/info").json()
    assert set(info["models"]) == {"convnext_tiny", "maxvit_tiny", "swin_tiny"}
    assert 0 < info["defer_cutoff"] < 1


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_predict_formats(client, fmt):
    r = client.post("/predict", files={"file": (f"lesion.{fmt.lower()}", image_bytes(fmt), f"image/{fmt.lower()}")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["decision"] in {"benign", "malignant", "refer"}
    assert body["message"] == api.MESSAGES[body["decision"]]
    assert 0 <= body["malignancy_prob"] <= 1 and 0 <= body["defer_degree"] <= 1
    assert body["inference_ms"] > 0


def test_matches_inference_module(client):
    raw = image_bytes(seed=3)
    body = client.post("/predict", files={"file": ("a.jpg", raw, "image/jpeg")}).json()
    direct = api.state["triage"].predict(Image.open(io.BytesIO(raw)).convert("RGB"))[0]
    for k in ("decision", "malignancy_prob", "defer_degree", "model_probs"):
        assert body[k] == direct[k]


def test_rejects_non_image(client):
    r = client.post("/predict", files={"file": ("notes.txt", b"not an image", "text/plain")})
    assert r.status_code == 400


def test_rejects_large_file(client, monkeypatch):
    monkeypatch.setattr(api, "MAX_UPLOAD_MB", 0.001)
    r = client.post("/predict", files={"file": ("a.jpg", image_bytes(), "image/jpeg")})
    assert r.status_code == 413


def test_api_key(client, monkeypatch):
    monkeypatch.setattr(api, "API_KEY", "secret")
    files = {"file": ("a.jpg", image_bytes(), "image/jpeg")}
    assert client.post("/predict", files=files).status_code == 401
    files = {"file": ("a.jpg", image_bytes(), "image/jpeg")}
    assert client.post("/predict", files=files, headers={"x-api-key": "secret"}).status_code == 200


def test_explain(client):
    import base64 as b64
    raw = image_bytes(seed=5)
    pred = client.post("/predict", files={"file": ("a.jpg", raw, "image/jpeg")}).json()
    r = client.post("/explain", files={"file": ("a.jpg", raw, "image/jpeg")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["decision"] == pred["decision"] and body["malignancy_prob"] == pred["malignancy_prob"]
    ex = body["explanation"]
    assert ex["summary"] and ex["confidence"]["level"] in {"low", "medium", "high"}
    assert 1 <= len(ex["rules"]) <= 3 and all(0 < rule["strength"] <= 1 for rule in ex["rules"])
    cfg, inf = api.state["triage"].cfg, api.state["inference"]
    rebuilt = float(inf.fis(np.array([ex["confidence"]["value"]]), np.array([ex["disagreement"]["value"]]),
                            cfg["fis_params"])[0])
    assert abs(rebuilt - body["defer_degree"]) < 2e-3
    assert set(body["heatmaps_png_base64"]) == {"convnext_tiny", "maxvit_tiny", "swin_tiny", "fused"}
    for png in body["heatmaps_png_base64"].values():
        img = Image.open(io.BytesIO(b64.b64decode(png)))
        assert img.format == "PNG" and max(img.size) <= 448
