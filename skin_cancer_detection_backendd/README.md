---
title: Skin Lesion Triage API
emoji: 🩺
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
pinned: false
short_description: Benign, malignant or refer for dermoscopic images
---

# Skin lesion triage API

A small web API around the trained system from `skin_cancer_triage.ipynb` (3 backbones, DE-weighted fusion, fuzzy defer degree, conformal cut-off). The mobile app and the web page send a photo to it and get back **benign**, **malignant** or **refer**, with the probabilities behind the decision.

Images are processed in memory and never saved.

## 1. Add the model files

Download `backend_bundle.zip` from the output of the Kaggle run (`/kaggle/working/backend_bundle.zip`) and unzip it into a folder called `model/`:

```
skin-triage-api/
├── app.py
├── Dockerfile
├── requirements.txt
└── model/
    ├── inference.py
    ├── deploy_config.json
    ├── model_convnext_tiny.pth
    ├── model_maxvit_tiny.pth
    └── model_swin_tiny.pth
```

## 2. Run it on your computer

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --port 7860
```

Open http://localhost:7860/docs, choose `POST /predict` → *Try it out*, and upload a HAM10000 image.

Check everything with `pip install pytest httpx && pytest test_api.py` (about one minute).

## 3. Put it online (existing Hugging Face Docker Space)

Upload this folder over an existing Docker Space (free CPU basic hardware), from inside the folder:

```bash
pip install -U huggingface_hub
hf auth login
hf upload <username>/<space-name> . . --repo-type=space --delete="*" --exclude=".gitignore" --exclude=".venv/*" --exclude="__pycache__/*" --exclude=".pytest_cache/*"
```

`--exclude=".gitignore"` matters: the Hub applies an uploaded `.gitignore`, which would skip the model files. The API is then at `https://<username>-<space-name>.hf.space`, with a test page at `/docs`. Optional, under *Settings → Variables and secrets*: set `API_KEY` so only your apps can call it, and `ALLOWED_ORIGINS` to your web page's address.

Free Spaces go to sleep after 48 hours without visitors, so the first request after a pause takes about a minute.

### Alternative: Modal

`modal_app.py` runs the same app on Modal (free Starter plan, $30 of compute credits a month). From inside this folder, with `model/` in place:

```bash
pip install modal
modal setup
modal deploy modal_app.py
```

Modal prints the public URL; the test page is at `<url>/docs`. The container uses 1 CPU core and 2 GiB of memory, starts when a request arrives, and shuts down after 5 idle minutes, so you pay only while it runs. To require an API key, create a Modal secret with `API_KEY` and add `secrets=[modal.Secret.from_name("<name>")]` to `@modal_app.function`.

## 4. GitHub

The three `.pth` files are over GitHub's 100 MB limit, so `.gitignore` leaves them out. Push the code and link to the Space for the weights. GitHub shows the settings block at the top of this file as a small table; that is expected.

## API

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | `{"status": "ok"}` when the models are loaded |
| GET | `/info` | models, fusion, threshold, cut-off, guarantee |
| POST | `/predict` | form field `file` = image (JPEG, PNG or WebP, up to 10 MB); header `x-api-key` if `API_KEY` is set |
| POST | `/explain` | same input as `/predict`; returns the prediction plus a plain-language explanation and Grad-CAM heatmaps (about 4 s on 2 CPUs) |

Example response:

```json
{
  "decision": "refer",
  "message": "The system is not confident about this lesion. Please have it examined by a dermatologist.",
  "malignancy_prob": 0.4881,
  "defer_degree": 0.831,
  "threshold": 0.365,
  "defer_cutoff": 0.76,
  "model_probs": {"convnext_tiny": 0.4757, "maxvit_tiny": 0.2962, "swin_tiny": 0.5925},
  "model_disagreement": 0.1219,
  "inference_ms": 2480.3,
  "disclaimer": "Research prototype for dermoscopic images. It does not provide a medical diagnosis."
}
```

Errors: `400` not an image, `401` wrong API key, `413` file too large.

### `/explain` (for the dermatologist's "Why?" screen)

Everything in `/predict`, plus:

- `explanation.summary`: one sentence, e.g. *"Referred to a dermatologist: confidence is low and the three models disagree strongly (standard deviation 0.19), so the defer degree 0.85 reached the cut-off 0.76."*
- `explanation.confidence` / `explanation.disagreement`: the two fuzzy inputs with their level (low, medium, high).
- `explanation.models_above_threshold`: which backbones call the lesion malignant.
- `explanation.rules`: the fuzzy rules that fired most strongly, with their strength and output.
- `heatmaps_png_base64`: Grad-CAM overlays (PNG, base64) for `convnext_tiny`, `maxvit_tiny`, `swin_tiny` and a `fused` map weighted like the fusion. They show where each model looked when scoring malignancy, not a clinical reason.

Flutter: `Image.memory(base64Decode(result['heatmaps_png_base64']['fused']))`.

**curl**

```bash
curl -F "file=@ISIC_0024306.jpg" https://<your-space>.hf.space/predict
```

**Flutter** (package `http`)

```dart
final req = http.MultipartRequest('POST', Uri.parse('$baseUrl/predict'))
  ..headers['x-api-key'] = apiKey
  ..files.add(await http.MultipartFile.fromPath('file', imagePath));
final res = await http.Response.fromStream(await req.send());
final result = jsonDecode(res.body);   // result['decision'], result['malignancy_prob'], ...
```

**Web (JavaScript)**

```js
const form = new FormData();
form.append("file", fileInput.files[0]);
const res = await fetch(`${baseUrl}/predict`, { method: "POST", body: form, headers: { "x-api-key": apiKey } });
const result = await res.json();
```

## Inference time for the paper

```bash
pip install requests
python benchmark.py --url https://<your-space>.hf.space --image ISIC_0024306.jpg
```

It prints the median time per image on the machine that serves the API. On a 2-CPU machine we measured about 2.5 s per image (3 backbones × 4 flipped views). Report the number you measure on your own deployment.

## Settings (environment variables)

| Name | Default | Meaning |
|---|---|---|
| `MODEL_DIR` | `./model` | folder with the unzipped bundle |
| `API_KEY` | empty (no key) | if set, requests must send it in `x-api-key` |
| `ALLOWED_ORIGINS` | `*` | comma-separated web origins allowed to call the API |
| `MAX_UPLOAD_MB` | `10` | largest accepted image |
| `TORCH_THREADS` | number of CPUs | CPU threads used for inference |
| `PORT` | `7860` | port inside Docker |
