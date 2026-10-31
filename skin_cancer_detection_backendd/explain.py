import base64
import io
import math

import numpy as np
import torch
from PIL import Image

LEVELS = ("low", "medium", "high")
NAMES = {"convnext_tiny": "ConvNeXt-Tiny", "maxvit_tiny": "MaxViT-Tiny", "swin_tiny": "Swin-Tiny"}
TARGET_LAYERS = {
    "convnext": lambda bb: bb.stages[-1].blocks[-1].conv_dw,
    "maxvit": lambda bb: bb.stages[-1].blocks[-1],
    "swin": lambda bb: bb.layers[-1].blocks[-1].norm2,
}


def fuzzy_explanation(inference, cfg, result):
    """Rebuild the fuzzy inference for one case and describe it in words."""
    order = cfg["model_order"]
    probs = np.array([result["model_probs"][k] for k in order])
    p, thr, P = result["malignancy_prob"], cfg["threshold"], cfg["fis_params"]
    c = float(inference.conf_input(np.array([p]), thr, cfg["fuzzy_conf"])[0])
    d = float(np.clip(probs.std() / P["dscale"], 0, 1))
    mc = [float(inference._trimf(c, *t)) for t in P["cmf"]]
    md = [float(inference._trimf(d, *t)) for t in P["dmf"]]
    rules = []
    for i in range(3):
        for j in range(3):
            w = min(mc[i], md[j])
            if w > 0:
                rules.append(dict(rule=f"IF confidence is {LEVELS[i]} AND disagreement is {LEVELS[j]}",
                                  strength=round(w, 3), defer_output=round(float(P["Z"][i][j]), 3)))
    rules.sort(key=lambda r: -r["strength"])
    c_level, d_level = LEVELS[int(np.argmax(mc))], LEVELS[int(np.argmax(md))]
    above = [NAMES.get(k, k) for k in order if result["model_probs"][k] >= thr]
    agree = {"low": "agree closely", "medium": "partly disagree", "high": "disagree strongly"}[d_level]

    dec = result["decision"]
    if dec == "refer":
        summary = (f"Referred to a dermatologist: confidence is {c_level} and the three models {agree} "
                   f"(standard deviation {probs.std():.2f}), so the defer degree {result['defer_degree']:.2f} "
                   f"reached the cut-off {cfg['defer_tau']:.2f}.")
    elif dec == "malignant":
        summary = (f"Reported as suspicious: the combined malignancy probability {p:.2f} is above the "
                   f"threshold {thr:.2f}; {len(above)} of 3 models are above it; confidence is {c_level} "
                   f"and the models {agree}.")
    else:
        summary = (f"Reported as likely benign: the combined malignancy probability {p:.2f} is below the "
                   f"threshold {thr:.2f}; {3 - len(above)} of 3 models are below it; confidence is {c_level} "
                   f"and the models {agree}.")
    return dict(summary=summary,
                confidence=dict(value=round(c, 3), level=c_level),
                disagreement=dict(value=round(d, 3), level=d_level, std=round(float(probs.std()), 4)),
                models_above_threshold=above,
                rules=rules[:3])


def _to_nchw(a):
    if a.dim() == 3:
        s = int(math.isqrt(a.shape[1]))
        a = a.reshape(a.shape[0], s, s, a.shape[2])
        return a.permute(0, 3, 1, 2)
    if a.shape[1] < a.shape[-1]:
        return a.permute(0, 3, 1, 2)
    return a


def gradcam(model, layer, x):
    store = {}

    def keep(module, inputs, output):
        store["act"] = output
        output.register_hook(lambda g: store.__setitem__("grad", g))

    handle = layer.register_forward_hook(keep)
    try:
        with torch.enable_grad():
            model.zero_grad(set_to_none=True)
            model(x)[0, 1].backward()
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
    act, grad = _to_nchw(store["act"].detach()), _to_nchw(store["grad"].detach())
    cam = torch.relu((grad.mean(dim=(2, 3), keepdim=True) * act).sum(1))[0].cpu().numpy()
    return cam / (cam.max() + 1e-8)


def _jet(v):
    return np.stack([np.clip(1.5 - np.abs(4 * v - k), 0, 1) for k in (3, 2, 1)], -1)


def _overlay_png(img, cam, max_side=448):
    scale = min(1.0, max_side / max(img.size))
    size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
    base = np.asarray(img.convert("RGB").resize(size, Image.BILINEAR), dtype=np.float32) / 255
    heat = np.asarray(Image.fromarray(cam.astype(np.float32), mode="F").resize(size, Image.BILINEAR))
    out = (0.55 * base + 0.45 * _jet(np.clip(heat, 0, 1))) * 255
    buf = io.BytesIO()
    Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def heatmaps(triage, img):
    """Grad-CAM for the malignant class of each backbone (original view), plus a fused map
    weighted like the fusion. Shows where the models looked, not a clinical reason."""
    x = triage.tfm(img.convert("RGB")).unsqueeze(0).to(triage.dev)
    cfg, order = triage.cfg, triage.cfg["model_order"]
    weights = np.array(cfg["de_weights"]) if cfg["fusion"] == "DE-weighted" else np.ones(len(order)) / len(order)
    cams, out = [], {}
    for k in order:
        family = next(f for f in TARGET_LAYERS if f in k)
        cam = gradcam(triage.models[k], TARGET_LAYERS[family](triage.models[k].backbone), x)
        cams.append(np.asarray(Image.fromarray(cam.astype(np.float32), mode="F").resize((224, 224), Image.BILINEAR)))
        out[k] = _overlay_png(img, cams[-1])
    fused = sum(w * c for w, c in zip(weights, cams))
    out["fused"] = _overlay_png(img, fused / (fused.max() + 1e-8))
    return out
