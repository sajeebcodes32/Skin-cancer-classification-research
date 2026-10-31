# inference.py - skin lesion triage (benign / malignant / refer)
import json, os
import numpy as np, torch, torch.nn as nn, timm
from PIL import Image
import torchvision.transforms as T

class SkinModel(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.backbone = timm.create_model(name, pretrained=False, num_classes=0)
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(self.backbone.num_features, 2))
    def forward(self, x):
        return self.head(self.backbone(x))

def _trimf(x, a, b, c):
    return np.clip(np.minimum((x - a) / (b - a + 1e-9), (c - x) / (c - b + 1e-9)), 0, 1)

def conf_input(p, thr, mode):
    if mode == "entropy":
        pc = np.clip(p, 1e-6, 1 - 1e-6)
        return np.clip(1 + (pc * np.log(pc) + (1 - pc) * np.log(1 - pc)) / np.log(2), 0, 1)
    if mode == "expected_error":
        e = np.where(p >= thr, 1 - p, p)
        return np.clip(1 - e / max(thr, 1 - thr), 0, 1)
    return np.clip(np.abs(p - thr) / max(thr, 1 - thr), 0, 1)

def fis(conf, dis, P):
    c = [_trimf(conf, *t) for t in P["cmf"]]; d = [_trimf(dis, *t) for t in P["dmf"]]
    num, den = 0.0, 0.0
    for i in range(3):
        for j in range(3):
            w = np.minimum(c[i], d[j]); num = num + w * P["Z"][i][j]; den = den + w
    return num / (den + 1e-9)

def defer_degree(p, X, thr, mode, P):
    return fis(conf_input(p, thr, mode), np.clip(X.std(1) / P["dscale"], 0, 1), P)

def _choquet(X, g, lam):
    idx = np.argsort(X, axis=1)
    Xs = np.take_along_axis(X, idx, axis=1); gs = g[idx]
    out = np.zeros(len(X)); prev = np.zeros(len(X))
    for i in range(X.shape[1]):
        tail = gs[:, i:]
        mu = tail.sum(1) if abs(lam) < 1e-8 else (np.prod(1 + lam * tail, axis=1) - 1) / lam
        out += (Xs[:, i] - prev) * np.clip(mu, 0, 1); prev = Xs[:, i]
    return np.clip(out, 0, 1)

class Triage:
    def __init__(self, folder, device=None):
        self.cfg = json.load(open(os.path.join(folder, "deploy_config.json")))
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.order = self.cfg["model_order"]
        self.models = {}
        for k in self.order:
            m = SkinModel(self.cfg["models"][k])
            m.load_state_dict(torch.load(os.path.join(folder, "model_" + k + ".pth"), map_location=self.dev))
            self.models[k] = m.to(self.dev).eval()
        s = self.cfg["img_size"]
        self.tfm = T.Compose([T.Resize((s, s)), T.ToTensor(), T.Normalize(self.cfg["mean"], self.cfg["std"])])

    @torch.no_grad()
    def model_probs(self, imgs):
        x = torch.stack([self.tfm(im.convert("RGB")) for im in imgs]).to(self.dev)
        views = [x] + ([torch.flip(x, [3]), torch.flip(x, [2]), torch.flip(x, [2, 3])] if self.cfg["tta"] else [])
        cols = []
        for k in self.order:
            ps = [torch.softmax(self.models[k](v), 1)[:, 1] for v in views]
            cols.append(torch.stack(ps, 0).mean(0).cpu().numpy())
        return np.stack(cols, 1)

    def fuse(self, X):
        f = self.cfg["fusion"]
        if f == "DE-weighted":
            return X @ np.array(self.cfg["de_weights"])
        if f == "Choquet (Sugeno)":
            return _choquet(X, np.array(self.cfg["sugeno_densities"]), self.cfg["sugeno_lambda"])
        if f == "stacking (LR)":
            z = X @ np.array(self.cfg["stacking_coef"]) + self.cfg["stacking_intercept"]
            return 1 / (1 + np.exp(-z))
        return X.mean(1)

    def predict(self, imgs):
        if not isinstance(imgs, (list, tuple)):
            imgs = [imgs]
        X = self.model_probs(imgs)
        p = self.fuse(X)
        thr = self.cfg["threshold"]
        dd = defer_degree(p, X, thr, self.cfg["fuzzy_conf"], self.cfg["fis_params"])
        out = []
        for i in range(len(p)):
            refer = bool(dd[i] >= self.cfg["defer_tau"])
            out.append(dict(
                decision="refer" if refer else ("malignant" if p[i] >= thr else "benign"),
                malignancy_prob=round(float(p[i]), 4),
                defer_degree=round(float(dd[i]), 4),
                model_probs={k: round(float(X[i, j]), 4) for j, k in enumerate(self.order)},
                model_disagreement=round(float(X[i].std()), 4)))
        return out
