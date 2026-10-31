import argparse
import statistics
import time

import requests

parser = argparse.ArgumentParser(description="Measure inference time per image for the paper.")
parser.add_argument("--url", default="http://localhost:7860")
parser.add_argument("--image", required=True, help="a dermoscopic image, e.g. one HAM10000 test image")
parser.add_argument("--n", type=int, default=30)
parser.add_argument("--api-key", default="")
args = parser.parse_args()

raw = open(args.image, "rb").read()
headers = {"x-api-key": args.api_key} if args.api_key else {}
server_ms, total_ms = [], []
for i in range(args.n + 3):
    t0 = time.perf_counter()
    r = requests.post(f"{args.url}/predict", files={"file": ("img.jpg", raw, "image/jpeg")}, headers=headers, timeout=120)
    r.raise_for_status()
    if i >= 3:
        total_ms.append((time.perf_counter() - t0) * 1000)
        server_ms.append(r.json()["inference_ms"])


def summary(v):
    v = sorted(v)
    return f"median {statistics.median(v):.0f} ms | p90 {v[int(0.9 * (len(v) - 1))]:.0f} ms"


print(f"{args.n} requests to {args.url}")
print(f"model inference (3 backbones x 4 views + fusion + fuzzy triage): {summary(server_ms)}")
print(f"full request including upload: {summary(total_ms)}")
print("last result:", r.json()["decision"], r.json()["malignancy_prob"])
