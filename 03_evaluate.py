"""
Evaluate a trained SIDNet on the test set (Sec. 7).

Metrics
  * classification accuracy over the 16 classes
  * per-direction precision, recall and F1 of the suppression prediction (s_hat_d > 0.5)
  * topology-reconstruction agreement: share of windows in which the predicted class and
    the thresholded suppression output s_hat give the same four directions
  * noise-strength MAE over suppressed directions only, and exact (+-0.05) and
    loose (+-0.10) noise-strength accuracy, per class
  * accuracy per noise strength (uniform regime)
  * confusion matrix (rows = true class), saved as CSV

Usage
  python 03_evaluate.py --regime uni  --checkpoint checkpoints/uni_phaseB4_best.pt
  python 03_evaluate.py --regime nuni --checkpoint checkpoints/nuni_phaseB4_best.pt
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from sid_common import (ALPHA_TOL_EXACT, ALPHA_TOL_LOOSE, DIRS, N_CLASSES, N_DIRS, NOISE_NAMES)
from sid_data import SIDDataset
from sidnet import PATTERNS, SIDNet


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    patterns = PATTERNS.to(device)
    conf = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    tp = np.zeros(N_DIRS); fp = np.zeros(N_DIRS); fn = np.zeros(N_DIRS)
    agree = n = 0
    err = defaultdict(list)                       # class -> absolute errors of alpha_d (suppressed d)
    per_alpha = defaultdict(lambda: [0, 0])       # shared strength -> [correct, total] (uniform)
    for delta, ws, wl, y, alpha in loader:
        delta, ws, wl, y, alpha = (t.to(device) for t in (delta, ws, wl, y, alpha))
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits, s_hat, a_hat = model(delta, ws, wl)
        pred = logits.argmax(-1)
        bits_hat = s_hat.float() > 0.5
        bits_true = patterns[y].bool()
        agree += (patterns[pred].bool() == bits_hat).all(-1).sum().item()
        n += y.numel()
        tp += (bits_hat & bits_true).sum(0).cpu().numpy()
        fp += (bits_hat & ~bits_true).sum(0).cpu().numpy()
        fn += (~bits_hat & bits_true).sum(0).cpu().numpy()
        for yi, pi in zip(y.tolist(), pred.tolist()):
            conf[yi, pi] += 1
        e = (a_hat.float() - alpha).abs()
        for i in range(y.numel()):
            act = bits_true[i]
            if act.any():
                err[NOISE_NAMES[y[i].item()]].extend(e[i][act].tolist())
                vals = alpha[i][act]
                if torch.all(vals == vals[0]):
                    key = round(vals[0].item(), 1)
                    per_alpha[key][0] += int(pred[i] == y[i]); per_alpha[key][1] += 1

    prec = tp / np.maximum(tp + fp, 1); rec = tp / np.maximum(tp + fn, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    res = {
        "accuracy": float(np.trace(conf) / conf.sum()),
        "topology_reconstruction_agreement": agree / n,
        "per_direction": {d: {"precision": float(prec[k]), "recall": float(rec[k]), "f1": float(f1[k])}
                          for k, d in enumerate(DIRS)},
        "per_class": {},
        "per_alpha_accuracy": {f"{a:.1f}": c / t for a, (c, t) in sorted(per_alpha.items())},
        "n_windows": int(conf.sum()),
    }
    for k, name in enumerate(NOISE_NAMES):
        tot = int(conf[k].sum())
        if tot == 0:
            continue
        row = {"accuracy": float(conf[k, k] / tot), "windows": tot}
        if err[name]:
            e = np.array(err[name])
            row.update(mae=float(e.mean()), exact=float((e <= ALPHA_TOL_EXACT + 1e-9).mean()),
                       loose=float((e <= ALPHA_TOL_LOOSE + 1e-9).mean()))
        res["per_class"][name] = row
    return res, conf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regime", required=True, choices=["uni", "nuni"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--out_dir", default="results")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SIDNet().to(device)
    model.load_state_dict(torch.load(a.checkpoint, map_location=device)["model"])
    ds = SIDDataset(a.data_dir, a.regime, a.split, NOISE_NAMES)
    dl = DataLoader(ds, 64, shuffle=False, num_workers=a.workers)
    res, conf = evaluate(model, dl, device)

    print(f"\n{a.regime} {a.split}: accuracy {100 * res['accuracy']:.2f}% "
          f"({np.trace(conf)} of {conf.sum()}), topology-reconstruction agreement "
          f"{100 * res['topology_reconstruction_agreement']:.2f}%")
    for d, m in res["per_direction"].items():
        print(f"  {d}: precision {100 * m['precision']:.1f}%  recall {100 * m['recall']:.1f}%  "
              f"F1 {100 * m['f1']:.1f}%")
    print(f"  {'class':<9}{'acc':>8}{'MAE':>8}{'exact':>8}{'loose':>8}{'windows':>9}")
    for name, m in res["per_class"].items():
        extra = (f"{m['mae']:8.3f}{100 * m['exact']:7.1f}%{100 * m['loose']:7.1f}%"
                 if "mae" in m else " " * 24)
        print(f"  {name:<9}{100 * m['accuracy']:7.2f}%{extra}{m['windows']:9d}")
    if a.regime == "uni":
        print("  accuracy per noise strength:",
              ", ".join(f"{k}: {100 * v:.1f}%" for k, v in res["per_alpha_accuracy"].items()))

    os.makedirs(a.out_dir, exist_ok=True)
    stem = os.path.join(a.out_dir, f"{a.regime}_{a.split}")
    with open(stem + "_metrics.json", "w") as f:
        json.dump(res, f, indent=2)
    with open(stem + "_confusion.csv", "w") as f:
        f.write("true\\pred," + ",".join(NOISE_NAMES) + "\n")
        for k, name in enumerate(NOISE_NAMES):
            f.write(name + "," + ",".join(str(v) for v in conf[k]) + "\n")
    print(f"\nsaved {stem}_metrics.json and {stem}_confusion.csv")


if __name__ == "__main__":
    main()
