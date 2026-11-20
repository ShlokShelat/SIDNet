"""
Fig. 11: test-set confusion matrices of SIDNet, (a) uniform and (b) non-uniform regime.

Reads results/uni_test_confusion.csv and results/nuni_test_confusion.csv written by
03_evaluate.py. Rows are true classes, normalised to percent; classes are grouped by noise
family (standard; single; neighbouring pairs; opposite pairs; three directions; NSEW).

Usage
  python 04_plot_confusion.py --results results --out results/fig_confusion.pdf
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "mathtext.fontset": "dejavusans",
                     "pdf.fonttype": 42, "ps.fonttype": 42})
ORDER = ["standard", "N", "S", "E", "W", "NE", "NW", "SE", "SW", "NS", "EW", "NSE", "NSW", "NEW", "SEW", "NSEW"]
SHOW = ["std" if c == "standard" else c for c in ORDER]
BREAKS = [1, 5, 9, 11, 15]
CMAP = LinearSegmentedColormap.from_list("mono", ["#ffffff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"])


def read_confusion(path):
    with open(path) as f:
        rows = [line.strip().split(",") for line in f if line.strip()]
    names = rows[0][1:]
    M = np.array([[int(v) for v in r[1:]] for r in rows[1:]], float)
    idx = [names.index(c) for c in ORDER]
    return M[np.ix_(idx, idx)]


def panel(ax, M, title):
    P = 100 * M / np.maximum(M.sum(1, keepdims=True), 1)
    im = ax.imshow(P, cmap=CMAP, vmin=0, vmax=100, interpolation="none")
    n = len(ORDER)
    for i in range(n):
        for j in range(n):
            if P[i, j] >= 0.5 or i == j:
                ax.text(j, i, "%.0f" % P[i, j], ha="center", va="center", fontsize=5.6,
                        color="white" if P[i, j] > 55 else "#222222")
    for b in BREAKS:
        ax.axhline(b - 0.5, color="#555555", lw=0.5); ax.axvline(b - 0.5, color="#555555", lw=0.5)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(SHOW, rotation=90, fontsize=6.5); ax.set_yticklabels(SHOW, fontsize=6.5)
    ax.tick_params(length=0, pad=2)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
    ax.set_xlabel("predicted class"); ax.set_ylabel("true class")
    ax.set_title("%s, accuracy %.2f%%" % (title, 100 * np.trace(M) / M.sum()), fontsize=8, pad=4)
    return im


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--split", default="test")
    p.add_argument("--out", default="results/fig_confusion.pdf")
    a = p.parse_args()
    U = read_confusion(os.path.join(a.results, f"uni_{a.split}_confusion.csv"))
    N = read_confusion(os.path.join(a.results, f"nuni_{a.split}_confusion.csv"))
    W, H, s, y0, cb = 7.16, 3.75, 2.85, 0.62, 0.09
    x1 = 0.45; x2 = x1 + s + 0.55
    fig = plt.figure(figsize=(W, H))
    a1 = fig.add_axes([x1 / W, y0 / H, s / W, s / H]); a2 = fig.add_axes([x2 / W, y0 / H, s / W, s / H])
    im = panel(a1, U, "(a) uniform regime")
    panel(a2, N, "(b) non-uniform regime"); a2.set_ylabel("")
    cax = fig.add_axes([(x2 + s + 0.10) / W, y0 / H, cb / W, s / H])
    c = fig.colorbar(im, cax=cax); c.ax.set_title("%", fontsize=7, pad=3)
    c.ax.tick_params(labelsize=6.5, length=2); c.outline.set_linewidth(0.5)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
