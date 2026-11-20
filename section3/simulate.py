#!/usr/bin/env python3
"""
Illustration runs of Sec. 3 (500 x 500 lattice, single central injection site)
==============================================================================

    Δ(r,c) = ( z̄_noisy(r,c) − z̄_ref(r,c) ) / ( z̄_ref(r,c) + 1 )

    z̄ = time-averaged height: grid summed after every grain's full relaxation
        (from grain --avg-start on) and divided by the number of samples.

PHYSICS  (Secs. 2 and 3.1)
  (size+2)² lattice with 1-cell absorbing boundary, pre-filled at BG,
  grains dropped one-by-one at cell full//2, full relaxation after each,
  synchronous toppling (threshold 4).  A grain sent in a noisy direction
  survives with probability α = 1 − noise, otherwise it is removed.

  GPU exactness: grains are added to a stable grid (h ≤ 3), so during
  relaxation h ≤ 7 and every cell topples at most once per wave; one random
  draw per toppling cell per noisy direction is therefore exact.  Without
  noise the result is bit-identical to the original Numba engine.

RUNS  (102 unique)
  standard                       1
  uniform  15 direction sets × 6 levels (0.01 0.02 0.05 0.10 0.30 0.50)   90
  mixed    the 11 non-uniform configurations of Sec. 3                  11

COMPARISONS
  vs_standard : every noisy run  vs  standard                (90 + 11)
  chain       : uniform  0.01→0.02→0.05→0.10→0.30→0.50 per direction set (75)
                mixed    each config vs the config with one fewer noisy
                         direction (2-dir configs vs uniform single-dir 0.01)  (11)

VIEWS for every comparison set
  clipped   high-contrast Δ, colour limit = --clip percentile of |Δ|
  sign      loss / no change / gain  (|Δ| > --thr)
  symlog    symmetric-log scale (small and large changes together)
  asymNS    Δ − (Δ mirrored N↔S about the drop site)
  asymEW    Δ − (Δ mirrored E↔W about the drop site)
  + z̄ maps, final snapshots, profiles, histograms, large single panels,
    summary.csv / summary.json, overview plot.

HPC USAGE
  # everything in one job
  python section3/simulate.py --mode all
  # split across a SLURM array (e.g. 4 GPUs), then analyse
  python section3/simulate.py --mode simulate --part $SLURM_ARRAY_TASK_ID --nparts 4
  python section3/simulate.py --mode analyze
  # progress
  python section3/simulate.py --mode status
  # quick test
  python section3/simulate.py --mode all --grid 60 --grains 2000 --out test_out

  Finished runs are skipped on restart; unfinished batches resume from the
  last checkpoint (saved every --ckpt-min minutes).

Requirements: torch (CUDA), numpy, matplotlib
"""

import os, re, csv, json, time, glob, math, pickle, hashlib, argparse
from collections import OrderedDict
import numpy as np

DIRS = ["N", "S", "E", "W"]

# ═══════════════════════════════════════════════════════════════════════════
# RUN DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════
UNIFORM_SETS = ["N", "S", "E", "W",
                "NS", "NE", "NW", "SE", "SW", "EW",
                "NSE", "NSW", "NEW", "SEW",
                "NSEW"]
UNIFORM_LEVELS = [0.01, 0.02, 0.05, 0.10, 0.30, 0.50]

# name → noise (1 - alpha_d) per direction: the 11 non-uniform configurations
MIXED = OrderedDict([
    ("F_N_0.01_S_0.05",               {"N": 0.01, "S": 0.05}),
    ("F_N_0.01_E_0.05",               {"N": 0.01, "E": 0.05}),
    ("F_N_0.01_W_0.05",               {"N": 0.01, "W": 0.05}),
    ("F_S_0.01_E_0.05",               {"S": 0.01, "E": 0.05}),
    ("F_S_0.01_W_0.05",               {"S": 0.01, "W": 0.05}),
    ("F_E_0.01_W_0.05",               {"E": 0.01, "W": 0.05}),
    ("F_N_0.01_S_0.05_E_0.10",        {"N": 0.01, "S": 0.05, "E": 0.10}),
    ("F_N_0.01_S_0.05_W_0.10",        {"N": 0.01, "S": 0.05, "W": 0.10}),
    ("F_S_0.01_E_0.05_W_0.10",        {"S": 0.01, "E": 0.05, "W": 0.10}),
    ("F_E_0.01_W_0.05_N_0.10",        {"E": 0.01, "W": 0.05, "N": 0.10}),
    ("F_N_0.01_S_0.05_E_0.10_W_0.30", {"N": 0.01, "S": 0.05, "E": 0.10, "W": 0.30}),
])

# mixed chain: config → config with one fewer noisy direction
MIXED_PARENT = OrderedDict([
    ("F_N_0.01_S_0.05",               "U_N_0.01"),
    ("F_N_0.01_E_0.05",               "U_N_0.01"),
    ("F_N_0.01_W_0.05",               "U_N_0.01"),
    ("F_S_0.01_E_0.05",               "U_S_0.01"),
    ("F_S_0.01_W_0.05",               "U_S_0.01"),
    ("F_E_0.01_W_0.05",               "U_E_0.01"),
    ("F_N_0.01_S_0.05_E_0.10",        "F_N_0.01_S_0.05"),
    ("F_N_0.01_S_0.05_W_0.10",        "F_N_0.01_S_0.05"),
    ("F_S_0.01_E_0.05_W_0.10",        "F_S_0.01_E_0.05"),
    ("F_E_0.01_W_0.05_N_0.10",        "F_E_0.01_W_0.05"),
    ("F_N_0.01_S_0.05_E_0.10_W_0.30", "F_N_0.01_S_0.05_E_0.10"),
])


def uname(dset, nl):
    return f"U_{dset}_{nl:.2f}"


def build_runs():
    """name → noise dict (empty = standard)."""
    runs = OrderedDict()
    runs["standard"] = {}
    for d in UNIFORM_SETS:
        for nl in UNIFORM_LEVELS:
            runs[uname(d, nl)] = {c: nl for c in d}
    for name, nd in MIXED.items():
        runs[name] = dict(nd)
    return runs


def check_definitions(runs):
    for child, parent in MIXED_PARENT.items():
        c, p = runs[child], runs[parent]
        extra = set(c) - set(p)
        assert len(extra) == 1 and all(c[k] == p[k] for k in p), \
            f"chain {parent} → {child} does not add exactly one direction"


def alphas_of(noise):
    return np.array([1.0 - noise.get(d, 0.0) for d in DIRS], dtype=np.float64)


def pretty(name, runs):
    if name == "standard":
        return "standard"
    nd = runs[name]
    if name.startswith("U_"):
        return f"({''.join(nd.keys())}, {next(iter(nd.values())):.2f})"
    return " ".join(f"{d}{nd[d]:.2f}" for d in DIRS if d in nd)


# ═══════════════════════════════════════════════════════════════════════════
# SIMULATION BACKENDS
# ═══════════════════════════════════════════════════════════════════════════

class TorchPile:
    """B sandpiles stepped in lock-step on a torch device (CUDA/MPS/CPU)."""

    def __init__(self, alphas, size, bg, seed, device):
        import torch
        self.T, self.dev = torch, device
        B, f = len(alphas), size + 2
        self.cx = f // 2
        self.mask = torch.zeros((1, f, f), dtype=torch.int32, device=device)
        self.mask[:, 1:-1, 1:-1] = 1
        self.grid = torch.full((B, f, f), bg, dtype=torch.int32, device=device) * self.mask
        self.acc = torch.zeros((B, f, f), dtype=torch.int32, device=device)
        a = torch.tensor(np.asarray(alphas), dtype=torch.float32, device=device)
        self.noisy = [d for d in range(4) if (np.asarray(alphas)[:, d] < 1.0).any()]
        self.alpha = {d: a[:, d].view(B, 1, 1) for d in self.noisy}
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(int(seed))

    def wave(self):
        T, g = self.T, self.grid
        t = g >= 4
        ti = t.to(T.int32)
        g -= 4 * ti
        if self.noisy:
            r = T.rand((len(self.noisy),) + tuple(g.shape),
                       device=self.dev, generator=self.gen)
        s = []
        j = 0
        for d in range(4):
            if d in self.alpha:
                s.append((t & (r[j] < self.alpha[d])).to(T.int32)); j += 1
            else:
                s.append(ti)
        g[:, :-1, :] += s[0][:, 1:, :]     # N: (r,c) → (r-1,c)
        g[:, 1:, :]  += s[1][:, :-1, :]    # S: (r,c) → (r+1,c)
        g[:, :, 1:]  += s[2][:, :, :-1]    # E: (r,c) → (r,c+1)
        g[:, :, :-1] += s[3][:, :, 1:]     # W: (r,c) → (r,c-1)
        g *= self.mask                     # absorbing boundary

    def unstable(self):
        return bool((self.grid >= 4).any())

    def drop(self):
        self.grid[:, self.cx, self.cx] += 1

    def accumulate(self):
        self.acc += self.grid

    def max_height(self):
        return int(self.grid.max())

    def results(self, n_avg):
        z = (self.acc.double() / max(n_avg, 1)).cpu().numpy()[:, 1:-1, 1:-1]
        fin = self.grid.cpu().numpy()[:, 1:-1, 1:-1].astype(np.int8)
        return z, fin

    def get_state(self):
        return {"grid": self.grid.cpu().numpy(), "acc": self.acc.cpu().numpy(),
                "rng": self.gen.get_state().cpu().numpy()}

    def set_state(self, st):
        T = self.T
        self.grid = T.from_numpy(st["grid"]).to(self.dev)
        self.acc = T.from_numpy(st["acc"]).to(self.dev)
        self.gen.set_state(T.from_numpy(st["rng"]))


class NumpyPile:
    """CPU fallback / testing backend with the same interface."""

    def __init__(self, alphas, size, bg, seed, device=None):
        B, f = len(alphas), size + 2
        self.cx = f // 2
        self.mask = np.zeros((1, f, f), np.int32); self.mask[:, 1:-1, 1:-1] = 1
        self.grid = np.full((B, f, f), bg, np.int32) * self.mask
        self.acc = np.zeros((B, f, f), np.int32)
        al = np.asarray(alphas)
        self.noisy = [d for d in range(4) if (al[:, d] < 1.0).any()]
        self.alpha = {d: al[:, d].reshape(B, 1, 1) for d in self.noisy}
        self.rng = np.random.default_rng(int(seed))

    def wave(self):
        g = self.grid
        t = g >= 4
        ti = t.astype(np.int32)
        g -= 4 * ti
        s = [((t & (self.rng.random(g.shape) < self.alpha[d])).astype(np.int32)
              if d in self.alpha else ti) for d in range(4)]
        g[:, :-1, :] += s[0][:, 1:, :]
        g[:, 1:, :]  += s[1][:, :-1, :]
        g[:, :, 1:]  += s[2][:, :, :-1]
        g[:, :, :-1] += s[3][:, :, 1:]
        g *= self.mask

    def unstable(self):
        return bool((self.grid >= 4).any())

    def drop(self):
        self.grid[:, self.cx, self.cx] += 1

    def accumulate(self):
        self.acc += self.grid

    def max_height(self):
        return int(self.grid.max())

    def results(self, n_avg):
        z = (self.acc / max(n_avg, 1))[:, 1:-1, 1:-1]
        return z, self.grid[:, 1:-1, 1:-1].astype(np.int8)

    def get_state(self):
        return {"grid": self.grid.copy(), "acc": self.acc.copy(),
                "rng": self.rng.bit_generator.state}

    def set_state(self, st):
        self.grid, self.acc = st["grid"].copy(), st["acc"].copy()
        self.rng.bit_generator.state = st["rng"]


def make_pile(alphas, args, seed):
    dev = args.device
    if dev == "numpy":
        return NumpyPile(alphas, args.grid, args.bg, seed), "numpy (CPU)"
    import torch
    if dev == "auto":
        dev = ("cuda" if torch.cuda.is_available() else
               "mps" if torch.backends.mps.is_available() else "cpu")
    name = dev
    if dev.startswith("cuda"):
        name += f" ({torch.cuda.get_device_name(torch.device(dev))})"
    return TorchPile(alphas, args.grid, args.bg, seed, torch.device(dev)), name


def relax(pile):
    """Run waves until stable.  Waves on a stable grid are no-ops, so we run
    them in growing chunks and only synchronise with the host between chunks."""
    chunk = 4
    while True:
        for _ in range(chunk):
            pile.wave()
        if not pile.unstable():
            return
        chunk = min(chunk * 2, 256)


# ═══════════════════════════════════════════════════════════════════════════
# SIMULATION DRIVER  (parts, batches, checkpoints, resume)
# ═══════════════════════════════════════════════════════════════════════════

def run_file(out, name, seed_idx):
    return os.path.join(out, "runs", f"{name}__s{seed_idx}.npz")


def all_jobs(runs, args):
    sel = re.compile(args.select) if args.select else None
    jobs = []
    for name in runs:
        if sel and not sel.search(name):
            continue
        nseeds = 1 if name == "standard" else args.seeds   # standard is deterministic
        for s in range(nseeds):
            jobs.append((name, s))
    return jobs


def simulate(runs, args):
    os.makedirs(os.path.join(args.out, "runs"), exist_ok=True)
    ck_dir = os.path.join(args.out, "checkpoints")
    os.makedirs(ck_dir, exist_ok=True)

    jobs = all_jobs(runs, args)[args.part::args.nparts]
    todo = [j for j in jobs if not os.path.exists(run_file(args.out, *j))]
    batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]

    print("=" * 72)
    print(f"  SIMULATE  part {args.part}/{args.nparts}:  {len(jobs)} jobs, "
          f"{len(jobs) - len(todo)} done, {len(todo)} to run in {len(batches)} batch(es)")
    print(f"  grid {args.grid}  grains {args.grains:,}  BG {args.bg}  "
          f"avg from grain {args.avg_start}  seeds/run {args.seeds}")
    print("=" * 72, flush=True)

    for bi, batch in enumerate(batches):
        ids = [f"{n}__s{s}" for n, s in batch]
        key = hashlib.md5(("|".join(ids) +
                           f"|{args.grid}|{args.grains}|{args.bg}|{args.avg_start}|{args.seed}"
                           ).encode()).hexdigest()[:12]
        ck_path = os.path.join(ck_dir, f"ckpt_{key}.pkl")
        alphas = np.stack([alphas_of(runs[n]) for n, _ in batch])
        pile, devname = make_pile(alphas, args, args.seed + int(key, 16) % 1_000_000)

        start = 0
        if os.path.exists(ck_path):
            with open(ck_path, "rb") as f:
                st = pickle.load(f)
            pile.set_state(st)
            start = st["i"]
            print(f"  resumed from checkpoint at grain {start:,}")

        print(f"\n  Batch {bi + 1}/{len(batches)}  ({len(batch)} runs on {devname})")
        print("   " + ", ".join(ids[:6]) + (" …" if len(ids) > 6 else ""), flush=True)

        t0, t_ck = time.time(), time.time()
        report = max(1, args.grains // 40)
        for i in range(start, args.grains):
            pile.drop()
            relax(pile)
            if i >= args.avg_start:
                pile.accumulate()
            done = i + 1
            if done % report == 0 or done == args.grains:
                el = time.time() - t0
                eta = el / (done - start) * (args.grains - done)
                print(f"    grain {done:>9,}/{args.grains:,}   {el / 60:7.1f} min   "
                      f"ETA {eta / 60:7.1f} min", flush=True)
            if time.time() - t_ck > args.ckpt_min * 60 and done < args.grains:
                st = pile.get_state(); st["i"] = done
                with open(ck_path + ".tmp", "wb") as f:
                    pickle.dump(st, f)
                os.replace(ck_path + ".tmp", ck_path)
                t_ck = time.time()

        assert pile.max_height() <= 3, "grid not stable at end of run"
        z, fin = pile.results(args.grains - args.avg_start)
        for k, (name, s) in enumerate(batch):
            np.savez_compressed(run_file(args.out, name, s), zbar=z[k], final=fin[k],
                                alphas=alphas[k], grid=args.grid, grains=args.grains,
                                bg=args.bg, avg_start=args.avg_start)
        if os.path.exists(ck_path):
            os.remove(ck_path)
        print(f"  Batch {bi + 1} saved ({(time.time() - t0) / 60:.1f} min)", flush=True)


def status(runs, args):
    jobs = all_jobs(runs, args)
    done = [j for j in jobs if os.path.exists(run_file(args.out, *j))]
    print(f"{len(done)}/{len(jobs)} run-seeds finished in {args.out}/runs")
    missing = sorted({n for n, s in jobs if (n, s) not in set(done)})
    if missing:
        print("missing:", ", ".join(missing))


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def load_runs(runs, args):
    data = {}
    for name in runs:
        files = sorted(glob.glob(os.path.join(args.out, "runs", f"{name}__s*.npz")))
        if not files:
            continue
        zs, fin = [], None
        for fp in files:
            d = np.load(fp)
            if int(d["grid"]) != args.grid or int(d["grains"]) != args.grains:
                print(f"  WARNING: {fp} was made with grid={int(d['grid'])}, "
                      f"grains={int(d['grains'])}; skipped")
                continue
            zs.append(d["zbar"])
            if fin is None:
                fin = d["final"]
        if zs:
            data[name] = {"zbar": np.mean(zs, axis=0), "final": fin, "nseeds": len(zs),
                          "zbar_std": np.std(zs, axis=0) if len(zs) > 1 else None}
    return data


def delta(zn, zr):
    return (zn - zr) / (zr + 1.0)


def box_smooth(a, k):
    if k <= 1:
        return a
    p = k // 2
    c = np.pad(np.pad(a, p, mode="edge").cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def build_groups(runs):
    groups = []
    for d in UNIFORM_SETS:
        names = [uname(d, nl) for nl in UNIFORM_LEVELS]
        groups.append({
            "folder": os.path.join("uniform", d),
            "title": f"Uniform noise, directions {d}",
            "sets": OrderedDict([
                ("vs_standard", [(n, "standard") for n in names]),
                ("chain", [(names[i + 1], names[i]) for i in range(len(names) - 1)]),
            ]),
            "context": ["standard"] + names})
    mixed = list(MIXED.keys())
    groups.append({
        "folder": "mixed",
        "title": "Mixed (per-direction) noise",
        "sets": OrderedDict([
            ("vs_standard", [(n, "standard") for n in mixed]),
            ("chain", [(c, p) for c, p in MIXED_PARENT.items()]),
        ]),
        "context": ["standard"] + mixed})
    return groups


class Geometry:
    def __init__(self, size):
        self.c0 = (size + 2) // 2 - 1                  # drop site in inner coords
        self.h = min(self.c0, size - 1 - self.c0)      # mirror window half-width

    def window(self, a):
        c, h = self.c0, self.h
        return a[c - h:c + h + 1, c - h:c + h + 1]

    def asym(self, d, axis):
        w = self.window(d)
        return w - (w[::-1, :] if axis == "NS" else w[:, ::-1])


def comp_stats(d, zn, zr, geo, thr):
    c = geo.c0
    north, south = d[:c].mean(), d[c + 1:].mean()
    west, east = d[:, :c].mean(), d[:, c + 1:].mean()
    return {"mean": d.mean(), "mean_abs": np.abs(d).mean(),
            "min": d.min(), "max": d.max(),
            "frac_loss": (d < -thr).mean(), "frac_gain": (d > thr).mean(),
            "north_mean": north, "south_mean": south,
            "east_mean": east, "west_mean": west,
            "NS_index": north - south, "EW_index": east - west,
            "mass_ratio": zn.sum() / zr.sum()}


def analyze(runs, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap, SymLogNorm

    DIV = LinearSegmentedColormap.from_list("strong_div", [
        "#001a66", "#0050c8", "#4da3ff", "#ffffff", "#ff8040", "#d01010", "#5c0000"])
    SIGN = ListedColormap(["#0050c8", "#f2f2f2", "#d01010"])
    PALETTE = np.array([[13, 27, 42], [0, 180, 216], [255, 183, 3], [255, 0, 110]],
                       dtype=np.uint8)
    VIEWS = ["clipped", "sign", "symlog", "asymNS", "asymEW"]

    data = load_runs(runs, args)
    print(f"\n  ANALYZE: {len(data)}/{len(runs)} runs available")
    if "standard" not in data:
        print("  standard run missing — nothing to compare against.")
        return
    geo = Geometry(args.grid)
    root = os.path.join(args.out, "analysis")
    rows_csv = []

    def P(n):
        return pretty(n, runs)

    # ── one multi-panel figure ─────────────────────────────────────────
    def grid_fig(items, view, suptitle, path):
        n = len(items)
        ncol = min(n, args.ncols)
        nrow = math.ceil(n / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 4.6 * nrow),
                                 squeeze=False)
        if view in ("asymNS", "asymEW"):
            arrs = [geo.asym(d, view[-2:]) for _, d in items]
        elif view == "sign":
            arrs = [np.sign(d) * (np.abs(d) > args.thr) for _, d in items]
        else:
            arrs = [d for _, d in items]
        flat = np.abs(np.concatenate([a.ravel() for a in arrs]))

        if view == "sign":
            kw = dict(cmap=SIGN, vmin=-1.5, vmax=1.5)
            cl = f"blue: Δ<−{args.thr}   white: |Δ|≤{args.thr}   red: Δ>+{args.thr}"
        elif view == "symlog":
            v = flat.max() or 1e-9
            kw = dict(cmap=DIV, norm=SymLogNorm(linthresh=args.linthresh,
                                                vmin=-v, vmax=v, base=10))
            cl = f"Δ  (symmetric log, linear within ±{args.linthresh})"
        else:
            v = np.percentile(flat, args.clip) or 1e-9
            kw = dict(cmap=DIV, vmin=-v, vmax=v)
            what = {"clipped": "Δ",
                    "asymNS": "Δ − Δ(N↔S mirror)",
                    "asymEW": "Δ − Δ(E↔W mirror)"}[view]
            cl = f"{what}  (±{v:.3g} = {args.clip:g}th pct)"

        for k, ax in enumerate(axes.ravel()):
            if k >= n:
                ax.axis("off"); continue
            im = ax.imshow(arrs[k], interpolation="nearest", **kw)
            t = items[k][0]
            if view == "sign":
                d = items[k][1]
                t += f"\nloss {100*(d < -args.thr).mean():.1f}%  " \
                     f"gain {100*(d > args.thr).mean():.1f}%"
            ax.set_title(t, fontsize=9.5)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_ylabel("N ↑", rotation=0, labelpad=10, fontsize=8)
        cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02,
                          extend="both" if view not in ("sign", "symlog") else "neither")
        cb.set_label(cl)
        if view == "sign":
            cb.set_ticks([-1, 0, 1]); cb.set_ticklabels(["loss", "≈0", "gain"])
        fig.suptitle(suptitle, fontsize=12)
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

    overview = {}   # uniform vs_standard stats for the overview plot

    for G in build_groups(runs):
        gdir = os.path.join(root, G["folder"])
        os.makedirs(os.path.join(gdir, "panels"), exist_ok=True)
        deltas = OrderedDict()

        for sname, pairs in G["sets"].items():
            items = []
            for noisy, ref in pairs:
                if noisy not in data or ref not in data:
                    print(f"  skip {noisy} vs {ref} (run missing)")
                    continue
                zn, zr = data[noisy]["zbar"], data[ref]["zbar"]
                d = box_smooth(delta(zn, zr), args.smooth)
                label = f"{P(noisy)}\nvs {P(ref)}"
                items.append((label, d))
                st = comp_stats(d, zn, zr, geo, args.thr)
                rows_csv.append({"group": G["folder"], "comparison": sname,
                                 "noisy": noisy, "reference": ref,
                                 "seeds_noisy": data[noisy]["nseeds"],
                                 "seeds_ref": data[ref]["nseeds"],
                                 **{k: float(v) for k, v in st.items()}})
                if sname == "vs_standard" and noisy.startswith("U_"):
                    overview.setdefault(G["folder"].split(os.sep)[-1], []).append(
                        (runs[noisy][next(iter(runs[noisy]))], st))
                v = np.percentile(np.abs(d), args.clip) or 1e-9
                plt.imsave(os.path.join(gdir, "panels", f"{sname}__{noisy}__vs__{ref}.png"),
                           np.kron(d, np.ones((4, 4))), cmap=DIV, vmin=-v, vmax=v)
            deltas[sname] = items
            if not items:
                continue
            ttl = {"vs_standard": "every run vs standard sandpile",
                   "chain": "each run vs the previous one"}[sname]
            for view in VIEWS:
                grid_fig(items, view,
                         f"{G['title']}  —  {ttl}  —  view: {view}   "
                         f"Δ = (z̄ − z̄_ref)/(z̄_ref + 1)",
                         os.path.join(gdir, f"{sname}_{view}.png"))

        # ── context: z̄ maps and final snapshots ──────────────────────────
        ctx = [n for n in G["context"] if n in data]
        n = len(ctx); ncol = min(n, args.ncols); nrow = math.ceil(n / ncol)
        for kind in ("zbar", "final"):
            fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 4.3 * nrow),
                                     squeeze=False)
            for k, ax in enumerate(axes.ravel()):
                if k >= n:
                    ax.axis("off"); continue
                if kind == "zbar":
                    im = ax.imshow(data[ctx[k]]["zbar"], cmap="magma", vmin=0, vmax=3,
                                   interpolation="nearest")
                else:
                    ax.imshow(PALETTE[np.clip(data[ctx[k]]["final"], 0, 3)],
                              interpolation="nearest")
                ax.set_title(P(ctx[k]), fontsize=10)
                ax.set_xticks([]); ax.set_yticks([])
            if kind == "zbar":
                fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02,
                             label="z̄ (time-averaged height)")
            fig.suptitle(f"{G['title']}  —  "
                         + ("time-averaged height z̄" if kind == "zbar"
                            else "final configuration (seed 0)"), fontsize=12)
            fig.savefig(os.path.join(gdir, f"{kind}_maps.png" if kind == "zbar"
                                     else "final_snapshots.png"),
                        dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)

        # ── profiles ────────────────────────────────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
        x = np.arange(args.grid) - geo.c0
        for col, sname in enumerate(["vs_standard", "chain"]):
            for lab, d in deltas.get(sname, []):
                lab1 = lab.replace("\n", " ")
                axes[0, col].plot(x, d.mean(axis=1), lw=1.2, label=lab1)
                axes[1, col].plot(x, d.mean(axis=0), lw=1.2, label=lab1)
            axes[0, col].set_title(f"{sname}: row-mean Δ  (negative offset = North)")
            axes[1, col].set_title(f"{sname}: column-mean Δ  (negative offset = West)")
            for a in axes[:, col]:
                a.axhline(0, color="k", lw=0.6); a.grid(alpha=0.3)
                a.set_xlabel("offset from drop site"); a.set_ylabel("mean Δ")
                a.legend(fontsize=7)
        fig.suptitle(G["title"] + "  —  profiles", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(gdir, "profiles.png"), dpi=args.dpi)
        plt.close(fig)

        # ── histograms ──────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
        for ax, sname in zip(axes, ["vs_standard", "chain"]):
            for lab, d in deltas.get(sname, []):
                ax.hist(d.ravel(), bins=200, histtype="step", lw=1.2,
                        label=lab.replace("\n", " "))
            ax.set_yscale("log"); ax.axvline(0, color="k", lw=0.6)
            ax.set_xlabel("Δ"); ax.set_ylabel("sites (log)")
            ax.set_title(sname); ax.grid(alpha=0.3); ax.legend(fontsize=7)
        fig.suptitle(G["title"] + "  —  distribution of Δ", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(gdir, "histograms.png"), dpi=args.dpi)
        plt.close(fig)
        print(f"  ✓ {G['folder']}", flush=True)

    # ── overview across all uniform direction sets ──────────────────────
    if overview:
        import matplotlib.cm as cm
        fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
        cols = cm.tab20(np.linspace(0, 1, len(overview)))
        keys = [("mean", "mean Δ"), ("frac_loss", "fraction of sites with loss"),
                ("NS_index", "NS index = mean Δ(North) − mean Δ(South)"),
                ("EW_index", "EW index = mean Δ(East) − mean Δ(West)")]
        for (dset, lst), c in zip(overview.items(), cols):
            lst = sorted(lst, key=lambda t: t[0])
            nl = [t[0] for t in lst]
            for ax, (k, _) in zip(axes, keys):
                ax.plot(nl, [t[1][k] for t in lst], "o-", color=c, lw=1.3, ms=4,
                        label=dset)
        for ax, (_, t) in zip(axes, keys):
            ax.set_xscale("log"); ax.set_xlabel("noise level"); ax.set_title(t)
            ax.axhline(0, color="k", lw=0.6); ax.grid(alpha=0.3)
        axes[-1].legend(fontsize=8, ncol=2)
        fig.suptitle("Uniform noise vs standard — all direction sets", fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(root, "overview_uniform_vs_standard.png"), dpi=args.dpi)
        plt.close(fig)

    # ── tables ──────────────────────────────────────────────────────────
    if rows_csv:
        with open(os.path.join(root, "summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
            w.writeheader(); w.writerows(rows_csv)
        with open(os.path.join(root, "summary.json"), "w") as f:
            json.dump({"settings": {k: getattr(args, k) for k in
                                    ("grid", "grains", "bg", "seeds", "avg_start",
                                     "clip", "thr", "linthresh", "smooth")},
                       "comparisons": rows_csv}, f, indent=1)
    n_png = len(glob.glob(os.path.join(root, "**", "*.png"), recursive=True))
    print(f"\n  {len(rows_csv)} comparisons, {n_png} images → {root}/")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Sandpile Δ analysis (GPU / HPC)")
    p.add_argument("--mode", default="all", choices=["all", "simulate", "analyze", "status"])
    p.add_argument("--out", default="output_section3")
    # physics (defaults = the setup of Sec. 3.1)
    p.add_argument("--grid", type=int, default=500)
    p.add_argument("--grains", type=int, default=50_000)
    p.add_argument("--bg", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=int, default=1,
                   help="independent noise realisations per noisy run")
    p.add_argument("--avg-start", type=int, default=0,
                   help="time-average from this grain index on")
    # compute
    p.add_argument("--device", default="auto", help="auto | cuda | cuda:1 | mps | cpu | numpy")
    p.add_argument("--batch", type=int, default=64, help="runs simulated together on the GPU")
    p.add_argument("--part", type=int, default=0)
    p.add_argument("--nparts", type=int, default=1)
    p.add_argument("--ckpt-min", type=float, default=10.0, help="checkpoint every N minutes")
    p.add_argument("--select", default=None,
                   help="regex on run names, e.g. '^U_N_|standard' (testing / subsets)")
    # plotting
    p.add_argument("--clip", type=float, default=99.0)
    p.add_argument("--thr", type=float, default=0.02)
    p.add_argument("--linthresh", type=float, default=0.01)
    p.add_argument("--smooth", type=int, default=1, help="box-filter size (odd), 1 = off")
    p.add_argument("--ncols", type=int, default=6)
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    runs = build_runs()
    check_definitions(runs)

    if args.mode == "status":
        status(runs, args); return
    if args.mode in ("all", "simulate"):
        simulate(runs, args)
    if args.mode == "analyze" or (args.mode == "all" and args.nparts == 1):
        analyze(runs, args)


if __name__ == "__main__":
    main()
