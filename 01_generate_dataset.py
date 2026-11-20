"""
Generate the SID-Uni and SID-nUni benchmark datasets (Sec. 4).

Each simulation
  * uses the 100 x 100 lattice with open boundaries and the 25 injection sites,
  * starts from an empty lattice and runs N_WARMUP = 30,000 steps as a standard
    sandpile, WITHOUT noise (the warm-up),
  * then switches the directional noise on and records 2,400 (SID-Uni) or
    3,600 (SID-nUni) steps; in each step one grain is added at a random
    injection site, the lattice topples until stable, and the stable
    configuration is saved as one frame.

Under directional noise, a grain toppled in direction d reaches its neighbour
with probability alpha_d and is removed with probability 1 - alpha_d (Sec. 2.2).

After all simulations are generated, the reference z_ref (the time-averaged
grain count of the standard sandpile, computed from the standard-class
simulations of the TRAINING split) is written to <regime>_reference.h5. The
same reference is used for every sample of every split (Sec. 4).

Usage
  python 01_generate_dataset.py --regime uni  --out_dir data --n_workers 8
  python 01_generate_dataset.py --regime nuni --out_dir data --n_workers 8
"""

import argparse
import os
from multiprocessing import Pool

import h5py
import numpy as np

from sid_common import (ALPHAS, DIRS, GRID, INJECTION_SITES, N_FRAMES, N_WARMUP,
                        NOISE_DIRS, NOISE_NAMES, NUNI_RUNS_NOISY, NUNI_RUNS_STANDARD,
                        UNI_RUNS_PER_STRENGTH, UNI_STANDARD_RUNS)

STEP = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}
NO_NOISE = {d: 1.0 for d in DIRS}


# --------------------------------------------------------------------------- simulation
def relax(grid, alpha, rng):
    """Topple every site with z >= 4 until the lattice is stable."""
    while True:
        unstable = np.argwhere(grid >= 4)
        if len(unstable) == 0:
            return grid
        for idx in rng.permutation(len(unstable)):
            o, c = unstable[idx]
            if grid[o, c] < 4:
                continue
            grid[o, c] -= 4
            for d in DIRS:
                no, nc = o + STEP[d][0], c + STEP[d][1]
                if not (0 <= no < GRID and 0 <= nc < GRID):
                    continue                          # open boundary: grain leaves the lattice
                a = alpha[d]
                if a >= 1.0 or rng.random() < a:      # xi_d ~ Bernoulli(alpha_d)
                    grid[no, nc] += 1                 # otherwise the grain is removed


def simulate(alpha, seed, n_frames, n_warmup=N_WARMUP):
    """Return the recorded frames [n_frames, H, W] (uint8) of one simulation."""
    rng = np.random.default_rng(seed)
    sites = rng.integers(len(INJECTION_SITES), size=n_warmup + n_frames)
    grid = np.zeros((GRID, GRID), dtype=np.int32)

    for i in range(n_warmup):                         # warm-up: standard sandpile, no noise
        o, c = INJECTION_SITES[sites[i]]
        grid[o, c] += 1
        relax(grid, NO_NOISE, rng)

    frames = np.empty((n_frames, GRID, GRID), dtype=np.uint8)
    for i in range(n_frames):                         # noise switched on after the warm-up
        o, c = INJECTION_SITES[sites[n_warmup + i]]
        grid[o, c] += 1
        relax(grid, alpha, rng)
        frames[i] = grid                              # stable configuration: z in {0,1,2,3}
    return frames


# --------------------------------------------------------------------------- tasks
def make_tasks(regime, out_dir, seed0):
    """List of (out_dir, regime, class, {dir: alpha}, seed, split)."""
    tasks, seed = [], seed0
    rng = np.random.default_rng(seed0)
    for split in ("train", "val", "test"):
        if regime == "uni":
            for _ in range(UNI_STANDARD_RUNS[split]):
                tasks.append((out_dir, regime, "standard", {}, seed, split)); seed += 1
            for name in NOISE_NAMES[1:]:
                for a in ALPHAS:                      # one shared strength for all active directions
                    for _ in range(UNI_RUNS_PER_STRENGTH[split]):
                        tasks.append((out_dir, regime, name,
                                      {d: a for d in NOISE_DIRS[name]}, seed, split)); seed += 1
        else:
            for _ in range(NUNI_RUNS_STANDARD[split]):
                tasks.append((out_dir, regime, "standard", {}, seed, split)); seed += 1
            for name in NOISE_NAMES[1:]:
                for _ in range(NUNI_RUNS_NOISY[split]):
                    # each active direction gets its own strength, drawn independently
                    # and uniformly at random from {0.0, ..., 0.9}
                    strengths = {d: float(rng.choice(ALPHAS)) for d in NOISE_DIRS[name]}
                    tasks.append((out_dir, regime, name, strengths, seed, split)); seed += 1
    return tasks


def file_name(regime, name, strengths, seed, split):
    tag = "_".join(f"{d}{strengths[d]:.1f}" for d in NOISE_DIRS[name]) or "std"
    return f"{regime}_{name}_{tag}_s{seed}_{split}.h5"


def run_task(task):
    out_dir, regime, name, strengths, seed, split = task
    path = os.path.join(out_dir, file_name(regime, name, strengths, seed, split))
    if os.path.exists(path):
        return path
    alpha = dict(NO_NOISE, **strengths)
    frames = simulate(alpha, seed, N_FRAMES[regime])
    tmp = path + ".tmp"
    with h5py.File(tmp, "w") as f:
        # chunks of 16 frames: a 64-frame sample reads only the 4-5 chunks it needs
        f.create_dataset("frames", data=frames, chunks=(16, GRID, GRID),
                         compression="gzip", compression_opts=1)
        f.create_dataset("z_mean", data=frames.astype(np.float64).mean(0).astype(np.float32))
        f.create_dataset("alpha_vec", data=np.array([alpha[d] for d in DIRS], np.float32))
        f.attrs.update(noise_name=name, split=split, regime=regime, seed=seed,
                       n_frames=N_FRAMES[regime], n_warmup=N_WARMUP)
    os.replace(tmp, path)
    print(f"  [ok] {os.path.basename(path)}", flush=True)
    return path


# --------------------------------------------------------------------------- reference
def write_reference(regime, out_dir):
    """z_ref = time-averaged grain count of the standard sandpile, from the
    standard-class simulations of the training split only."""
    files = sorted(f for f in os.listdir(out_dir)
                   if f.startswith(f"{regime}_standard_") and f.endswith("_train.h5"))
    if not files:
        raise RuntimeError("no standard-class training simulations found")
    acc = np.zeros((GRID, GRID), np.float64)
    for f in files:
        with h5py.File(os.path.join(out_dir, f), "r") as h:
            acc += h["z_mean"][:]
    z_ref = (acc / len(files)).astype(np.float32)
    path = os.path.join(out_dir, f"{regime}_reference.h5")
    with h5py.File(path, "w") as h:
        h.create_dataset("z_ref", data=z_ref)
        h.attrs["n_simulations"] = len(files)
    print(f"  reference from {len(files)} standard training simulations -> {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regime", required=True, choices=["uni", "nuni"])
    p.add_argument("--out_dir", default="data")
    p.add_argument("--n_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=None,
                   help="first seed (default 1000 for uni, 200000 for nuni)")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    seed0 = a.seed if a.seed is not None else (1_000 if a.regime == "uni" else 200_000)
    tasks = make_tasks(a.regime, a.out_dir, seed0)
    counts = {s: sum(t[5] == s for t in tasks) for s in ("train", "val", "test")}
    print(f"SID-{'Uni' if a.regime == 'uni' else 'nUni'}: {len(tasks)} simulations {counts}, "
          f"{N_FRAMES[a.regime]} frames each, warm-up {N_WARMUP} steps without noise")
    if a.n_workers <= 1:
        for t in tasks:
            run_task(t)
    else:
        with Pool(a.n_workers) as pool:
            pool.map(run_task, tasks, chunksize=1)
    write_reference(a.regime, a.out_dir)
    done = [f for f in os.listdir(a.out_dir) if f.startswith(a.regime + "_") and f.endswith(
        ("_train.h5", "_val.h5", "_test.h5"))]
    print(f"done: {len(done)} of {len(tasks)} simulations in {a.out_dir}")


if __name__ == "__main__":
    main()
