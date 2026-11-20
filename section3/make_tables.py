"""
Tables 1-3 of the paper (Sec. 3.3) from the 102 illustration runs in section3/runs/.

Definitions (Sec. 3.2)
  avalanche region  sites whose time-averaged grain count differs from the starting 2 grains
  area              number of sites in the avalanche region
  reach (N/S/E/W)   distance from the injection site to the farthest site of the region along d
  height x width    (reach_N + reach_S + 1) x (reach_E + reach_W + 1)
  centre shift      distance from the injection site to the centre (mean position) of the region,
                    with the closest compass direction
  lean              centre shift / radius, radius = average of half-height and half-width;
                    given only when the shift exceeds the noise floor
  noise floor       largest centre shift among the NS, EW and NSEW runs (0.6 sites)
  step              change from the previous row: reach change, shift before -> after, and the share
                    of the 500 x 500 lattice where the step comparison is below -0.02 or above +0.02

Run names: U_<dirs>_<1 - alpha> (uniform), F_<d>_<1 - alpha_d>_... (non-uniform), standard.

Usage
  python section3/make_tables.py                 # prints Tables 1-3
  python section3/make_tables.py --csv out.csv   # also writes every run's measures
"""

import argparse
import math
import os

import numpy as np

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
C0 = 250                 # injection site (centre of the 500 x 500 lattice)
BG = 2.0                 # starting grains per site
THR = 0.02               # threshold for a changed site in the step comparison
LEVELS = ["0.01", "0.02", "0.05", "0.10", "0.30", "0.50"]   # 1 - alpha


def zbar(key, runs=RUNS):
    return np.load(os.path.join(runs, f"{key}__s0.npz"))["zbar"]


def measures(z):
    rr, cc = np.nonzero(np.abs(z - BG) > 1e-12)
    reach = dict(N=C0 - rr.min(), S=rr.max() - C0, E=cc.max() - C0, W=C0 - cc.min())
    dy, dx = rr.mean() - C0, cc.mean() - C0           # dy > 0: south, dx > 0: east
    h, w = reach["N"] + reach["S"] + 1, reach["E"] + reach["W"] + 1
    shift = math.hypot(dy, dx)
    if abs(dy) >= abs(dx):
        compass = "south" if dy > 0 else "north"
    else:
        compass = "east" if dx > 0 else "west"
    return dict(reach=reach, height=h, width=w, area=len(rr), dy=dy, dx=dx, shift=shift,
                compass=compass, lean=shift / ((h / 2 + w / 2) / 2))


def step_share(z, z_prev):
    d = (z - z_prev) / (z_prev + 1.0)
    return float(((d < -THR) | (d > THR)).mean())


def noise_floor(runs=RUNS):
    return max(measures(zbar(f"U_{c}_{l}", runs))["shift"] for c in ("NS", "EW", "NSEW") for l in LEVELS)


def fmt_shift(m, floor):
    s = f"{m['shift']:.1f}"
    return f"{s} (floor)" if m["shift"] <= floor else f"{s} {m['compass']}"


def row(label, m, floor, prev=None, z=None, z_prev=None):
    r = m["reach"]
    reach = "/".join(str(r[d]) for d in "NSEW")
    lean = f"{100 * m['lean']:.0f}%" if m["shift"] > floor else "--"
    step = "--"
    if prev is not None:
        dr = "/".join(f"{r[d] - prev['reach'][d]:+d}" for d in "NSEW")
        before = f"{prev['shift']:.1f}" + ("" if prev["shift"] <= floor else " " + prev["compass"][0].upper())
        after = f"{m['shift']:.1f}" + ("" if m["shift"] <= floor else " " + m["compass"][0].upper())
        step = f"{dr}; {before} -> {after}; {100 * step_share(z, z_prev):.1f}%"
    return (f"  {label:<34}{reach:<17}{m['height']:>4} x {m['width']:<5}{m['area']:>9,}  "
            f"{fmt_shift(m, floor):<13}{lean:>5}   {step}")


def uniform_block(cls, floor, runs, with_standard=False):
    lines = []
    prev = z_prev = None
    if with_standard:
        z_std = zbar("standard", runs)
        lines.append(row("1 (standard)", measures(z_std), floor))
    for lev in LEVELS:
        z = zbar(f"U_{cls}_{lev}", runs)
        m = measures(z)
        lines.append(row(f"{cls}, {1 - float(lev):.2f}", m, floor, prev, z, z_prev))
        prev, z_prev = m, z
    return lines


def pair(first, second, label1, label2, floor, runs):
    z1, z2 = zbar(first, runs), zbar(second, runs)
    m1, m2 = measures(z1), measures(z2)
    return [row(label1, m1, floor), row(label2, m2, floor, m1, z2, z1)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=RUNS)
    p.add_argument("--csv", default=None)
    a = p.parse_args()
    floor = noise_floor(a.runs)
    head = (f"  {'':<34}{'reach N/S/E/W':<17}{'height x width':<14}{'area':>6}  "
            f"{'centre shift':<13}{'lean':>5}   step: reach; shift; changed")
    print(f"noise floor: {floor:.2f} sites (largest shift of NS, EW, NSEW)\n")
    print("Table 1  uniform regime, class N"); print(head)
    print("\n".join(uniform_block("N", floor, a.runs, with_standard=True)))
    print("\nTable 2  uniform regime, classes NS and NSE"); print(head)
    print("\n".join(uniform_block("NS", floor, a.runs)))
    print("\n".join(uniform_block("NSE", floor, a.runs)))
    print("\nTable 3  non-uniform regime"); print(head)
    for first, second, l1, l2 in [
            ("U_N_0.01", "F_N_0.01_S_0.05", "aN=0.99", "aN=0.99, aS=0.95"),
            ("U_N_0.01", "F_N_0.01_W_0.05", "aN=0.99", "aN=0.99, aW=0.95"),
            ("F_N_0.01_S_0.05_E_0.10", "F_N_0.01_S_0.05_E_0.10_W_0.30",
             "aN=0.99, aS=0.95, aE=0.90", "aN=0.99, aS=0.95, aE=0.90, aW=0.70")]:
        print("\n".join(pair(first, second, l1, l2, floor, a.runs)))

    if a.csv:
        keys = sorted(f[:-8] for f in os.listdir(a.runs) if f.endswith("__s0.npz"))
        with open(a.csv, "w") as f:
            f.write("run,reach_N,reach_S,reach_E,reach_W,height,width,area,shift_south,shift_east,"
                    "centre_shift,compass,lean\n")
            for k in keys:
                m = measures(zbar(k, a.runs)); r = m["reach"]
                f.write(f"{k},{r['N']},{r['S']},{r['E']},{r['W']},{m['height']},{m['width']},{m['area']},"
                        f"{m['dy']:.4f},{m['dx']:.4f},{m['shift']:.4f},{m['compass']},{m['lean']:.4f}\n")
        print(f"\nwrote {a.csv} ({len(keys)} runs)")


if __name__ == "__main__":
    main()
