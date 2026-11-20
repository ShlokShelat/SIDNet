"""
Shared constants for the Sandpile Identification Problem (SID) code.

Paper: "The Sandpile Identification Problem: Recovering Directional Noise
from Space-time Dynamics" (Salvi, Shelat, Roy).
"""

# Lattice and simulation (Secs. 2.1 and 4)
GRID = 100                       # 100 x 100 benchmark lattice
N_WARMUP = 30_000                # warm-up steps, run WITHOUT noise
N_FRAMES = {"uni": 2_400,        # recorded steps after the warm-up, SID-Uni
            "nuni": 3_600}       # SID-nUni
INJECTION_ROWS = (38, 44, 50, 56, 62)
INJECTION_SITES = [(o, c) for o in INJECTION_ROWS for c in INJECTION_ROWS]   # 5 x 5 = 25 sites

# Noise classes (Sec. 2.2): all 2^4 = 16 subsets of {N, S, E, W}
DIRS = ["N", "S", "E", "W"]
NOISE_NAMES = [
    "standard",
    "N", "S", "E", "W",
    "NS", "NE", "NW", "SE", "SW", "EW",
    "NSE", "NSW", "NEW", "SEW",
    "NSEW",
]
NOISE_DIRS = {name: ([] if name == "standard" else list(name)) for name in NOISE_NAMES}
NOISE_CLASS = {name: i for i, name in enumerate(NOISE_NAMES)}
N_CLASSES = len(NOISE_NAMES)     # 16
N_DIRS = 4

# Noise strengths {0.0, 0.1, ..., 0.9}
ALPHAS = [round(0.1 * i, 1) for i in range(10)]
ALPHA_MIN, ALPHA_MAX = 0.0, 0.9

# Simulations per split (Sec. 4)
UNI_RUNS_PER_STRENGTH = {"train": 3, "val": 1, "test": 1}     # every noisy class x strength
UNI_STANDARD_RUNS = {"train": 30, "val": 10, "test": 10}      # 50 runs, as often as each noisy class
NUNI_RUNS_NOISY = {"train": 35, "val": 8, "test": 8}          # each of the 15 noisy classes
NUNI_RUNS_STANDARD = {"train": 5, "val": 1, "test": 1}

# Samples (Sec. 4)
T_SHORT = 16
T_LONG = 64
STRIDE = {"train": 8, "val": 32, "test": 32}

# Evaluation tolerances (Sec. 6)
ALPHA_TOL_EXACT = 0.05
ALPHA_TOL_LOOSE = 0.10


def direction_bits(name):
    """4-bit pattern (N, S, E, W) of the suppressed directions of a class."""
    return [1 if d in NOISE_DIRS[name] else 0 for d in DIRS]


def n_windows(n_frames, stride, t_long=T_LONG):
    """Samples cut from one simulation: floor((L - T_long) / stride) + 1."""
    return (n_frames - t_long) // stride + 1
