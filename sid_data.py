"""
Samples of the SID datasets (Sec. 4).

A sample is cut from one simulation and has three parts:
  * a short window of T_short = 16 consecutive frame-level maps Delta_t,
  * a long window of T_long = 64 consecutive frame-level maps, centred on the
    short window,
  * the time-averaged map Delta of the simulation, averaged over all its
    recorded time steps after the warm-up.

Both maps use the same reference z_ref (Eq. normdiff):
  Delta   = (z_mean  - z_ref) / (z_ref + 1)      time-averaged map
  Delta_t = (z_t     - z_ref) / (z_ref + 1)      frame-level map

Training samples start 8 frames apart, validation and test samples 32 frames
apart, so one simulation of L frames gives floor((L - 64) / stride) + 1 samples.
"""

from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from sid_common import NOISE_CLASS, STRIDE, T_LONG, T_SHORT


def load_reference(data_dir, regime):
    path = Path(data_dir) / f"{regime}_reference.h5"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: generate the data first with "
                                f"python 01_generate_dataset.py --regime {regime} --out_dir {data_dir}")
    with h5py.File(path, "r") as h:
        return h["z_ref"][:].astype(np.float32)


class SIDDataset(Dataset):
    def __init__(self, data_dir, regime, split, classes, stride=None,
                 t_short=T_SHORT, t_long=T_LONG):
        self.t_short, self.t_long = t_short, t_long
        self.offset = (t_long - t_short) // 2           # short window centred in the long one
        stride = stride or STRIDE[split]
        self.z_ref = load_reference(data_dir, regime)
        self.files, self.delta, self.labels, self.alphas, self.windows = [], [], [], [], []
        allowed = set(classes)
        for path in sorted(Path(data_dir).glob(f"{regime}_*_{split}.h5")):
            with h5py.File(path, "r") as h:
                name = str(h.attrs["noise_name"])
                if name not in allowed:
                    continue
                n_frames = h["frames"].shape[0]
                z_mean = h["z_mean"][:].astype(np.float32)
                alpha = h["alpha_vec"][:].astype(np.float32)
            k = len(self.files)
            self.files.append(str(path))
            self.delta.append((z_mean - self.z_ref) / (self.z_ref + 1.0))
            self.labels.append(NOISE_CLASS[name])
            self.alphas.append(alpha)
            for start in range(0, n_frames - t_long + 1, stride):
                self.windows.append((k, start))
        if not self.windows:
            raise RuntimeError(f"no {split} samples of {regime} for classes {classes} in {data_dir}")
        self._handles = {}
        print(f"  [{regime}] {split:5s}: {len(self.files):4d} simulations, "
              f"{len(self.windows):8,d} samples")

    def __len__(self):
        return len(self.windows)

    def _frames(self, k, start):
        h = self._handles.get(k)
        if h is None:                                   # one open handle per file and worker
            h = self._handles[k] = h5py.File(self.files[k], "r")
        return h["frames"][start:start + self.t_long].astype(np.float32)

    def __getitem__(self, i):
        k, start = self.windows[i]
        long_win = (self._frames(k, start) - self.z_ref) / (self.z_ref + 1.0)
        short_win = long_win[self.offset:self.offset + self.t_short]
        return (torch.from_numpy(self.delta[k]),
                torch.from_numpy(np.ascontiguousarray(short_win)),
                torch.from_numpy(long_win),
                torch.tensor(self.labels[k], dtype=torch.long),
                torch.from_numpy(self.alphas[k]))

    def sample_labels(self):
        return [self.labels[k] for k, _ in self.windows]

    def __getstate__(self):                             # h5py handles are not picklable
        s = self.__dict__.copy()
        s["_handles"] = {}
        return s


def class_balanced_sampler(ds):
    """Draws about the same number of samples from each class in every batch."""
    labels = ds.sample_labels()
    counts = Counter(labels)
    weights = [1.0 / counts[y] for y in labels]
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)
