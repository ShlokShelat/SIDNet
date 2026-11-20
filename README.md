# SIDNet: The Sandpile Identification Problem

Code for the paper **"The Sandpile Identification Problem: Recovering Directional Noise from Space-time Dynamics"** by Shrey Salvi, Shlok Shelat and Souvik Roy (Ahmedabad University).

An Abelian sandpile is given directional noise: a grain toppled in direction *d* ∈ {N, S, E, W} reaches its neighbour with probability α_d and is removed with probability 1 − α_d. The **Sandpile Identification Problem** is to recover, from a finite window of observed frames, which of the 16 subsets of directions is suppressed (the noise class) and the strength α_d of each suppressed direction. This repository contains the generator of the two benchmark datasets (SID-Uni and SID-nUni), the SIDNet model, and the training and evaluation code.

## Repository layout

| File | Contents | Paper |
|---|---|---|
| `sid_common.py` | lattice, noise classes, dataset sizes, window settings | Secs. 2, 4 |
| `01_generate_dataset.py` | simulates the sandpile with directional noise and writes SID-Uni / SID-nUni | Sec. 4 |
| `sid_data.py` | builds the samples (short window, long window, time-averaged map) | Sec. 4 |
| `sidnet.py` | the SIDNet model | Sec. 5, Fig. 10, Table 5 |
| `sid_losses.py` | classification, direction and noise-strength losses | Sec. 6 |
| `02_train.py` | five-phase curriculum training | Sec. 6, Table 6 |
| `03_evaluate.py` | test-set metrics and confusion matrix | Sec. 7 |
| `scripts/run_all.sh` | full pipeline (also usable as a SLURM job) | |

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Python 3.10 or later. Training needs a CUDA GPU; mixed precision is used automatically when one is available.

## Datasets (Sec. 4)

Every simulation uses a 100 × 100 lattice with open boundaries. Grains are added one at a time at a site chosen uniformly at random from 25 injection sites (rows and columns {38, 44, 50, 56, 62}). Each simulation starts from an empty lattice and runs 30,000 steps as a standard sandpile, **without noise** (the warm-up). The noise is then switched on, and 2,400 (SID-Uni) or 3,600 (SID-nUni) more steps are recorded, one stable configuration (frame) per step.

| | SID-Uni | SID-nUni |
|---|---|---|
| Noise strengths | one shared α ∈ {0.0, …, 0.9} for all suppressed directions | an independent α_d ∈ {0.0, …, 0.9} for each suppressed direction |
| Runs per split (train/val/test) | 3/1/1 per noisy class and strength; standard class 30/10/10 | 35/8/8 per noisy class; standard class 5/1/1 |
| Simulations | 800 | 772 |
| Frames per simulation | 2,400 | 3,600 |
| Samples (train/val/test) | 140,640 / 11,840 / 11,840 | 234,790 / 13,431 / 13,431 |

The split is made by simulation, so every test sample comes from a sandpile never seen in training. A sample has three parts:

* a short window of 16 consecutive frame-level maps Δ_t;
* a long window of 64 consecutive maps, centred on the short window;
* the time-averaged map Δ of the simulation.

Both maps use Δ = (z̄ − z̄_ref) / (z̄_ref + 1), where the reference z̄_ref is the time-averaged grain count of the standard sandpile, computed from the standard-class simulations of the training split. The same reference is used for every split. Training samples start 8 frames apart and validation/test samples 32 frames apart, so a simulation of L frames gives ⌊(L − 64)/stride⌋ + 1 samples.

```bash
python 01_generate_dataset.py --regime uni  --out_dir data --n_workers 8
python 01_generate_dataset.py --regime nuni --out_dir data --n_workers 8
```

Each simulation is written to its own HDF5 file (`frames`, `z_mean`, `alpha_vec`). Finished files are skipped, so an interrupted run can simply be restarted. Use as many workers as the node has CPU cores. The reference is written last, to `data/<regime>_reference.h5`.

## Model (Sec. 5)

SIDNet combines:

* an **explicit direction detector**: boundary-strip scores A_d and 28 related statistics, with two MLPs (32 → 16 → 4);
* a **spatial encoder**: four 3 × 3 conv layers (1 → 32 → 64 → 128 → 256) with batch normalisation, ReLU and max-pooling;
* a **patch embedding**: 100 patches × 14 statistics → 256;
* a **dual-window Vision Transformer**: 4 blocks, 8 heads, feed-forward size 512, one branch for each window;
* a **direction GNN**: 4 nodes of dimension 64, fully connected, 3 message-passing rounds;
* a **learned fusion gate** for the suppression probabilities, with fixed 0.4 / 0.6 weights for the noise strengths;
* a **classification head**: D_cat = 1,076 → 256 → 128 → 16, plus an embedding of the nearest class and a residual shortcut.

```bash
python sidnet.py      # prints the parameter count and runs a forward pass
```

## Training (Sec. 6)

| Phase | Classes | Epochs (Uni / nUni) | LR | β_cls | β_dir, β_alp (Uni / nUni) |
|---|---|---|---|---|---|
| A | all 16 (no class loss) | 25 / 40 | 3e-4 | 0 | 3.0, 2.0 / 4.0, 3.0 |
| B1 | standard only | 20 | 3e-4 | 1 | 2.0, 1.0 |
| B2 | + N, S, E, W, NS, EW | 30 | 2e-4 | 1 | 2.0, 1.0 / 2.5, 1.5 |
| B3 | + NE, NW, SE, SW | 30 / 40 | 1e-4 | 1 | 2.0, 1.0 / 2.5, 1.5 |
| B4 | + three-direction, NSEW | 40 / 60 | 1e-4 | 1 | 2.0, 1.0 / 2.5, 1.5 |

The optimiser and selection settings are:

* AdamW with weight decay 1e-4, and a cosine schedule down to 5% of the starting rate within each phase.
* Batch size 32, with gradients accumulated over 4 steps in B3–B4.
* Gradient clipping at 1.0, dropout 0.1, and a class-balanced sampler.
* The best model of each phase is chosen on the validation set: by mean direction F1 in Phase A, lowest validation loss in B1, and accuracy in B2–B4. It is the starting point of the next phase.

```bash
python 02_train.py --regime uni  --data_dir data --out_dir checkpoints
python 02_train.py --regime nuni --data_dir data --out_dir checkpoints
```

## Evaluation (Sec. 7)

```bash
python 03_evaluate.py --regime uni  --checkpoint checkpoints/uni_phaseB4_best.pt
python 03_evaluate.py --regime nuni --checkpoint checkpoints/nuni_phaseB4_best.pt
```

The script reports classification accuracy, per-direction precision, recall and F1, topology-reconstruction agreement, noise-strength MAE and exact (±0.05) / loose (±0.10) accuracy per class, and accuracy per noise strength. It writes `results/<regime>_test_metrics.json` and `results/<regime>_test_confusion.csv`.

## Full pipeline

```bash
bash scripts/run_all.sh uni      # SID-Uni: generate, train, evaluate
bash scripts/run_all.sh nuni     # SID-nUni

# on a SLURM cluster (edit the #SBATCH lines in the script for your cluster)
VENV=/path/to/venv sbatch --job-name=sidnet_uni  scripts/run_all.sh uni
VENV=/path/to/venv sbatch --job-name=sidnet_nuni scripts/run_all.sh nuni
```

The two regimes are independent and can run as two jobs at the same time. Paths are set with the environment variables `DATA`, `CKPT`, `RESULTS` and `WORKERS` (defaults: `data`, `checkpoints`, `results`, `8`). Every step can be restarted. Finished simulations are skipped. If training stops, for example at a job time limit, the same command continues from the last finished epoch (use `--no_resume` with `02_train.py` to start again from scratch).

## Outputs

| Path | Contents |
|---|---|
| `data/<regime>_*.h5`, `data/<regime>_reference.h5` | simulations and the reference z̄_ref |
| `checkpoints/<regime>_phase<P>_best.pt` | best model of each phase; `<regime>_phaseB4_best.pt` is the final model |
| `checkpoints/<regime>_phase<P>_history.json` | training loss and validation metrics after every epoch |
| `checkpoints/<regime>_last.pt` | full training state, used to resume |
| `results/<regime>_<split>_metrics.json`, `results/<regime>_<split>_confusion.csv` | evaluation results |
| `logs/` | job logs |

## Citation

```bibtex
@article{salvi_sandpile_identification,
  title   = {The Sandpile Identification Problem: Recovering Directional Noise from Space-time Dynamics},
  author  = {Salvi, Shrey and Shelat, Shlok and Roy, Souvik},
  note    = {Preprint submitted to Elsevier},
  year    = {2026}
}
```
