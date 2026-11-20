# SIDNet: The Sandpile Identification Problem

Code and data for the paper **"The Sandpile Identification Problem: Recovering Directional Noise from Space-time Dynamics"** by Shrey Salvi, Shlok Shelat and Souvik Roy (Ahmedabad University).

An Abelian sandpile is given directional noise: a grain toppled in direction *d* ∈ {N, S, E, W} reaches its neighbour with probability α_d and is removed with probability 1 − α_d. The **Sandpile Identification Problem** is to recover, from a finite window of observed frames, which of the 16 subsets of directions is suppressed (the noise class) and the strength α_d of each suppressed direction.

This repository contains:

- the illustration runs, tables and figures of Sec. 3;
- the generator of the two benchmark datasets, SID-Uni and SID-nUni (Sec. 4);
- the SIDNet model (Sec. 5), its training (Sec. 6) and its evaluation (Sec. 7).

## Reproducing the paper

| Paper | Command | Hardware |
|---|---|---|
| Tables 1–3 | `python section3/make_tables.py` | CPU, seconds |
| Figs. 1, 3–9 | `python section3/make_figures.py --out figures_section3` | CPU, seconds |
| The 102 illustration runs behind Tables 1–3 and Figs. 1, 3–9 (included in `section3/runs/`) | `python section3/simulate.py --mode simulate --batch 102` | GPU (about 4.5 h on one RTX 5090) |
| SID-Uni / SID-nUni datasets (Table 4) | `python 01_generate_dataset.py --regime uni` (and `nuni`) | CPU cores |
| SIDNet training (Table 6) | `python 02_train.py --regime uni` (and `nuni`) | GPU |
| Table 7 and the other results of Sec. 7 | `python 03_evaluate.py --regime uni --checkpoint checkpoints/uni_phaseB4_best.pt` (and `nuni`) | GPU |
| Fig. 11 | `python 04_plot_confusion.py` | CPU, seconds |
| Everything for one regime | `bash scripts/run_all.sh uni` (and `nuni`) | GPU node |

Fig. 2 (the five noise families) and Fig. 10 (the SIDNet diagram) are drawings and are not generated from data.

## Installation

```bash
git clone https://github.com/ShlokShelat/SIDNet.git
cd SIDNet
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Python 3.10 or later. Training and the Sec. 3 simulation need a CUDA GPU. Everything else runs on a CPU.

To check that every step runs (a few minutes on a laptop CPU), run the quick test:

```bash
bash scripts/quick_test.sh
```

It makes tiny datasets (one short simulation per class and split), trains one short epoch per phase, evaluates, draws Fig. 11, and builds the Sec. 3 tables and figures. Its accuracy numbers only show that the pipeline runs; they are not results.

## Repository layout

| Path | Contents | Paper |
|---|---|---|
| `section3/simulate.py` | GPU simulation of the 102 illustration runs (500 × 500 lattice) | Sec. 3.1 |
| `section3/runs/` | the 102 runs used in the paper (time-averaged and final grain counts) | Sec. 3 |
| `section3/make_tables.py` | Tables 1–3 | Sec. 3.3 |
| `section3/make_figures.py` | Figs. 1 and 3–9 | Sec. 3 |
| `section3/checks/` | independent checks of the simulation physics | |
| `sid_common.py` | lattice, noise classes, dataset sizes, window settings | Secs. 2, 4 |
| `01_generate_dataset.py` | simulates the sandpile with directional noise and writes SID-Uni / SID-nUni | Sec. 4 |
| `sid_data.py` | builds the samples (short window, long window, time-averaged map) | Sec. 4 |
| `sidnet.py` | the SIDNet model | Sec. 5, Table 5 |
| `sid_losses.py` | classification, direction and noise-strength losses | Sec. 6 |
| `02_train.py` | five-phase curriculum training | Sec. 6, Table 6 |
| `03_evaluate.py` | test-set metrics and confusion matrices | Sec. 7 |
| `04_plot_confusion.py` | Fig. 11 | Sec. 7 |
| `scripts/run_all.sh` | full pipeline for one regime (also a SLURM job script) | |
| `scripts/quick_test.sh` | pipeline test on a CPU | |

## Sec. 3: illustration runs, tables and figures

The illustration setup (Sec. 3.1) is a 500 × 500 lattice with a single injection site at its centre. Every site starts with two grains, and 50,000 grains are added one at a time. Each grain is followed by full relaxation with synchronous toppling. The time average z̄ is taken over all 50,000 steps. The 102 runs are:

- the standard sandpile;
- the 15 noisy classes at α_d ∈ {0.99, 0.98, 0.95, 0.90, 0.70, 0.50};
- the 11 non-uniform configurations.

They are stored in `section3/runs/`. Each file is named after its noise, for example `U_N_0.30` = class N with 1 − α_N = 0.30, or `F_N_0.01_S_0.05` = α_N = 0.99, α_S = 0.95. Tables 1–3 and Figs. 1 and 3–9 are computed directly from these files:

```bash
python section3/make_tables.py               # prints Tables 1-3 (add --csv measures.csv for every run)
python section3/make_figures.py --out figures_section3
```

To regenerate the runs themselves on a GPU (seed 42, all 102 runs in one batch, as in the paper):

```bash
python section3/simulate.py --mode simulate --batch 102 --out output_section3
python section3/make_tables.py --runs output_section3/runs
```

`python section3/checks/check_physics.py` checks the simulation engine on a CPU, and `section3/checks/check_gpu.py` checks it on a GPU. Without noise, the batched engine gives exactly the same result as a plain one-site-at-a-time sandpile. Runs in the same batch do not affect each other, and noise removes grains at the rate λ = Σ_d (1 − α_d) per toppling.

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

- a short window of 16 consecutive frame-level maps Δ_t;
- a long window of 64 consecutive maps, centred on the short window;
- the time-averaged map Δ of the simulation.

Both maps use Δ = (z̄ − z̄_ref) / (z̄_ref + 1). The reference z̄_ref is the time-averaged grain count of the standard sandpile, computed from the standard-class simulations of the training split, and the same reference is used for every split. Training samples start 8 frames apart and validation/test samples 32 frames apart, so a simulation of L frames gives ⌊(L − 64)/stride⌋ + 1 samples.

```bash
python 01_generate_dataset.py --regime uni  --out_dir data --n_workers 32
python 01_generate_dataset.py --regime nuni --out_dir data --n_workers 32
```

- **Seeds:** every simulation has its own fixed seed (from 1,000 for SID-Uni and from 200,000 for SID-nUni). The SID-nUni strengths are drawn from a fixed seed too, so the datasets are identical on every machine.
- **Files:** each simulation is written to its own HDF5 file (`frames`, `z_mean`, `alpha_vec`), and the reference is written last to `data/<regime>_reference.h5`.
- **Restarting:** finished files are skipped, so an interrupted run can simply be restarted. Use as many workers as the node has CPU cores.
- **Storage:** the raw frames are 19 GB for SID-Uni and 28 GB for SID-nUni before compression. They are stored gzip-compressed.

## Model (Sec. 5)

SIDNet has the following parts:

- **explicit direction detector:** boundary-strip scores A_d and 28 related statistics, followed by two MLPs (32 → 16 → 4);
- **spatial encoder:** four 3 × 3 conv layers (1 → 32 → 64 → 128 → 256) with batch normalisation, ReLU and max-pooling;
- **patch embedding:** 100 patches × 14 statistics → 256;
- **dual-window Vision Transformer:** 4 blocks, 8 heads, feed-forward size 512, one branch for each window;
- **direction GNN:** 4 nodes of dimension 64, fully connected, 3 message-passing rounds;
- **learned fusion gate:** for the suppression probabilities, with fixed 0.4 / 0.6 weights for the noise strengths;
- **classification head:** D_cat = 1,076 → 256 → 128 → 16, plus an embedding of the nearest class and a residual shortcut.

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

The settings are:

- AdamW with weight decay 1e-4, and a cosine schedule down to 5 % of the starting rate within each phase;
- batch size 32, with gradients accumulated over 4 steps in B3–B4;
- gradient clipping at 1.0, dropout 0.1 and a class-balanced sampler.

The best model of each phase is chosen on the validation set and starts the next phase:

- Phase A: mean direction F1;
- Phase B1: lowest validation loss;
- Phases B2–B4: accuracy.

```bash
python 02_train.py --regime uni  --data_dir data --out_dir checkpoints
python 02_train.py --regime nuni --data_dir data --out_dir checkpoints
```

## Evaluation (Sec. 7)

```bash
python 03_evaluate.py --regime uni  --checkpoint checkpoints/uni_phaseB4_best.pt
python 03_evaluate.py --regime nuni --checkpoint checkpoints/nuni_phaseB4_best.pt
python 04_plot_confusion.py --results results --out results/fig_confusion.pdf
```

`03_evaluate.py` reports:

- the rows of the results summary table: overall accuracy, topology-reconstruction agreement, mean per-direction F1, the lowest class accuracy of each noise family, and the single-direction noise-strength MAE;
- per-direction precision, recall and F1;
- per-class accuracy, MAE and exact (±0.05) / loose (±0.10) noise-strength accuracy;
- accuracy per noise strength (uniform regime).

It writes `results/<regime>_<split>_metrics.json` and `results/<regime>_<split>_confusion.csv`, and `04_plot_confusion.py` draws Fig. 11 from the two confusion matrices.

## Full pipeline

```bash
bash scripts/run_all.sh uni      # SID-Uni: generate, train, evaluate
bash scripts/run_all.sh nuni     # SID-nUni

# on a SLURM cluster, from the repository root (edit the #SBATCH lines for your cluster)
VENV=/path/to/venv sbatch --job-name=sidnet_uni  scripts/run_all.sh uni
VENV=/path/to/venv sbatch --job-name=sidnet_nuni scripts/run_all.sh nuni
```

- **Two jobs:** the two regimes are independent and can run as two jobs at the same time.
- **Paths:** set with the environment variables `DATA`, `CKPT`, `RESULTS` and `WORKERS` (defaults: `data`, `checkpoints`, `results`, `8`).
- **Restarting:** every step can be restarted. Finished simulations are skipped. If training stops, for example at a job time limit, the same command continues from the last finished epoch. Use `--no_resume` with `02_train.py` to start again from scratch.

## Reproducibility

- **Data:** the simulations depend only on their seeds, so regenerating the datasets gives bit-identical files.
- **Training:**
  - The model initialisation, the order in which samples are drawn and dropout are seeded (`--seed`, default 0).
  - The sampling order and dropout of each epoch depend only on the seed, the phase and the epoch, so a run that is stopped and resumed follows the same sequence as one that is not.
  - On a CPU, two runs with the same seed give identical weights.
  - On a GPU, cuDNN autotuning and TF32 are used by default for speed. `--deterministic` switches them off and uses deterministic kernels, at some cost in speed.
- **Run record:** each training run records its software versions, GPU, git commit and settings in `checkpoints/<regime>_run_info.json`.

## Outputs

| Path | Contents |
|---|---|
| `data/<regime>_*.h5`, `data/<regime>_reference.h5` | simulations and the reference z̄_ref |
| `checkpoints/<regime>_phase<P>_best.pt` | best model of each phase; `<regime>_phaseB4_best.pt` is the final model |
| `checkpoints/<regime>_phase<P>_history.json` | training loss and validation metrics after every epoch |
| `checkpoints/<regime>_last.pt` | full training state, used to resume |
| `checkpoints/<regime>_run_info.json` | software, hardware and settings of the run |
| `results/<regime>_<split>_metrics.json`, `results/<regime>_<split>_confusion.csv` | evaluation results |
| `logs/` | job logs |

## Citation

If you use this code or the SID datasets, please cite the paper (see also `CITATION.cff`):

```bibtex
@article{salvi_sandpile_identification,
  title   = {The Sandpile Identification Problem: Recovering Directional Noise from Space-time Dynamics},
  author  = {Salvi, Shrey and Shelat, Shlok and Roy, Souvik},
  year    = {2026}
}
```

## License

MIT, see `LICENSE`.
