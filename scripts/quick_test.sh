#!/bin/bash
# Pipeline test on a CPU in a few minutes: tiny datasets (one short simulation per class
# and split), one short epoch per phase, evaluation, Fig. 11, and the Sec. 3 tables and
# figures. The numbers it prints only show that every step runs; they are not results.
set -euo pipefail
cd "$(dirname "$0")/.."
python sidnet.py
for REGIME in uni nuni; do
    python 01_generate_dataset.py --regime $REGIME --out_dir data_quick --n_workers 4 --quick
    python 02_train.py --regime $REGIME --data_dir data_quick --out_dir checkpoints_quick \
                       --workers 0 --log_every 0 --quick
    python 03_evaluate.py --regime $REGIME --data_dir data_quick --out_dir results_quick \
                          --checkpoint checkpoints_quick/${REGIME}_phaseB4_best.pt --workers 0
done
python 04_plot_confusion.py --results results_quick --out results_quick/fig_confusion.pdf
python section3/make_tables.py > results_quick/section3_tables.txt
python section3/make_figures.py --out results_quick/figures_section3
echo "quick test passed"
