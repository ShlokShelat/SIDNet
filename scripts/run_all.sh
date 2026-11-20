#!/bin/bash
# SIDNet pipeline for one regime: generate the dataset, train, evaluate.
#
#   bash   scripts/run_all.sh uni          # SID-Uni
#   sbatch scripts/run_all.sh nuni         # SID-nUni as a SLURM job (submit from the repository root)
#
# Settings (environment variables): DATA, CKPT, RESULTS, WORKERS, VENV.
# Every step can be restarted: finished simulations are skipped, and training
# continues from the last finished epoch.
#
#SBATCH --job-name=sidnet
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=168:00:00
#SBATCH --output=logs/sidnet_%x_%j.out

set -euo pipefail
REGIME=${1:?usage: run_all.sh uni|nuni}
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
DATA=${DATA:-data}
CKPT=${CKPT:-checkpoints}
RESULTS=${RESULTS:-results}
WORKERS=${WORKERS:-8}
[ -n "${VENV:-}" ] && source "$VENV/bin/activate"
mkdir -p logs "$DATA" "$CKPT" "$RESULTS"

echo "SIDNet [$REGIME]  host $(hostname)  start $(date)"
python sidnet.py

echo "== 1/3 dataset ($(date))"
if [ -f "$DATA/${REGIME}_reference.h5" ]; then
    echo "dataset found in $DATA, skipping generation"
else
    python -u 01_generate_dataset.py --regime "$REGIME" --out_dir "$DATA" --n_workers "$WORKERS"
fi

echo "== 2/3 training ($(date))"
python -u 02_train.py --regime "$REGIME" --data_dir "$DATA" --out_dir "$CKPT" --workers "$WORKERS"

echo "== 3/3 evaluation ($(date))"
for SPLIT in val test; do
    python -u 03_evaluate.py --regime "$REGIME" --split $SPLIT --data_dir "$DATA" --out_dir "$RESULTS" \
                             --checkpoint "$CKPT/${REGIME}_phaseB4_best.pt" --workers "$WORKERS"
done
echo "done $(date): results in $RESULTS/"
