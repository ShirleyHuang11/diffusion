#!/usr/bin/env bash
# Submit multi-seed training runs to SLURM.
#
# Usage: scripts/submit_runs.sh <config.yaml> [seeds...]
# Example: scripts/submit_runs.sh configs/gate_mappo_cramped.yaml 0 1 2
set -euo pipefail

CONFIG=${1:?usage: submit_runs.sh <config.yaml> [seeds...]}
shift
SEEDS=("${@:-0}")
[ $# -eq 0 ] && SEEDS=(0 1 2)

PYTHON=${REAP_PYTHON:-/n/home12/shirleyhuang/conda-envs/grove-marl/bin/python}
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
NAME=$(basename "$CONFIG" .yaml)
mkdir -p "$REPO_DIR/slurm-logs"

for SEED in "${SEEDS[@]}"; do
  sbatch <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${NAME}_s${SEED}
#SBATCH --partition=${REAP_PARTITION:-sapphire}
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=36:00:00
#SBATCH --output=${REPO_DIR}/slurm-logs/${NAME}_s${SEED}_%j.out
cd "$REPO_DIR"
PYTHONPATH=. exec "$PYTHON" -m reap.train --config "$CONFIG" --seed "$SEED"
EOF
done
