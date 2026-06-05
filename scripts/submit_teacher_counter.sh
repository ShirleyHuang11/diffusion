#!/usr/bin/env bash
# Submit the counter-scope hybrid teacher pipeline (GPU) once the
# h4_rnd_counter fallback rung checkpoint exists. Mirrors the forced-scope
# pipeline parameters exactly (d256/L6/h8, 60k steps, window 16).
set -euo pipefail

PYTHON=${REAP_PYTHON:-/n/home12/shirleyhuang/conda-envs/grove-marl/bin/python}
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$REPO_DIR/slurm-logs"

sbatch <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=teacher_hybrid_counter
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=${REPO_DIR}/slurm-logs/teacher_hybrid_counter_%j.out
cd "$REPO_DIR"
PYTHONPATH=. exec "$PYTHON" scripts/teacher_pipeline.py \
  --hybrid \
  --layout counter_circuit \
  --vanilla-run runs/hardness_mappo_counter/seed0 \
  --rnd-run runs/h4_rnd_counter/seed0 \
  --teacher-steps 60000 \
  --d-model 256 --num-layers 6 --nhead 8 \
  --samples-per-state 8 \
  --window 16 \
  --device cuda
EOF
