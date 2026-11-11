#!/bin/bash
#SBATCH --job-name=student-grpo
#SBATCH --partition=openlab-queue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --nodelist=gpgpu01
#SBATCH --output=logs/%A_%a.out
#
# Full single-A100 GRPO run: SFT-merged policy + fresh LoRA, style/content
# rewards (src/students/shared/grpo_rewards.py). Output: models/qwen3-grpo/ (adapter,
# run_config.json, reward_components.jsonl, checkpoint-*/). Run
# grpo_smoke.slurm first. Extra args go straight to train_grpo.py.
#
# Usage:
#   sbatch src/students/shared/hpc/train_grpo.slurm
#   FAMILY=gemma4 sbatch src/students/shared/hpc/train_grpo.slurm   # Gemma 4 E4B student (see train_grpo_gemma.slurm)
#   sbatch src/students/shared/hpc/train_grpo.slurm --max-steps 200 --style-weight 0.6 --content-weight 0.4

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -f pyproject.toml ]] || { echo "Submit from the repo root: sbatch <path to this .slurm>" >&2; exit 1; }
source src/students/shared/hpc/grpo_common.sh
grpo_preflight

run_in_container '
    $PY -m students.shared.merge_sft_adapter --family "$FAMILY"
    $PY -m students.shared.train_grpo --family "$FAMILY" "$@"
' "$@"
